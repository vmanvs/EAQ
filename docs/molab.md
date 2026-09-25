# Molab preflight

[Molab](https://molab.marimo.io/) is marimo's hosted notebook service and the
current target for EAQ experiments. The preflight notebook checks the
environment before any full model download or LIBERO rollout. It is a marimo
`.py` file; GitHub is the source of truth for its code.

## Running the preflight

1. Push `notebooks/01_molab_preflight.py`, `scripts/cloud_preflight.py`, and
   `requirements/preflight.txt` to the same branch.
2. In Molab, use the **new notebook** dropdown to add the notebook from its
   GitHub URL. Molab re-syncs it after each push. Open the **Server** preview;
   the static and WebAssembly previews cannot use the GPU.
3. Attach the GPU with the **notebook specs** button in the header.
4. From the sidebar's **Packages** panel, install the missing preflight
   packages pinned in `requirements/preflight.txt`, typically `mujoco`,
   `cmake`, and `ninja`. Keep Molab's PyTorch. Molab sessions are temporary,
   so repeat this in each new session.
5. Add a read-only `HF_TOKEN` to Molab **Secrets**. The account needs
   approved access to `facebook/dinov3-vitb16-pretrain-lvd1689m` (LaWAM) and
   `google/paligemma-3b-pt-224` (Pi 0.5 tokenizer). Never put the token in a
   cell or commit.
6. Click **Install CUDA build headers**, then **Run bounded preflight**. The
   run stops after at most 20 minutes.
7. Click **Download preflight artifacts (.zip)** before the session ends, even
   if checks fail.

The ZIP contains `report.json` (every check with its details, plus the
ActQuant verdict), `manifest.json` (commit, script source, header install),
`packages.txt`, `preflight.log`, and `render.png` when headless rendering
works. Review logs before sharing them.

### How the notebook gets its scripts

Molab's GitHub sync brings down only the notebook file, not the rest of the
repository. The notebook therefore downloads `scripts/cloud_preflight.py` and
`requirements/preflight.txt` through the GitHub API. It resolves the branch to
one commit first and records that commit as the run's `eaq_commit`, so a later
push cannot silently change a finished run. Set `EAQ_GIT_REF` (default `main`)
or `EAQ_GITHUB_REPO` in Molab Secrets to change the source. A private
repository, or an exhausted unauthenticated GitHub API limit (60 requests per
hour per IP), needs a read-only `GITHUB_TOKEN`. The inventory's repository line
shows where the scripts came from, or why fetching failed.

`cloud_preflight.py` is deliberately a single self-contained file: Molab's
Python may run with `PYTHONSAFEPATH`, which prevents a script from importing
modules that sit next to it.

## What the preflight checks

General checks: CUDA discovery, FP16 and BF16 attention, MuJoCo EGL rendering,
and pinned Hugging Face model access. The `actquant_*` checks establish whether
[ActQuant](https://github.com/arashakb/ActQuant)'s Pi 0.5 pipeline, a
llama.cpp/ggml fork built with CMake and nvcc, can be built and run:

| Check | What it establishes |
| --- | --- |
| `actquant_toolchain` | `nvcc` (system or pip-installed) and its CUDA headers, CMake ≥ 3.18, g++, ninja/make, git, conda, uv |
| `actquant_system` | Linux, root or passwordless sudo, apt and its package lists, libEGL, Python 3.8/3.11 |
| `actquant_disk` | About 60 GiB writable scratch, or that the sandbox cannot report it |
| `actquant_cuda_nvcc` | A small FP16 kernel compiled and run for the GPU's own architecture, and as compute_80 PTX |
| `actquant_cuda_cmake` | A CMake project using `find_package(CUDAToolkit)` and cuBLAS, built and run |
| `actquant_local_server` | Loopback sockets, used by the Pi 0.5 policy server (not Molab's usage policy) |
| `actquant_sources` | ActQuant, openpi, and LIBERO reachable, and whether each is still at its pinned commit |
| `access_actquant_*`, `access_pi05_*`, `access_paligemma_*` | Pinned checkpoint and gated-tokenizer access |

`actquant_verdict` summarizes them as `ready`, `ready_with_adaptations`, or
`blocked`, listing each blocker and each deviation from ActQuant's documented
setup that a run must record. Temporary build folders are deleted; only their
output is kept. To run just these checks on another Linux GPU host:

```bash
python scripts/cloud_preflight.py --output artifacts/actquant-probe --only actquant
```

## Observed Molab environment

Observed on 2026-09-25 (EAQ commit `1ad08ef`); verdict
`ready_with_adaptations`, all checks passing. Molab can change any of this.

| Item | Observation |
| --- | --- |
| GPU | NVIDIA RTX PRO 6000 Blackwell Server Edition, 95 GiB, compute capability 12.0 (sm_120) |
| Driver | 595.71.05, supporting CUDA 13.2 |
| Host | gVisor sandbox, Debian 13, glibc 2.41, 160 GiB RAM, root with apt |
| Python | 3.13; PyTorch 2.11.0 built for CUDA 13.0 |
| CUDA toolkit | No system toolkit. Pip packages install a mixed toolkit under `site-packages/nvidia/cu13`: nvcc 13.3 with the 13.0 runtime headers PyTorch pins |
| Disk | Not measurable: the sandbox reports a placeholder size |

Five behaviours shaped the recipe below:

- **Two Python environments.** The Packages panel installs into an overlay
  environment (`/tmp/uv-venv`), while PyTorch and its CUDA toolkit live in the
  base environment (`/usr/local/lib/python3.13/site-packages`). nvcc and CMake
  only search their own toolkit, so CUDA headers installed through the panel
  are invisible to them. The **Install CUDA build headers** button installs
  `nvidia-cuda-cccl` next to PyTorch's toolkit instead, with `--no-deps`.
- **Incomplete pip toolkit.** PyTorch pulls in nvcc but not the CCCL headers
  (`nv/target`, `include/cccl`) that `cuda_fp16.h` and CMake require.
- **Mixed pip toolkit versions.** nvcc is 13.3 (`nvidia-cuda-nvcc`), but the
  runtime headers are 13.0 (`nvidia-cuda-runtime`, pinned by PyTorch). Small
  kernels compile, but CCCL's CUB refuses the mix ("CUDA compiler and CUDA
  toolkit headers are incompatible"), which failed the first ActQuant build
  in ggml's `mean.cu`. The build therefore uses its own toolkit, pinned to
  one CUDA release.
- **nvcc newer than the driver.** Code compiled for sm_120 runs, but the
  driver cannot JIT-compile PTX produced by the newer nvcc ("PTX was compiled
  with an unsupported toolchain"). Builds must include sm_120 device code.
- **Libraries outside the loader path.** The pip toolkit ships only versioned
  libraries (`libcudart.so.13`) in a directory the dynamic loader does not
  search. Linking needs unversioned `lib*.so` links, and running needs an
  RPATH or `LD_LIBRARY_PATH`.

## ActQuant build recipe for Molab

Derived from the preflight and the first build, and applied by the build
notebook below.

1. Install `cmake` and `ninja` from the Packages panel.
2. Install the CUDA 13.0 wheels pinned in `requirements/actquant-cuda-toolkit.txt`
   with `--target` into a private folder, and use
   `TOOLKIT=<folder>/nvidia/cu13`. This leaves PyTorch's environment
   untouched, and the toolkit is not newer than the driver (13.2).
3. Create unversioned links for `$TOOLKIT/lib/lib*.so.*` in a scratch folder
   and pass it as `-DCMAKE_LIBRARY_PATH`.
4. Configure with `-G Ninja -DCMAKE_CUDA_COMPILER=$TOOLKIT/bin/nvcc
   -DCUDAToolkit_ROOT=$TOOLKIT -DCMAKE_CUDA_ARCHITECTURES=120
   -DCMAKE_BUILD_RPATH=$TOOLKIT/lib`.
5. Use uv instead of conda. Install Python 3.8 for the LIBERO client with
   `uv python install 3.8`. Run `apt-get update` before any apt install.
6. Record each item above as a deviation from ActQuant's documented setup
   (Ubuntu 22.04, CUDA 12.6, conda).

Open questions: whether ActQuant's own ggml kernels build and run under this
setup (the first build reached 40 of 234 steps before the version mix
stopped it), real scratch-disk capacity, and whether long policy-server rollouts fit
Molab's [usage restrictions](https://molab.marimo.io/pages/molab/restrictions)
and 12-hour session limit.

## Building ActQuant

`notebooks/02_actquant_build.py` applies the recipe to ActQuant itself. Like the
preflight, it fetches its script (`scripts/actquant_build.py`) and pins
(`requirements/actquant-build.txt`, `requirements/actquant-cuda-toolkit.txt`)
through the GitHub API.

1. Attach the GPU, open the **Server** preview, and install `cmake`, `ninja`,
   `huggingface_hub`, and optionally `pybind11` from the Packages panel. Do
   not install CUDA packages; the `toolkit` stage installs its own.
2. Choose stages and click **Run ActQuant build**. Output streams into the
   notebook while it runs.
3. Download the run artifacts before the session ends.

| Stage | What it does |
| --- | --- |
| `toolkit` | Installs the pinned CUDA 13.0 wheels (about 0.5 GB) into the work folder. It then checks that nvcc, the runtime headers and CCCL agree and that nvcc is not newer than the driver. It adds a `lib64 → lib` link, because the 13.0 nvcc wheel's `nvcc.profile` expects the system-install layout. Finally it compiles, links and runs a CUB + `cuda_fp16` kernel on the GPU |
| `source` | Fetches ActQuant at its pinned commit and unpacks `vendor/tokenizers-cpp.zip` |
| `configure` | Runs CMake with the recipe's flags and `LLAMA_CURL=OFF`; enables the `pi05.so` binding when `pybind11` and `Python.h` are available, and retries without it otherwise |
| `build` | Builds `pi05`, `llama-quantize` and, if enabled, `pi05.so`; checks with `ldd` that every library resolves |
| `download` | Fetches `pi05.gguf`, `tokenizer.model` and `norm_stats.json` of `ActQuant-Pi05-LIBERO-3bpw` at a pinned revision and verifies SHA-256 |
| `infer` | Runs the `pi05` CLI once on `CUDA0` with a synthetic image and a fixed prompt, and requires finite actions. If the binding was built, it also loads it and runs it twice, for warm latency and repeatability |

Sources, build trees and the checkpoint stay in a work folder (`EAQ_WORK_DIR`,
default `/tmp/eaq-actquant`) for the session. Stages can be rerun on their
own, and an interrupted build resumes where it stopped. The run ZIP holds
`report.json` (stage results, host and toolkit details, and the list of
deviations from ActQuant's documented setup), `manifest.json`, and the
configure, build and inference logs. Binaries and weights are not included.

The **CPU comparison** option runs the CLI on the CPU as well. The flow's noise
uses a fixed seed, so the two outputs should be close, though not identical.
The **no-VMM** option builds a separate tree with `GGML_CUDA_NO_VMM=ON`, in
case CUDA virtual memory management does not work under gVisor.

The smoke test shows that the runtime executes and produces finite actions on
this GPU. It is not a LIBERO observation and says nothing about task success.

Molab documentation: [GitHub mirroring and server previews](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github),
[GPU and session limits](https://docs.marimo.io/guides/molab/#compute), and
[storage behavior](https://marimo.io/pages/molab/storage).
