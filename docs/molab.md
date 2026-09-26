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
| Host | gVisor sandbox, Debian 13, glibc 2.41, root with apt. The sandbox reports 20 CPUs and 160 GiB, but Molab's documentation allots each notebook 4 CPUs and 32 GiB |
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

## ActQuant runtime

ActQuant's Pi 0.5 runtime is compiled in GitHub Actions and only run on
Molab. Compiling it in a Molab notebook failed repeatedly. Every attempt,
with 2 to 20 parallel jobs, ended with the whole sandbox reset partway
through, within a few minutes of starting. Memory stayed low throughout: 3–5 GiB
resident and about 2 GiB of files. Molab does not document a cause. Its
[restrictions](https://molab.marimo.io/pages/molab/restrictions) allow
terminating notebooks used for "non-interactive jobs", and a long compile
launched from one button resembles such a job. Repeated terminations also risk
the account, so heavy compiles stay off Molab.

### Building the package (GitHub Actions)

The **ActQuant build** workflow (`.github/workflows/actquant-build.yml`) is
started by hand from the repository's **Actions** tab. It runs on a free
GitHub-hosted runner, in a `python:3.13-slim-trixie` container. That image
matches Molab's Debian 13, glibc 2.41, gcc 14 and Python 3.13, so the
binaries and the `pi05.so` binding load there. It then runs
`scripts/actquant_build.py` with these settings:

- **Toolkit:** the private CUDA 13.0 toolkit from the wheels pinned in
  `requirements/actquant-cuda-toolkit.txt`.
- **No GPU on the runner:** `--no-gpu --cuda-arch 120`. The toolkit probe is
  compiled but not run.
- **No VMM:** `--no-vmm` (`GGML_CUDA_NO_VMM=ON`). ggml links the CUDA driver
  library only for virtual memory management, and the runner has no driver.
  ggml then allocates GPU memory without VMM.
- **Portable CPU code:** with `--no-gpu` the script also sets `GGML_NATIVE=OFF`,
  so ggml's CPU code targets a portable AVX2 baseline rather than the runner's CPU.
- **FlashAttention stubs:** `--no-flash-attn` (`GGML_CUDA_FA=OFF`, a workflow
  input, on by default) compiles ggml's FlashAttention CUDA kernels as stubs.
  ActQuant's `tools/pi0.5` never calls `ggml_flash_attn_ext`.
- **Package:** the `package` stage collects `pi05`, `llama-quantize`,
  `pi05.so` and the ggml/llama libraries. It sets their RUNPATH to
  `$ORIGIN:$ORIGIN/../cuda/lib` and writes `eaq-package.json` (commits,
  options, toolchain, deviations, and per-file SHA-256). Then it creates a
  tarball, installs it and checks with `ldd` that every library resolves.
- **CPU smoke test:** downloads the checkpoint and runs one CPU inference
  from the installed package (a workflow input, on by default).

A second job publishes the tarball, its `.sha256` and `report.json` as a
GitHub release. The run summary shows a pin to copy into
`requirements/actquant-prebuilt.json`. Once the pin is committed, notebook 02
uses the new package. The fetch stage refuses a download whose SHA-256
differs from the pin.

### Running on Molab

`notebooks/02_actquant_build.py` fetches its script and pins through the
GitHub API, like the preflight.

1. Attach the GPU, open the **Server** preview, and install `huggingface_hub`
   from the Packages panel. Do not install CUDA packages.
2. Click **Run ActQuant**. Output streams into the notebook; the run takes a
   few minutes.
3. Download the run artifacts before the session ends.

| Stage | What it does |
| --- | --- |
| `toolkit` | Installs the pinned CUDA 13.0 wheels (about 0.5 GB) into the work folder. It checks that nvcc, the runtime headers and CCCL agree, and that nvcc is not newer than the driver (13.2). It adds a `lib64 → lib` link, because the 13.0 nvcc wheel's `nvcc.profile` expects the system-install layout. It then compiles, links and runs a CUB + `cuda_fp16` kernel on the GPU |
| `fetch` | Downloads the pinned release package, verifies its SHA-256, and unpacks it into the work folder. Links its `cuda` folder to the toolkit and checks with `ldd` that every library resolves. Refuses a package built for another ActQuant commit, GPU architecture or CUDA version. Adds the package's build-time deviations to the report |
| `download` | Fetches `pi05.gguf`, `tokenizer.model` and `norm_stats.json` of `ActQuant-Pi05-LIBERO-3bpw` at a pinned revision and verifies SHA-256 |
| `infer` | Runs the `pi05` CLI once on `CUDA0` with a synthetic image and a fixed prompt, and requires finite actions. If the package has the binding, it also loads `pi05.so` and runs it twice, for warm latency and repeatability |

The toolkit, package and checkpoint stay in a work folder (`EAQ_WORK_DIR`,
default `/tmp/eaq-actquant`) for the session, so stages can be rerun on their
own. The script runs as a separate process writing to a log file. The status
cell refreshes itself and picks a run up again after a reconnect, and the run
ZIP can be downloaded at any time. The ZIP holds `report.json` (stage results,
host and toolkit details, and the deviations from ActQuant's documented
setup), `manifest.json`, and the inference logs. Binaries and weights are not
included.

The **CPU comparison** option runs the CLI on the CPU as well. The flow's noise
uses a fixed seed, so the two outputs should be close, though not identical.

The smoke test shows that the runtime executes and produces finite actions on
this GPU. It is not a LIBERO observation and says nothing about task success.

Deviations from ActQuant's documented setup (Ubuntu 22.04, CUDA 12.6, conda)
are all listed in the report. The main ones:
- the pinned pip CUDA 13.0 toolkit;
- sm_120 device code only;
- `LLAMA_CURL=OFF`;
- `GGML_CUDA_NO_VMM=ON`;
- `GGML_NATIVE=OFF`;
- FlashAttention stubs;
- `pi05.so` for Python 3.13, where ActQuant's server uses 3.11;
- a runtime built in CI rather than on the GPU host.

## LIBERO rollouts

`notebooks/03_libero_rollout.py` runs closed-loop LIBERO episodes of the
released 3-bit checkpoint the way ActQuant evaluates it. ActQuant's
`tools/pi0.5/serve_policy.py` wraps the `pi05.so` binding in a WebSocket
policy server, and openpi's `examples/libero/main.py` drives the LIBERO
simulator as its client. Both files run unmodified. They are fetched at pinned
commits (ActQuant `b647911`, openpi `215abfb`) and checked against pinned
SHA-256 digests in `scripts/libero_rollout.py`.

1. Attach the GPU, open the **Server** preview, and install `huggingface_hub`
   from the Packages panel. Molab ships `uv`; install it there only if the
   notebook reports it missing.
2. Choose stages, suites and trials per task, then click **Run LIBERO
   rollout**. Start with one trial per task on `libero_spatial` (10 episodes).
3. Download the run ZIP. It holds `report.json`, `episodes.jsonl` (one line
   per episode), the client and server logs, the render-probe frame, the first
   image the server received, and `main.py`'s replay videos.

| Stage | What it does |
| --- | --- |
| `runtime` | Runs `actquant_build.py --stages toolkit fetch download`: the CUDA runtime, the pinned package and the checkpoint, reusing the work folder |
| `client` | Installs Python 3.8 with uv and `requirements/libero-client.txt` without dependency resolution. Fetches LIBERO at openpi's submodule commit (`f78abd6`) and writes its config file, which LIBERO otherwise asks for on stdin. Renders one `libero_spatial` scene with EGL, then Mesa EGL, then OSMesa; if none works it installs Mesa with apt and tries again |
| `server` | Creates a venv for the Python `pi05.so` was built for, with `requirements/libero-server.txt`. Starts `serve_policy.py` on `CUDA0` with 10 flow steps, as `run_libero_eval.sh` does, and sends three observations over the openpi protocol |
| `rollout` | Starts the server and runs `main.py` once per suite with `--args.port`, 5 replan steps and seed 7. Records every episode |

**Validity.** `serve_policy.py` answers a failed inference with zero actions,
which it then unnormalizes, so the client cannot tell them from real ones. The
script watches the server log during the rollout. Any `Inference failed`
line, handler error or missing normalization stats, or a server exit, stops
the run. A suite is valid only if none of those happened, no episode ended in
a client exception, and every expected episode finished. Success rates come
with a 95% Wilson interval and ActQuant's reported rate for 500 trials
(`libero_spatial` 98.2%, `libero_object` 98.8%, `libero_goal` 95.0%,
`libero_10` 87.2%).

**Client environment.** `requirements/libero-client.in` lists the client's
direct requirements. It is locked for Python 3.8 with openpi's
`examples/libero/requirements.txt` as constraints, so openpi's versions are
kept, with these exceptions:
- torch 2.4.1 CPU instead of 1.11.0 cu113. LIBERO only uses torch to load its
  init states, and glibc 2.41 or newer refuses to load 1.11's
  `libtorch_cpu.so`, which requests an executable stack.
- `opencv-python-headless` instead of `opencv-python`, which needs libGL.
- robosuite's keyboard teleoperation packages (`pynput`, `evdev`,
  `python-xlib`) are left out; `evdev` has no wheels.
- Only LIBERO's runtime requirements are installed, not its training ones.

Other deviations the report records:
- the server runs in a Python 3.13 venv instead of openpi's 3.11;
- a single server on one GPU, where ActQuant's launcher shards tasks across
  one server per GPU;
- fewer than 50 trials per task, when chosen;
- the rendering backend, if it is not EGL.

A full suite at 50 trials per task is 500 episodes and takes hours. Molab
resets sandboxes that run long jobs, and a reset wipes the work folder and
the run, so run suites in pieces and download each ZIP.

Molab documentation: [GitHub mirroring and server previews](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github),
[GPU and session limits](https://docs.marimo.io/guides/molab/#compute), and
[storage behavior](https://marimo.io/pages/molab/storage).
