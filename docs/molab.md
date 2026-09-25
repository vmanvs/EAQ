# Molab preflight

Molab is the current target for interactive EAQ experiments. This first run
checks its environment before any full model download or LIBERO rollout. The
notebook is a marimo `.py` file; GitHub is the source of truth for its code.

## After pushing to GitHub

1. Confirm that `notebooks/01_molab_preflight.py`,
   `scripts/cloud_preflight.py`, and `requirements/preflight.txt` are on the
   same branch. Copy the pushed commit SHA for the run record.
2. Sign in to [Molab](https://molab.marimo.io/). On the home page, use the
   **new notebook** dropdown to add a notebook from GitHub. Paste the GitHub
   URL of `notebooks/01_molab_preflight.py`. Later pushes to that branch update
   the synced notebook. Open its **Server** preview or start it on an ephemeral
   server; the static and WebAssembly previews cannot run this GPU check.
3. In the running notebook, use the **notebook specs** button in the header to
   attach the GPU. Molab currently advertises an RTX Pro 6000 Blackwell with
   96 GB VRAM; record the device actually assigned to this run.
4. Review the notebook's runtime and package inventory. If a direct preflight
   dependency is missing, install the version listed in
   `requirements/preflight.txt` from the sidebar's **Packages** panel. On
   Molab that is typically `mujoco`, `cmake`, `ninja`, and `nvidia-cuda-cccl`
   (CUDA headers PyTorch omits); reinstall them in each new session. Keep Molab's installed PyTorch; it also supplies the CUDA
   13 toolkit (`nvcc` under `site-packages/nvidia/cu13`) that the ActQuant
   checks use. Do not install LaWAM's full requirements yet.
5. For the gated access checks, ensure the Hugging Face account has
   approved access to `facebook/dinov3-vitb16-pretrain-lvd1689m` (LaWAM) and
   `google/paligemma-3b-pt-224` (tokenizer for ActQuant's Pi 0.5). Set a
   read-only `HF_TOKEN` in Molab Secrets so it is available to the server
   process. Do not put the token in a cell, GitHub commit, or shared output.
6. Click **Run bounded preflight** once. It may run for several minutes and
   stops after 20 minutes. Read `report.json` and any failed-check summary.
   Click **Download preflight artifacts (.zip)** even if a check fails, before
   the session ends. Keep the ZIP and the pushed commit SHA together.

The ZIP contains `report.json`, `manifest.json`, `packages.txt`, and
`preflight.log`; it also contains `render.png` if headless rendering succeeds.
The manifest's commit field can be `unknown` if Molab's GitHub mirror does not
include Git metadata, so retain the SHA copied in step 1. Review logs before
sharing them publicly. Google Drive is a suitable place to upload the ZIP
manually for now; automatic Drive artifact transfer is a later step.

Molab's GitHub sync brings down only the notebook file, not the rest of the
repository. When `scripts/cloud_preflight.py` is not next to the notebook, the
notebook downloads `scripts/cloud_preflight.py` and
`requirements/preflight.txt` from `vmanvs/EAQ` through the GitHub API.
It first resolves the branch to a commit, then records that commit as the
run's `eaq_commit`. Set `EAQ_GIT_REF` (default `main`) or `EAQ_GITHUB_REPO`
in Molab Secrets to change the source. A private repository also needs a
read-only `GITHUB_TOKEN`. The repository line in the inventory shows where the
scripts came from, or why fetching failed. If CUDA is unavailable, check the GPU toggle and restart the server.
For attention, EGL, or Hugging Face failures, preserve the report and log so
the runtime or access issue can be diagnosed before downloading large weights.

The preflight checks CUDA discovery, FP16 and available BF16 attention,
MuJoCo EGL rendering, and pinned model configuration access. It does not
download full checkpoints or datasets, run LaWAM inference, or evaluate LIBERO.

## ActQuant probe

The same run probes whether the host can build and evaluate
[ActQuant](https://github.com/arashakb/ActQuant)'s Pi 0.5 path, which needs a
CUDA toolkit, CMake, a local policy server, and a Python 3.8 LIBERO client.
The `actquant_*` checks record:

| Check | What it establishes |
| --- | --- |
| `actquant_toolchain` | `nvcc` and version, CMake ≥ 3.18, g++, ninja/make, git, conda, uv |
| `actquant_system` | Linux, root or passwordless sudo, `apt-get` simulation of `libegl1-mesa-dev`, libEGL, Python 3.8/3.11 |
| `actquant_disk` | A writable location with about 60 GiB free |
| `actquant_cuda_nvcc` | A small FP16 kernel compiled and run natively for the GPU, and as compute_80 PTX (the fork's default JIT path on Blackwell) |
| `actquant_cuda_cmake` | CMake `find_package(CUDAToolkit)` plus a cuBLAS link, built and run |
| `actquant_local_server` | Loopback socket echo (technical only, not Molab's usage policy) |
| `actquant_sources` | ActQuant, openpi and LIBERO reachable; whether HEAD still matches the pin |
| `access_actquant_*`, `access_pi05_*`, `access_paligemma_*` | Pinned checkpoint and gated-tokenizer access |

`actquant_verdict` in `report.json` summarizes them as `ready`,
`ready_with_adaptations`, or `blocked`, listing each blocker and each
deviation from ActQuant's documented setup that a run must record. Compile
folders are deleted after each check; only their output is kept.

To run only the probe on another Linux GPU host, such as a self-hosted runner:

```bash
python scripts/cloud_preflight.py --output artifacts/actquant-probe --only actquant
```
A passing preflight is the gate for planning the Google Drive round trip and
reference rollout, not a policy result.

Molab documentation: [GitHub mirroring and server previews](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github),
[GPU and session limits](https://docs.marimo.io/guides/molab/#compute), and
[storage behavior](https://marimo.io/pages/molab/storage).
