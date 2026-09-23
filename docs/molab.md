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
   `requirements/preflight.txt` with Molab's package manager. Keep Molab's
   installed PyTorch for this first check; do not install LaWAM's full
   requirements yet.
5. For the gated DINOv3 access check, ensure the Hugging Face account has
   approved access to `facebook/dinov3-vitb16-pretrain-lvd1689m`. Set a
   read-only `HF_TOKEN` in Molab Secrets so it is available to the server
   process. Do not put the token in a cell, GitHub commit, or shared output.
6. Click **Run bounded preflight** once. It may run for several minutes and
   stops after 15 minutes. Read `report.json` and any failed-check summary.
   Click **Download preflight artifacts (.zip)** even if a check fails, before
   the session ends. Keep the ZIP and the pushed commit SHA together.

The ZIP contains `report.json`, `manifest.json`, `packages.txt`, and
`preflight.log`; it also contains `render.png` if headless rendering succeeds.
The manifest's commit field can be `unknown` if Molab's GitHub mirror does not
include Git metadata, so retain the SHA copied in step 1. Review logs before
sharing them publicly. Google Drive is a suitable place to upload the ZIP
manually for now; automatic Drive artifact transfer is a later step.

If the notebook cannot find `scripts/cloud_preflight.py` and
`requirements/preflight.txt`, preserve the error and the Molab notebook URL.
That means the server did not expose the mirrored repository in the expected
layout. If CUDA is unavailable, check the GPU toggle and restart the server.
For attention, EGL, or Hugging Face failures, preserve the report and log so
the runtime or access issue can be diagnosed before downloading large weights.

The preflight checks CUDA discovery, FP16 and available BF16 attention,
MuJoCo EGL rendering, and pinned model configuration access. It does not
download full checkpoints or datasets, run LaWAM inference, or evaluate LIBERO.
A passing preflight is the gate for planning the Google Drive round trip and
reference rollout, not a policy result.

Molab documentation: [GitHub mirroring and server previews](https://docs.marimo.io/guides/molab/#mirror-notebooks-from-github),
[GPU and session limits](https://docs.marimo.io/guides/molab/#compute), and
[storage behavior](https://marimo.io/pages/molab/storage).
