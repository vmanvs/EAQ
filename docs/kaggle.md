# Kaggle validation

Import `notebooks/01_kaggle_preflight.ipynb` into Kaggle after pushing the notebook,
`scripts/cloud_preflight.py`, and `requirements/preflight.txt` to GitHub.

1. Enable Internet and choose a GPU accelerator.
2. Add the `HF_TOKEN` secret and grant the notebook access. The associated account
   needs approved access to `facebook/dinov3-vitb16-pretrain-lvd1689m`.
3. Set `EAQ_REF` to the pushed commit SHA. The default `main` is a convenience;
   every run records its resolved commit and uses a detached checkout.
4. Run the notebook from top to bottom.
5. Save the output ZIP before ending the session.

The launcher assumes the GitHub repository is public. Private repositories need
an authenticated checkout configured separately; never embed a token in a URL or
notebook cell. No repository push is performed by the notebook.

## Scope

The first notebook validates CUDA discovery, FP16 (and native BF16 when available)
SDPA against a simple FP32 calculation, MuJoCo EGL rendering, and access to pinned
model configuration files. It downloads no full checkpoints or datasets and
installs no FlashAttention. It preserves Kaggle's torch/torchvision installation.
Direct preflight dependencies are pinned; inherited and transitive package
versions are recorded, not fully locked. This is not the final model environment.

Its output contains `report.json`, `manifest.json`, `packages.txt`, `preflight.log`,
and, when rendering succeeds, `render.png`. Results are session artifacts, not
automatically committed research results. Paths under `/kaggle/working` must be
saved as notebook outputs or downloaded to survive session removal.

## Failure interpretation

| Check | Next step |
| --- | --- |
| CUDA unavailable | Select a GPU, restart, and rerun. |
| Attention failed | Inspect torch/CUDA versions and the GPU model; preserve the error. |
| Rendering failed | Inspect EGL/driver availability; do not interpret this as model failure. |
| DINO access failed | Confirm model approval and the notebook's enabled read token. |
| Public metadata failed | Check Internet and service availability. |

A pass is permission to investigate model loading, not evidence of model fit or
successful robot control. Full inference and LIBERO evaluation remain pending.
