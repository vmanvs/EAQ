# Embodiment-Aware Quantization

Investigating how sensing conditions and actuator dynamics should influence
precision allocation in pretrained world-action models.

## Research question

At a fixed deployment resource budget, can calibration that accounts for sensor
uncertainty and mechanical response preserve closed-loop task performance better
than conventional or action-sensitive quantization?

## Scope

- Pretrained policies evaluated in simulation.
- Controlled camera corruption and actuator-response perturbations.
- Static mixed-precision allocation per deployment configuration.
- Comparisons at matched resource budgets and calibration-data access.

The initial study tests whether deployment conditions change module sensitivity
enough to make a different precision allocation useful. Condition-specific
allocations will be compared with a universal allocation across conditions.

## Status

Early research development; no quantization results yet.

The first feasibility target is the released Pi 0.5 pipeline from
[ActQuant](https://github.com/arashakb/ActQuant) on LIBERO. ActQuant allocates
precision by action sensitivity, which makes it the natural action-sensitive
comparison and starting point for this study. World-action models such as
[LaWAM](https://github.com/RLinf/LaWAM) remain a later target.

As of 2026-09-29, ActQuant's released 3-bit Pi 0.5 checkpoint runs closed-loop
on LIBERO through ActQuant's policy server and openpi's LIBERO client, both
unmodified, on a free Google Colab T4. A 10-episode `libero_spatial` smoke run
succeeded in 10 of 10 episodes, twice. That shows the pipeline works; it is not yet a
baseline measurement. With MuJoCo rendering on the GPU, an episode takes
about 18 s, so a full suite (500 episodes) fits in one Colab session. The next
milestone is reproducing ActQuant's reported success rates over full suites
(50 trials per task).

## Cloud validation

ActQuant's Pi 0.5 runtime is compiled by
[a GitHub Actions workflow](.github/workflows/actquant-build.yml) and
published as a release, one build per GPU architecture. Cloud GPU hosts only
download and run it.

- **LIBERO rollouts: Google Colab.** `cloud/colab_libero.py` runs
  [the rollout script](scripts/libero_rollout.py) on a Colab T4 through the
  Colab CLI and copies each run back. See the [Colab guide](docs/colab.md).
- **Environment checks: Molab.** [Molab](https://molab.marimo.io/), marimo's
  hosted notebook service, was the first target.
  - [The preflight notebook](notebooks/01_molab_preflight.py) checks GPU
    execution, headless rendering, model access, and whether ActQuant's
    pipeline can be built.
  - [The runtime notebook](notebooks/02_actquant_build.py) fetches the pinned
    package, downloads the 3-bit checkpoint, and runs one CUDA inference.
  - [The LIBERO rollout notebook](notebooks/03_libero_rollout.py) ran the
    first successful episodes. Molab's sandboxes reset a few minutes after
    starting, so rollouts moved to Colab. See the [Molab guide](docs/molab.md).
- **Other options.** `cloud/modal_libero.py` targets Modal's L4 GPUs; it
  needs a payment method on the Modal account and has not been run. The earlier
  [Kaggle preflight](docs/kaggle.md) remains available.

## Experimental approach

1. Reproduce a reference policy on a defined task subset.
2. Validate sensing and actuator perturbations independently.
3. Measure quantization sensitivity across deployment conditions.
4. Compare precision allocations using held-out, paired episodes.
5. Report task success, uncertainty, and measured deployment resources.

See [the evaluation protocol](docs/evaluation-protocol.md) for comparison and
reporting requirements.
