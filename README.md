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

Early research development; no policy baseline or quantization results yet.

The first feasibility target is the released Pi 0.5 pipeline from
[ActQuant](https://github.com/arashakb/ActQuant) on LIBERO. ActQuant allocates
precision by action sensitivity, which makes it the natural action-sensitive
comparison and starting point for this study. World-action models such as
[LaWAM](https://github.com/RLinf/LaWAM) remain a later target.

As of 2026-09-25 the Molab environment check reports
`ready_with_adaptations` for that pipeline: its CUDA code can be compiled for
and run on Molab's Blackwell GPU, with recorded deviations from ActQuant's
documented setup. The next milestone is building ActQuant there and running a
single inference with its released 3-bit checkpoint, followed by a
reference-precision LIBERO rollout.

## Cloud validation

Experiments run on [Molab](https://molab.marimo.io/), marimo's hosted notebook
service. [The Molab preflight notebook](notebooks/01_molab_preflight.py) checks
GPU execution, headless rendering, model access, and whether ActQuant's
pipeline can be built, before any large download. See the
[Molab guide](docs/molab.md) for running it, the observed environment, and the
resulting build recipe. It validates infrastructure only; it does not run a
policy or benchmark. ActQuant's Pi 0.5 runtime is compiled by
[a GitHub Actions workflow](.github/workflows/actquant-build.yml) in a
Molab-matching container and published as a release. [The ActQuant runtime
notebook](notebooks/02_actquant_build.py) fetches that pinned package, downloads
the released 3-bit checkpoint, and runs one CUDA inference as a smoke test. The earlier [Kaggle preflight](docs/kaggle.md) remains
available as an alternative.

## Experimental approach

1. Reproduce a reference policy on a defined task subset.
2. Validate sensing and actuator perturbations independently.
3. Measure quantization sensitivity across deployment conditions.
4. Compare precision allocations using held-out, paired episodes.
5. Report task success, uncertainty, and measured deployment resources.

See [the evaluation protocol](docs/evaluation-protocol.md) for comparison and
reporting requirements.
