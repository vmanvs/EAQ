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

Early research development. Model and benchmark feasibility are being evaluated.
A reproducible policy baseline and quantization results are not yet available.

## Cloud validation

The first cloud run uses [the Molab preflight notebook](notebooks/01_molab_preflight.py)
to check GPU execution, headless rendering, and model-resource access before
downloading weights. It also probes whether the host can build and run
[ActQuant](https://github.com/arashakb/ActQuant)'s Pi 0.5 pipeline. After pushing it to GitHub, follow the
[Molab setup and run guide](docs/molab.md). This validates infrastructure; it
does not run the policy or benchmark. The earlier
[Kaggle preflight](docs/kaggle.md) remains available as an alternative.

## Experimental approach

1. Reproduce a reference policy on a defined task subset.
2. Validate sensing and actuator perturbations independently.
3. Measure quantization sensitivity across deployment conditions.
4. Compare precision allocations using held-out, paired episodes.
5. Report task success, uncertainty, and measured deployment resources.

See [the evaluation protocol](docs/evaluation-protocol.md) for comparison and
reporting requirements.
