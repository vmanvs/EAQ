# Evaluation protocol

This document defines the intended evaluation. It does not report experimental
results. Model, benchmark, precision backend, and perturbation severities remain
to be selected and recorded before comparative evaluation.

## Reference and deployment conditions

Establish a working pretrained reference policy before applying quantization.
Record source and checkpoint revisions, preprocessing, action normalization,
controller semantics, and evaluation settings.

Begin with four conditions: nominal, camera noise, actuator lag, and their
combination. Run the reference policy in every condition to distinguish
domain-shift failures from quantization damage.

Define camera corruption by its distribution, magnitude, affected cameras,
injection point, temporal correlation, and random seed. Define actuator lag at
a controller signal with explicit units, update rate, initialization, clipping,
and reset behavior. Validate that zero perturbation reproduces the nominal path.

## Comparisons

- Reference precision.
- Uniform supported low precision.
- Action-sensitive mixed precision using identical calibration data.
- A universal allocation across calibration conditions.
- Physical-context allocation, if the sensitivity study supports it.

Match checkpoint, candidate modules, resource budget, calibration samples, and
access to simulator information. Report additional calibration-compute costs.
Evaluate complete allocations because module errors may interact.

## Data separation

Separate calibration, validation, and frozen test episodes by trajectory or
initial state. Pair initial states and random streams across methods. Once
trajectories diverge, paired seeds do not imply identical observations.

Use the pilot to plan sample size around a declared useful effect. Report
per-task outcomes and uncertainty; small debugging samples cannot establish
small performance improvements. Reserve unseen perturbation severities for
generalization checks.

## Reporting

Report absolute task success alongside the difference between reference and
quantized success in each condition. Preserve failures and termination reasons.
Document model/runtime revisions, precision manifests, seeds, and resource
measurement procedures with each experiment.

Distinguish fake quantization from packed execution. Memory and speed claims
require measurements of the deployed implementation, including metadata,
unquantized modules, and runtime buffers. Separate model inference latency from
preprocessing and simulation time.

Simulation results establish behavior only under the specified simulated
conditions. Cross-robot transfer and physical hardware robustness require
separate evidence.
