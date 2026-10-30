# Reference models: what went wrong, and how it was found

ActQuant's 3-bit Pi 0.5 release runs closed-loop on LIBERO on a Colab T4 and
reproduces its reported `libero_spatial` success (489/500, 97.8%; ActQuant
reports 98.2%). To measure what the 3-bit quantization costs, the same
episodes must be run with a higher-precision model on the same runtime: the
release's unquantized parent (FP16) and a Q8_0 version of it. This document
records how those references were built, the four problems found along the
way, the evidence for each, and the first result from the corrected setup. The setup itself is described in
[the Colab guide](colab.md#reference-models).

## Status (2026-10-30)

| Item | State |
| --- | --- |
| Runtime fits FP16 on a T4 | Fixed by a loader patch (release r5) and verified on the GPU |
| Export metadata matches the release | Fixed (`action_horizon` 10) and checked automatically |
| Base checkpoint | **Corrected to `lerobot/pi05_libero_base`**. On the 10 smoke-test scenes, FP16 from it succeeded in 9 and Q8_0 in 10 (section 5) |
| FP16 against 3-bit, full `libero_spatial` | 489/500 for both; 7 against 7 discordant episodes, p = 1.0 (section 6) |
| fp16 accumulation in ggml's cuBLAS path | Real, but no measurable effect on the corrected FP16 model either. A patch is written but not built, and is not needed for now |

The runs before 2026-10-30 used the wrong base checkpoint, so they do not
measure the cost of quantization. They are kept below because they show how
each problem was found.

## 1. The runtime put three copies of the model on the GPU

**Symptom.** The FP16 export (6.25 GB) failed when the policy server started:
`cudaMalloc failed: out of memory`, while allocating 6,714,025,600 bytes after
the vision encoder and projector had taken 12.5 GB.

**Cause.** ActQuant's pi0.5 runtime loads three parts: the vision encoder, the
projector and the action expert. Each opens the whole `pi05.gguf`.
`BaseModel::load_tensors` (`tools/pi0.5/model_defs.cpp`) then allocated a
device buffer for every tensor in the file, filling only its own. GPU memory
was therefore three times the file size: about 7 GB for the 3-bit release, so
it fitted, but about 19 GB for FP16, against the T4's 15 GB.

The text embedding had the same pattern on the host:
`tools/pi0.5/text_embed.hpp` read the whole file into RAM
(`no_alloc = false`) to copy one table out of it, and kept the file in memory
for the life of the process.

**Fix.** `ACTQUANT_PATCH` in `scripts/actquant_build.py`, applied by the
source stage:

- each part builds a context holding only the tensors it loads, and allocates
  that;
- the text embedding reads `embed.weight` from its file offset.

Packages built with it are named `...-loadfix`. The sm_75 pin is release r5.

**Verification.**

- **Unchanged output on CPU.** The r5 build's test inference (3-bit model, CPU)
  gave exactly the same actions as the unpatched r4 build.
- **FP16 fits on the T4.** The FP16 model now loads on the T4. The patched
  loader reports 437, 2 and 370 tensors for the three parts, and inference
  takes about 0.5 s.

## 2. The export used a different action horizon

**Symptom.** The first FP16 run reached 4/10 on ten starting scenes the 3-bit
release solved 10/10. For the same probe input, the 3-bit server returned
actions of shape (10, 7) and the FP16 server (50, 7).

**Cause.** `export_pi05.py` takes `action_horizon` from `config.json`, falling
back to `chunk_size`. The lerobot config has `chunk_size` 50, while the
release's header has `pi05_action.action_horizon = 10` (openpi's
`pi05_libero` value).

**Fix.**

- **Release horizon.** The export reads a `config.json` with
  `"action_horizon": 10` added (`EXPORT_CONFIG`). ActQuant's script is not
  modified.
- **Metadata check.** The structure check now compares every metadata key with
  the release's header, not only the tensor names. It fails on any
  difference other than the quantization keys, the vision mean/std
  (`[0.5, 0.5, 0.5]` against `[0.5]`) and the `GGUF.*` header fields the
  release's merge step copied in.
- **Cache.** The override is recorded in the export manifest, so a cached
  export without it is rejected.

## 3. The export came from the wrong checkpoint

**Symptom.** With the horizon fixed, FP16 still reached only 5/10 on the same
scenes, and Q8_0 6/10. Their outputs for the probe input nearly coincide. Both
are far from the release's:

| Model | First action of the probe (raw, first 6 of 7) |
| --- | --- |
| 3-bit release | `0.426  0.611  0.215 -0.097  0.219 -0.233` |
| FP16, from v044 | `1.951  1.407  0.525 -0.041  0.463  0.156` |
| Q8_0, from v044 | `1.946  1.408  0.517 -0.042  0.474  0.151` |

Quantization error alone would not move the outputs this far.

**Cause.** ActQuant's `tools/pi0.5/02_prepare_models.sh` downloads
`lerobot/pi05_libero_finetuned_v044`, and the references were exported from
it. The release was not made from that checkpoint:

- **The weights match `pi05_libero_base`.** The release keeps the action
  expert and projector in F16 and the vision biases in F32, unquantized. Cast
  to the same types, `lerobot/pi05_libero_base` (`a217bfd`, F32) reproduces
  them bit for bit. The exception is `action.action_out.weight`, where the
  difference is within one fp16 rounding step.
- **v044 is a sibling fine-tune.** Its weights correlate with the release's at
  0.99, but differ by up to 0.04. `lerobot/pi05_libero_finetuned_quantiles_v044`
  behaves the same way.
- **v044 normalizes differently.** It was trained with mean/std normalization
  of state and actions (`normalization_mapping` in its config). The runs fed
  it quantile-normalized state and decoded its actions with quantile stats,
  because the release's `norm_stats.json` is quantile.
- **The release's stats are openpi's.** The release's `norm_stats.json` equals
  openpi's published `pi05_libero` stats
  (`gs://openpi-assets/checkpoints/pi05_libero/assets/physical-intelligence/libero/norm_stats.json`)
  to float32 rounding.

To reproduce the comparison (it reads a few MB with HTTP range requests):

```bash
python scripts/checkpoint_provenance.py lerobot/pi05_libero_base a217bfd3b14673cf2ce597e69997ab21866438dd
python scripts/checkpoint_provenance.py lerobot/pi05_libero_finetuned_v044 8e174154ef5f6c60a8da12ae99c303d8963138c1
```

| Tensor | `pi05_libero_base` | `pi05_libero_finetuned_v044` |
| --- | --- | --- |
| `action.action_out.weight` | max diff 0.000244 | max diff 0.0208, corr 0.997 |
| `action.blk.0.attn_q.weight` | identical | max diff 0.0402, corr 0.995 |
| `action.blk.17.ffn_down.weight` | identical | max diff 0.0338, corr 0.990 |
| `v.blk.0.attn_q.bias` | identical | max diff 0.0152 |
| `mm.0.weight` | identical | max diff 0.0224, corr 0.992 |

**Fix.**

- **Base checkpoint.** The export now uses `lerobot/pi05_libero_base` at
  `a217bfd3b14673cf2ce597e69997ab21866438dd` (`model.safetensors`, 14.5 GB,
  SHA-256 pinned).
- **Normalization stats.** openpi's `pi05_libero` `norm_stats.json` (SHA-256
  pinned) is placed where `export_pi05.py` looks for openpi assets, so the
  export writes the same `norm.*` tensors the release carries.
- **Stricter structure check.** The check no longer tolerates missing `norm.*`
  tensors.
- **Cache.** The Drive cache is keyed by the base revision, so v044 exports
  are ignored.

## 4. ggml accumulates F16 products in fp16

**Finding.** In ActQuant's ggml (`ggml/src/ggml-cuda/ggml-cuda.cu`), a matrix
product with F16 weights and a large batch takes the cuBLAS path. On NVIDIA
GPUs that path accumulates and writes in fp16 (`CUBLAS_COMPUTE_16F`), for
example over 16,384 terms in the Gemma MLP's down projection. The batched
cuBLAS path does the same for F16. Nothing in `tools/pi0.5` requests
`GGML_PREC_F32`. Quantized weights use ggml's int8 kernels (MMQ), which
accumulate in fp32.

**Effect here.** None measurable. FP16 (fp16 accumulation) and Q8_0 (fp32
accumulation), both from v044, produce nearly identical probe actions (table
in section 3).

**Status.** A patch is written: both cuBLAS paths accumulate and write in
fp32, as ggml already does on AMD CDNA. It would be named `loadfix-fp32acc`
and released as r6. It is not built. Whether it is needed will be decided
after the FP16 reference from the correct checkpoint has run.

## 5. The references from the corrected checkpoint

Runs `20261030-174710-t1-fp16` and `20261030-180829-t1-q8` exported FP16 and
Q8_0 from `pi05_libero_base` on the VM. Both passed the structure check,
including the `norm.*` tensors. Each export took 2–3 minutes, and the whole
preparation stage, download included, about 8 minutes. The exports are
cached on Drive as `eaq-cache/pi05-libero-{fp16,q8}-a217bfd-b647911`. The
Q8_0 file is 3.70 GB.

- **Episodes.** On the 10 scenes, FP16 succeeded in 9 (`111111111.`) and
  Q8_0 in all 10. FP16's failure, "pick up the black bowl on the wooden
  cabinet", is a task the release also failed once in 50 trials (49/50 in
  `20261002-175720-t50`). With a single discordant scene, exact McNemar
  p = 1.0: neither run shows a difference from the release.
- **Probe.** Both references now lie close to the release in the
  translation components, where the v044 exports were off by up to 1.5:

  | Model | First action of the probe (raw, first 6 of 7) |
  | --- | --- |
  | 3-bit release | `0.426  0.611  0.215 -0.097  0.219 -0.233` |
  | FP16, from `pi05_libero_base` | `0.381  0.451  0.180 -0.003  0.020 -0.001` |
  | Q8_0, from `pi05_libero_base` | `0.392  0.457  0.191 -0.004  0.021 -0.001` |
  | FP16, from v044 | `1.951  1.407  0.525 -0.041  0.463  0.156` |

  FP16 and Q8_0 differ by at most 0.012. The 3-bit release differs from
  both by up to 0.23, mostly in the rotation components. That is the
  difference the references exist to measure; one probe input cannot show
  how much it matters in closed loop.
- **fp32 accumulation.** FP16 runs its large matrix products through cuBLAS
  with fp16 accumulation, Q8_0 through MMQ with fp32 accumulation (section
  4). Their probe actions agree to 0.012, against a 0.23 gap to the release,
  so the r6 patch is not needed for this comparison.
- **Speed.** pi05 took 500 ms (FP16) and 494 ms (Q8_0) per inference (p50)
  during the rollout, against 563 ms for the 3-bit release. Successful
  episodes took 13–19 s.

## 6. FP16 against the 3-bit release on the full `libero_spatial` suite

Run `20261030-185844-t50-fp16` ran FP16 from `pi05_libero_base` on all 500
`libero_spatial` episodes (50 trials per task), with the same initial states
as the 3-bit run `20261002-175720-t50`. Both runs are valid, with no server
problems or client exceptions.

- **Result.** FP16 succeeded in 489 of 500 (97.8%, 95% Wilson interval
  96.1–98.8%), exactly the 3-bit release's total.
- **Paired.** The two models disagreed on 14 of the 500 initial states: 7
  solved only by the 3-bit release and 7 only by FP16. Exact McNemar
  p = 1.0. On this suite the 3-bit quantization costs nothing measurable.

| Task | 3-bit | FP16 | Only 3-bit | Only FP16 |
| --- | --- | --- | --- | --- |
| between the plate and the ramekin | 50 | 50 | 0 | 0 |
| next to the ramekin | 43 | 46 | 1 | 4 |
| from table center | 50 | 50 | 0 | 0 |
| on the cookie box | 49 | 50 | 0 | 1 |
| in the top drawer of the wooden cabinet | 50 | 49 | 1 | 0 |
| on the ramekin | 50 | 50 | 0 | 0 |
| next to the cookie box | 50 | 50 | 0 | 0 |
| on the stove | 48 | 50 | 0 | 2 |
| next to the plate | 50 | 50 | 0 | 0 |
| on the wooden cabinet | 49 | 44 | 5 | 0 |
| **Total** | **489** | **489** | **7** | **7** |

The discordant episodes cluster in two tasks, in opposite directions: FP16
did better on "next to the ramekin" and worse on "on the wooden cabinet",
where all 5 disagreements favour the 3-bit model (exact p ≈ 0.06 for that
task alone, chosen after seeing the data, so not evidence of a difference).
Whether quantization shifts *which* tasks fail, while leaving the total
unchanged, is a question for more trials or a harder suite.

`libero_spatial` is near its ceiling for both models (97.8%), so it can show
only large differences. `libero_10`, with longer tasks, is the more sensitive
test.

- **Speed.** pi05 took 496 ms per inference (p50). Episodes averaged 17.0 s,
  and failed ones 33.4 s. The rollout took 2.6 hours.

## Runs

All runs are on a Colab T4, `libero_spatial`. The "10 scenes" are trial 0 of
each of the 10 tasks, the same initial states in every run (`1` = success).

| Run | Model | Base | Runtime | 10 scenes | Notes |
| --- | --- | --- | --- | --- | --- |
| `20260928-202117-t1` | 3-bit release | — | r4 | `1111111111` | |
| `20261002-172743-t1` | 3-bit release | — | r4 | `1111111111` | pi05 p50 563 ms |
| `20261002-175720-t50` | 3-bit release | — | r4 | 489/500 | Wilson 95% 96.1–98.8% |
| `20261002-211525-t1-fp16` | FP16 | v044 | r4 | — | Out of GPU memory at server start (section 1) |
| `20261003-062154-t1-fp16` | FP16 | v044 | r5 | `111..1....` | horizon 50 (section 2) |
| `20261003-065227-t1-fp16` | FP16 | v044 | r5 | `1.11.1..1.` | horizon 10; pi05 p50 483 ms |
| `20261003-071712-t1-q8` | Q8_0 | v044 | r5 | — | VM lost in the server stage, no logs (below) |
| `20261003-081936-t1-q8` | Q8_0 | v044 | r5 | `1.11.11.1.` | pi05 p50 518 ms |
| `20261030-174710-t1-fp16` | FP16 | `pi05_libero_base` | r5 | `111111111.` | pi05 p50 500 ms (section 5) |
| `20261030-180829-t1-q8` | Q8_0 | `pi05_libero_base` | r5 | `1111111111` | pi05 p50 494 ms (section 5) |
| `20261030-185844-t50-fp16` | FP16 | `pi05_libero_base` | r5 | 489/500 | Paired with the 3-bit run: 7 against 7, p = 1.0 (section 6) |

On the same 10 scenes:

- FP16 from `pi05_libero_base` disagreed with the release once (p = 1.0),
  and Q8_0 from it not at all;
- the horizon-50 FP16 run disagreed with the release 6 times, every time
  failing where the release succeeded (exact McNemar p ≈ 0.03);
- the horizon-10 FP16 run disagreed 5 times;
- Q8_0 disagreed 4 times.

Ten episodes are enough to show that something is wrong, not to rank models.
The comparison itself uses 500 paired episodes per suite (section 6).

**The lost VM.** In `20261003-071712-t1-q8`, the server stage started and the
run printed nothing more. The Colab kernel stopped answering about 9 minutes
later, and the session was gone after about 23 minutes. The VM's disk, with
the server log, was lost with it. A retry of the same configuration passed the
server stage in 24 s. Since then, runs with `--drive` write the VM's free
memory, the largest processes, GPU memory and the end of `server.log` to
`MyDrive/eaq-cache/diag/<run>.log` every 10 s.

## Next

1. **FP16 smoke test.** Done: 9 of 10 (section 5).
2. **Q8_0 smoke test.** Done: 10 of 10 (section 5).
3. **Decide on fp32 accumulation.** Not needed for now: FP16 and Q8_0 agree
   to 0.012 on the probe (section 5). Revisit if the 500-episode runs show
   FP16 trailing Q8_0.
4. **Paired comparison.** FP16 on `libero_spatial`: done, 489/500, no
   difference from the 3-bit release (section 6). Still to run: Q8_0 on
   `libero_spatial`, then the 3-bit release, FP16 and Q8_0 on `libero_10`,
   on identical initial states, compared with McNemar's test on the
   discordant episodes.
5. **Pin the exports.** Record each export's SHA-256 in `EXPORT_SHA256`.
