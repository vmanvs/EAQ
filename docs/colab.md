# Colab LIBERO rollouts

LIBERO rollouts of ActQuant's released 3-bit Pi 0.5 checkpoint now run on a
Google Colab T4. They are driven from a terminal through the Colab CLI
(`google-colab-cli`), not from a browser notebook. `cloud/colab_libero.py`
runs `scripts/libero_rollout.py` unchanged on the Colab VM, with the same
stages, validity checks and report as on Molab (see
[LIBERO rollouts](molab.md#libero-rollouts)).

Molab is no longer used for rollouts. Its sandboxes reset a few minutes after
starting, even when the script was only sleeping (see
[Molab](molab.md#libero-rollouts)).

## Why the CLI and not a notebook

A Colab session is a VM running a Jupyter kernel. The browser notebook and the
CLI drive the same kind of VM, but what keeps it alive differs:

- A free browser session disconnects after a period without interaction in the
  tab, even while code is running.
- The CLI's sessions stay alive while the kernel is busy executing a cell.
- Both are limited to 12 hours per session and a GPU allowance Colab does not
  publish.
- In both, the VM's disk is erased when the session ends.

`cloud/colab_libero.py` therefore does three things:

- It starts the rollout as a detached process on the VM.
- It keeps the kernel busy with "watch" cells of at most `--watch-minutes`
  (default 15). Each cell streams the run's log and ends on its own.
- After each watch cell it downloads the run folder, without videos, so the
  local copy is never more than one cell behind.

## Setup

The Colab CLI runs on Linux and macOS; on Windows, use WSL.

```bash
uv tool install google-colab-cli
```

The first `colab new` opens a Google sign-in. Its token is stored under
`~/.config/colab-cli/`, outside the repository. No Hugging Face or GitHub token
is needed: the checkpoint and the runtime release are public.

## Running

```bash
python3 cloud/colab_libero.py --suites libero_spatial --trials 1
```

1. Creates the session (`colab new -s eaq-libero --gpu T4`), or reuses it if it
   is already running.
2. Uploads the two scripts, the sm_75 package pin, and the client, server and
   toolkit requirement locks to `/content/eaq`.
3. Starts `libero_rollout.py` with the runtime, client, server and rollout
   stages. Its work folder is `/content/eaq/work`.
4. Watches it and downloads each copy to `runs/colab/<run name>/`. When the run
   ends, it downloads the full folder including replay videos and prints each
   suite's result.
5. Stops the session, unless `--keep` is given.

| Option | Default | Meaning |
| --- | --- | --- |
| `--suites` | `libero_spatial` | Comma-separated suite names, or `all`; run in sequence |
| `--trials` | 1 | Trials per task; ActQuant uses 50 |
| `--stages` | `runtime,client,server,rollout` | `libero_rollout.py` stages; add `profile` to time inference before the rollout |
| `--profile-requests` | 20 | Requests per phase of the profile stage |
| `--model` | `actquant-3bpw` | `actquant-3bpw`, or a reference exported on the VM: `fp16` or `q8` (see [Reference models](#reference-models)) |
| `--drive` | off | Mount Google Drive and keep reference exports in `MyDrive/eaq-cache/`, so later sessions copy them instead of exporting again |
| `--session` | `eaq-libero` | Colab session name |
| `--attach RUN` | | Resume watching a run already started in the session |
| `--keep` | off | Leave the session running at the end |
| `--watch-minutes` | 15 | Length of each watch cell |
| `--suite-hours` | 11 | `libero_rollout.py` time limit per suite |
| `--client-threads`, `--server-threads` | 1, 1 | Thread caps; the T4 VM has one physical core |

If the local command is interrupted, the run keeps going on the VM. To resume
watching it, use `--attach <run name>` while the session is still alive. Once
the session ends, anything not yet downloaded is lost.

Check `colab sessions` afterwards, and run `colab stop -s eaq-libero` if a
session is still listed: a session left running counts against the GPU
allowance even when idle.

## Reference models

To measure what 3-bit quantization costs, the same episodes are run with a
higher-precision model on the same runtime. `actquant_build.py` exports it on
the VM from the release's base checkpoint (`lerobot/pi05_libero_finetuned_v044`)
with ActQuant's own `tools/pi0.5/export_pi05.py`, the script whose output the
3-bit release is built from. The download and export take about 5 minutes.

| `--model` | Export flags | Size | What it measures |
| --- | --- | --- | --- |
| `fp16` | none | 6.25 GB | The full-precision policy |
| `q8` | `--quant_llm q8 --quant_vision q8 --quant_embedding q8` | about 3.7 GB (estimated) | The parts the release quantizes to 2–3 bits, at Q8_0 instead; the action expert stays FP16 in both |

Both keep the 3-bit release's `norm_stats.json`, so only the weights differ.
Use `--drive`: a VM's disk is wiped when its session stops, so without it the
export is lost with the session.

As released, ActQuant's runtime cannot run `fp16` on a T4. Its vision encoder,
projector and action expert each open the whole `pi05.gguf` and allocated
every tensor in it on the GPU, so the GPU held three copies of the file: about
19 GB for FP16, against the T4's 15 GB. The source stage applies a patch
(`ACTQUANT_PATCH` in `actquant_build.py`) so that each part allocates only its
own tensors. The same patch has the text embedding read just its table rather
than loading the whole file into RAM, where it stayed for the life of the
process. Packages built with the patch have `-loadfix` in their name.

## Runtime package

The T4 is sm_75. Its package is built by the
[ActQuant build workflow](molab.md#building-the-package-github-actions) with
`cuda_arch=75` and pinned in `requirements/actquant-prebuilt-sm75.json`
(release r4). The package is built in a Debian 13 container but loads on
Colab's Ubuntu 24.04: the highest glibc symbol versions it needs are
GLIBC_2.38 and GLIBCXX_3.4.32, and Colab has 2.39 and 3.4.33.

| Pin | GPU | Used by |
| --- | --- | --- |
| `requirements/actquant-prebuilt.json` | sm_120, RTX PRO 6000 | Molab notebooks 02 and 03 |
| `requirements/actquant-prebuilt-sm75.json` | sm_75, T4 | `cloud/colab_libero.py` |
| `requirements/actquant-prebuilt-sm89.json` | sm_89, L4 | `cloud/modal_libero.py` |

`cloud/modal_libero.py` runs the same rollout on [Modal](https://modal.com/),
one L4 container per suite. It needs a Modal account with a payment method and
has not been run.

## Observed Colab environment

Observed on 2026-09-28 on the free tier. Colab can change any of this.

| Item | Observation |
| --- | --- |
| GPU | Tesla T4, 15 GB, compute capability 7.5 |
| Driver | 580.82.07, supporting CUDA 13.0 |
| Host | A full VM (not a gVisor sandbox), Ubuntu 24.04, glibc 2.39, root with apt |
| CPU, memory, disk | 2 vCPUs (one physical core), 12 GB RAM, about 66 GB free disk |
| Python | 3.13, with uv |
| Rendering | NVIDIA's EGL library is in `/usr/lib64-nvidia`, but only Mesa's vendor file is registered, so plain EGL renders with llvmpipe on the CPU. The client stage's `egl-nvidia` backend registers NVIDIA's library and renders on the T4 |

## Results

| | CPU rendering (2026-09-28) | GPU rendering (2026-10-02) |
| --- | --- | --- |
| Run | `20260928-202117-t1` | `20261002-172743-t1` |
| Renderer | Mesa llvmpipe | NVIDIA EGL on the T4 (`egl-nvidia`) |
| Render probe, per control step | 516 ms | 64 ms |
| `libero_spatial`, 1 trial per task | 10/10, `rollout_ok` | 10/10, `rollout_ok` |
| Time per episode | ~88 s | ~18 s (14.5–21.6 s) |
| Policy inference, per call (median) | 621 ms | 578 ms |
| Share of episode time in inference | ~17% | ~68% |

Ten episodes only show that the pipeline works. The full suite,
`20261002-175720-t50` (50 trials per task, GPU rendering), gives a number that
can be compared with ActQuant's:

- **Result:** 489/500 = 97.8%, 95% Wilson interval [0.961, 0.988]. ActQuant
  reports 98.2%, which is inside the interval. The run was valid, with no
  server problems or client exceptions.
- **Per task:** most failures were on "pick up the black bowl next to the
  ramekin and place it on the plate", at 43/50. "On the stove" had 48/50, and
  "on the cookie box" and "on the wooden cabinet" 49/50 each; the other six
  tasks were 50/50.
- **Time:** the rollout took 2.8 hours. Episodes averaged 18.8 s; failed ones
  averaged 36.3 s, because they run to the step limit.
 Rendering on the GPU is not a
deviation from ActQuant's setup, which renders with EGL. The NVIDIA renderer
can produce slightly different pixels from Mesa.

At about 18 s per episode, a full `libero_spatial` suite (500 episodes) takes
about 2.5 hours, well inside Colab's 12-hour session limit. `libero_10`'s
tasks are longer, so expect it to take a few times as long.

### Where an inference spends its time

pi05 prints a timing breakdown for every inference into the server log. The
`profile` stage (`--stages runtime,client,server,profile`) sends the server
three phases of requests, with no simulator running:
- **same:** one observation repeated;
- **distinct:** a new image every time;
- **distinct_cpu_load:** a new image every time, with a busy process on the CPU.

It matches each timing summary and an `nvidia-smi` sample to its phase.

| Part of one inference, distinct images (median) | ms |
| --- | --- |
| Vision encoder, two cameras | ~76 |
| Prefix forward: 18 language-model layers over 777 tokens | ~314 |
| Action expert: 10 flow steps of ~12 ms | ~123 |
| **Total** (round trip 521 ms) | **~515** |

- **Repeated observations are much faster.** A repeated observation returns in
  about 200 ms because pi05 skips the prefix forward when its input has not
  changed. Its printed summary then repeats the last prefix time, so latency
  from repeated inputs, as in the server stage's probe, understates real
  inference.
- **CPU load barely matters.** A busy CPU core adds about 8 ms.
- **The T4 hits its 70 W power limit.** Under load it draws its full 70 W, and
  `nvidia-smi` reports the software power cap as the clock limit. The SM
  clock runs at 1.1–1.3 GHz instead of its 1.59 GHz maximum.
- **Inference is slower inside a rollout.** It was 578 ms per call, against
  521 ms back to back. The GPU idles between requests while the simulator
  steps, and the warm-up phase shows its clock ramping up from 585 MHz. This
  explanation is inferred, not measured.
