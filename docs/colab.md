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
| Rendering | Only Mesa's EGL is registered. MuJoCo renders with llvmpipe on the CPU |

## First result

Smoke run `20260928-202117-t1` (`libero_spatial`, one trial per task):

- **Result:** 10/10 successes, status `rollout_ok`, 95% Wilson interval
  [0.72, 1.00]. There were no server problems or client exceptions.
- **Stage times:** runtime 74 s, client 78 s, server 18 s, rollout 887 s.
- **Session:** the VM stayed up for the whole run, more than 35 minutes.

Ten episodes say the pipeline works; they cannot be compared with ActQuant's
reported 98.2% over 500 episodes.

**Speed is the open problem.**

- An episode takes about 88 s.
- Policy inference is 622 ms per call on the T4, which is about 17% of the
  rollout time.
- Most of the rest is the simulator, above all rendering the camera images with
  llvmpipe on the single core: the render probe measured 516 ms per step.
- At this speed, one suite at 50 trials per task (500 episodes) takes about as
  long as Colab's 12-hour session limit, and `libero_10`'s longer tasks take
  more.

Rendering on the GPU through NVIDIA's EGL library, if the VM provides it, is
being investigated first. A full suite can also be split across several
sessions.
