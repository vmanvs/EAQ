"""Run LIBERO rollouts of ActQuant's 3-bit Pi 0.5 checkpoint through its pi05.so policy server.

This follows ActQuant's tools/pi0.5 evaluation: its serve_policy.py (a WebSocket policy server
wrapping the pi05.so binding) runs unmodified next to openpi's unmodified
examples/libero/main.py client. Both files are fetched at pinned commits and checked against
pinned SHA-256 digests. Stages run in order and each writes `report.json`, so a partial run
still leaves evidence:

  runtime  make sure the ActQuant runtime is in the work folder: runs actquant_build.py with
           --stages toolkit fetch download (pinned CUDA runtime, prebuilt package, checkpoint)
  client   Python 3.8 venv with the locked client requirements, LIBERO at openpi's pinned
           commit and openpi's main.py; picks a MuJoCo rendering backend (EGL, Mesa EGL,
           OSMesa) by rendering a LIBERO scene and saves that frame
  server   venv with the locked server requirements for the Python pi05.so was built for,
           serve_policy.py; starts the server and sends observations over the openpi protocol
  rollout  starts the server, runs main.py once per suite, and parses every episode

serve_policy.py answers a failed inference with zero actions, which it then unnormalizes, so
the client cannot tell them apart from real ones. The server log is therefore watched during
the rollout: any "Inference failed" line (or a server crash) stops the run and marks it
invalid. A rollout is also invalid if the client caught an exception or finished fewer
episodes than expected. Only then is a success rate reported.

The work folder is shared with actquant_build.py (EAQ_WORK_DIR). This is one self-contained
file: Molab fetches it alone and may run Python with PYTHONSAFEPATH, which blocks
sibling-module imports.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request


OPENPI_COMMIT = "215abfb217dbac7d5f1273282331b9b1866c0479"
OPENPI_MAIN_URL = (f"https://raw.githubusercontent.com/Physical-Intelligence/openpi/{OPENPI_COMMIT}/"
                   "examples/libero/main.py")
OPENPI_MAIN_SHA256 = "d53748c95804ada559e3ca867305b8f755d8651c4c7f4e2723ef80aea55a3943"
# openpi's third_party/libero submodule at OPENPI_COMMIT.
LIBERO_URL = "https://github.com/Lifelong-Robot-Learning/LIBERO.git"
LIBERO_COMMIT = "f78abd68ee283de9f9be3c8f7e2a9ad60246e95c"
ACTQUANT_COMMIT = "b64791125070652fe6b554e244fe809c79ef5246"
SERVE_POLICY_URL = (f"https://raw.githubusercontent.com/arashakb/ActQuant/{ACTQUANT_COMMIT}/"
                    "tools/pi0.5/serve_policy.py")
SERVE_POLICY_SHA256 = "a82c2671209e28df052c5926558b3d642a9ad102b6e2afc34bb2fed98bdb7ab4"
# Where actquant_build.py's download stage puts the checkpoint.
# Model name: checkpoint folder in the work folder (as in actquant_build.py's MODELS). "fp16" is the
# reference exported from the 3-bit release's base checkpoint by ActQuant's export_pi05.py.
CHECKPOINTS = {"actquant-3bpw": Path("checkpoints") / "actquant-pi05-libero-3bpw",
               "fp16": Path("checkpoints") / "pi05-libero-fp16"}
CLIENT_PYTHON = "3.8"
FLOW_STEPS = 10  # run_libero_eval.sh: FLOW_STEPS=10
STAGES = ("runtime", "client", "server", "profile", "rollout", "hold")
DEFAULT_STAGES = ("runtime", "client", "server", "rollout")  # "profile" and "hold" are diagnostics
SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
TASKS_PER_SUITE = 10
# Success rates on the checkpoint's model card (500 trials per suite, pi05.so binding).
REPORTED_SUCCESS = {"libero_spatial": 0.982, "libero_object": 0.988, "libero_goal": 0.950, "libero_10": 0.872}
PROBE_PROMPT = "put the black bowl on the plate"
# Rendering backends tried in order: MUJOCO_GL value and extra environment. "egl-nvidia" registers
# the driver's own EGL library with glvnd (its extra environment is built in _nvidia_egl_env): Colab
# ships libEGL_nvidia.so.0 in /usr/lib64-nvidia but only Mesa's vendor file, so plain "egl" renders
# with llvmpipe on the CPU (96 ms per 256x256 frame on Colab's T4 VM, against 0.8 ms on the GPU).
GL_CANDIDATES = (
    ("egl-nvidia", None),
    ("egl", {}),
    ("egl-mesa", {"__EGL_VENDOR_LIBRARY_FILENAMES": "/usr/share/glvnd/egl_vendor.d/50_mesa.json"}),
    ("osmesa", {}),
)
# Where NVIDIA's EGL library may be when no vendor file registers it (Colab: /usr/lib64-nvidia).
NVIDIA_EGL_DIRS = ("/usr/lib64-nvidia", "/usr/local/nvidia/lib64", "/usr/lib/x86_64-linux-gnu", "/usr/lib64")
GL_APT_PACKAGES = ("libegl1", "libegl-mesa0", "libgl1", "libgl1-mesa-dri", "libosmesa6")
NICENESS = 10
# serve_policy.py log lines that make a rollout invalid.
SERVER_PROBLEMS = {
    "inference_failed": re.compile(r"Inference failed:"),
    "handler_error": re.compile(r"ERROR:[\w.]+:Error:"),
    "unexpected_action_size": re.compile(r"Unexpected action size"),
    "no_norm_stats": re.compile(r"No norm(alization)? stats"),
}
SERVER_INFER_MS = re.compile(r"Inference time: ([\d.]+)ms")

RENDER_PROBE = r'''
import json, os, pathlib, sys, time
started = time.monotonic()
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import imageio
import numpy as np
suite = benchmark.get_benchmark_dict()["libero_spatial"]()
task = suite.get_task(0)
init_states = suite.get_task_init_states(0)
bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
env.seed(7)
env.reset()
obs = env.set_init_state(init_states[0])
for _ in range(10):  # main.py's num_steps_wait; also warms up numba
    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
ready = time.monotonic()
for _ in range(20):
    obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
stepped = time.monotonic()
image = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])  # as main.py sends it
imageio.imwrite(sys.argv[1], image)
renderer = None
try:  # "llvmpipe" means Mesa rendering on the CPU; an NVIDIA name means the GPU renders
    from OpenGL.GL import GL_RENDERER, GL_VENDOR, glGetString
    renderer = " / ".join(glGetString(name).decode() for name in (GL_VENDOR, GL_RENDERER))
except Exception as exc:
    renderer = f"unknown ({type(exc).__name__})"
env.close()
print("EAQ_PROBE " + json.dumps({
    "task": task.language, "n_tasks": suite.n_tasks, "init_states": len(init_states),
    "image_shape": list(image.shape), "image_std": float(image.std()), "renderer": renderer,
    "setup_seconds": round(ready - started, 1), "step_ms": round((stepped - ready) * 50, 1)}))
'''

# openpi's WebSocket protocol, shared by the probe clients below.
WIRE = r'''
import json, sys, time
import msgpack
import numpy as np
from websockets.sync.client import connect

def pack(obj):  # openpi_client.msgpack_numpy encoding
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": list(obj.shape)}
    if isinstance(obj, dict):
        return {key: pack(value) for key, value in obj.items()}
    return obj

def unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=tuple(obj[b"shape"]))
    return obj

'''

SERVER_PROBE = WIRE + r'''
uri, prompt, image_path = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    import cv2
    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
    source = image_path
except Exception:
    image = np.random.default_rng(0).integers(0, 256, (224, 224, 3), dtype=np.uint8)
    source = "random"
observation = {"observation/image": image, "observation/wrist_image": image.copy(),
               "observation/state": np.zeros(8, dtype=np.float64), "prompt": prompt}
round_trips, outputs = [], []
with connect(uri, compression=None, max_size=None, open_timeout=60) as websocket:
    metadata = msgpack.unpackb(websocket.recv(), object_hook=unpack)
    for _ in range(3):
        started = time.monotonic()
        websocket.send(msgpack.packb(pack(observation)))
        reply = websocket.recv()
        round_trips.append(round((time.monotonic() - started) * 1000, 1))
        if isinstance(reply, str):
            print("EAQ_PROBE " + json.dumps({"server_error": reply[-3000:]}))
            sys.exit(1)
        outputs.append(msgpack.unpackb(reply, object_hook=unpack))
    # Optional: keep requesting for argv[4] seconds, at about a rollout's pace (the "hold" stage).
    hold_until, hold_requests = time.monotonic() + (float(sys.argv[4]) if len(sys.argv) > 4 else 0), 0
    while time.monotonic() < hold_until:
        websocket.send(msgpack.packb(pack(observation)))
        if isinstance(websocket.recv(), str):
            print("EAQ_PROBE " + json.dumps({"server_error": "error reply during hold"}))
            sys.exit(1)
        hold_requests += 1
        time.sleep(0.25)
actions = np.asarray(outputs[-1]["actions"], dtype=np.float64)
print("EAQ_PROBE " + json.dumps({"hold_requests": hold_requests,
    "metadata": metadata, "image": source, "actions_shape": list(actions.shape),
    "finite": bool(np.isfinite(actions).all()), "min": float(actions.min()), "max": float(actions.max()),
    "first_action": [round(float(v), 4) for v in actions[0]],
    "repeat_max_abs_diff": float(np.abs(np.asarray(outputs[0]["actions"]) - actions).max()),
    "round_trip_ms": round_trips, "server_timing": outputs[-1].get("server_timing")}))
'''

# The profile stage's client: phases of requests to the policy server, each printed with its wall-clock
# span, so the stage can match pi05's per-inference timing summaries and GPU samples to a phase.
PROFILE_PROBE = WIRE + r'''
import subprocess
uri, prompt, image_path, requests = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
try:
    import cv2
    base = cv2.resize(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB), (224, 224))
except Exception:
    base = np.full((224, 224, 3), 128, dtype=np.uint8)
rng = np.random.default_rng(0)

def observation(distinct):
    image = base
    if distinct:  # a different image each time, as in a rollout; the content stays a LIBERO scene
        image = np.clip(base.astype(np.int16) + rng.integers(-8, 9, base.shape), 0, 255).astype(np.uint8)
    return {"observation/image": image, "observation/wrist_image": image[::-1].copy(),
            "observation/state": rng.normal(0, 0.1, 8), "prompt": prompt}

# same: one observation repeated (the server stage's probe does this); distinct: a new image per
# request; distinct_cpu_load: the same with a busy process on the CPU, as the simulator is in a rollout.
phases = [("warmup", False, 2, False), ("same", False, requests, False), ("distinct", True, requests, False),
          ("distinct_cpu_load", True, requests, True)]
results = []
with connect(uri, compression=None, max_size=None, open_timeout=60) as websocket:
    websocket.recv()  # metadata
    for name, distinct, count, load in phases:
        burner = subprocess.Popen([sys.executable, "-c", "while True: pass"]) if load else None
        try:
            started, round_trips = time.time(), []
            obs = observation(distinct)
            for _ in range(count):
                if distinct:
                    obs = observation(True)
                begin = time.monotonic()
                websocket.send(msgpack.packb(pack(obs)))
                reply = websocket.recv()
                round_trips.append(round((time.monotonic() - begin) * 1000, 1))
                if isinstance(reply, str):
                    print("EAQ_PROFILE " + json.dumps({"server_error": reply[-3000:]}))
                    sys.exit(1)
                actions = np.asarray(msgpack.unpackb(reply, object_hook=unpack)["actions"])
                if not np.isfinite(actions).all():
                    print("EAQ_PROFILE " + json.dumps({"server_error": "non-finite actions"}))
                    sys.exit(1)
            results.append({"phase": name, "requests": count, "start": started, "end": time.time(),
                            "round_trip_ms": round_trips})
        finally:
            if burner:
                burner.kill()
print("EAQ_PROFILE " + json.dumps({"phases": results}))
'''


class StageError(RuntimeError):
    """A stage failed; the message is shown to the user, details stay in the report."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def log(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


def _now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


# Interpreter settings of the notebook's own Python (Molab puts its 3.13 venv on PYTHONPATH). A child
# running another interpreter must not inherit them: uv's Python 3.8 build environment imported the
# notebook's 3.13-only `packaging` and failed to build openpi-client.
INHERITED_PYTHON_VARS = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV")


# Thread pools in the client that otherwise size themselves to the 20 CPUs gVisor reports (Molab has
# about 4 real ones). Sustained all-core load is the best explanation for Molab's sandbox resets.
THREAD_VARS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS",
               "NUMBA_NUM_THREADS", "LP_NUM_THREADS")  # LP_: Mesa llvmpipe's rendering threads


def _cpu_seconds(pid="self"):
    """CPU seconds (user + system) of one process, or None."""
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return (int(fields[11]) + int(fields[12])) / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError, AttributeError):
        return None


def _uptime_minutes():
    try:
        return round(float(Path("/proc/uptime").read_text().split()[0]) / 60, 1)
    except (OSError, ValueError, IndexError):
        return None


class CpuMonitor:
    """Cores in use by all processes (from /proc), logged every `every` seconds."""

    def __init__(self, server_pid, every=30):
        self.server_pid, self.every = server_pid, every
        self.last = None
        self.samples = []

    @staticmethod
    def _total():
        total = 0.0
        for entry in Path("/proc").iterdir():
            if entry.name.isdigit():
                total += _cpu_seconds(entry.name) or 0.0
        return total

    def poll(self):
        now = time.monotonic()
        if self.last is not None and now - self.last[0] < self.every:
            return None
        try:
            current = (now, self._total(), _cpu_seconds(self.server_pid) or 0.0)
        except OSError:
            return None
        previous, self.last = self.last, current
        if previous is None:
            return None
        wall = current[0] - previous[0]
        # Exited processes drop out of the total, so a sample can dip; clamp at zero.
        sample = {"uptime_min": _uptime_minutes(),
                  "cores_all": round(max(current[1] - previous[1], 0.0) / wall, 2),
                  "cores_server": round(max(current[2] - previous[2], 0.0) / wall, 2)}
        self.samples.append(sample)
        log(f"cpu: {sample['cores_all']} cores in use ({sample['cores_server']} policy server), "
            f"sandbox uptime {sample['uptime_min']} min")
        return sample

    def summary(self):
        if not self.samples:
            return {}
        cores = sorted(sample["cores_all"] for sample in self.samples)
        return {"samples": len(cores), "cores_median": cores[len(cores) // 2], "cores_max": cores[-1],
                "server_cores_max": max(sample["cores_server"] for sample in self.samples)}


def _child_env(**overrides):
    env = {key: value for key, value in os.environ.items() if key not in INHERITED_PYTHON_VARS}
    env.update(overrides)
    return env


def _run(command, timeout=60, cwd=None, env=None):
    try:
        result = subprocess.run([str(part) for part in command], capture_output=True, text=True,
                                errors="replace", check=False, timeout=timeout, cwd=cwd, env=env)
    except FileNotFoundError:
        return {"returncode": None, "output": "not found"}
    except OSError as exc:
        return {"returncode": None, "output": f"could not start: {type(exc).__name__}: {exc}"}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "output": f"timed out after {timeout} s"}
    return {"returncode": result.returncode, "output": (result.stdout + result.stderr).strip()}


def _stream(command, log_path, timeout, cwd=None, env=None, on_line=None, stop=None):
    """Run a command, appending its output to log_path and echoing it; stop() may end it early."""
    command = [str(part) for part in command]
    started = time.monotonic()
    tail = []
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        log_file.flush()
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdin=subprocess.DEVNULL,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       errors="replace", bufsize=1)
        except OSError as exc:
            return {"returncode": None, "seconds": 0.0, "timed_out": False, "stopped": None,
                    "tail": f"could not start: {type(exc).__name__}: {exc}"}

        def pump():
            for line in process.stdout:
                log_file.write(line)
                tail.append(line.rstrip("\n"))
                del tail[:-80]
                print(line, end="", flush=True)
                if on_line is not None:
                    on_line(line.rstrip("\n"))

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out, stopped = False, None
        while process.poll() is None:
            if time.monotonic() - started > timeout:
                timed_out = True
            elif stop is not None:
                stopped = stop()
            if timed_out or stopped:
                process.terminate()
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    process.kill()
                break
            time.sleep(2)
        process.wait()
        reader.join(timeout=30)
        log_file.flush()
    return {"returncode": process.returncode, "seconds": round(time.monotonic() - started, 1),
            "timed_out": timed_out, "stopped": stopped, "tail": "\n".join(tail)}


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _download(url, target, sha256):
    """Download url to target unless it is already there with the pinned SHA-256."""
    if target.is_file() and _sha256(target) == sha256:
        return {"url": url, "sha256": sha256, "cached": True}
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "eaq-libero-rollout"})
    with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as handle:
        shutil.copyfileobj(response, handle)
    digest = _sha256(partial)
    if digest != sha256:
        partial.unlink()
        raise StageError(f"{url} has SHA-256 {digest}, not the pinned {sha256}.")
    partial.replace(target)
    return {"url": url, "sha256": sha256, "cached": False}


def _which(name):
    return shutil.which(name) or shutil.which(name, path=str(Path(sys.executable).parent))


def _free_port(preferred):
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                continue
            return probe.getsockname()[1]
    raise StageError("No free TCP port for the policy server.")


def _nvidia_egl_library():
    """Path of NVIDIA's EGL vendor library (libEGL_nvidia.so.0), or None."""
    dirs = [part for part in os.environ.get("LD_LIBRARY_PATH", "").split(":") if part] + list(NVIDIA_EGL_DIRS)
    for folder in dirs:
        path = Path(folder) / "libEGL_nvidia.so.0"
        if path.exists():
            return path
    ldconfig = shutil.which("ldconfig") or "/sbin/ldconfig"
    match = re.search(r"libEGL_nvidia\.so\.0 .*=> (\S+)", _run([ldconfig, "-p"])["output"])
    return Path(match[1]) if match else None


def _wilson(successes, trials, z=1.96):
    """95% Wilson score interval for a success rate."""
    if trials == 0:
        return None
    rate = successes / trials
    centre = (rate + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


class ServerLog:
    """Counts serve_policy.py problem lines and inference times in a growing log file."""

    def __init__(self, path):
        self.path = path
        self.offset = path.stat().st_size if path.exists() else 0
        self.counts = {key: 0 for key in SERVER_PROBLEMS}
        self.infer_ms = []
        self.examples = []

    def poll(self):
        try:
            with open(self.path, "rb") as handle:
                handle.seek(self.offset)
                data = handle.read()
        except OSError:
            return self.counts
        # Only complete lines; the rest is read next time.
        end = data.rfind(b"\n") + 1
        self.offset += end
        for line in data[:end].decode("utf-8", errors="replace").splitlines():
            for key, pattern in SERVER_PROBLEMS.items():
                if pattern.search(line):
                    self.counts[key] += 1
                    if len(self.examples) < 5:
                        self.examples.append(line[-300:])
            match = SERVER_INFER_MS.search(line)
            if match:
                self.infer_ms.append(float(match[1]))
        return self.counts

    def summary(self):
        self.poll()
        timing = None
        if self.infer_ms:
            timing = {"mean": round(sum(self.infer_ms) / len(self.infer_ms), 1),
                      "p50": _percentile(self.infer_ms, 0.5), "p95": _percentile(self.infer_ms, 0.95),
                      "max": max(self.infer_ms)}
        return {"inferences": len(self.infer_ms), "infer_ms": timing, "problems": dict(self.counts),
                "problem_examples": self.examples}

    @property
    def problem_count(self):
        return sum(self.counts.values())


TIMING_SECTIONS = {"Level 1: Chain": "chain", "Level 2: compute_kv_cache": "kv_cache",
                   "Level 2: ODE per step": "ode_step"}
TIMING_LINE = re.compile(r"^\s+(\S.*?)\s*:\s*([\d.]+) ms")


def _timing_summaries(path, offset=0):
    """pi05's "Inference Timing Summary" blocks in a server log after byte `offset`, one
    {"section/name": ms} per inference."""
    summaries, current, section = [], None, None
    try:
        with open(path, "rb") as handle:
            handle.seek(offset)
            lines = handle.read().decode("utf-8", errors="replace").splitlines()
    except OSError:
        return summaries
    for line in lines:
        if "=== Inference Timing Summary ===" in line:
            current, section = {}, None
        elif current is None:
            continue
        elif line.startswith("====="):
            summaries.append(current)
            current = None
        elif line.startswith("["):
            section = next((key for prefix, key in TIMING_SECTIONS.items() if line[1:].startswith(prefix)), None)
        elif section and (match := TIMING_LINE.match(line)):
            current[f"{section}/{match[1]}"] = float(match[2])
    return summaries


class GpuSampler:
    """nvidia-smi samples (clocks, power, utilisation, throttle reasons) twice a second, with timestamps."""

    FIELDS = ("clocks.sm", "clocks.mem", "power.draw", "utilization.gpu", "temperature.gpu", "pstate",
              "clocks_event_reasons.active")

    def __init__(self):
        self.samples = []
        smi = _which("nvidia-smi")
        self.process = None if not smi else subprocess.Popen(
            [smi, f"--query-gpu={','.join(self.FIELDS)}", "--format=csv,noheader,nounits", "-lms", "500"],
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
        self.thread = threading.Thread(target=self._read, daemon=True)
        if self.process:
            self.thread.start()

    def _read(self):
        for line in self.process.stdout:
            values = [part.strip() for part in line.split(",")]
            if len(values) == len(self.FIELDS):
                self.samples.append((time.time(), dict(zip(self.FIELDS, values))))

    def stop(self):
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=10)

    def between(self, start, end):
        rows = [values for stamp, values in self.samples if start <= stamp <= end]
        summary = {"samples": len(rows)}
        for field in ("clocks.sm", "clocks.mem", "power.draw", "utilization.gpu", "temperature.gpu"):
            numbers = [float(row[field]) for row in rows if re.fullmatch(r"[\d.]+", row[field])]
            if numbers:
                summary[field] = {"mean": round(sum(numbers) / len(numbers), 1), "min": min(numbers),
                                  "max": max(numbers)}
        summary["pstates"] = sorted({row["pstate"] for row in rows})
        summary["throttle_reasons"] = sorted({row["clocks_event_reasons.active"] for row in rows})
        return summary


class EpisodeParser:
    """Follows openpi main.py's log lines and records one entry per finished episode."""

    def __init__(self, suite, episodes_path, on_episode):
        self.suite = suite
        self.episodes_path = episodes_path
        self.on_episode = on_episode
        self.task = None
        self.task_index = -1
        self.task_episode = 0
        self.started = None
        self.exception = None
        self.episodes = []

    def __call__(self, line):
        # main.py logs "\nTask: ...", so the description is on its own line after "INFO:root:".
        text = line.split("INFO:root:", 1)[-1].split("ERROR:root:", 1)[-1].strip()
        if text.startswith("Task: "):
            description = text[len("Task: "):]
            if description != self.task:
                self.task = description
                self.task_index += 1
                self.task_episode = 0
        elif text.startswith("Starting episode"):
            self.started = time.monotonic()
            self.exception = None
        elif text.startswith("Caught exception:"):
            self.exception = text[len("Caught exception:"):].strip()[:500]
        elif text.startswith("Success:"):
            episode = {"suite": self.suite, "task_index": self.task_index, "task": self.task,
                       "trial": self.task_episode, "success": text.split(":", 1)[1].strip() == "True",
                       "exception": self.exception,
                       "seconds": round(time.monotonic() - self.started, 1) if self.started else None}
            self.task_episode += 1
            self.started = None
            self.exception = None
            self.episodes.append(episode)
            with open(self.episodes_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(episode) + "\n")
            self.on_episode(episode)


class Rollout:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.work_dir).resolve()
        self.output = Path(args.output).resolve()
        self.libero_root = self.work / "libero"
        self.checkpoint = self.work / CHECKPOINTS[args.model]
        self.deviations = {}
        self.server_process = None
        self.report = {
            "schema_version": 1,
            "scope": "LIBERO closed-loop rollouts of ActQuant-Pi05-LIBERO-3bpw through ActQuant's "
                     "serve_policy.py and openpi's examples/libero/main.py",
            "started_utc": _now(),
            "pins": {"actquant": ACTQUANT_COMMIT, "openpi": OPENPI_COMMIT, "libero": LIBERO_COMMIT,
                     "serve_policy_sha256": SERVE_POLICY_SHA256, "openpi_main_sha256": OPENPI_MAIN_SHA256},
            "options": {"model": args.model, "stages": args.stages, "suites": args.suites, "trials_per_task": args.trials,
                        "replan_steps": args.replan_steps, "seed": args.seed, "device": args.device,
                        "gl": args.gl, "work_dir": str(self.work)},
            "host": self._host(),
            "stages": {},
        }

    # ------------------------------------------------------------------ helpers

    def _host(self):
        try:
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                                  "--format=csv,noheader"], capture_output=True, text=True, timeout=10,
                                 check=False).stdout.strip() or None
        except (OSError, subprocess.SubprocessError):
            gpu = None
        try:
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            cpus = os.cpu_count()
        return {"python": sys.version.split()[0], "platform": platform.platform(), "glibc": "-".join(
            platform.libc_ver()), "cpus": cpus, "gpu": gpu, "uv": _run([self._uv_path() or "uv", "--version"])["output"]}

    @staticmethod
    def _uv_path():
        return _which("uv")

    def _uv(self):
        uv = self._uv_path()
        if not uv:
            raise StageError("uv is not installed; install it from the Packages panel (pip package `uv`).")
        return uv

    def _uv_env(self):
        env = _child_env()
        # uv-managed interpreters (Python 3.8 for the client) live in the work folder.
        env["UV_PYTHON_INSTALL_DIR"] = str(self.libero_root / "python")
        return env

    def deviation(self, key, text):
        self.deviations[key] = text

    def save(self):
        self.report["deviations"] = list(self.deviations.values())
        self.report["finished_utc"] = _now()
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")

    def _lock(self):
        try:
            import fcntl
        except ImportError:
            return None
        self._lock_file = open(self.work / ".libero.lock", "a+", encoding="utf-8")
        try:
            fcntl.flock(self._lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self._lock_file.seek(0)
            return self._lock_file.read().strip() or "another run"
        self._lock_file.seek(0)
        self._lock_file.truncate()
        self._lock_file.write(f"pid {os.getpid()}, run folder {self.output}\n")
        self._lock_file.flush()
        return None

    def _package(self):
        """The installed ActQuant package: its bin/ folder and manifest."""
        current = _read_json(self.work / "prebuilt" / "current.json")
        if not current:
            raise StageError("No ActQuant package is installed in the work folder; run the runtime stage.")
        root = Path(current["path"])
        manifest = _read_json(root / "eaq-package.json") or {}
        if not (root / "bin" / "pi05.so").is_file():
            raise StageError(f"The installed package {root.name} has no pi05.so binding.")
        return root / "bin", manifest

    def _install_requirements(self, python, requirements, stamp):
        """Install a locked requirements file exactly (no dependency resolution), once per lock."""
        digest = _sha256(requirements)
        if stamp.is_file() and stamp.read_text(encoding="utf-8").strip() == digest:
            return {"requirements": str(requirements), "sha256": digest, "cached": True}
        result = _run([self._uv(), "pip", "install", "--python", python, "--no-deps",
                       "--index-strategy", "unsafe-best-match", "-r", requirements],
                      timeout=3600, env=self._uv_env())
        if result["returncode"] != 0:
            raise StageError(f"Installing {requirements.name} failed.", {"output": result["output"][-3000:]})
        stamp.write_text(digest + "\n", encoding="utf-8")
        return {"requirements": str(requirements), "sha256": digest, "cached": False,
                "output_tail": result["output"][-1000:]}

    def _venv(self, path, python):
        if (path / "bin" / "python").exists():
            return path / "bin" / "python"
        if path.exists():  # without an interpreter: left over from an interrupted run
            shutil.rmtree(path)
        result = _run([self._uv(), "venv", "--python", python, path], timeout=1800, env=self._uv_env())
        if result["returncode"] != 0:
            raise StageError(f"Creating the Python {python} venv failed.", {"output": result["output"][-3000:]})
        return path / "bin" / "python"

    # ------------------------------------------------------------------ run

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.libero_root.mkdir(parents=True, exist_ok=True)
        holder = self._lock()
        if holder:
            log(f"work folder {self.work} is in use by {holder}; not starting a second run")
            self.report["status"] = "failed"
            self.report["error"] = f"LIBERO work folder in use by {holder}"
            self.save()
            (self.output / "exit_code").write_text("1\n", encoding="utf-8")
            return 1
        try:
            self.report["niceness"] = os.nice(NICENESS)
        except (AttributeError, OSError):
            self.report["niceness"] = None
        requested = [stage for stage in STAGES if stage in self.args.stages]
        failed = None
        self.report["status"] = "running"
        try:
            for stage in requested:
                if failed:
                    self.report["stages"][stage] = {"status": "skipped", "reason": f"stage '{failed}' failed"}
                    continue
                log(f"=== stage: {stage} ===")
                self.report["stages"][stage] = {"status": "running", "started_utc": _now()}
                self.save()
                started = time.monotonic()
                try:
                    details = getattr(self, f"stage_{stage}")() or {}
                    entry = {"status": "passed", **details}
                except StageError as exc:
                    failed = stage
                    entry = {"status": "failed", "error": str(exc), **exc.details}
                    log(f"stage {stage} FAILED: {exc}")
                except Exception as exc:  # keep the report even for unexpected errors
                    failed = stage
                    entry = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                    log(f"stage {stage} FAILED unexpectedly: {type(exc).__name__}: {exc}")
                entry["seconds"] = round(time.monotonic() - started, 1)
                self.report["stages"][stage] = entry
                self.save()
        finally:
            self._stop_server()
        if failed:
            self.report["status"] = "failed"
            self.report["failed_stage"] = failed
        elif "rollout" in requested:
            self.report["status"] = "rollout_ok"
        else:
            self.report["status"] = "stages_passed"
        self.save()
        log(f"status: {self.report['status']}")
        code = 0 if not failed else 1
        (self.output / "exit_code").write_text(f"{code}\n", encoding="utf-8")
        return code

    # ------------------------------------------------------------------ stages

    def stage_runtime(self):
        script = Path(self.args.actquant_script)
        if not script.is_file():
            raise StageError(f"{script} not found.")
        runtime_output = self.output / "runtime"
        command = [sys.executable, script, "--output", runtime_output, "--work-dir", self.work,
                   "--stages", "toolkit", "fetch", "download"]
        if self.args.toolkit_requirements:
            command += ["--toolkit-requirements", self.args.toolkit_requirements]
        if self.args.prebuilt_pin:
            command += ["--prebuilt-pin", self.args.prebuilt_pin]
        command += ["--model", self.args.model]
        if self.args.model_cache:
            command += ["--model-cache", self.args.model_cache]
        env = {**os.environ, "PYTHONUNBUFFERED": "1"}  # the notebook's own interpreter: keep its settings
        result = _stream(command, self.output / "runtime.log", 7200, env=env)
        report = _read_json(runtime_output / "report.json") or {}
        stages = {name: entry.get("status") for name, entry in report.get("stages", {}).items()}
        details = {"stages": stages, "returncode": result["returncode"]}
        if result["returncode"] != 0:
            raise StageError("actquant_build.py could not prepare the runtime; see runtime/report.json.",
                             {**details, "tail": result["tail"][-3000:]})
        bin_dir, manifest = self._package()
        for index, text in enumerate(report.get("deviations", [])):
            self.deviation(f"runtime_{index}", text)
        return {**details, "package": manifest.get("name"), "bin": str(bin_dir),
                "checkpoint": str(self.checkpoint)}

    def _client_env(self, gl_name=None):
        client = _read_json(self.libero_root / "client.json") or {}
        gl_name = gl_name or client.get("gl")
        env = _child_env()
        # LIBERO asks on stdin for a dataset path unless its config file exists.
        env["LIBERO_CONFIG_PATH"] = str(self.libero_root / "config")
        env["PYTHONPATH"] = str(self.libero_root / "LIBERO")
        env["TQDM_DISABLE"] = "1"  # main.py's progress bars only garble the log
        env["PYTHONUNBUFFERED"] = "1"
        env.update({name: str(self.args.client_threads) for name in THREAD_VARS})
        for name, extra in GL_CANDIDATES:
            if name == gl_name:
                env["MUJOCO_GL"] = name.split("-")[0]
                env["PYOPENGL_PLATFORM"] = name.split("-")[0]
                env.update(self._nvidia_egl_env(env) if extra is None else extra)
        return env

    def _nvidia_egl_env(self, env):
        """glvnd settings that make EGL load NVIDIA's library, or {} if the driver has none."""
        library = _nvidia_egl_library()
        if library is None:
            return {}
        vendor = self.libero_root / "egl_vendor.d" / "10_nvidia.json"
        vendor.parent.mkdir(parents=True, exist_ok=True)
        vendor.write_text(json.dumps({"file_format_version": "1.0.0",
                                      "ICD": {"library_path": str(library)}}) + "\n", encoding="utf-8")
        # The library's own dependencies (libnvidia-eglcore, ...) sit next to it, outside the
        # loader's search path on Colab; without them EGL finds no device.
        search = [str(library.parent)] + [part for part in env.get("LD_LIBRARY_PATH", "").split(":")
                                           if part and part != str(library.parent)]
        return {"__EGL_VENDOR_LIBRARY_FILENAMES": str(vendor), "LD_LIBRARY_PATH": ":".join(search)}

    def _fetch_libero(self):
        source = self.libero_root / "LIBERO"
        git = _which("git")
        if not git:
            raise StageError("git is not installed.")
        if not (source / ".git").exists():
            source.mkdir(parents=True, exist_ok=True)
            _run([git, "init", "-q", source])
            _run([git, "-C", source, "remote", "add", "origin", LIBERO_URL])
        head = _run([git, "-C", source, "rev-parse", "HEAD"])["output"].strip()
        if head != LIBERO_COMMIT:
            log(f"fetching LIBERO @ {LIBERO_COMMIT[:12]} (about 0.7 GB with assets)")
            fetch = _run([git, "-C", source, "fetch", "--depth", "1", "origin", LIBERO_COMMIT], timeout=1800)
            if fetch["returncode"] != 0:
                raise StageError("git fetch of the pinned LIBERO commit failed.", {"output": fetch["output"][-2000:]})
            checkout = _run([git, "-C", source, "checkout", "-q", "--detach", "FETCH_HEAD"], timeout=600)
            if checkout["returncode"] != 0:
                raise StageError("git checkout of LIBERO failed.", {"output": checkout["output"][-2000:]})
        head = _run([git, "-C", source, "rev-parse", "HEAD"])["output"].strip()
        if head != LIBERO_COMMIT:
            raise StageError(f"LIBERO is at {head}, not the pinned {LIBERO_COMMIT}.")
        benchmark_root = source / "libero" / "libero"
        config = self.libero_root / "config"
        config.mkdir(exist_ok=True)
        # The same paths LIBERO's get_default_path_dict() writes on its first (interactive) import.
        (config / "config.yaml").write_text("".join(f"{key}: {value}\n" for key, value in (
            ("benchmark_root", benchmark_root), ("bddl_files", benchmark_root / "bddl_files"),
            ("init_states", benchmark_root / "init_files"), ("datasets", source / "libero" / "datasets"),
            ("assets", benchmark_root / "assets"))), encoding="utf-8")
        return {"commit": head, "path": str(source)}

    def _render_probe(self, python, gl_name):
        probe = self.libero_root / "render_probe.py"
        probe.write_text(RENDER_PROBE, encoding="utf-8")
        image = self.output / f"render_probe_{gl_name}.png"
        result = _run([python, probe, image], timeout=1200, cwd=self.libero_root, env=self._client_env(gl_name))
        line = next((line for line in result["output"].splitlines() if line.startswith("EAQ_PROBE ")), None)
        details = json.loads(line[len("EAQ_PROBE "):]) if line else None
        ok = result["returncode"] == 0 and details is not None and details["image_std"] > 1.0
        return ok, {"gl": gl_name, "returncode": result["returncode"], "probe": details,
                    "image": image.name if ok else None,
                    **({} if ok else {"output_tail": result["output"][-2500:]})}

    def stage_client(self):
        uv_env = self._uv_env()
        install = _run([self._uv(), "python", "install", CLIENT_PYTHON], timeout=1800, env=uv_env)
        if install["returncode"] != 0:
            raise StageError(f"uv could not install Python {CLIENT_PYTHON}.", {"output": install["output"][-2000:]})
        python = self._venv(self.libero_root / "client-venv", CLIENT_PYTHON)
        version = _run([python, "--version"])["output"]
        requirements = self._install_requirements(python, Path(self.args.client_requirements),
                                                  self.libero_root / "client-venv" / ".eaq-requirements")
        libero = self._fetch_libero()
        main = _download(OPENPI_MAIN_URL, self.libero_root / "openpi" / "main.py", OPENPI_MAIN_SHA256)

        attempts = []
        candidates = [name for name, _ in GL_CANDIDATES if self.args.gl in ("auto", name)]
        chosen = None
        for round_ in ("installed", "apt"):
            if round_ == "apt":
                apt = _which("apt-get")
                if not apt or os.geteuid() != 0:
                    break
                log("no rendering backend worked; installing Mesa EGL/OSMesa with apt-get")
                update = _run([apt, "update"], timeout=900)
                result = _run([apt, "install", "-y", "--no-install-recommends", *GL_APT_PACKAGES], timeout=1800,
                              env=_child_env(DEBIAN_FRONTEND="noninteractive"))
                attempts.append({"apt_install": list(GL_APT_PACKAGES), "returncode": result["returncode"],
                                 "update_returncode": update["returncode"], "output_tail": result["output"][-1500:]})
                if result["returncode"] != 0:
                    break
                self.deviation("apt", "Mesa rendering libraries installed with apt-get: " + ", ".join(GL_APT_PACKAGES))
            for name in candidates:
                if name == "egl-mesa" and not Path(dict(GL_CANDIDATES)[name]["__EGL_VENDOR_LIBRARY_FILENAMES"]).is_file():
                    continue
                if name == "egl-nvidia" and (round_ == "apt" or _nvidia_egl_library() is None):
                    continue
                log(f"render probe with MUJOCO_GL={name}")
                ok, attempt = self._render_probe(python, name)
                attempts.append(attempt)
                if ok:
                    chosen = attempt
                    break
            if chosen:
                break
        if not chosen:
            raise StageError("No MuJoCo rendering backend could render a LIBERO scene.", {"attempts": attempts})
        if chosen["gl"] not in ("egl", "egl-nvidia"):  # egl-nvidia is EGL on the GPU, only registered by us
            self.deviation("gl", f"MuJoCo rendering with {chosen['gl']} (ActQuant's README uses EGL).")
        shutil.copy2(self.output / chosen["image"], self.libero_root / "render_probe.png")
        (self.libero_root / "client.json").write_text(json.dumps(
            {"python": str(python), "gl": chosen["gl"], "main": str(self.libero_root / "openpi" / "main.py")},
            indent=2) + "\n", encoding="utf-8")
        self.deviation("client", f"LIBERO client in a uv-managed Python {CLIENT_PYTHON} venv (ActQuant: conda "
                                 "env openpi-libero) from requirements/libero-client.txt: openpi's pins, except "
                                 "torch 2.4.1 CPU for 1.11.0 cu113 (glibc 2.41 refuses 1.11's executable-stack "
                                 "library), opencv-python-headless, no keyboard-teleoperation packages, and only "
                                 "LIBERO's runtime (not training) requirements.")
        return {"python": version, "requirements": requirements, "libero": libero, "openpi_main": main,
                "gl": chosen["gl"], "render": chosen["probe"], "render_image": chosen["image"],
                "gl_attempts": attempts}

    def _start_server(self, port, log_name="server.log"):
        bin_dir, _ = self._package()
        python = self.libero_root / "server-venv" / "bin" / "python"
        serve = self.libero_root / "actquant" / "serve_policy.py"
        if not (python.exists() and serve.is_file()):
            raise StageError("The policy server is not installed; run the server stage.")
        if not (self.checkpoint / "pi05.gguf").is_file():
            raise StageError(f"No checkpoint at {self.checkpoint}; run the runtime stage.")
        env = _child_env()
        env["PYTHONPATH"] = str(bin_dir)  # as run_libero_eval.sh: the folder holding pi05.so
        env["PYTHONUNBUFFERED"] = "1"
        command = [python, serve, "--model-dir", self.checkpoint, "--host", "127.0.0.1", "--port", port,
                   "--device", self.args.device, "--steps", FLOW_STEPS, "--threads", self.args.server_threads]
        log_path = self.output / log_name
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(f"$ {' '.join(str(part) for part in command)}\n")
            handle.flush()
            self.server_process = subprocess.Popen([str(part) for part in command], env=env, cwd=self.output,
                                                   stdin=subprocess.DEVNULL, stdout=handle,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if self.server_process.poll() is not None:
                raise StageError(f"The policy server exited with code {self.server_process.returncode} "
                                 "before it was ready.", {"tail": self._tail(log_path)})
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=5) as response:
                    if response.status == 200:
                        return log_path
            except OSError:
                pass
            time.sleep(2)
        raise StageError("The policy server did not become ready in 10 minutes.", {"tail": self._tail(log_path)})

    def _stop_server(self):
        process, self.server_process = self.server_process, None
        if process is None or process.poll() is not None:
            return
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=30)
        except (OSError, subprocess.TimeoutExpired):
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass

    @staticmethod
    def _tail(path, lines=60):
        try:
            return "\n".join(Path(path).read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
        except OSError:
            return ""

    def stage_server(self):
        bin_dir, manifest = self._package()
        binding_python = manifest.get("python") or ""
        match = re.match(r"(\d+)\.(\d+)", binding_python)
        if not match:
            raise StageError("The package manifest does not say which Python pi05.so was built for.")
        minor = f"{match[1]}.{match[2]}"
        python = self._venv(self.libero_root / "server-venv", minor)
        requirements = self._install_requirements(python, Path(self.args.server_requirements),
                                                  self.libero_root / "server-venv" / ".eaq-requirements")
        serve = _download(SERVE_POLICY_URL, self.libero_root / "actquant" / "serve_policy.py", SERVE_POLICY_SHA256)
        env = _child_env(PYTHONPATH=str(bin_dir))
        imports = _run([python, "-c", "import sys, pi05, numpy, cv2, websockets, msgpack; print(sys.version.split()[0], "
                        "numpy.__version__, cv2.__version__, websockets.__version__, msgpack.version)"],
                       timeout=120, env=env)
        if imports["returncode"] != 0:
            raise StageError("The server venv cannot import pi05 and its dependencies.",
                             {"output": imports["output"][-2000:]})
        port = _free_port(self.args.port)
        server_log = ServerLog(self.output / "server.log")
        log_path = self._start_server(port)
        probe_script = self.libero_root / "server_probe.py"
        probe_script.write_text(SERVER_PROBE, encoding="utf-8")
        # A rendered LIBERO frame if the client stage has run, else random pixels.
        image = self.libero_root / "render_probe.png"
        probe = _run([python, probe_script, f"ws://127.0.0.1:{port}", PROBE_PROMPT, image],
                     timeout=600, env=env)
        self._stop_server()
        line = next((line for line in probe["output"].splitlines() if line.startswith("EAQ_PROBE ")), None)
        result = json.loads(line[len("EAQ_PROBE "):]) if line else None
        summary = server_log.summary()
        norm_loaded = "Loaded QUANTILE norm stats" in log_path.read_text(encoding="utf-8", errors="replace")
        details = {"python": imports["output"], "requirements": requirements, "serve_policy": serve,
                   "port": port, "probe": result, "server": summary, "norm_stats_quantile": norm_loaded}
        problems = []
        if probe["returncode"] != 0 or result is None or "server_error" in (result or {}):
            problems.append("the protocol probe failed")
        else:
            if not result["finite"]:
                problems.append("non-finite actions")
            if result["actions_shape"][-1:] != [7] or result["actions_shape"][0] < self.args.replan_steps:
                problems.append(f"unexpected action shape {result['actions_shape']}")
        if server_log.problem_count:
            problems.append(f"server log problems {summary['problems']} (zero-action fallback or errors)")
        if not norm_loaded:
            problems.append("serve_policy.py did not load quantile norm stats")
        if problems:
            raise StageError("Policy server check failed: " + "; ".join(problems) + ".",
                             {**details, "probe_output": probe["output"][-2500:], "tail": self._tail(log_path)})
        if not minor.startswith("3.11"):
            self.deviation("server", f"serve_policy.py runs in a Python {minor} venv (the Python pi05.so was built "
                                     "for) with requirements/libero-server.txt; ActQuant uses openpi's Python 3.11 "
                                     "venv.")
        return details

    def stage_profile(self):
        """Diagnostic: where a policy inference spends its time, without the simulator.

        Sends --profile-requests observations per phase (one repeated, distinct ones, distinct ones
        with a busy process on the CPU) and matches pi05's own per-inference timing summaries and
        nvidia-smi samples to each phase.
        """
        if not (self.libero_root / "server-venv" / ".eaq-requirements").is_file():
            raise StageError("The policy server is not set up; run the server stage.")
        bin_dir, manifest = self._package()
        port = _free_port(self.args.port)
        log_path = self.output / "profile_server.log"
        offset = log_path.stat().st_size if log_path.exists() else 0  # the log is appended to
        self._start_server(port, log_name=log_path.name)
        gpu = GpuSampler()
        probe_script = self.libero_root / "profile_probe.py"
        probe_script.write_text(PROFILE_PROBE, encoding="utf-8")
        image = self.libero_root / "render_probe.png"
        try:
            result = _run([self.libero_root / "server-venv" / "bin" / "python", probe_script,
                           f"ws://127.0.0.1:{port}", PROBE_PROMPT, image, self.args.profile_requests],
                          timeout=3600, env=_child_env(PYTHONPATH=str(bin_dir)))
        finally:
            gpu.stop()
            self._stop_server()
        line = next((line for line in result["output"].splitlines() if line.startswith("EAQ_PROFILE ")), None)
        probe = json.loads(line[len("EAQ_PROFILE "):]) if line else None
        if result["returncode"] != 0 or not probe or "server_error" in probe:
            raise StageError("The profile client failed.", {"output": result["output"][-2500:],
                                                            "server_log_tail": self._tail(log_path)})
        summaries = _timing_summaries(log_path, offset)
        expected = sum(phase["requests"] for phase in probe["phases"])
        if len(summaries) != expected:
            raise StageError(f"The server log has {len(summaries)} timing summaries for {expected} requests.",
                             {"server_log_tail": self._tail(log_path)})
        with open(log_path, "rb") as handle:
            handle.seek(offset)
            log_text = handle.read().decode("utf-8", errors="replace")
        phases, index = {}, 0
        for phase in probe["phases"]:
            block = summaries[index:index + phase["requests"]]
            index += phase["requests"]
            timing = {}
            for key in block[0]:
                values = [entry[key] for entry in block if key in entry]
                timing[key] = {"mean": round(sum(values) / len(values), 2), "p50": _percentile(values, 0.5),
                               "min": min(values), "max": max(values)}
            trips = phase["round_trip_ms"]
            phases[phase["phase"]] = {
                "requests": phase["requests"], "pi05_ms": timing,
                "round_trip_ms": {"mean": round(sum(trips) / len(trips), 1), "p50": _percentile(trips, 0.5),
                                  "max": max(trips)},
                "gpu": gpu.between(phase["start"], phase["end"])}
            chain = {key.split("/", 1)[1]: value["p50"] for key, value in timing.items() if key.startswith("chain/")}
            log(f"profile {phase['phase']}: round trip p50 {phases[phase['phase']]['round_trip_ms']['p50']} ms; "
                + ", ".join(f"{name} {ms:.0f}" for name, ms in chain.items()))
        smi = _run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.max.sm,clocks.max.mem,power.limit",
                    "--format=csv,noheader"])["output"].strip()
        return {"requests_per_phase": self.args.profile_requests, "phases": phases, "gpu": smi,
                "ggml": [line for line in log_text.splitlines() if line.startswith("ggml_cuda_init")
                         or "compute capability" in line],
                "package": manifest.get("name"), "server_log": Path(log_path).name}

    def stage_hold(self):
        """Diagnostic: keep the sandbox in one controlled state for --hold-minutes, logging uptime.

        idle: this script only sleeps. server: the policy server on the GPU answers a request about
        every 0.3 s, with no simulator. Comparing which of these outlives a Molab reset separates
        the platform from the parts of a rollout.
        """
        seconds = self.args.hold_minutes * 60
        mode = self.args.hold_mode
        log(f"hold: {mode} for {self.args.hold_minutes} min, sandbox uptime {_uptime_minutes()} min")
        details = {"mode": mode, "minutes": self.args.hold_minutes, "uptime_min_start": _uptime_minutes()}
        self.report["stages"]["hold"].update(details)
        self.save()
        probe = None
        if mode == "server":
            bin_dir, _ = self._package()
            port = _free_port(self.args.port)
            self._start_server(port)
            probe_script = self.libero_root / "server_probe.py"
            probe_script.write_text(SERVER_PROBE, encoding="utf-8")
            probe = subprocess.Popen(
                [str(self.libero_root / "server-venv" / "bin" / "python"), str(probe_script),
                 f"ws://127.0.0.1:{port}", PROBE_PROMPT, str(self.libero_root / "render_probe.png"), str(seconds)],
                env=_child_env(PYTHONPATH=str(bin_dir)), stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        monitor = CpuMonitor(self.server_process.pid if self.server_process else None)
        monitor.poll()
        started = time.monotonic()
        while time.monotonic() - started < seconds:
            if probe is not None and probe.poll() is not None:
                break
            if monitor.poll():
                self.report["stages"]["hold"]["uptime_min_last"] = _uptime_minutes()
                self.save()
            time.sleep(2)
        details["uptime_min_end"] = _uptime_minutes()
        details["cpu"] = monitor.summary()
        if probe is not None:
            output = probe.communicate(timeout=120)[0]
            self._stop_server()
            line = next((line for line in output.splitlines() if line.startswith("EAQ_PROBE ")), None)
            details["probe"] = json.loads(line[len("EAQ_PROBE "):]) if line else None
            if probe.returncode != 0 or not line or "server_error" in details["probe"]:
                raise StageError("The policy server stopped answering during the hold.",
                                 {**details, "probe_output": output[-2500:]})
        log(f"hold: finished, sandbox uptime {details['uptime_min_end']} min")
        return details

    def stage_rollout(self):
        client = _read_json(self.libero_root / "client.json")
        if not client:
            raise StageError("The LIBERO client is not set up; run the client stage.")
        if not (self.libero_root / "server-venv" / ".eaq-requirements").is_file():
            raise StageError("The policy server is not set up; run the server stage.")
        port = _free_port(self.args.port)
        server_log = ServerLog(self.output / "server.log")
        log_path = self._start_server(port)
        monitor = CpuMonitor(self.server_process.pid)
        monitor.poll()
        episodes_path = self.output / "episodes.jsonl"
        progress_path = self.output / "progress.json"
        total_expected = TASKS_PER_SUITE * self.args.trials * len(self.args.suites)
        results = {}
        self.report["stages"]["rollout"]["results"] = results
        progress = {"suites": self.args.suites, "trials_per_task": self.args.trials,
                    "episodes_expected": total_expected, "episodes_done": 0, "successes": 0,
                    "by_suite": {}, "server_problems": server_log.counts}

        def write_progress():
            progress["updated_utc"] = _now()
            tmp = progress_path.with_suffix(".tmp")
            tmp.write_text(json.dumps(progress, indent=2) + "\n", encoding="utf-8")
            tmp.replace(progress_path)

        write_progress()
        problems = []
        for suite in self.args.suites:
            suite_state = {"episodes": 0, "successes": 0, "expected": TASKS_PER_SUITE * self.args.trials}
            progress["by_suite"][suite] = suite_state
            progress["suite"] = suite

            def on_episode(episode, suite_state=suite_state):
                suite_state["episodes"] += 1
                suite_state["successes"] += int(episode["success"])
                progress["episodes_done"] += 1
                progress["successes"] += int(episode["success"])
                progress["last_episode"] = episode
                write_progress()

            parser = EpisodeParser(suite, episodes_path, on_episode)

            def stop():
                server_log.poll()
                sample = monitor.poll()
                if sample:
                    progress["cpu"] = sample
                write_progress()
                if self.server_process is None or self.server_process.poll() is not None:
                    return "policy server exited"
                if server_log.problem_count:
                    return f"server log problems {dict(server_log.counts)}"
                return None

            suite_log = ServerLog(log_path)
            command = [client["python"], client["main"], "--args.task-suite-name", suite,
                       "--args.num-trials-per-task", self.args.trials, "--args.replan-steps", self.args.replan_steps,
                       "--args.host", "127.0.0.1", "--args.port", port,
                       "--args.video-out-path", self.output / "videos" / suite, "--args.seed", self.args.seed]
            log(f"rollout {suite}: {TASKS_PER_SUITE} tasks x {self.args.trials} trials")
            result = _stream(command, self.output / f"client_{suite}.log", self.args.suite_hours * 3600,
                             cwd=self.output, env=self._client_env(), on_line=parser, stop=stop)
            episodes = parser.episodes
            successes = sum(episode["success"] for episode in episodes)
            exceptions = [episode for episode in episodes if episode["exception"]]
            per_task = {}
            for episode in episodes:
                task = per_task.setdefault(episode["task"], {"episodes": 0, "successes": 0})
                task["episodes"] += 1
                task["successes"] += int(episode["success"])
            suite_problems = []
            if result["stopped"]:
                suite_problems.append(f"stopped: {result['stopped']}")
            if result["timed_out"]:
                suite_problems.append(f"timed out after {self.args.suite_hours} h")
            if result["returncode"] != 0 and not result["stopped"]:
                suite_problems.append(f"client exit code {result['returncode']}")
            if len(episodes) != suite_state["expected"]:
                suite_problems.append(f"{len(episodes)} of {suite_state['expected']} episodes finished")
            if exceptions:
                suite_problems.append(f"{len(exceptions)} episodes ended in a client exception")
            server_summary = suite_log.summary()
            if suite_log.problem_count and not result["stopped"]:
                suite_problems.append(f"server log problems {server_summary['problems']}")
            results[suite] = {
                "valid": not suite_problems, "problems": suite_problems,
                "episodes": len(episodes), "expected": suite_state["expected"], "successes": successes,
                "success_rate": round(successes / len(episodes), 4) if episodes else None,
                "wilson95": _wilson(successes, len(episodes)),
                "actquant_reported": REPORTED_SUCCESS.get(suite),
                "per_task": per_task, "exceptions": [episode["exception"] for episode in exceptions][:10],
                "client_returncode": result["returncode"], "seconds": result["seconds"],
                "server": server_summary,
            }
            self.save()
            problems += [f"{suite}: {problem}" for problem in suite_problems]
            if result["stopped"] or (self.server_process is None or self.server_process.poll() is not None):
                break
        self._stop_server()
        debug_image = Path(tempfile.gettempdir()) / "debug_base_image.png"
        if debug_image.is_file():  # serve_policy.py saves the first image it receives here
            shutil.copy2(debug_image, self.output / "server_first_image.png")
        server_total = server_log.summary()
        if self.args.trials < 50:
            self.deviation("trials", f"{self.args.trials} trial(s) per task (initial states 0-{self.args.trials - 1}) "
                                     "instead of ActQuant's 50; success rates are far less precise.")
        self.deviation("launcher", "One policy server on one GPU and openpi's main.py with --args.port; "
                                   "ActQuant's run_libero_eval.sh shards tasks over one server per GPU (--args.ports).")
        details = {"results": results, "port": port, "server": server_total, "cpu": monitor.summary(),
                   "threads": {"client": self.args.client_threads, "server": self.args.server_threads}}
        if problems:
            raise StageError("Rollout invalid: " + "; ".join(problems) + ".", details)
        return details


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repository = Path(__file__).resolve().parent.parent
    parser.add_argument("--output", required=True, help="run folder for report.json and logs")
    parser.add_argument("--work-dir", default=os.environ.get("EAQ_WORK_DIR",
                                                             str(Path(tempfile.gettempdir()) / "eaq-actquant")))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(DEFAULT_STAGES))
    parser.add_argument("--suites", nargs="+", choices=SUITES, default=["libero_spatial"])
    parser.add_argument("--trials", type=int, default=1, help="trials per task (ActQuant uses 50)")
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="CUDA0", help="ggml device for the policy server")
    parser.add_argument("--model", choices=list(CHECKPOINTS), default="actquant-3bpw",
                        help="the released 3-bit checkpoint, or the FP16 reference exported from its base")
    parser.add_argument("--model-cache", help="folder to reuse an FP16 export from and store one in")
    parser.add_argument("--gl", default="auto", choices=["auto", *(name for name, _ in GL_CANDIDATES)])
    parser.add_argument("--client-threads", type=int, default=1,
                        help="cap for the client's thread pools (OpenMP, BLAS, numba, Mesa llvmpipe)")
    parser.add_argument("--server-threads", type=int, default=2,
                        help="ggml CPU threads in the policy server (the GPU does the model's work)")
    parser.add_argument("--hold-mode", choices=["idle", "server"], default="idle",
                        help="the hold stage: sleep only, or keep the GPU policy server answering requests")
    parser.add_argument("--hold-minutes", type=float, default=15.0)
    parser.add_argument("--profile-requests", type=int, default=20, help="requests per phase of the profile stage")
    parser.add_argument("--suite-hours", type=float, default=10.0, help="time limit per suite")
    parser.add_argument("--actquant-script", default=str(repository / "scripts" / "actquant_build.py"))
    parser.add_argument("--toolkit-requirements", default=str(repository / "requirements" / "actquant-cuda-toolkit.txt"))
    parser.add_argument("--prebuilt-pin", default=str(repository / "requirements" / "actquant-prebuilt.json"))
    parser.add_argument("--client-requirements", default=str(repository / "requirements" / "libero-client.txt"))
    parser.add_argument("--server-requirements", default=str(repository / "requirements" / "libero-server.txt"))
    args = parser.parse_args()
    if not 1 <= args.trials <= 50:
        parser.error("--trials must be between 1 and 50 (LIBERO has 50 initial states per task)")
    def on_sigterm(signum, frame):
        raise KeyboardInterrupt

    # A stop request (SIGTERM) unwinds normally, so the policy server is shut down and the report saved.
    signal.signal(signal.SIGTERM, on_sigterm)
    rollout = Rollout(args)
    try:
        return rollout.run()
    except KeyboardInterrupt:
        rollout._stop_server()
        rollout.report["status"] = "stopped"
        for entry in rollout.report["stages"].values():
            if entry.get("status") == "running":
                entry["status"] = "stopped"
        rollout.save()
        (rollout.output / "exit_code").write_text("130\n", encoding="utf-8")
        log("stopped")
        return 130


if __name__ == "__main__":
    sys.exit(main())
