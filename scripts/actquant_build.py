"""Build ActQuant's Pi 0.5 runtime on a Linux GPU host and run one inference.

Applies the Molab build recipe established by `cloud_preflight.py --only
actquant`, with one change found by the first build: PyTorch's pip CUDA
toolkit mixes nvcc 13.3 with 13.0 runtime headers, which CCCL (CUB) rejects,
so the build uses a private, version-consistent CUDA toolkit installed from
pinned pip wheels into the work folder. Device code is built for the attached
GPU only, with unversioned library links and a toolkit RPATH. Stages run in
order and each writes `report.json`, so a partial run still leaves evidence:

  toolkit    install the pinned CUDA toolkit wheels into the work folder and
             check that nvcc, runtime headers and CCCL agree (compiles CUB)
  source     fetch ActQuant at its pinned commit and unpack vendored code
  configure  CMake + Ninja configure of the pi05 runtime
  build      build pi05, llama-quantize and, when possible, the pi05.so binding
  download   fetch the pinned 3-bit checkpoint and verify its SHA-256
  infer      one CUDA inference through the CLI (and the binding, if built)

The work directory keeps sources, build trees and checkpoints between runs in
one session, so stages can be rerun individually and builds are incremental.
A lock in the work directory stops two runs from building the same tree; the
notebook starts this script as a detached process so a disconnected notebook
does not kill the build.
This is one self-contained file: Molab fetches it alone and may run Python
with PYTHONSAFEPATH, which blocks sibling-module imports.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import glob
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time
import zipfile


ACTQUANT_URL = "https://github.com/arashakb/ActQuant"
ACTQUANT_COMMIT = "b64791125070652fe6b554e244fe809c79ef5246"
CHECKPOINT_REPO = "NU-World-Model-Embodied-AI/ActQuant-Pi05-LIBERO-3bpw"
CHECKPOINT_REVISION = "4d03f36f1ac019ad5e314dea584697ab29644171"
# SHA-256 of the LFS files at CHECKPOINT_REVISION; None = small file, not LFS-tracked.
CHECKPOINT_FILES = {
    "pi05.gguf": "7061b410272bbd35ad75e5919140fc55d2c5ed158f7f2c1ab45311ff39e54555",
    "tokenizer.model": "8986bb4f423f07f8c7f70d0dbe3526fb2316056c17bae71b1ea975e77a168fc6",
    "norm_stats.json": None,
}
BUILD_TARGETS = ("pi05", "llama-quantize")
DEFAULT_PROMPT = "put the black bowl on the plate"
STAGES = ("toolkit", "source", "configure", "build", "download", "infer")
# Compiled, linked and run by the toolkit stage, as CMake's compiler check and the build will:
# build 1 failed in the one ggml file using CUB, build 2 at CMake's link test.
TOOLKIT_PROBE = """
#include <cstdio>
#include <cuda_fp16.h>
#include <cub/cub.cuh>
__global__ void eaq_probe(const half *x, float *y) { y[threadIdx.x] = __half2float(x[threadIdx.x]); }
int main() {
    half *x; float *y;
    if (cudaMalloc(&x, 32 * sizeof(half)) != cudaSuccess || cudaMalloc(&y, 32 * sizeof(float)) != cudaSuccess) {
        printf("FAIL malloc\\n"); return 2;
    }
    cudaMemset(x, 0, 32 * sizeof(half));
    eaq_probe<<<1, 32>>>(x, y);
    cudaError_t err = cudaDeviceSynchronize();
    int runtime = 0, driver = 0; cudaRuntimeGetVersion(&runtime); cudaDriverGetVersion(&driver);
    printf("%s runtime=%d driver=%d\\n", err == cudaSuccess ? "OK" : cudaGetErrorString(err), runtime, driver);
    return err == cudaSuccess ? 0 : 3;
}
"""
TIMEOUTS = {"configure": 15 * 60, "build": 3 * 60 * 60, "infer": 20 * 60}
# Molab resets the whole sandbox partway through long builds, while memory use stays low; builds
# with more parallel jobs got further (20 jobs: step 232 of 234; 2 jobs: about 51), which looks
# like a time budget. So the default favours speed: as many jobs as reported CPUs, limited by
# memory. The build runs at low priority so the notebook server keeps getting CPU time.
DEFAULT_MAX_JOBS = 16
GIB_PER_JOB = 4
BUILD_NICENESS = 10


class StageError(RuntimeError):
    """A stage failed; the message is shown to the user, details stay in the report."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details or {}


def log(message):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)


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


def _read_bytes_value(path):
    try:
        text = Path(path).read_text(encoding="utf-8").strip()
    except OSError:
        return None
    # "max" (v2) or a near-2^63 value (v1) both mean "no limit".
    return int(text) if text.isdigit() and int(text) < 2**60 else None


def _meminfo():
    try:
        with open("/proc/meminfo", encoding="utf-8") as handle:
            return {key: int(value.split()[0]) for key, value in
                    (line.split(":", 1) for line in handle if ":" in line)}
    except (OSError, ValueError, IndexError):
        return {}


def _uptime():
    try:
        return float(Path("/proc/uptime").read_text(encoding="utf-8").split()[0])
    except (OSError, ValueError, IndexError):
        return None


def _tree_gib(root):
    """Total size of the files under root, without following links; None if missing."""
    if not os.path.isdir(root):
        return None
    total = 0
    for folder, _, files in os.walk(root, onerror=lambda error: None):
        for name in files:
            try:
                total += os.lstat(os.path.join(folder, name)).st_size
            except OSError:
                pass
    return round(total / 2**30, 2)


def resource_sample(with_files=False):
    """What might end a sandboxed session, in GiB: resident memory of all visible processes, the
    kernel's memory counters (in gVisor, files in the sandbox filesystem can be held in memory),
    load average and process count, cgroup usage and limit when exposed, and optionally the size
    of the scratch folders."""
    sample = {}
    if os.path.isdir("/proc"):
        pages, processes = 0, 0
        for statm in glob.glob("/proc/[0-9]*/statm"):
            try:
                with open(statm, encoding="utf-8") as handle:
                    pages += int(handle.read().split()[1])
                processes += 1
            except (OSError, ValueError, IndexError):
                pass
        sample["processes"] = processes
        sample["processes_rss_gib"] = round(pages * os.sysconf("SC_PAGE_SIZE") / 2**30, 1)
        # gVisor leaves /proc/loadavg at zero; its /proc/uptime is the sandbox's, which dates resets.
        uptime = _uptime()
        if uptime is not None:
            sample["uptime_min"] = round(uptime / 60, 1)
    meminfo = _meminfo()
    for field, key in (("MemAvailable", "available_gib"), ("MemFree", "free_gib"),
                       ("Cached", "cached_gib"), ("Shmem", "shmem_gib"), ("AnonPages", "anon_gib")):
        if field in meminfo:
            sample[key] = round(meminfo[field] / 2**20, 1)
    for key, paths in (("cgroup_used_gib", ("/sys/fs/cgroup/memory.current",
                                            "/sys/fs/cgroup/memory/memory.usage_in_bytes")),
                       ("cgroup_limit_gib", ("/sys/fs/cgroup/memory.max",
                                             "/sys/fs/cgroup/memory/memory.limit_in_bytes"))):
        for path in paths:
            value = _read_bytes_value(path)
            if value is not None:
                sample[key] = round(value / 2**30, 1)
                break
    if with_files:
        sample["tmp_files_gib"] = _tree_gib(tempfile.gettempdir())
        sample["cache_files_gib"] = _tree_gib(os.path.expanduser("~/.cache"))
    return sample


class ResourceWatch:
    """Log resources every few seconds during a long stage, so a killed session leaves evidence."""

    def __init__(self, log_path, interval=15, files_every=4, high_rss_gib=24.0):
        self.log_path, self.interval, self.files_every = log_path, interval, files_every
        self.high_rss_gib = high_rss_gib
        self.peaks = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=60)

    def _loop(self):
        last_warning, count = 0.0, 0
        with open(self.log_path, "a", encoding="utf-8") as handle:
            while True:
                sample = resource_sample(with_files=count % self.files_every == 0)
                count += 1
                handle.write(json.dumps({"time": dt.datetime.now().strftime("%H:%M:%S"), **sample}) + "\n")
                handle.flush()
                for key in ("processes_rss_gib", "processes", "cached_gib", "shmem_gib",
                            "tmp_files_gib", "cgroup_used_gib"):
                    if sample.get(key) is not None:
                        self.peaks[key] = max(sample[key], self.peaks.get(key, sample[key]))
                if "available_gib" in sample:
                    self.peaks["min_available_gib"] = min(sample["available_gib"],
                                                          self.peaks.get("min_available_gib", sample["available_gib"]))
                rss = sample.get("processes_rss_gib")
                if rss is not None and rss > self.high_rss_gib and time.monotonic() - last_warning > 60:
                    last_warning = time.monotonic()
                    log(f"high memory use: {rss} GiB resident in all processes")
                if self._stop.wait(self.interval):
                    break

    @property
    def summary(self):
        return {"peaks": self.peaks, "log": self.log_path.name}


def cpu_quota():
    """CPUs allowed by the cgroup (cpu.max or v1 cfs quota), or None when unlimited or hidden."""
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text(encoding="utf-8").split()[:2]
    except (OSError, ValueError):
        quota = _read_bytes_value("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
        period = _read_bytes_value("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    try:
        return round(int(quota) / int(period), 1)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _stream(command, log_path, timeout, cwd=None, env=None, echo=True):
    """Run a long command, appending its output to log_path and echoing it for the notebook."""
    command = [str(part) for part in command]
    started = time.monotonic()
    tail = collections.deque(maxlen=80)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n")
        log_file.flush()
        try:
            process = subprocess.Popen(command, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, errors="replace", bufsize=1)
        except OSError as exc:
            return {"returncode": None, "seconds": 0.0, "timed_out": False,
                    "tail": f"could not start: {type(exc).__name__}: {exc}"}

        def pump():
            for line in process.stdout:
                log_file.write(line)
                tail.append(line.rstrip("\n"))
                if echo:
                    print(line, end="", flush=True)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        try:
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            process.kill()
            process.wait()
        reader.join(timeout=30)
        log_file.flush()
    return {"returncode": process.returncode, "seconds": round(time.monotonic() - started, 1),
            "timed_out": timed_out, "tail": "\n".join(tail)}


def _version_tuple(text):
    match = re.search(r"(\d+)\.(\d+)", text or "")
    return (int(match[1]), int(match[2])) if match else None


def _package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _which(name):
    """Find a tool on PATH or beside this interpreter, where pip puts cmake/ninja entry points."""
    return shutil.which(name) or shutil.which(name, path=os.pathsep.join(
        {str(Path(sys.executable).parent), sysconfig.get_paths()["scripts"]}))


def _toolkit_info(nvcc, source):
    """Versions that must agree: nvcc itself and the CUDART_VERSION of the runtime headers it will use."""
    root = nvcc.resolve().parent.parent
    match = re.search(r"release (\d+\.\d+)", _run([nvcc, "--version"], timeout=30)["output"])
    runtime = None
    try:
        define = re.search(r"#define\s+CUDART_VERSION\s+(\d+)",
                           (root / "include" / "cuda_runtime_api.h").read_text(encoding="utf-8", errors="replace"))
        if define:
            runtime = f"{int(define[1]) // 1000}.{int(define[1]) % 1000 // 10}"
    except OSError:
        pass
    return {"nvcc": str(nvcc), "version": match[1] if match else None, "runtime_headers": runtime,
            "cccl_headers": (root / "include" / "nv" / "target").is_file(), "root": str(root), "source": source}


def find_toolkit(private_prefix):
    """The private pinned toolkit if installed, else a system CUDA install. PyTorch's mixed pip toolkit is never used."""
    candidates = [(path, "pinned pip wheels (private)") for path in sorted(Path(private_prefix).glob("nvidia/cu*/bin/nvcc"))]
    candidates += [(Path(path), "system") for path in
                   filter(None, [shutil.which("nvcc"), *sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))])
                   if "site-packages" not in path and "dist-packages" not in path]
    for nvcc, source in candidates:
        if nvcc.is_file():
            return _toolkit_info(nvcc, source)
    return None


def _toolkit_problems(toolkit):
    problems = []
    if not toolkit["cccl_headers"]:
        problems.append("CCCL headers (include/nv/target) are missing")
    if toolkit["runtime_headers"] != toolkit["version"]:
        problems.append(f"nvcc {toolkit['version']} does not match runtime headers {toolkit['runtime_headers']}; "
                        "CCCL rejects this mix")
    return problems


def gpu_info():
    result = _run(["nvidia-smi", "--query-gpu=name,compute_cap,driver_version,memory.total",
                   "--format=csv,noheader"], timeout=30)
    if result["returncode"] != 0 or not result["output"]:
        return None
    name, capability, driver, memory = [part.strip() for part in result["output"].splitlines()[0].split(",")]
    cuda = re.search(r"CUDA Version:\s*(\d+\.\d+)", _run(["nvidia-smi"], timeout=30)["output"])
    return {"name": name, "compute_capability": capability, "driver": driver, "memory": memory,
            "driver_cuda": cuda[1] if cuda else None}


def _find_libcuda():
    """The driver library, which lives outside the toolkit; CMake's CUDA::cuda_driver needs libcuda.so."""
    result = _run(["ldconfig", "-p"], timeout=30)
    for line in result["output"].splitlines():
        if "libcuda.so.1 " in line and "=>" in line:
            return line.split("=>")[-1].strip()
    for pattern in ("/usr/lib/x86_64-linux-gnu/libcuda.so.1", "/usr/lib64/libcuda.so.1",
                    "/usr/local/nvidia/lib64/libcuda.so.1", "/usr/lib/wsl/lib/libcuda.so.1"):
        if Path(pattern).exists():
            return pattern
    return None


def _library_shim(toolkit_root, shim_dir):
    """Link lib*.so -> lib*.so.N: pip toolkits ship only versioned sonames, which the linker ignores."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for library in sorted(Path(toolkit_root, "lib").glob("lib*.so.*")):
        unversioned = library.name.split(".so.")[0] + ".so"
        if not (library.parent / unversioned).exists() and not (shim_dir / unversioned).exists():
            (shim_dir / unversioned).symlink_to(library)
            linked.append(unversioned)
    libcuda = _find_libcuda()
    if libcuda and not (Path(libcuda).parent / "libcuda.so").exists() and not (shim_dir / "libcuda.so").exists():
        (shim_dir / "libcuda.so").symlink_to(libcuda)
        linked.append(f"libcuda.so -> {libcuda}")
    return linked, libcuda


def _read_pins(path):
    if not path or not Path(path).is_file():
        return []
    return [line.partition("#")[0].strip() for line in Path(path).read_text(encoding="utf-8").splitlines()
            if "==" in line.partition("#")[0]]


class Builder:
    def __init__(self, args):
        self.args = args
        self.work = Path(args.work_dir).resolve()
        self.output = Path(args.output).resolve()
        self.source = self.work / "ActQuant"
        self.checkpoint = self.work / "checkpoints" / "actquant-pi05-libero-3bpw"
        self.toolkit_prefix = self.work / "cuda-toolkit"
        self.toolkit = find_toolkit(self.toolkit_prefix)
        self.gpu = gpu_info()
        capability = (self.gpu or {}).get("compute_capability") or ""
        self.arch = args.cuda_arch or capability.replace(".", "") or None
        self.deviations = {}
        self.report = {
            "schema_version": 2,
            "scope": "ActQuant Pi 0.5 build and a single-inference smoke test; no LIBERO rollout",
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "actquant": {"url": ACTQUANT_URL, "commit": ACTQUANT_COMMIT},
            "checkpoint": {"repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION},
            "options": {"stages": args.stages, "cuda_arch": self.arch, "no_vmm": args.no_vmm,
                        "no_flash_attn": args.no_flash_attn,
                        "jobs": args.jobs, "cpu_check": args.cpu_check, "prompt": args.prompt,
                        "work_dir": str(self.work)},
            "host": self._host(),
            "stages": {},
        }

    @property
    def _suffix(self):
        # One tree per architecture, toolkit version and VMM setting: CMake cannot switch compilers in place.
        version = (self.toolkit or {}).get("version") or "none"
        return (f"sm{self.arch}-cu{version}" + ("-novmm" if self.args.no_vmm else "")
                + ("-nofa" if self.args.no_flash_attn else ""))

    @property
    def build_dir(self):
        return self.work / f"build-{self._suffix}"

    @property
    def shim_dir(self):
        return self.work / f"libshim-{self._suffix}"

    def _host(self):
        memory = None
        try:
            with open("/proc/meminfo", encoding="utf-8") as handle:
                fields = dict(line.split(":", 1) for line in handle if ":" in line)
            memory = {key: round(int(fields[key].split()[0]) / 2**20, 1) for key in ("MemTotal", "MemAvailable")}
        except (OSError, KeyError, ValueError):
            pass
        if memory is not None and "cgroup_limit_gib" in (sample := resource_sample()):
            memory["cgroup_limit_gib"] = sample["cgroup_limit_gib"]
        try:
            usage = shutil.disk_usage(self.work.parent if not self.work.exists() else self.work)
            disk = {"free_gib": round(usage.free / 2**30, 1), "total_gib": round(usage.total / 2**30, 1)}
            if usage.total > 2**50:
                disk["note"] = "sandbox reports an implausible filesystem size; free space is not measurable"
        except OSError:
            disk = None
        try:
            cpus = len(os.sched_getaffinity(0))
        except AttributeError:
            cpus = os.cpu_count()
        return {"python": sys.version.split()[0], "executable": sys.executable,
                "platform": platform.platform(), "cpus": cpus, "cpu_quota": cpu_quota(),
                "uptime_min_at_start": round(_uptime() / 60, 1) if _uptime() is not None else None,
                "memory_gib": memory, "disk": disk,
                "gpu": self.gpu, "tools": {name: self._tool_version(name) for name in ("cmake", "ninja", "gcc", "git")},
                "packages": {name: _package_version(name) for name in (
                    "torch", "nvidia-cuda-nvcc", "nvidia-cuda-runtime", "pybind11", "huggingface_hub",
                    "cmake", "ninja")}}

    @staticmethod
    def _tool_version(name):
        path = _which(name)
        if not path:
            return None
        lines = _run([path, "--version"], timeout=30)["output"].splitlines()
        return lines[0] if lines else path

    # ------------------------------------------------------------------ report

    def deviation(self, key, text):
        self.deviations[key] = text

    def save(self):
        self.report["toolkit"] = self.toolkit
        self.report["options"]["build_dir"] = str(self.build_dir)
        self.report["deviations"] = list(self.deviations.values())
        self.report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        (self.output / "report.json").write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")

    def _lock(self):
        """Hold an exclusive lock on the work folder, or return a description of its holder."""
        try:
            import fcntl
        except ImportError:  # not Linux; nothing to guard against locally
            return None
        self._lock_file = open(self.work / ".build.lock", "a+", encoding="utf-8")
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

    def run(self):
        self.output.mkdir(parents=True, exist_ok=True)
        self.work.mkdir(parents=True, exist_ok=True)
        holder = self._lock()
        if holder:
            log(f"work folder {self.work} is in use by {holder}; not starting a second run")
            self.report["status"] = "failed"
            self.report["error"] = f"work folder in use by {holder}"
            self.save()
            (self.output / "exit_code").write_text("1\n", encoding="utf-8")
            return 1
        # Low priority, so the notebook server keeps getting CPU time (children inherit it).
        try:
            self.report["niceness"] = os.nice(BUILD_NICENESS)
        except (AttributeError, OSError):
            self.report["niceness"] = None
        requested = [stage for stage in STAGES if stage in self.args.stages]
        failed = None
        # "running" stays in report.json if the process is killed, showing where it stopped.
        self.report["status"] = "running"
        for stage in requested:
            if failed:
                self.report["stages"][stage] = {"status": "skipped", "reason": f"stage '{failed}' failed"}
                continue
            log(f"=== stage: {stage} ===")
            self.report["stages"][stage] = {"status": "running",
                                            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
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
        if failed:
            self.report["status"] = "failed"
            self.report["failed_stage"] = failed
        elif "infer" in requested:
            self.report["status"] = "inference_ok"
        elif "build" in requested:
            self.report["status"] = "built"
        else:
            self.report["status"] = "stages_passed"
        self.save()
        log(f"status: {self.report['status']}")
        code = 0 if not failed else 1
        (self.output / "exit_code").write_text(f"{code}\n", encoding="utf-8")
        return code

    # ------------------------------------------------------------------ stages

    def _require_toolkit(self):
        if self.toolkit is None:
            raise StageError("No CUDA toolkit found; run the toolkit stage.")
        problems = _toolkit_problems(self.toolkit)
        if problems:
            raise StageError("Inconsistent CUDA toolkit: " + "; ".join(problems) + ".", {"toolkit": self.toolkit})
        return self.toolkit

    def stage_toolkit(self):
        pins = _read_pins(self.args.toolkit_requirements)
        if not pins:
            raise StageError(f"No toolkit pins found in {self.args.toolkit_requirements}.")
        nvcc_pin = next((pin for pin in pins if pin.startswith("nvidia-cuda-nvcc==")), None)
        wanted = _version_tuple(nvcc_pin.split("==")[1]) if nvcc_pin else None
        driver = _version_tuple((self.gpu or {}).get("driver_cuda"))
        details = {"pins": pins, "prefix": str(self.toolkit_prefix), "driver_cuda": (self.gpu or {}).get("driver_cuda")}
        if wanted is None:
            raise StageError("The toolkit pins do not include nvidia-cuda-nvcc.", details)
        if driver and wanted > driver:
            raise StageError(f"Pinned nvcc {wanted[0]}.{wanted[1]} is newer than the driver's CUDA "
                             f"{driver[0]}.{driver[1]}.", details)

        marker = self.toolkit_prefix / "eaq-pins.txt"
        if marker.is_file() and marker.read_text(encoding="utf-8").splitlines() == pins:
            details["note"] = "Pinned toolkit already installed in the work folder."
        else:
            if self.toolkit_prefix.exists():
                shutil.rmtree(self.toolkit_prefix)
            commands = []
            if _which("uv"):
                commands.append([_which("uv"), "pip", "install", "--python", sys.executable,
                                 "--target", self.toolkit_prefix, "--no-deps", *pins])
            commands.append([sys.executable, "-m", "pip", "install", "--target", self.toolkit_prefix,
                             "--no-deps", *pins])
            attempts = []
            for command in commands:
                log(f"installing the pinned CUDA toolkit ({len(pins)} wheels, about 0.5 GB)")
                result = _run(command, timeout=1800)
                attempts.append({"command": [str(part) for part in command], "returncode": result["returncode"],
                                 "output": result["output"][-1500:]})
                if result["returncode"] == 0:
                    break
                shutil.rmtree(self.toolkit_prefix, ignore_errors=True)
            details["attempts"] = attempts
            if attempts[-1]["returncode"] != 0:
                raise StageError("Could not install the pinned CUDA toolkit wheels.", details)
            marker.write_text("\n".join(pins) + "\n", encoding="utf-8")

        self.toolkit = find_toolkit(self.toolkit_prefix)
        details["toolkit"] = self.toolkit
        if self.toolkit is None or self.toolkit["source"] == "system":
            raise StageError("The installed wheels did not produce nvidia/cu*/bin/nvcc.", details)
        if _version_tuple(self.toolkit["version"]) != wanted:
            raise StageError(f"Installed nvcc reports {self.toolkit['version']}, expected "
                             f"{wanted[0]}.{wanted[1]}.", details)
        self._require_toolkit()

        # The 13.0 nvcc wheel's nvcc.profile still searches lib64 (the system-install layout), while
        # the wheels install into lib; later wheels fixed the profile. A link keeps nvcc's default
        # link step (-lcudadevrt -lcudart_static) working, which CMake's compiler check relies on.
        root = Path(self.toolkit["root"])
        if not (root / "lib64").exists():
            (root / "lib64").symlink_to("lib", target_is_directory=True)
            details["lib64_link"] = "created lib64 -> lib"

        # Compile, link and run CUB + cuda_fp16 for this GPU: seconds here instead of a failed long build.
        if self.arch:
            probe_dir = self.output / "toolkit_probe"
            probe_dir.mkdir(exist_ok=True)
            (probe_dir / "probe.cu").write_text(TOOLKIT_PROBE, encoding="utf-8")
            result = _run([self.toolkit["nvcc"], "-std=c++17", f"-arch=sm_{self.arch}",
                           probe_dir / "probe.cu", "-o", probe_dir / "probe"], timeout=300)
            details["probe_build"] = {"returncode": result["returncode"], "output": result["output"][-2000:]}
            if result["returncode"] != 0:
                raise StageError("The pinned toolkit cannot compile and link CUB for this GPU "
                                 "(see probe_build).", details)
            result = _run([probe_dir / "probe"], timeout=120)
            details["probe_run"] = {"returncode": result["returncode"], "output": result["output"][-500:]}
            if result["returncode"] != 0:
                raise StageError("The probe built by the pinned toolkit does not run on this GPU "
                                 "(see probe_run).", details)
        self.deviation("toolkit", f"CUDA toolkit {self.toolkit['version']} from pinned pip wheels in a private "
                                  f"folder ({self.toolkit_prefix}), instead of the system CUDA 12.6 ActQuant "
                                  "documents. PyTorch's own pip toolkit mixes nvcc 13.3 with 13.0 runtime "
                                  "headers, which CCCL rejects.")
        return details

    def stage_source(self):
        git = _which("git")
        if not git:
            raise StageError("git is not installed.")
        self.source.mkdir(parents=True, exist_ok=True)
        if not (self.source / ".git").is_dir():
            _run([git, "init", "-q", self.source])
            _run([git, "-C", self.source, "remote", "add", "origin", ACTQUANT_URL])
        head = _run([git, "-C", self.source, "rev-parse", "HEAD"])["output"].strip()
        if head != ACTQUANT_COMMIT:
            log(f"fetching {ACTQUANT_URL} @ {ACTQUANT_COMMIT[:12]}")
            fetch = _run([git, "-C", self.source, "fetch", "--depth", "1", "origin", ACTQUANT_COMMIT], timeout=900)
            if fetch["returncode"] != 0:
                raise StageError("git fetch of the pinned ActQuant commit failed.",
                                 {"output": fetch["output"][-2000:]})
            checkout = _run([git, "-C", self.source, "checkout", "-q", "--detach", "FETCH_HEAD"], timeout=300)
            if checkout["returncode"] != 0:
                raise StageError("git checkout failed.", {"output": checkout["output"][-2000:]})
        head = _run([git, "-C", self.source, "rev-parse", "HEAD"])["output"].strip()
        if head != ACTQUANT_COMMIT:
            raise StageError(f"Checked-out commit {head} is not the pinned {ACTQUANT_COMMIT}.")
        dirty = _run([git, "-C", self.source, "status", "--porcelain", "--untracked-files=no"])["output"]
        vendored = self.source / "vendor" / "tokenizers-cpp"
        if not vendored.is_dir():
            # As in ActQuant's README: unzip vendor/tokenizers-cpp.zip -d vendor/
            with zipfile.ZipFile(self.source / "vendor" / "tokenizers-cpp.zip") as archive:
                archive.extractall(self.source / "vendor")
        return {"commit": head, "path": str(self.source), "modified_tracked_files": dirty.splitlines(),
                "tokenizers_cpp": vendored.is_dir()}

    def _binding_plan(self):
        """Build pi05.so only if this interpreter can: pybind11 importable and Python.h present."""
        include = Path(sysconfig.get_paths()["include"])
        reasons = []
        if importlib.util.find_spec("pybind11") is None:
            reasons.append("pybind11 is not installed (install it from the Packages panel)")
        if not (include / "Python.h").is_file():
            reasons.append(f"Python.h not found in {include}")
        return (not reasons and not self.args.no_binding), reasons

    def stage_configure(self):
        toolkit = self._require_toolkit()
        cmake, ninja = _which("cmake"), _which("ninja")
        if not cmake:
            raise StageError("cmake is not installed; install the pin from requirements/actquant-build.txt.")
        if not (self.source / "CMakeLists.txt").is_file():
            raise StageError("ActQuant sources are missing; run the source stage.")
        if not self.arch:
            raise StageError("No GPU compute capability found; attach a GPU or pass --cuda-arch.")
        linked, libcuda = _library_shim(toolkit["root"], self.shim_dir)
        binding, binding_reasons = self._binding_plan()
        base = [cmake, "-S", self.source, "-B", self.build_dir,
                *(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"] if ninja else ["-G", "Unix Makefiles"]),
                "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=ON",
                f"-DCMAKE_CUDA_COMPILER={toolkit['nvcc']}", f"-DCUDAToolkit_ROOT={toolkit['root']}",
                f"-DCMAKE_CUDA_ARCHITECTURES={self.arch}",
                f"-DCMAKE_LIBRARY_PATH={self.shim_dir}",
                f"-DCMAKE_BUILD_RPATH={Path(toolkit['root']) / 'lib'}",
                "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
                "-DLLAMA_CURL=OFF",
                f"-DPython_EXECUTABLE={sys.executable}",
                *(["-DGGML_CUDA_NO_VMM=ON"] if self.args.no_vmm else []),
                *(["-DGGML_CUDA_FA=OFF"] if self.args.no_flash_attn else [])]
        log_path = self.output / "configure.log"
        attempts = []
        result = _stream([*base, f"-DBUILD_PI05_PYTHON={'ON' if binding else 'OFF'}"], log_path,
                         TIMEOUTS["configure"])
        attempts.append({"binding": binding, "returncode": result["returncode"], "seconds": result["seconds"]})
        if result["returncode"] != 0 and binding:
            log("configure failed with the Python binding enabled; retrying without it")
            binding_reasons.append("CMake could not configure the binding (see configure.log)")
            binding = False
            result = _stream([*base, "-DBUILD_PI05_PYTHON=OFF"], log_path, TIMEOUTS["configure"])
            attempts.append({"binding": False, "returncode": result["returncode"], "seconds": result["seconds"]})
        details = {"attempts": attempts, "library_links": linked, "libcuda": libcuda,
                   "binding": binding, "binding_skipped_because": binding_reasons, "generator":
                   "Ninja" if ninja else "Unix Makefiles"}
        if result["returncode"] != 0:
            raise StageError("CMake configure failed; see configure.log.", {**details, "tail": result["tail"]})
        architectures = re.findall(r"Using CUDA architectures: (.*)", log_path.read_text(encoding="utf-8"))
        details["cuda_architectures_reported"] = architectures[-1] if architectures else None
        self.report["binding"] = {"enabled": binding, "skipped_because": binding_reasons}
        self.deviations.setdefault("toolkit", f"CUDA toolkit: {toolkit['source']} nvcc {toolkit['version']} "
                                              f"at {toolkit['root']} (ActQuant documents CUDA 12.6 for Pi 0.5).")
        self.deviation("arch", f"CMAKE_CUDA_ARCHITECTURES={self.arch}: device code for this GPU only "
                               "(ActQuant's README leaves it to ggml's default).")
        self.deviation("libs", "Unversioned library links in a scratch folder (CMAKE_LIBRARY_PATH) and "
                               "CMAKE_BUILD_RPATH to the toolkit's lib folder.")
        self.deviation("curl", "LLAMA_CURL=OFF: the runtime does not download models and libcurl "
                               "headers are not assumed.")
        if self.args.no_vmm:
            self.deviation("vmm", "GGML_CUDA_NO_VMM=ON: CUDA virtual memory management disabled.")
        if self.args.no_flash_attn:
            self.deviation("flash_attn", "GGML_CUDA_FA=OFF: ggml's FlashAttention CUDA kernels compiled as "
                           "stubs to shorten the build (tools/pi0.5 never calls ggml_flash_attn_ext).")
        self.deviation("binding", f"pi05.so built for Python {platform.python_version()} "
                                  "(ActQuant's policy server uses Python 3.11)." if binding else
                       "pi05.so not built: " + "; ".join(binding_reasons))
        return details

    def _jobs(self):
        if self.args.jobs:
            return self.args.jobs
        host = self.report["host"]
        # The cgroup quota, when visible, is the real CPU count; otherwise use what is reported.
        cpus = int(host["cpu_quota"] or host["cpus"] or DEFAULT_MAX_JOBS)
        memory = host["memory_gib"] or {}
        budgets = [value for value in (memory.get("MemAvailable"), memory.get("cgroup_limit_gib")) if value]
        # CUDA template instances take 2-4 GiB each to compile.
        by_memory = int(min(budgets) // GIB_PER_JOB) if budgets else cpus
        return max(1, min(cpus, by_memory, DEFAULT_MAX_JOBS))

    def stage_build(self):
        cmake = _which("cmake")
        if not cmake or not (self.build_dir / "CMakeCache.txt").is_file():
            raise StageError("No configured build tree; run the configure stage.")
        cache = (self.build_dir / "CMakeCache.txt").read_text(encoding="utf-8", errors="replace")
        binding = "BUILD_PI05_PYTHON:BOOL=ON" in cache
        targets = [*BUILD_TARGETS, *(["pi05_py"] if binding else [])]
        jobs = self._jobs()
        log(f"building {', '.join(targets)} with {jobs} parallel jobs; the first build takes a while")
        with ResourceWatch(self.output / "resources.log") as resources:
            result = _stream([cmake, "--build", self.build_dir, "--target", *targets, "-j", str(jobs)],
                             self.output / "build.log", self.args.build_timeout)
        details = {"targets": targets, "jobs": jobs, "niceness": self.report.get("niceness"),
                   "returncode": result["returncode"], "build_seconds": result["seconds"],
                   "resources": resources.summary}
        if result["timed_out"]:
            raise StageError(f"Build timed out after {self.args.build_timeout} s; rerun to continue "
                             "(the build is incremental).", {**details, "tail": result["tail"][-4000:]})
        if result["returncode"] != 0:
            errors = [line for line in result["tail"].splitlines() if "error" in line.lower()]
            raise StageError("Build failed; see build.log.",
                             {**details, "errors": errors[-20:], "tail": result["tail"][-4000:]})
        binary_dir = self.build_dir / "bin"
        details["outputs"] = {path.name: path.stat().st_size for path in sorted(binary_dir.iterdir())
                              if path.is_file()} if binary_dir.is_dir() else {}
        ldd = _run(["ldd", binary_dir / "pi05"], timeout=60)["output"]
        details["pi05_libraries"] = [line.strip() for line in ldd.splitlines()
                                     if re.search(r"cuda|cublas|ggml|llama|not found", line)]
        if "not found" in ldd:
            raise StageError("pi05 links against libraries the loader cannot find (see pi05_libraries).",
                             details)
        return details

    def stage_download(self):
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise StageError("huggingface_hub is not installed.") from exc
        self.checkpoint.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or None  # not required: the checkpoint is public
        files = {}
        for name, expected in CHECKPOINT_FILES.items():
            target = self.checkpoint / name
            if not (target.is_file() and (expected is None or self._sha256(target) == expected)):
                log(f"downloading {name}")
                started = time.monotonic()
                try:
                    path = hf_hub_download(CHECKPOINT_REPO, name, revision=CHECKPOINT_REVISION, token=token,
                                           local_dir=str(self.checkpoint))
                except Exception as exc:  # never log tokens or raw HTTP exceptions
                    code = getattr(getattr(exc, "response", None), "status_code", None)
                    raise StageError(f"Download of {name} failed ({type(exc).__name__}, HTTP {code}).")
                target = Path(path)
                files[name] = {"download_seconds": round(time.monotonic() - started, 1)}
            digest = self._sha256(target)
            files.setdefault(name, {}).update(bytes=target.stat().st_size, sha256=digest)
            if expected is not None and digest != expected:
                raise StageError(f"{name} SHA-256 mismatch: expected {expected}, got {digest}.", {"files": files})
        try:
            resolved = HfApi().model_info(CHECKPOINT_REPO, revision=CHECKPOINT_REVISION, token=token).sha
        except Exception:
            resolved = None
        return {"path": str(self.checkpoint), "files": files, "revision_resolved": resolved}

    @staticmethod
    def _sha256(path):
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(16 * 2**20), b""):
                digest.update(block)
        return digest.hexdigest()

    def _smoke_image(self):
        """A synthetic 224x224 RGB image: a gradient table with two coloured blocks. Written as binary PPM."""
        path = self.output / "smoke_input.ppm"
        width = height = 224
        pixels = bytearray()
        for y in range(height):
            for x in range(width):
                if 60 <= x < 110 and 120 <= y < 170:
                    pixels += bytes((20, 20, 20))       # dark "bowl"
                elif 140 <= x < 200 and 130 <= y < 180:
                    pixels += bytes((200, 200, 210))    # light "plate"
                else:
                    pixels += bytes((120 + x // 4, 90 + y // 5, 60))
        path.write_bytes(f"P6 {width} {height} 255\n".encode() + bytes(pixels))
        return path

    @staticmethod
    def _parse_cli(output):
        parsed = {}
        match = re.search(r"Total inference time:\s*([\d.]+)\s*ms", output)
        parsed["inference_ms"] = float(match[1]) if match else None
        match = re.search(r"First timestep actions:\s*\n\s*\[([^\]]*)\]", output)
        if match:
            parsed["first_actions"] = [float(value) for value in re.findall(r"[-+]?(?:nan|inf|\d+\.?\d*(?:e[-+]?\d+)?)",
                                                                             match[1], flags=re.I)]
        matches = re.findall(r"min:\s*(\S+),\s*max:\s*(\S+),\s*mean:\s*(\S+)", output)
        if matches:
            parsed["stats"] = {key: float(value) for key, value in zip(("min", "max", "mean"), matches[-1])}
        match = re.search(r"using (\S+) backend", output)
        parsed["backend"] = match[1] if match else None
        match = re.search(r"Output actions \((\d+) dim x (\d+) horizon", output)
        if match:
            parsed["action_dim"], parsed["action_horizon"] = int(match[1]), int(match[2])
        parsed["completed"] = "Inference completed successfully." in output
        return parsed

    def _cli(self, device, image, label):
        binary = self.build_dir / "bin" / "pi05"
        threads = min(8, self.report["host"]["cpus"] or 4)
        result = _stream([binary, "-m", self.checkpoint, "-i", image, "-p", self.args.prompt,
                          "-d", device, "-n", str(threads), "-s", "10"],
                         self.output / f"infer_{label}.log", TIMEOUTS["infer"])
        text = (self.output / f"infer_{label}.log").read_text(encoding="utf-8", errors="replace")
        return {"device": device, "returncode": result["returncode"], "seconds": result["seconds"],
                "timed_out": result["timed_out"], **self._parse_cli(text)}

    @staticmethod
    def _finite(values):
        return bool(values) and all(math.isfinite(value) for value in values)

    def stage_infer(self):
        binary = self.build_dir / "bin" / "pi05"
        if not binary.is_file():
            raise StageError(f"{binary} does not exist; run the build stage.")
        missing = [name for name in CHECKPOINT_FILES if not (self.checkpoint / name).is_file()]
        if missing:
            raise StageError(f"Checkpoint files missing ({', '.join(missing)}); run the download stage.")
        image = self._smoke_image()
        self.deviation("smoke", "Smoke-test input is a synthetic 224x224 image and a fixed prompt, "
                                "not a LIBERO observation; it checks execution, not task behaviour.")
        details = {"image": image.name, "prompt": self.args.prompt}
        cuda = self._cli("CUDA0", image, "cuda")
        details["cli_cuda"] = cuda
        problems = []
        if cuda["returncode"] != 0 or not cuda["completed"]:
            problems.append(f"pi05 CLI exited with {cuda['returncode']} (see infer_cuda.log)")
        if cuda["backend"] != "CUDA0":
            problems.append(f"pi05 ran on backend {cuda['backend']!r}, not CUDA0")
        if not self._finite(cuda.get("first_actions", [])) or not self._finite(list(cuda.get("stats", {}).values())):
            problems.append("actions missing or not finite")

        binding_file = next(iter(sorted((self.build_dir / "bin").glob("pi05*.so"))), None)
        if binding_file is not None and not problems:
            details["binding"] = self._binding_run(binding_file, image)
            if details["binding"].get("first") and cuda.get("first_actions"):
                count = min(len(details["binding"]["first"]), len(cuda["first_actions"]))
                details["binding_vs_cli_max_abs_diff"] = max(
                    abs(a - b) for a, b in zip(details["binding"]["first"][:count], cuda["first_actions"][:count]))
        elif binding_file is None:
            details["binding"] = {"status": "not built"}

        if self.args.cpu_check and not problems:
            cpu = self._cli("CPU", image, "cpu")
            details["cli_cpu"] = cpu
            if cpu.get("first_actions") and cuda.get("first_actions"):
                count = min(len(cpu["first_actions"]), len(cuda["first_actions"]))
                details["cpu_vs_cuda_max_abs_diff"] = max(
                    abs(a - b) for a, b in zip(cpu["first_actions"][:count], cuda["first_actions"][:count]))
        if problems:
            raise StageError("; ".join(problems), details)
        return details

    def _binding_run(self, binding_file, image):
        """Load pi05.so in a fresh interpreter: the path serve_policy.py uses. Two runs give warm latency."""
        code = f"""
import json, sys, time
sys.path.insert(0, {str(binding_file.parent)!r})
import pi05
t = time.monotonic()
pipeline = pi05.Pi05Pipeline(model_path={str(self.checkpoint / 'pi05.gguf')!r},
                             tokenizer_path={str(self.checkpoint / 'tokenizer.model')!r},
                             device_name="CUDA0", n_threads=4, num_flow_steps=10)
load = time.monotonic() - t
runs, times = [], []
for _ in range(2):
    t = time.monotonic()
    runs.append([float(v) for v in pipeline.run({str(image)!r}, {self.args.prompt!r})])
    times.append(time.monotonic() - t)
print("EAQ_BINDING " + json.dumps({{
    "load_seconds": load, "run_seconds": times, "values": len(runs[0]),
    "action_horizon": pipeline.action_horizon, "action_dim": pipeline.action_dim,
    "first": runs[0][:pipeline.action_dim][:10],
    "repeat_max_abs_diff": max(abs(a - b) for a, b in zip(runs[0], runs[1])),
    "finite": all(v == v and abs(v) != float("inf") for v in runs[0])}}))
"""
        result = _stream([sys.executable, "-c", code], self.output / "infer_binding.log", TIMEOUTS["infer"])
        text = (self.output / "infer_binding.log").read_text(encoding="utf-8", errors="replace")
        match = re.search(r"^EAQ_BINDING (.*)$", text, flags=re.M)
        if result["returncode"] != 0 or not match:
            return {"status": "failed", "returncode": result["returncode"], "tail": result["tail"][-2000:]}
        return {"status": "passed", "file": binding_file.name, **json.loads(match[1])}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", required=True, type=Path, help="Run folder for report.json and logs.")
    parser.add_argument("--work-dir", type=Path,
                        default=Path(os.environ.get("EAQ_WORK_DIR", Path(tempfile.gettempdir()) / "eaq-actquant")),
                        help="Persistent folder for sources, build trees and checkpoints.")
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--toolkit-requirements", type=Path,
                        default=Path(__file__).resolve().parents[1] / "requirements" / "actquant-cuda-toolkit.txt",
                        help="Pinned CUDA toolkit wheels installed by the toolkit stage.")
    parser.add_argument("--cuda-arch", help="Override the GPU architecture, e.g. 120.")
    parser.add_argument("--jobs", type=int, help=f"Parallel build jobs (default: CPUs, limited by memory, at most {DEFAULT_MAX_JOBS}).")
    parser.add_argument("--no-vmm", action="store_true", help="Build with GGML_CUDA_NO_VMM=ON.")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Build with GGML_CUDA_FA=OFF (FlashAttention kernels as stubs; shorter build).")
    parser.add_argument("--no-binding", action="store_true", help="Do not build the pi05.so Python binding.")
    parser.add_argument("--cpu-check", action="store_true", help="Also run the CLI on CPU and compare.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--build-timeout", type=int, default=TIMEOUTS["build"])
    args = parser.parse_args()
    return Builder(args).run()


if __name__ == "__main__":
    sys.exit(main())
