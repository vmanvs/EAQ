"""Probe whether a Linux GPU host can build and run ActQuant's Pi 0.5 path.

ActQuant (https://github.com/arashakb/ActQuant) is a llama.cpp fork whose
Pi 0.5 runtime is built with CMake + nvcc and evaluated through a local
WebSocket policy server plus a Python 3.8 LIBERO client. These checks establish
whether this host can do that: toolchain, system-package access, disk, CUDA
compilation for the attached GPU (including Blackwell), local sockets and
source reachability. They do not clone ActQuant, download weights, install
packages or run LIBERO.

Each check returns details with optional `blockers` (the stock path cannot
work here) and `adaptations` (it can work with a recorded deviation).
"""
from __future__ import annotations

import ctypes.util
import glob
import os
from pathlib import Path
import platform
import re
import shutil
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading


SOURCES = {
    "actquant": ("https://github.com/arashakb/ActQuant", "b64791125070652fe6b554e244fe809c79ef5246"),
    "openpi": ("https://github.com/Physical-Intelligence/openpi", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    "libero": ("https://github.com/Lifelong-Robot-Learning/LIBERO", "8f1084e3132a39270c3a13ebe37270a43ece2a01"),
}
# (repo, revision, small file fetched to prove access). Consumed by cloud_preflight.access_check.
MODELS = {
    "actquant_pi05_3bpw": ("NU-World-Model-Embodied-AI/ActQuant-Pi05-LIBERO-3bpw",
                           "4d03f36f1ac019ad5e314dea584697ab29644171", "norm_stats.json"),
    "pi05_libero_base": ("lerobot/pi05_libero_finetuned_v044",
                         "8e174154ef5f6c60a8da12ae99c303d8963138c1", "config.json"),
    # Manually gated; ActQuant's Pi 0.5 runtime needs its tokenizer.model.
    "paligemma_tokenizer": ("google/paligemma-3b-pt-224",
                            "35e4f46485b4d07967e7e9935bc3786aad50687c", "config.json"),
}
# ActQuant README: ~50 GB for build outputs and intermediate GGUFs, plus the
# 7.5 GB Pi 0.5 base checkpoint and 2.4 GB released 3 bpw checkpoint.
MIN_DISK_GIB = 60
MIN_CMAKE = (3, 18)  # ggml-cuda's cmake_minimum_required
BLACKWELL_MIN_NVCC = (12, 8)
DEFAULT_REAL_ARCHS = ("86", "89")  # device code in the fork's default CMAKE_CUDA_ARCHITECTURES

KERNEL_SOURCE = r"""
#include <cstdio>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#ifdef EAQ_WITH_CUBLAS
#include <cublas_v2.h>
#endif

__global__ void axpy_half(int n, float a, const __half *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * __half2float(x[i]) + y[i];
}

int main() {
    const int n = 1 << 16;
    __half *x; float *y;
    if (cudaMallocManaged(&x, n * sizeof(__half)) != cudaSuccess) { printf("FAIL alloc\n"); return 2; }
    cudaMallocManaged(&y, n * sizeof(float));
    for (int i = 0; i < n; ++i) { x[i] = __float2half(1.0f); y[i] = 2.0f; }
    axpy_half<<<(n + 255) / 256, 256>>>(n, 3.0f, x, y);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) err = cudaGetLastError();
    if (err != cudaSuccess) { printf("FAIL kernel %s\n", cudaGetErrorString(err)); return 3; }
    for (int i = 0; i < n; ++i) if (y[i] != 5.0f) { printf("FAIL value %d %f\n", i, y[i]); return 4; }
#ifdef EAQ_WITH_CUBLAS
    cublasHandle_t handle;
    if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS) { printf("FAIL cublas\n"); return 5; }
    cublasDestroy(handle);
#endif
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int runtime = 0, driver = 0; cudaRuntimeGetVersion(&runtime); cudaDriverGetVersion(&driver);
    printf("OK %s sm_%d%d runtime=%d driver=%d\n", prop.name, prop.major, prop.minor, runtime, driver);
    cudaFree(x); cudaFree(y);
    return 0;
}
"""


def _run(command, timeout=60, cwd=None, env=None):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False,
                                timeout=timeout, cwd=cwd, env=env)
    except FileNotFoundError:
        return {"returncode": None, "output": "not found"}
    except subprocess.TimeoutExpired:
        return {"returncode": None, "output": f"timed out after {timeout} s"}
    output = (result.stdout + result.stderr).strip()
    return {"returncode": result.returncode, "output": output[-2000:]}


def _first_line(command):
    result = _run(command, timeout=20)
    if result["returncode"] != 0:
        return None
    return result["output"].splitlines()[0] if result["output"] else ""


def _parse_version(text):
    match = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", text or "")
    return tuple(int(part) for part in match.groups() if part is not None) if match else None


def find_nvcc():
    candidates = [shutil.which("nvcc")]
    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        if os.environ.get(variable):
            candidates.append(str(Path(os.environ[variable]) / "bin" / "nvcc"))
    candidates += sorted(glob.glob("/usr/local/cuda*/bin/nvcc")) + sorted(glob.glob("/opt/cuda*/bin/nvcc"))
    found = {}
    for candidate in filter(None, candidates):
        path = Path(candidate)
        if not path.is_file():
            continue
        resolved = str(path.resolve())
        if resolved in found:
            continue
        result = _run([resolved, "--version"], timeout=20)
        match = re.search(r"release (\d+)\.(\d+)", result["output"])
        found[resolved] = {"path": str(path), "version": f"{match[1]}.{match[2]}" if match else None}
    return sorted(found.values(), key=lambda item: _parse_version(item["version"]) or (0, 0), reverse=True)


def gpu_compute_capability():
    try:
        import torch

        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0))
    except Exception:
        pass
    line = _first_line(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"])
    version = _parse_version(line)
    return version[:2] if version else None


def driver_cuda_version():
    result = _run(["nvidia-smi"], timeout=20)
    match = re.search(r"CUDA Version:\s*(\d+\.\d+)", result["output"])
    return match[1] if match else None


def toolchain_check():
    tools = {name: _first_line(command) for name, command in {
        "cmake": ["cmake", "--version"], "ninja": ["ninja", "--version"], "make": ["make", "--version"],
        "gcc": ["gcc", "--version"], "g++": ["g++", "--version"], "git": ["git", "--version"],
        "conda": ["conda", "--version"], "mamba": ["mamba", "--version"],
        "micromamba": ["micromamba", "--version"], "uv": ["uv", "--version"],
    }.items()}
    nvcc = find_nvcc()
    blockers, adaptations = [], []
    if not nvcc:
        blockers.append("No CUDA toolkit (nvcc) found; ActQuant's C++/CUDA runtime cannot be built.")
    cmake_version = _parse_version(tools["cmake"])
    if cmake_version is None:
        blockers.append("cmake missing (pip install cmake is a user-space fix; record it).")
    elif cmake_version[:2] < MIN_CMAKE:
        blockers.append(f"cmake {tools['cmake']} is older than {MIN_CMAKE[0]}.{MIN_CMAKE[1]}.")
    if not tools["g++"]:
        blockers.append("No C++ compiler (g++).")
    if not (tools["ninja"] or tools["make"]):
        blockers.append("No build tool (ninja or make); pip install ninja is a user-space fix.")
    if not tools["git"]:
        blockers.append("git missing.")
    if not (tools["conda"] or tools["mamba"] or tools["micromamba"]):
        adaptations.append("No conda: ActQuant's 01_setup_env.sh assumes conda; use uv/venv or micromamba and record it.")
    if not tools["uv"]:
        adaptations.append("uv missing: needed by the Pi 0.5 setup (pip install uv).")
    return {"tools": tools, "nvcc": nvcc, "blockers": blockers, "adaptations": adaptations}


def system_access_check():
    euid = getattr(os, "geteuid", lambda: None)()
    sudo = _run(["sudo", "-n", "true"], timeout=10) if shutil.which("sudo") else {"returncode": None, "output": "not found"}
    can_elevate = euid == 0 or sudo["returncode"] == 0
    apt = shutil.which("apt-get")
    apt_simulation = _run(["apt-get", "-s", "install", "libegl1-mesa-dev"], timeout=60) if apt else None
    egl_library = ctypes.util.find_library("EGL")
    purelib = sysconfig.get_paths()["purelib"]
    os_release = {}
    if Path("/etc/os-release").is_file():
        for line in Path("/etc/os-release").read_text(encoding="utf-8").splitlines():
            key, _, value = line.partition("=")
            os_release[key] = value.strip('"')
    pythons = {version: shutil.which(f"python{version}") for version in ("3.8", "3.10", "3.11")}
    if shutil.which("uv"):
        for version in ("3.8", "3.11"):
            found = _run(["uv", "python", "find", version], timeout=20)
            if found["returncode"] == 0 and not pythons[version]:
                pythons[version] = found["output"].splitlines()[-1]
    blockers, adaptations = [], []
    if platform.system() != "Linux":
        blockers.append(f"{platform.system()} host; ActQuant's build and LIBERO scripts target Linux.")
    if not egl_library and not (can_elevate and apt):
        blockers.append("No libEGL and no root/apt to install libegl1-mesa-dev; headless LIBERO rendering needs EGL.")
    if not can_elevate:
        adaptations.append("No root or passwordless sudo: only user-space installs (pip/uv/conda) are possible.")
    if not pythons["3.8"]:
        adaptations.append("No Python 3.8 found; the LIBERO client needs it (uv python install 3.8 downloads one).")
    return {"os": os_release.get("PRETTY_NAME", platform.platform()), "glibc": "-".join(platform.libc_ver()),
            "euid": euid, "passwordless_sudo": sudo["returncode"] == 0, "can_install_system_packages": bool(can_elevate and apt),
            "apt_get": apt, "apt_simulate_libegl": apt_simulation, "libEGL": egl_library,
            "site_packages_writable": os.access(purelib, os.W_OK), "pythons": pythons,
            "blockers": blockers, "adaptations": adaptations}


def disk_check(output_dir):
    anchors = [output_dir, Path.cwd(), Path.home(), Path(tempfile.gettempdir()), Path("/tmp"), Path("/mnt"), Path("/content")]
    locations, seen = [], set()
    for anchor in anchors:
        try:
            anchor = anchor.resolve()
            device = anchor.stat().st_dev
            usage = shutil.disk_usage(anchor)
        except OSError:
            continue
        if device in seen:
            continue
        seen.add(device)
        locations.append({"path": str(anchor), "free_gib": round(usage.free / 2**30, 2),
                          "total_gib": round(usage.total / 2**30, 2), "writable": os.access(anchor, os.W_OK)})
    best = max((item["free_gib"] for item in locations if item["writable"]), default=0)
    blockers = [] if best >= MIN_DISK_GIB else [
        f"Largest writable free space is {best:.1f} GiB; ActQuant's Pi 0.5 path needs about {MIN_DISK_GIB} GiB."]
    return {"required_gib": MIN_DISK_GIB, "locations": locations, "blockers": blockers,
            "note": "Free space is a snapshot on possibly ephemeral scratch storage."}


def _compile_and_run(nvcc, workdir, name, gencode):
    source = workdir / "probe.cu"
    source.write_text(KERNEL_SOURCE, encoding="utf-8")
    binary = workdir / name
    compile_result = _run([nvcc, *gencode, "-O2", "-o", str(binary), str(source)], timeout=300)
    if compile_result["returncode"] != 0:
        return {"gencode": gencode, "compiled": False, "ran": False, "output": compile_result["output"]}
    env = dict(os.environ, CUDA_CACHE_DISABLE="1")  # force a real JIT for PTX-only binaries
    run_result = _run([str(binary)], timeout=120, env=env)
    return {"gencode": gencode, "compiled": True,
            "ran": run_result["returncode"] == 0 and run_result["output"].startswith("OK"),
            "output": run_result["output"]}


def cuda_compile_check(workdir):
    nvcc = find_nvcc()
    capability = gpu_compute_capability()
    if not nvcc:
        return {"blockers": ["No nvcc; cannot test CUDA compilation."]}
    if capability is None:
        return {"nvcc": nvcc[0], "blockers": ["No visible NVIDIA GPU (torch/nvidia-smi); attach the GPU and restart."]}
    arch = f"{capability[0]}{capability[1]}"
    compiler = nvcc[0]["path"]
    native = _compile_and_run(compiler, workdir, "probe_native", ["-gencode", f"arch=compute_{arch},code=sm_{arch}"])
    # The ActQuant fork's default CUDA arch list ends in 75/80-virtual (PTX) and 86/89-real;
    # on newer GPUs only the PTX path can run, via driver JIT.
    ptx = _compile_and_run(compiler, workdir, "probe_ptx80", ["-gencode", "arch=compute_80,code=compute_80"])
    blockers, adaptations = [], []
    if not (native["ran"] or ptx["ran"]):
        blockers.append("Neither native nor PTX-JIT CUDA binaries ran on this GPU.")
    elif not native["ran"]:
        adaptations.append(f"nvcc {nvcc[0]['version']} cannot target sm_{arch} natively; only the PTX-JIT build runs "
                           "(slower first load, possibly slower kernels). Install a newer CUDA toolkit if possible.")
    if capability >= (12, 0) and (_parse_version(nvcc[0]["version"]) or (0, 0)) < BLACKWELL_MIN_NVCC:
        adaptations.append("Blackwell GPU with CUDA toolkit < 12.8; ActQuant was tested with 12.6 (Pi 0.5).")
    if native["ran"] and arch not in DEFAULT_REAL_ARCHS:
        adaptations.append(f"Build ActQuant with -DCMAKE_CUDA_ARCHITECTURES={arch}; the fork's default list has no "
                           f"sm_{arch} device code and would fall back to PTX JIT.")
    return {"nvcc": nvcc[0], "compute_capability": f"{capability[0]}.{capability[1]}",
            "driver_cuda_version": driver_cuda_version(), "native": native, "ptx_jit_compute_80": ptx,
            "blockers": blockers, "adaptations": adaptations}


def cmake_cuda_check(workdir):
    nvcc = find_nvcc()
    capability = gpu_compute_capability()
    if not nvcc or capability is None or not shutil.which("cmake"):
        return {"blockers": ["Requires nvcc, a visible GPU and cmake; see the toolchain and CUDA checks."]}
    (workdir / "probe.cu").write_text(KERNEL_SOURCE, encoding="utf-8")
    (workdir / "CMakeLists.txt").write_text(
        "cmake_minimum_required(VERSION 3.18)\n"
        "project(eaq_probe LANGUAGES CXX CUDA)\n"
        "find_package(CUDAToolkit REQUIRED)\n"
        "add_executable(probe probe.cu)\n"
        "target_compile_definitions(probe PRIVATE EAQ_WITH_CUBLAS)\n"
        "target_link_libraries(probe PRIVATE CUDA::cudart CUDA::cublas)\n", encoding="utf-8")
    arch = f"{capability[0]}{capability[1]}"
    generator = ["-G", "Ninja"] if shutil.which("ninja") else []
    configure = _run(["cmake", "-S", str(workdir), "-B", str(workdir / "build"), *generator,
                      f"-DCMAKE_CUDA_COMPILER={nvcc[0]['path']}", f"-DCMAKE_CUDA_ARCHITECTURES={arch}",
                      "-DCMAKE_BUILD_TYPE=Release"], timeout=300)
    result = {"nvcc": nvcc[0], "cuda_architectures": arch, "configure": configure}
    if configure["returncode"] != 0:
        result["blockers"] = ["CMake could not configure a CUDA + cuBLAS project (ggml-cuda needs both)."]
        return result
    build = _run(["cmake", "--build", str(workdir / "build")], timeout=600)
    result["build"] = build
    if build["returncode"] != 0:
        result["blockers"] = ["CMake CUDA + cuBLAS build failed."]
        return result
    binary = next((path for path in (workdir / "build").rglob("probe*")
                   if path.is_file() and path.name in ("probe", "probe.exe")), None)
    run = _run([str(binary)], timeout=120) if binary else {"returncode": None, "output": "binary not found"}
    result["run"] = run
    result["blockers"] = [] if run["returncode"] == 0 and run["output"].startswith("OK") else [
        "CMake-built CUDA + cuBLAS binary did not run correctly."]
    return result


def local_server_check():
    """Technical check only; Molab's policy on long-running local servers is a separate question."""
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    port = server.getsockname()[1]

    def echo():
        connection, _ = server.accept()
        with connection:
            connection.sendall(connection.recv(64))

    thread = threading.Thread(target=echo, daemon=True)
    thread.start()
    with socket.create_connection(("127.0.0.1", port), timeout=10) as client:
        client.sendall(b"eaq-probe")
        reply = client.recv(64)
    thread.join(timeout=10)
    server.close()
    if reply != b"eaq-probe":
        raise RuntimeError("Loopback echo returned unexpected data.")
    return {"loopback_port": port, "note": "Pi 0.5 eval uses a local WebSocket policy server; this tests loopback only, not service policy."}


def sources_check():
    results, blockers = {}, []
    for label, (url, pinned) in SOURCES.items():
        remote = _run(["git", "ls-remote", url, "HEAD"], timeout=60)
        head = remote["output"].split()[0] if remote["returncode"] == 0 and remote["output"] else None
        results[label] = {"url": url, "pinned": pinned, "remote_head": head,
                          "head_matches_pin": head == pinned if head else None}
        if head is None:
            blockers.append(f"{label}: git ls-remote failed ({remote['output'][:200]}).")
    return {"sources": results, "blockers": blockers,
            "note": "A moved HEAD is fine; experiments use the pinned commit."}


def _in_scratch(check):
    """Run a compile check in a throwaway folder so build files stay out of the run artifacts."""
    def run(output_dir):
        workdir = Path(tempfile.mkdtemp(prefix="eaq-build-probe-", dir=output_dir))
        try:
            return check(workdir)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)
    return run


CHECKS = {
    "toolchain": lambda output_dir: toolchain_check(),
    "system": lambda output_dir: system_access_check(),
    "disk": disk_check,
    "cuda_nvcc": _in_scratch(cuda_compile_check),
    "cuda_cmake": _in_scratch(cmake_cuda_check),
    "local_server": lambda output_dir: local_server_check(),
    "sources": lambda output_dir: sources_check(),
}


def verdict(checks):
    """Summarize ActQuant readiness from report checks named actquant_* and access_* for MODELS."""
    relevant = {name: item for name, item in checks.items()
                if name.startswith("actquant_") or name.removeprefix("access_") in MODELS}
    blockers = [f"{name}: {text}" for name, item in relevant.items()
                for text in item.get("details", {}).get("blockers", [])]
    blockers += [f"{name}: {item['error']}" for name, item in relevant.items()
                 if item["status"] == "failed" and not item.get("details", {}).get("blockers")]
    adaptations = [f"{name}: {text}" for name, item in relevant.items()
                   for text in item.get("details", {}).get("adaptations", [])]
    status = "blocked" if blockers else "ready_with_adaptations" if adaptations else "ready"
    return {"status": status, "blockers": blockers, "adaptations": adaptations}


if __name__ == "__main__":
    print("Run through cloud_preflight.py: python scripts/cloud_preflight.py --output DIR --only actquant", file=sys.stderr)
    raise SystemExit(2)
