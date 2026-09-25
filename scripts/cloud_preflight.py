"""Validate a Linux GPU runtime before downloading LaWAM or ActQuant weights.

This checks infrastructure, not model correctness or LIBERO task success.
The ActQuant group (`--only actquant`) tests whether ActQuant's Pi 0.5
build-and-evaluate path can run on this host. This is one self-contained file:
Molab fetches it alone and may run Python with PYTHONSAFEPATH, which blocks
sibling-module imports.
"""
from __future__ import annotations

import argparse
import ctypes.util
import datetime as dt
import glob
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import site
import socket
import subprocess
import sys
import sysconfig
import tempfile
import threading
import time


MODELS = {
    "policy": ("jialei02/lawam_libero_sft_release", "b09a207ffcde367e71439621eb1679f941b4f026", "config.yaml"),
    "lam": ("jialei02/lawam_lam", "bd993da2a0861afaac5a95ac86d2555b1313ab8c", "dino_large_vae.yaml"),
    "qwen": ("Qwen/Qwen3-VL-2B-Instruct", "89644892e4d85e24eaac8bacfd4f463576704203", "config.json"),
    "dino": ("facebook/dinov3-vitb16-pretrain-lvd1689m", "5931719e67bbdb9737e363e781fb0c67687896bc", "config.json"),
}
LAWAM_REVISION = "7d27b9607c22034934a4b70347f8bfaba92bf692"


def runtime_check():
    import psutil
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Select a GPU accelerator and restart the session.")
    devices = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        devices.append({"index": index, "name": props.name,
                        "vram_gib": props.total_memory / 2**30,
                        "compute_capability": list(torch.cuda.get_device_capability(index))})
    ram = psutil.virtual_memory()
    return {"torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "devices": devices, "ram_total_gib": ram.total / 2**30,
            "ram_available_gib": ram.available / 2**30,
            "note": "GPU memory is per device; two GPUs do not automatically pool VRAM."}


def attention_check():
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; attention test requires a GPU.")
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    # A bounded synthetic test, not an exhaustive LaWAM shape/operator check.
    shape = (1, 8, 128, 64)
    results = {}
    dtypes = [torch.float16]
    if torch.cuda.get_device_capability(0)[0] >= 8 and torch.cuda.is_bf16_supported():
        dtypes.append(torch.bfloat16)
    for dtype in dtypes:
        q, k, v = [torch.randn(shape, device=device, dtype=dtype) for _ in range(3)]
        reference = torch.softmax((q.float() @ k.float().transpose(-2, -1)) / 8, dim=-1) @ v.float()
        torch.cuda.synchronize()
        start = time.perf_counter()
        with sdpa_kernel(SDPBackend.MATH):
            output = F.scaled_dot_product_attention(q, k, v, dropout_p=0)
        torch.cuda.synchronize()
        tolerance = 0.01 if dtype == torch.float16 else 0.05
        torch.testing.assert_close(output.float(), reference, atol=tolerance, rtol=tolerance)
        if not torch.isfinite(output).all():
            raise RuntimeError("Non-finite attention output")
        results[str(dtype)] = {"max_abs_error": (output.float() - reference).abs().max().item(),
                              "single_call_seconds": time.perf_counter() - start}
    return {"backend": "PyTorch SDPA MATH; no flash-attn package", "shape": list(shape),
            "results": results, "note": "One synthetic call; timing is not a benchmark. LaWAM precision changes remain unvalidated."}


def render_check(output_dir):
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco
    import numpy as np
    from PIL import Image

    xml = '''<mujoco><option timestep="0.002"/>
      <worldbody><light pos="0 0 3"/>
      <camera name="view" pos="1.5 -2 1.5" xyaxes="0.8 0.6 0 -0.3 0.4 0.86"/>
      <geom type="plane" size="2 2 .1" rgba=".3 .3 .3 1"/>
      <body pos="0 0 .5"><freejoint/><geom type="box" size=".05 .05 .05" mass=".1" rgba=".2 .7 .9 1"/></body>
      </worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    if not np.isfinite(data.qpos).all() or not 0.04 < data.qpos[2] < 0.06 or data.ncon == 0:
        raise RuntimeError("Physics/contact check failed")
    with mujoco.Renderer(model, height=240, width=320) as renderer:
        renderer.update_scene(data, camera="view")
        rgb = renderer.render().copy()
    if rgb.shape != (240, 320, 3) or rgb.dtype != np.uint8 or rgb.std() < 1:
        raise RuntimeError("RGB rendering returned an invalid or blank frame")
    Image.fromarray(rgb).save(output_dir / "render.png")
    return {"mujoco": mujoco.__version__, "backend": os.environ["MUJOCO_GL"],
            "frame": "render.png", "note": "Simple MuJoCo scene, not a LIBERO compatibility test."}


def access_check(label, output_dir):
    from huggingface_hub import HfApi, hf_hub_download

    repo, revision, config = ALL_MODELS[label]
    # Never log the token, environment, headers, or raw HTTP exceptions.
    token = os.environ.get("HF_TOKEN") or False
    try:
        info = HfApi().model_info(repo, revision=revision, files_metadata=True, token=token)
        path = hf_hub_download(repo, config, revision=revision, token=token,
                               cache_dir=str(output_dir.parent / "hf-metadata-cache"))
    except Exception as exc:
        code = getattr(getattr(exc, "response", None), "status_code", None)
        raise RuntimeError(f"{label}: metadata access failed ({type(exc).__name__}, HTTP {code}). Check Internet, HF_TOKEN, and model access approval.") from None
    return {"repo": repo, "revision": info.sha, "config_bytes": Path(path).stat().st_size,
            "weight_files": [{"name": f.rfilename, "bytes": f.size} for f in info.siblings
                             if f.rfilename.endswith((".pt", ".safetensors", ".gguf"))]}


# ---------------------------------------------------------------------------
# ActQuant probe
#
# ActQuant (https://github.com/arashakb/ActQuant) is a llama.cpp fork whose
# Pi 0.5 runtime is built with CMake + nvcc and evaluated through a local
# WebSocket policy server plus a Python 3.8 LIBERO client. These checks
# establish whether this host can do that: toolchain, system-package access,
# disk, CUDA compilation for the attached GPU (including Blackwell), local
# sockets and source reachability. They do not clone ActQuant, download
# weights, install packages or run LIBERO. Each returns details with optional
# `blockers` (the stock path cannot work here) and `adaptations` (it can work
# with a recorded deviation).
# ---------------------------------------------------------------------------

ACTQUANT_SOURCES = {
    "actquant": ("https://github.com/arashakb/ActQuant", "b64791125070652fe6b554e244fe809c79ef5246"),
    "openpi": ("https://github.com/Physical-Intelligence/openpi", "215abfb217dbac7d5f1273282331b9b1866c0479"),
    "libero": ("https://github.com/Lifelong-Robot-Learning/LIBERO", "8f1084e3132a39270c3a13ebe37270a43ece2a01"),
}
# (repo, revision, small file fetched to prove access), as in MODELS.
ACTQUANT_MODELS = {
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
IMPLAUSIBLE_DISK_BYTES = 2**50  # 1 PiB
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
    except OSError as exc:  # not executable, wrong binary format, ...
        return {"returncode": None, "output": f"could not start: {type(exc).__name__}: {exc}"}
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


def _site_dirs():
    dirs = {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}
    try:
        dirs.update(site.getsitepackages())
        dirs.add(site.getusersitepackages())
    except AttributeError:  # some virtualenvs lack getsitepackages
        pass
    # Molab layers a package venv (/tmp/uv-venv) over the base interpreter; sys.path sees both.
    dirs.update(entry for entry in sys.path if entry.endswith(("site-packages", "dist-packages")))
    return sorted(dirs)


def find_cccl_includes():
    """Include dirs of pip-installed CCCL (nvidia-cuda-cccl), which may sit in a different site-packages than nvcc."""
    found = []
    for directory in _site_dirs():
        for target in sorted(glob.glob(os.path.join(directory, "nvidia", "cu*", "include", "nv", "target"))):
            include = str(Path(target).parent.parent.resolve())
            if include not in found:
                found.append(include)
    return found


def _owning_python(toolkit_root):
    """Best guess at the interpreter whose site-packages holds this pip toolkit."""
    site_packages = next((parent for parent in Path(toolkit_root).parents
                          if parent.name in ("site-packages", "dist-packages")), None)
    if site_packages is None:
        return None
    candidate = site_packages.parent.parent.parent / "bin" / site_packages.parent.name  # .../lib/python3.X
    return str(candidate) if candidate.is_file() else None


def _which(name):
    """Find a tool on PATH or beside this interpreter, where pip puts cmake/ninja entry points."""
    return shutil.which(name) or shutil.which(name, path=os.pathsep.join(
        {str(Path(sys.executable).parent), sysconfig.get_paths()["scripts"]}))


def find_nvcc():
    candidates = [shutil.which("nvcc")]
    for variable in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
        if os.environ.get(variable):
            candidates.append(str(Path(os.environ[variable]) / "bin" / "nvcc"))
    candidates += sorted(glob.glob("/usr/local/cuda*/bin/nvcc")) + sorted(glob.glob("/opt/cuda*/bin/nvcc"))
    # CUDA 13 pip wheels (pulled in by PyTorch cu13x) install a toolkit-shaped tree at
    # site-packages/nvidia/cu13/{bin,include,lib,nvvm}; CUDA 12 wheels use nvidia/cuda_nvcc.
    for directory in _site_dirs():
        candidates += sorted(glob.glob(os.path.join(directory, "nvidia", "cu*", "bin", "nvcc")))
        candidates += sorted(glob.glob(os.path.join(directory, "nvidia", "cuda_nvcc", "bin", "nvcc")))
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
        found[resolved] = {"path": str(path), "version": f"{match[1]}.{match[2]}" if match else None,
                           "toolkit_root": str(path.parent.parent),
                           "source": "pip wheel" if "site-packages" in resolved or "dist-packages" in resolved
                           else "system"}
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
    tools = {name: _first_line([_which(name) or name, "--version"]) for name in (
        "cmake", "ninja", "make", "gcc", "g++", "git", "conda", "mamba", "micromamba", "uv")}
    nvcc = find_nvcc()
    blockers, adaptations = [], []
    if not nvcc:
        blockers.append("No CUDA toolkit (nvcc) found; ActQuant's C++/CUDA runtime cannot be built.")
    cmake_version = _parse_version(tools["cmake"])
    if cmake_version is None:
        blockers.append("cmake missing; install the pinned cmake from requirements/preflight.txt.")
    elif cmake_version[:2] < MIN_CMAKE:
        blockers.append(f"cmake {tools['cmake']} is older than {MIN_CMAKE[0]}.{MIN_CMAKE[1]}.")
    if not tools["g++"]:
        blockers.append("No C++ compiler (g++).")
    if not (tools["ninja"] or tools["make"]):
        blockers.append("No build tool (ninja or make); install the pinned ninja from requirements/preflight.txt.")
    if not tools["git"]:
        blockers.append("git missing.")
    if not (tools["conda"] or tools["mamba"] or tools["micromamba"]):
        adaptations.append("No conda: ActQuant's 01_setup_env.sh assumes conda; use uv/venv or micromamba and record it.")
    if not tools["uv"]:
        adaptations.append("uv missing: needed by the Pi 0.5 setup (pip install uv).")
    if nvcc and nvcc[0]["source"] == "pip wheel":
        adaptations.append(f"CUDA toolkit comes from pip wheels at {nvcc[0]['toolkit_root']}, not a system install; "
                           "build ActQuant with -DCUDAToolkit_ROOT pointing there and record it.")
        # PyTorch pulls in nvcc but not CCCL (libcu++/CUB/Thrust); cuda_fp16.h needs its <nv/target>.
        toolkit_include = Path(nvcc[0]["toolkit_root"], "include")
        if not (toolkit_include / "nv" / "target").is_file():
            elsewhere = [include for include in find_cccl_includes() if Path(include) != toolkit_include.resolve()]
            if elsewhere:
                owner = _owning_python(nvcc[0]["toolkit_root"]) or "the interpreter that owns it"
                blockers.append(
                    f"CCCL headers are installed in a different environment ({elsewhere[0]}) from nvcc's toolkit "
                    f"({toolkit_include}); nvcc and CMake only search their own tree. Install nvidia-cuda-cccl into "
                    f"nvcc's environment, e.g. uv pip install --python {owner} --system --break-system-packages "
                    "nvidia-cuda-cccl==<pinned version>.")
            else:
                blockers.append(f"pip CUDA toolkit lacks CCCL headers (nv/target); install nvidia-cuda-cccl matching "
                                f"nvcc {nvcc[0]['version']} (pinned in requirements/preflight.txt).")
    return {"tools": tools, "nvcc": nvcc, "blockers": blockers, "adaptations": adaptations}


def system_access_check():
    euid = getattr(os, "geteuid", lambda: None)()
    sudo = _run(["sudo", "-n", "true"], timeout=10) if shutil.which("sudo") else {"returncode": None, "output": "not found"}
    can_elevate = euid == 0 or sudo["returncode"] == 0
    apt = shutil.which("apt-get")
    apt_simulation = None
    # "Unable to locate package" for everything usually means apt-get update never ran.
    apt_lists_present = any(Path("/var/lib/apt/lists").glob("*_Packages*"))
    if apt:
        # ActQuant's README names libegl1-mesa-dev; Debian 13 ships the headers as libegl-dev.
        for package in ("libegl1-mesa-dev", "libegl-dev"):
            apt_simulation = {"package": package, **_run(["apt-get", "-s", "install", package], timeout=60)}
            if apt_simulation["returncode"] == 0:
                break
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
    elif apt and not apt_lists_present:
        adaptations.append("apt package lists are empty; run apt-get update before any apt install.")
    if not pythons["3.8"]:
        adaptations.append("No Python 3.8 found; the LIBERO client needs it (uv python install 3.8 downloads one).")
    return {"os": os_release.get("PRETTY_NAME", platform.platform()), "glibc": "-".join(platform.libc_ver()),
            "euid": euid, "passwordless_sudo": sudo["returncode"] == 0, "can_install_system_packages": bool(can_elevate and apt),
            "apt_get": apt, "apt_lists_present": apt_lists_present, "apt_simulate_libegl": apt_simulation,
            "libEGL": egl_library,
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
                          "total_gib": round(usage.total / 2**30, 2), "writable": os.access(anchor, os.W_OK),
                          # Sandboxes such as gVisor (Molab) report placeholder sizes of many PiB.
                          "size_plausible": usage.total < IMPLAUSIBLE_DISK_BYTES})
    measurable = [item for item in locations if item["writable"] and item["size_plausible"]]
    best = max((item["free_gib"] for item in measurable), default=0)
    blockers, adaptations = [], []
    if best >= MIN_DISK_GIB:
        pass
    elif any(item["writable"] and not item["size_plausible"] for item in locations):
        adaptations.append("Sandboxed filesystem reports a placeholder size; real scratch capacity is unknown. "
                           "Watch free space and RAM while staging the first weights.")
    else:
        blockers.append(f"Largest writable free space is {best:.1f} GiB; ActQuant's Pi 0.5 path needs about "
                        f"{MIN_DISK_GIB} GiB.")
    return {"required_gib": MIN_DISK_GIB, "locations": locations, "blockers": blockers, "adaptations": adaptations,
            "note": "Free space is a snapshot on possibly ephemeral scratch storage."}


def _compile_and_run(nvcc, workdir, name, gencode, extra_flags=()):
    source = workdir / "probe.cu"
    source.write_text(KERNEL_SOURCE, encoding="utf-8")
    binary = workdir / name
    compile_result = _run([nvcc, *gencode, *extra_flags, "-O2", "-o", str(binary), str(source)], timeout=300)
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
    # If CCCL sits in another environment than nvcc, point nvcc at it explicitly so this check still
    # answers whether the GPU runs freshly compiled code; the toolchain check reports the install split.
    extra_flags = []
    if not Path(nvcc[0]["toolkit_root"], "include", "nv", "target").is_file():
        cccl = find_cccl_includes()
        if cccl:
            extra_flags = ["-I", cccl[0]]
    native = _compile_and_run(compiler, workdir, "probe_native", ["-gencode", f"arch=compute_{arch},code=sm_{arch}"],
                              extra_flags)
    # The ActQuant fork's default CUDA arch list ends in 75/80-virtual (PTX) and 86/89-real;
    # on newer GPUs only the PTX path can run, via driver JIT.
    ptx = _compile_and_run(compiler, workdir, "probe_ptx80", ["-gencode", "arch=compute_80,code=compute_80"],
                           extra_flags)
    blockers, adaptations = [], []
    if extra_flags:
        adaptations.append(f"Compiled with an explicit CCCL include ({extra_flags[1]}) because it is not in nvcc's "
                           "own toolkit; ActQuant's CMake build needs it installed there instead.")
    if not (native["ran"] or ptx["ran"]):
        blockers.append("Neither native nor PTX-JIT CUDA binaries ran on this GPU.")
    elif not native["ran"]:
        adaptations.append(f"nvcc {nvcc[0]['version']} cannot target sm_{arch} natively; only the PTX-JIT build runs "
                           "(slower first load, possibly slower kernels). Install a newer CUDA toolkit if possible.")
    driver_cuda = driver_cuda_version()
    if (_parse_version(nvcc[0]["version"]) or (0, 0)) > (_parse_version(driver_cuda) or (99, 0)):
        adaptations.append(f"nvcc {nvcc[0]['version']} is newer than the driver's CUDA {driver_cuda}: native "
                           "device code runs, but the driver cannot JIT this nvcc's PTX. Build for the exact GPU "
                           "architecture, or use a toolkit no newer than the driver.")
    if capability >= (12, 0) and (_parse_version(nvcc[0]["version"]) or (0, 0)) < BLACKWELL_MIN_NVCC:
        adaptations.append("Blackwell GPU with CUDA toolkit < 12.8; ActQuant was tested with 12.6 (Pi 0.5).")
    if native["ran"] and arch not in DEFAULT_REAL_ARCHS:
        adaptations.append(f"Build ActQuant with -DCMAKE_CUDA_ARCHITECTURES={arch}; the fork's default list has no "
                           f"sm_{arch} device code and would fall back to PTX JIT.")
    return {"nvcc": nvcc[0], "compute_capability": f"{capability[0]}.{capability[1]}",
            "driver_cuda_version": driver_cuda, "native": native, "ptx_jit_compute_80": ptx,
            "blockers": blockers, "adaptations": adaptations}


def _unversioned_library_shim(toolkit_root, shim_dir):
    """Link lib*.so -> lib*.so.N for pip toolkits, which ship only versioned sonames."""
    shim_dir.mkdir(parents=True, exist_ok=True)
    linked = []
    for library in sorted(Path(toolkit_root, "lib").glob("lib*.so.*")):
        unversioned = library.name.split(".so.")[0] + ".so"
        if not (library.parent / unversioned).exists() and not (shim_dir / unversioned).exists():
            (shim_dir / unversioned).symlink_to(library)
            linked.append(unversioned)
    return linked


def cmake_cuda_check(workdir):
    nvcc = find_nvcc()
    capability = gpu_compute_capability()
    cmake, ninja = _which("cmake"), _which("ninja")
    if not nvcc or capability is None or not cmake:
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
    toolkit = nvcc[0]
    base_arguments = [cmake, "-S", str(workdir),
                      *(["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"] if ninja else []),
                      f"-DCMAKE_CUDA_COMPILER={toolkit['path']}", f"-DCUDAToolkit_ROOT={toolkit['toolkit_root']}",
                      f"-DCMAKE_CUDA_ARCHITECTURES={arch}", "-DCMAKE_BUILD_TYPE=Release"]
    attempts = [("as installed", [])]
    toolkit_lib = str(Path(toolkit["toolkit_root"], "lib"))
    if toolkit["source"] == "pip wheel":
        # pip toolkits ship only versioned sonames (link step) and live off the loader path (run step):
        # PyTorch preloads them itself, but a freshly built binary must carry an RPATH to them.
        attempts.append(("with unversioned .so links and toolkit RPATH", None))
    result = {"nvcc": toolkit, "cuda_architectures": arch, "attempts": []}
    for label, extra in attempts:
        build_dir = workdir / f"build-{len(result['attempts'])}"
        attempt = {"label": label}
        if extra is None:
            shim = workdir / "libshim"
            attempt["linked"] = _unversioned_library_shim(toolkit["toolkit_root"], shim)
            extra = [f"-DCMAKE_LIBRARY_PATH={shim}", f"-DCMAKE_BUILD_RPATH={toolkit_lib}"]
        attempt["configure"] = _run([*base_arguments, "-B", str(build_dir), *extra], timeout=300)
        if attempt["configure"]["returncode"] == 0:
            attempt["build"] = _run([cmake, "--build", str(build_dir)], timeout=600)
            if attempt["build"]["returncode"] == 0:
                binary = next((path for path in build_dir.rglob("probe*")
                               if path.is_file() and path.name in ("probe", "probe.exe")), None)
                attempt["run"] = (_run([str(binary)], timeout=120) if binary
                                  else {"returncode": None, "output": "binary not found"})
                if binary and "error while loading shared libraries" in attempt["run"]["output"]:
                    # Diagnostic only: does the binary work once the loader can see the toolkit's libraries?
                    env = dict(os.environ, LD_LIBRARY_PATH=os.pathsep.join(
                        filter(None, [toolkit_lib, os.environ.get("LD_LIBRARY_PATH")])))
                    attempt["run_with_ld_library_path"] = _run([str(binary)], timeout=120, env=env)
        result["attempts"].append(attempt)
        run = attempt.get("run", {})
        if run.get("returncode") == 0 and run.get("output", "").startswith("OK"):
            result["working_setup"] = label
            break
    blockers, adaptations = [], []
    if "working_setup" not in result:
        blockers.append("No CMake setup could configure, build and run a CUDA + cuBLAS binary (ggml-cuda needs this).")
    elif result["working_setup"] != "as installed":
        adaptations.append("Pip CUDA toolkit: before building ActQuant, create unversioned lib*.so links and pass "
                           f"-DCMAKE_BUILD_RPATH={toolkit_lib} (or set LD_LIBRARY_PATH at run time); record both.")
    result.update(blockers=blockers, adaptations=adaptations)
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
    for label, (url, pinned) in ACTQUANT_SOURCES.items():
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


ACTQUANT_CHECKS = {
    "toolchain": lambda output_dir: toolchain_check(),
    "system": lambda output_dir: system_access_check(),
    "disk": disk_check,
    "cuda_nvcc": _in_scratch(cuda_compile_check),
    "cuda_cmake": _in_scratch(cmake_cuda_check),
    "local_server": lambda output_dir: local_server_check(),
    "sources": lambda output_dir: sources_check(),
}


def actquant_verdict(checks):
    """Summarize ActQuant readiness from report checks named actquant_* and access_* for ACTQUANT_MODELS."""
    relevant = {name: item for name, item in checks.items()
                if name.startswith("actquant_") or name.removeprefix("access_") in ACTQUANT_MODELS}
    blockers = [f"{name}: {text}" for name, item in relevant.items()
                for text in item.get("details", {}).get("blockers", [])]
    blockers += [f"{name}: {item['error']}" for name, item in relevant.items()
                 if item["status"] == "failed" and not item.get("details", {}).get("blockers")]
    adaptations = [f"{name}: {text}" for name, item in relevant.items()
                   for text in item.get("details", {}).get("adaptations", [])]
    status = "blocked" if blockers else "ready_with_adaptations" if adaptations else "ready"
    return {"status": status, "blockers": blockers, "adaptations": adaptations}


ALL_MODELS = {**MODELS, **ACTQUANT_MODELS}


def build_checks(output_dir, group):
    base = {"runtime": runtime_check, "attention": attention_check,
            "render": lambda: render_check(output_dir)}
    access = {f"access_{label}": lambda label=label: access_check(label, output_dir) for label in ALL_MODELS}
    actquant = {f"actquant_{name}": lambda check=check: check(output_dir)
                for name, check in ACTQUANT_CHECKS.items()}
    actquant.update({name: check for name, check in access.items()
                     if name.removeprefix("access_") in ACTQUANT_MODELS})
    if group == "all":
        return {**base, **access, **actquant}
    if group == "access":
        return access
    if group == "actquant":
        return actquant
    return {group: base[group]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=["all", "runtime", "attention", "render", "access", "actquant"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=False)
    report = {"schema_version": 2, "scope": "preflight only; no policy inference or task success",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "eaq_commit": git.stdout.strip() if git.returncode == 0 else None,
              "lawam_revision": LAWAM_REVISION, "actquant_sources": ACTQUANT_SOURCES,
              "python": platform.python_version(),
              "platform": platform.system(), "disk_free_gib": shutil.disk_usage(args.output).free / 2**30,
              "checks": {}}
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    (args.output / "packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    for name, check in build_checks(args.output, args.only).items():
        try:
            details = check()
            blockers = details.get("blockers") if isinstance(details, dict) else None
            if blockers:
                # Probe checks keep their evidence even when they find a blocker.
                report["checks"][name] = {"status": "failed", "error": "; ".join(blockers), "details": details}
            else:
                report["checks"][name] = {"status": "passed", "details": details}
        except Exception as exc:
            # Access errors are sanitized above; do not serialize arbitrary remote responses.
            message = str(exc) if name.startswith("access_") else f"{type(exc).__name__}: {exc}"
            report["checks"][name] = {"status": "failed", "error": message}
        report["passed"] = all(item["status"] == "passed" for item in report["checks"].values())
        if args.only in ("all", "actquant"):
            report["actquant_verdict"] = actquant_verdict(report["checks"])
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {report['checks'][name]['status']}", flush=True)
    if "actquant_verdict" in report:
        print(f"ActQuant verdict: {report['actquant_verdict']['status']}")
    print(f"Report: {args.output / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
