"""Build ActQuant's Pi 0.5 runtime, package it, and run one inference.

The runtime is compiled off Molab, in GitHub Actions (Debian 13 and Python
3.13, as on Molab, but no GPU), and published as a release. Molab only
fetches that package and runs inference: long compiles in a notebook get
the Molab sandbox reset. Both sides use this script and the same private,
version-consistent CUDA toolkit from pinned pip wheels (PyTorch's own pip
toolkit mixes nvcc 13.3 with 13.0 runtime headers, which CCCL rejects).
Stages run in order and each writes `report.json`, so a partial run still
leaves evidence:

  toolkit    install the pinned CUDA toolkit wheels into the work folder and
             check that nvcc, runtime headers and CCCL agree (compiles CUB;
             runs it too when a GPU is present)
  source     fetch ActQuant at its pinned commit and unpack vendored code
  configure  CMake + Ninja configure of the pi05 runtime
  build      build pi05, llama-quantize and, when possible, the pi05.so binding
  package    tar the executables and libraries with RUNPATH
             $ORIGIN:$ORIGIN/../cuda/lib, and install that package locally
  fetch      download the pinned release package, verify its SHA-256 and
             install it (the Molab path; replaces source/configure/build/package)
  download   fetch the pinned 3-bit checkpoint and verify its SHA-256; with --model fp16
             or q8, export that reference from the base checkpoint with ActQuant's
             export_pi05.py instead (or copy it from --model-cache)
  infer      one inference through the CLI (and the binding, if present),
             from the installed package or else the build tree

CI:    --no-gpu --cuda-arch 120 --no-vmm --stages toolkit source configure build
       package download infer --infer-device CPU
Molab: --stages toolkit fetch download infer

An installed package is a folder with bin/ and a `cuda` link to the pinned
toolkit's root, so no LD_LIBRARY_PATH is needed. The work directory keeps
sources, build trees, packages and checkpoints between runs in one session.
A lock in the work directory stops two runs from using it at once. This is
one self-contained file: Molab fetches it alone and may run Python with
PYTHONSAFEPATH, which blocks sibling-module imports.
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
import struct
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import threading
import time
import urllib.parse
import urllib.request
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
# The FP16 reference: ActQuant's export_pi05.py run on the base checkpoint it quantizes. Its 3-bit
# release is this FP16 file with the PaliGemma LLM swapped for a quantized one (merge_pi05_llm.py).
BASE_REPO = "lerobot/pi05_libero_finetuned_v044"
BASE_REVISION = "8e174154ef5f6c60a8da12ae99c303d8963138c1"
BASE_FILES = {
    "model.safetensors": "877b3ec1130548b69af7f8aeef3ec9d3fc7738040f0b9beb490857ec970997ae",
    "config.json": None,
    "policy_preprocessor.json": None,
    "policy_preprocessor_step_2_normalizer_processor.safetensors":
        "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
    "policy_postprocessor.json": None,
    "policy_postprocessor_step_0_unnormalizer_processor.safetensors":
        "a002c0df7f79c5b169c5a899ad151d4ea1bed246c7d82bd93ed1556558d517a9",
}
MODELS = {"actquant-3bpw": "actquant-pi05-libero-3bpw", "fp16": "pi05-libero-fp16",
          "q8": "pi05-libero-q8"}  # name: checkpoint folder
# References exported from the base checkpoint by export_pi05.py: name -> its quantization flags.
# q8 quantizes what the 3-bit release quantizes (PaliGemma LLM, SigLIP vision, text embedding), to
# Q8_0 instead of 2-3 bits, and keeps the action expert, projector and norms in FP16 as the release does.
EXPORTS = {"fp16": [], "q8": ["--quant_llm", "q8", "--quant_vision", "q8", "--quant_embedding", "q8"]}
# SHA-256 of each export's pi05.gguf, once known; later exports and cached copies must match.
EXPORT_SHA256 = {"fp16": None, "q8": None}
# Added to the base checkpoint's config.json for the export, so the export's metadata matches the 3-bit
# release's. export_pi05.py reads action_horizon first and falls back to chunk_size, which is 50 in
# lerobot's config.json; the release has pi05_action.action_horizon = 10 (openpi's pi05_libero). A
# horizon-50 FP16 export scored 4/10 on scenes the release solved 10/10.
EXPORT_CONFIG = {"action_horizon": 10}
# Metadata allowed to differ from the release's: what was quantized, and the vision mean/std, which the
# export writes as [0.5, 0.5, 0.5] and the release as [0.5]. GGUF.* keys (the release's merge step copied
# GGUFReader's virtual header fields in as metadata) are ignored too.
METADATA_MAY_DIFFER = {"pi05.quant_llm", "pi05.quant_vision", "pi05.quant_embedding",
                       "clip.vision.image_mean", "clip.vision.image_std"}

# Applied to the ActQuant checkout by the source stage. The pi05 runtime's vision, projector and
# action expert each open the whole pi05.gguf and allocated every tensor in it on the device, so
# the device held three copies of the file (6.25 GB FP16: ~19 GB, more than a T4's 15 GB). The
# patch allocates each model's own tensors only, and has the text embedding read embed.weight
# instead of the whole file into host RAM, where it stayed for the life of the process ("loadfix").
# It also makes ggml's cuBLAS matrix products with F16 weights accumulate and write in fp32
# ("fp32acc"). With fp16 accumulation, the default on NVIDIA, the FP16 export succeeded in 5 of 10
# episodes whose starting scenes the 3-bit release solved 10 of 10; quantized weights take ggml's int8
# kernels, which accumulate in fp32 anyway.
ACTQUANT_PATCH_ID = "loadfix-fp32acc"
ACTQUANT_PATCH = r'''diff --git a/ggml/src/ggml-cuda/ggml-cuda.cu b/ggml/src/ggml-cuda/ggml-cuda.cu
index fb69152..d7a1cab 100644
--- a/ggml/src/ggml-cuda/ggml-cuda.cu
+++ b/ggml/src/ggml-cuda/ggml-cuda.cu
@@ -1290,7 +1290,9 @@ static void ggml_cuda_op_mul_mat_cublas(
 
         CUBLAS_CHECK(cublasSetStream(ctx.cublas_handle(id), stream));
 
-        if (GGML_CUDA_CC_IS_CDNA(cc) || GGML_CUDA_CC_IS_RDNA4(cc)) {
+        // EAQ: accumulate and write in fp32 on every GPU, as on CDNA/RDNA4. With fp16 accumulation, Pi 0.5's
+        // FP16 export (16384-term dot products in the Gemma MLP) lost to its own 3-bit quantization.
+        if (true) {
             const float alpha = 1.0f;
             const float beta = 0.0f;
             CUBLAS_CHECK(
@@ -1892,7 +1894,8 @@ static void ggml_cuda_mul_mat_batched_cublas_impl(ggml_backend_cuda_context & ct
     const float alpha_f32 = 1.0f;
     const float beta_f32 = 0.0f;
 
-    if (dst->op_params[0] == GGML_PREC_DEFAULT) {
+    // EAQ: F16 inputs accumulate and write in fp32 (the GGML_PREC_F32 branch), as in ggml_cuda_op_mul_mat_cublas.
+    if (dst->op_params[0] == GGML_PREC_DEFAULT && src0_type != GGML_TYPE_F16) {
         if constexpr (src0_type == GGML_TYPE_F32) {
             dst_t = (char *) dst_ddf;  // Direct F32 output
         } else {
diff --git a/tools/pi0.5/model_defs.cpp b/tools/pi0.5/model_defs.cpp
index f2b2420..394cb0c 100644
--- a/tools/pi0.5/model_defs.cpp
+++ b/tools/pi0.5/model_defs.cpp
@@ -9,8 +9,7 @@ bool BaseModel::load_tensors(ModelLoader &model_loader, ContextManager &ctx_mana
     std::map<std::string, size_t> tensor_offset;
     gguf_context_ptr &ctx_gguf = model_loader.ctx_gguf_;
 
-    ctx_manager.ctx_data_ = std::move(model_loader.ctx_meta_);
-    ggml_context *ctx = ctx_manager.ctx_data_.get();
+    ggml_context_ptr ctx_meta = std::move(model_loader.ctx_meta_);
 
     for (int64_t i = 0; i < gguf_get_n_tensors(ctx_gguf.get()); ++i) {
         const char *name = gguf_get_tensor_name(ctx_gguf.get(), i);
@@ -18,6 +17,23 @@ bool BaseModel::load_tensors(ModelLoader &model_loader, ContextManager &ctx_mana
                               gguf_get_tensor_offset(ctx_gguf.get(), i);
     }
 
+    // Allocate this model's tensors only. Vision, projector and action expert each open the
+    // whole pi05.gguf; allocating its metadata context put one copy of the file per model on
+    // the device. The second get_tensors_to_load points the model at the copied tensors.
+    std::vector<ggml_tensor *> wanted = get_tensors_to_load(ctx_meta.get());
+    ggml_init_params ctx_params = {
+        /*.mem_size   =*/ (wanted.size() + 1) * ggml_tensor_overhead(),
+        /*.mem_buffer =*/ nullptr,
+        /*.no_alloc   =*/ true,
+    };
+    ctx_manager.ctx_data_.reset(ggml_init(ctx_params));
+    ggml_context *ctx = ctx_manager.ctx_data_.get();
+    for (ggml_tensor *t : wanted) {
+        if (!ggml_get_tensor(ctx, t->name)) {
+            ggml_set_name(ggml_dup_tensor(ctx, t), t->name);
+        }
+    }
+
     std::vector<ggml_tensor *> tensors_to_load = get_tensors_to_load(ctx);
 
     std::vector<uint8_t> read_buf;
diff --git a/tools/pi0.5/text_embed.hpp b/tools/pi0.5/text_embed.hpp
index 9a38162..451b8d8 100644
--- a/tools/pi0.5/text_embed.hpp
+++ b/tools/pi0.5/text_embed.hpp
@@ -14,6 +14,7 @@
 #include "ggml-quants.h"
 #include <cmath>
 #include <cstring>
+#include <fstream>
 #include <stdexcept>
 #include <string>
 #include <vector>
@@ -222,7 +223,7 @@ private:
         // Load GGUF file
         ggml_context *meta = nullptr;
         gguf_init_params params;
-        params.no_alloc = false;  // We need the data
+        params.no_alloc = true;  // embed.weight is read below; the rest of the file is not needed
         params.ctx = &meta;
 
         gguf_context *ctx_gguf = gguf_init_from_file(model_path.c_str(), params);
@@ -269,7 +270,16 @@ private:
         embed_type_ = embed_tensor->type;
         size_t data_size = ggml_nbytes(embed_tensor);
         embed_buffer_.resize(data_size);
-        memcpy(embed_buffer_.data(), embed_tensor->data, data_size);
+        const int64_t tensor_id = gguf_find_tensor(ctx_gguf, "embed.weight");
+        std::ifstream fin(model_path, std::ios::binary);
+        fin.seekg(gguf_get_data_offset(ctx_gguf) + gguf_get_tensor_offset(ctx_gguf, tensor_id));
+        fin.read(reinterpret_cast<char *>(embed_buffer_.data()), data_size);
+        if (!fin) {
+            fprintf(stderr, "TextEmbed: failed to read embed.weight from %s\n", model_path.c_str());
+            gguf_free(ctx_gguf);
+            ggml_free(meta);
+            return false;
+        }
         embed_data_ = embed_buffer_.data();
 
         printf("TextEmbed: Loaded embedding table (%.2f MB, type=%d)\n",
'''

# Runs export_pi05.py unmodified (passing on any quantization flags given after the folders), with
# two memory changes that do not alter its output: the 7.5 GB
# bf16 state dict is read lazily, one tensor at a time, and GGUFWriter spills tensors to a temporary
# file instead of keeping all 7 GB of FP16 arrays in RAM. As written it needs about 16-17 GB.
EXPORT_WRAPPER = r'''
import runpy, sys
source, model_dir, output_dir, *quant_flags = sys.argv[1:]
sys.path[:0] = [f"{source}/tools/pi0.5", f"{source}/gguf-py"]
import gguf
import safetensors.torch
from safetensors import safe_open


class LazyStateDict(dict):
    """A dict (export_pi05.py checks isinstance(st, dict)) that reads each tensor when asked."""

    def __init__(self, path):
        super().__init__()
        self._file = safe_open(path, framework="pt")
        self._keys = list(self._file.keys())

    def __getitem__(self, key):
        return self._file.get_tensor(key)

    def keys(self):
        return list(self._keys)

    def __iter__(self):
        return iter(self._keys)

    def __len__(self):
        return len(self._keys)

    def __contains__(self, key):
        return key in self._keys


eager = safetensors.torch.load_file
safetensors.torch.load_file = lambda path, device="cpu": (
    LazyStateDict(path) if str(path).endswith("model.safetensors") else eager(path, device=device))


class TempFileWriter(gguf.GGUFWriter):
    def __init__(self, *args, **kwargs):
        kwargs["use_temp_file"] = True
        super().__init__(*args, **kwargs)


gguf.GGUFWriter = TempFileWriter
script = f"{source}/tools/pi0.5/export_pi05.py"
sys.argv = [script, "-d", model_dir, "-o", output_dir, *quant_flags]
runpy.run_path(script, run_name="__main__")
'''

GGUF_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 8: "Q8_0", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K", 13: "Q5_K",
              14: "Q6_K", 16: "IQ2_XXS", 17: "IQ2_XS", 18: "IQ3_XXS", 21: "IQ3_S", 22: "IQ2_S", 30: "BF16"}


def gguf_header(data):
    """({key: value}, {name: (shape, type)}) from the start of a GGUF file (metadata and tensor infos)."""
    pos = 0

    def take(fmt):
        nonlocal pos
        values = struct.unpack_from("<" + fmt, data, pos)
        pos += struct.calcsize("<" + fmt)
        return values[0]

    def text():
        nonlocal pos
        size = take("Q")
        pos += size
        return data[pos - size:pos].decode("utf-8", "replace")

    scalar = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}

    def value(kind):
        if kind == 8:
            return text()
        if kind == 9:
            item, count = take("I"), take("Q")
            return [value(item) for _ in range(count)]
        return take(scalar[kind])

    if data[:4] != b"GGUF":
        raise ValueError("not a GGUF file")
    pos = 4
    take("I")  # version
    n_tensors, n_kv = take("Q"), take("Q")
    metadata = {}
    for _ in range(n_kv):
        key = text()
        metadata[key] = value(take("I"))
    table = {}
    for _ in range(n_tensors):
        name = text()
        shape = [take("Q") for _ in range(take("I"))]
        kind = take("I")
        take("Q")  # offset
        table[name] = (shape, GGUF_TYPES.get(kind, str(kind)))
    return metadata, table


BUILD_TARGETS = ("pi05", "llama-quantize")
DEFAULT_PROMPT = "put the black bowl on the plate"
STAGES = ("toolkit", "source", "configure", "build", "package", "fetch", "download", "infer")
PACKAGE_RUNPATH = "$ORIGIN:$ORIGIN/../cuda/lib"
# Files taken from the build tree's bin/ into a package.
PACKAGE_FILES = re.compile(r"^(pi05|llama-quantize|pi05\.so|lib[\w.+-]*\.so(\.\d+)*)$")
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
TIMEOUTS = {"configure": 15 * 60, "build": 3 * 60 * 60, "infer": 20 * 60, "fetch": 30 * 60}
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
        self.checkpoint = self.work / "checkpoints" / MODELS[args.model]
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
            "checkpoint": ({"repo": CHECKPOINT_REPO, "revision": CHECKPOINT_REVISION} if args.model not in EXPORTS
                           else {"base_repo": BASE_REPO, "base_revision": BASE_REVISION,
                                 "export": "tools/pi0.5/export_pi05.py", "export_flags": EXPORTS[args.model]}),
            "options": {"model": args.model,"stages": args.stages, "cuda_arch": self.arch, "no_vmm": args.no_vmm,
                        "no_flash_attn": args.no_flash_attn, "no_gpu": args.no_gpu,
                        "infer_device": args.infer_device,
                        "jobs": args.jobs, "cpu_check": args.cpu_check, "prompt": args.prompt,
                        "work_dir": str(self.work)},
            "host": self._host(),
            "stages": {},
        }

    @property
    def _suffix(self):
        # One tree per architecture, toolkit version and option set: CMake cannot switch compilers in place.
        version = (self.toolkit or {}).get("version") or "none"
        return (f"sm{self.arch}-cu{version}" + ("-novmm" if self.args.no_vmm else "")
                + ("-nofa" if self.args.no_flash_attn else "") + ("-portable" if self.args.no_gpu else ""))

    @property
    def build_dir(self):
        return self.work / f"build-{self._suffix}"

    @property
    def shim_dir(self):
        return self.work / f"libshim-{self._suffix}"

    @property
    def package_name(self):
        return f"actquant-pi05-{self._suffix}-{ACTQUANT_COMMIT[:7]}-{ACTQUANT_PATCH_ID}"

    @property
    def runtime_bin(self):
        """bin/ of the installed package (fetch or package stage), else of the build tree."""
        current = self.work / "prebuilt" / "current.json"
        try:
            installed = Path(json.loads(current.read_text(encoding="utf-8"))["path"]) / "bin"
        except (OSError, ValueError, KeyError):
            installed = None
        if installed is not None and (installed / "pi05").is_file():
            return installed
        return self.build_dir / "bin"

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
            if self.args.no_gpu:
                details["probe_run"] = "skipped: --no-gpu"
            else:
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
        # Reset tracked files, then apply ACTQUANT_PATCH, so a rerun never patches twice.
        _run([git, "-C", self.source, "checkout", "-q", "--", "."], timeout=300)
        patch_file = self.work / f"actquant-{ACTQUANT_PATCH_ID}.patch"
        patch_file.write_text(ACTQUANT_PATCH, encoding="utf-8", newline="\n")
        applied = _run([git, "-C", self.source, "apply", "--ignore-whitespace", patch_file])
        if applied["returncode"] != 0:
            raise StageError("Applying ACTQUANT_PATCH failed.", {"output": applied["output"][-2000:]})
        self.deviation("patch", f"ActQuant patched ({ACTQUANT_PATCH_ID}): each pi05 sub-model allocates only "
                                "its own tensors, the text embedding reads only embed.weight, and cuBLAS "
                                "products with F16 weights accumulate in fp32 instead of fp16.")
        dirty = _run([git, "-C", self.source, "status", "--porcelain", "--untracked-files=no"])["output"]
        vendored = self.source / "vendor" / "tokenizers-cpp"
        if not vendored.is_dir():
            # As in ActQuant's README: unzip vendor/tokenizers-cpp.zip -d vendor/
            with zipfile.ZipFile(self.source / "vendor" / "tokenizers-cpp.zip") as archive:
                archive.extractall(self.source / "vendor")
        return {"commit": head, "path": str(self.source), "patch": ACTQUANT_PATCH_ID,
                "patch_sha256": hashlib.sha256(ACTQUANT_PATCH.encode()).hexdigest(),
                "modified_tracked_files": dirty.splitlines(),
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
        if libcuda is None and not self.args.no_vmm:
            raise StageError("The CUDA driver library (libcuda.so.1) was not found, and ggml links it for "
                             "virtual memory management; pass --no-vmm on hosts without a driver.")
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
                *(["-DGGML_CUDA_FA=OFF"] if self.args.no_flash_attn else []),
                # Without a GPU this is not the target host, so -march=native could emit instructions
                # the target CPU lacks; GGML_NATIVE=OFF targets ggml's portable AVX2 baseline.
                *(["-DGGML_NATIVE=OFF"] if self.args.no_gpu else [])]
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
        if self.args.no_gpu:
            self.deviation("native", "GGML_NATIVE=OFF: ggml's CPU code built for a portable AVX2 baseline "
                                     "instead of -march=native, since the build host is not the target.")
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

    def _install_package(self, tarball, name):
        """Unpack a package into work/prebuilt/<name>, link its `cuda` folder to the pinned toolkit,
        check that every library resolves, and make it the runtime used by the infer stage."""
        toolkit = self._require_toolkit()
        root = self.work / "prebuilt"
        target = root / name
        if target.exists():
            shutil.rmtree(target)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball) as archive:
            names = archive.getnames()
            if not names or any(member.split("/")[0] != name for member in names):
                raise StageError(f"{Path(tarball).name} does not contain a single {name}/ folder.")
            archive.extractall(root, filter="data")
        (target / "cuda").symlink_to(toolkit["root"], target_is_directory=True)
        try:
            manifest = json.loads((target / "eaq-package.json").read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StageError(f"The package has no readable eaq-package.json ({type(exc).__name__}).")
        libraries, unresolved = {}, []
        for binary in ("pi05", "pi05.so"):
            if (target / "bin" / binary).is_file():
                output = _run(["ldd", target / "bin" / binary], timeout=60)["output"]
                libraries[binary] = [line.strip() for line in output.splitlines()
                                     if re.search(r"cuda|cublas|ggml|llama|gomp|stdc\+\+|not found", line)]
                unresolved += [f"{binary}: {line.strip()}" for line in output.splitlines() if "not found" in line]
        details = {"path": str(target), "libraries": libraries}
        if unresolved:
            raise StageError("Libraries in the package do not resolve: " + "; ".join(unresolved), details)
        (root / "current.json").write_text(json.dumps({"name": name, "path": str(target)}) + "\n",
                                           encoding="utf-8")
        return details, manifest

    def stage_package(self):
        patchelf = _which("patchelf")
        if not patchelf:
            raise StageError("patchelf is not installed; it sets the package's RUNPATH.")
        binary_dir = self.build_dir / "bin"
        if not (binary_dir / "pi05").is_file():
            raise StageError(f"{binary_dir / 'pi05'} does not exist; run the build stage.")
        toolkit = self._require_toolkit()
        name = self.package_name
        # Staged in the work folder so the output (uploaded by CI) holds only the tarball.
        staging = self.work / "package-staging" / name
        if staging.exists():
            shutil.rmtree(staging)
        (staging / "bin").mkdir(parents=True)
        files = {}
        for path in sorted(binary_dir.iterdir()):
            if not PACKAGE_FILES.match(path.name):
                continue
            target = staging / "bin" / path.name
            if path.is_symlink():
                target.symlink_to(os.readlink(path))
                files[path.name] = {"link": os.readlink(path)}
                continue
            shutil.copy2(path, target)
            result = _run([patchelf, "--set-rpath", PACKAGE_RUNPATH, target], timeout=120)
            if result["returncode"] != 0:
                raise StageError(f"patchelf failed on {path.name}: {result['output'][-500:]}")
            files[path.name] = {"bytes": target.stat().st_size, "sha256": self._sha256(target),
                                "runpath": _run([patchelf, "--print-rpath", target])["output"]}
        binding = "pi05.so" in files
        self.deviation("package", "Runtime compiled without a GPU in a Debian 13 / Python 3.13 container "
                                  "(GitHub Actions) and packaged with RUNPATH "
                                  f"{PACKAGE_RUNPATH}; `cuda` links to the pinned toolkit at install.")
        manifest = {
            "schema_version": 1,
            "name": name,
            "built_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "eaq_commit": os.environ.get("GITHUB_SHA"),
            "eaq_run": (f"{os.environ['GITHUB_SERVER_URL']}/{os.environ['GITHUB_REPOSITORY']}/actions/runs/"
                        f"{os.environ['GITHUB_RUN_ID']}" if os.environ.get("GITHUB_RUN_ID") else None),
            "actquant": {"url": ACTQUANT_URL, "commit": ACTQUANT_COMMIT},
            "cuda_arch": self.arch,
            "toolkit": {"version": toolkit["version"], "pins": _read_pins(self.args.toolkit_requirements)},
            "options": {"no_vmm": self.args.no_vmm, "no_flash_attn": self.args.no_flash_attn,
                        "binding": binding},
            "python": platform.python_version() if binding else None,
            "glibc": "-".join(platform.libc_ver()),
            "compiler": self.report["host"]["tools"].get("gcc"),
            "layout": f"bin/ holds the executables and libraries, with RUNPATH {PACKAGE_RUNPATH}. Link "
                      "<package>/cuda to the root of the pinned CUDA toolkit (nvidia/cu13) before use.",
            "deviations": list(self.deviations.values()),
            "files": files,
        }
        (staging / "eaq-package.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        tarball = self.output / "package" / f"{name}.tar.gz"
        tarball.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tarball, "w:gz") as archive:
            archive.add(staging, arcname=name)
        digest = self._sha256(tarball)
        (self.output / "package" / f"{name}.tar.gz.sha256").write_text(f"{digest}  {tarball.name}\n",
                                                                       encoding="utf-8")
        installed, _ = self._install_package(tarball, name)
        return {"name": name, "tarball": str(tarball), "bytes": tarball.stat().st_size, "sha256": digest,
                "files": sorted(files), "binding": binding, "installed": installed}

    def stage_fetch(self):
        try:
            pin = json.loads(Path(self.args.prebuilt_pin).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise StageError(f"Cannot read the package pin {self.args.prebuilt_pin} ({type(exc).__name__}).")
        missing = [key for key in ("repository", "tag", "asset", "sha256") if not pin.get(key)]
        if missing:
            raise StageError("No prebuilt package is pinned yet (missing " + ", ".join(missing) + " in "
                             f"{Path(self.args.prebuilt_pin).name}); run the ActQuant build workflow and "
                             "copy the pin from its summary.")
        if not pin["asset"].endswith(".tar.gz"):
            raise StageError(f"The pinned asset {pin['asset']} is not a .tar.gz package.")
        name = pin["asset"][:-len(".tar.gz")]
        url = (f"https://github.com/{pin['repository']}/releases/download/"
               f"{urllib.parse.quote(pin['tag'])}/{urllib.parse.quote(pin['asset'])}")
        tarball = self.work / "downloads" / pin["asset"]
        tarball.parent.mkdir(parents=True, exist_ok=True)
        details = {"url": url, "sha256_pinned": pin["sha256"]}
        if not (tarball.is_file() and self._sha256(tarball) == pin["sha256"]):
            log(f"downloading {pin['asset']}")
            started = time.monotonic()
            partial = tarball.with_name(tarball.name + ".part")
            request = urllib.request.Request(url, headers={"User-Agent": "eaq-actquant-build"})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, open(partial, "wb") as handle:
                    shutil.copyfileobj(response, handle, 16 * 2**20)
            except OSError as exc:
                raise StageError(f"Download of {url} failed ({type(exc).__name__}: {exc}).", details)
            partial.replace(tarball)
            details["download_seconds"] = round(time.monotonic() - started, 1)
        digest = self._sha256(tarball)
        details.update(bytes=tarball.stat().st_size, sha256=digest)
        if digest != pin["sha256"]:
            raise StageError(f"Package SHA-256 mismatch: pinned {pin['sha256']}, got {digest}.", details)
        installed, manifest = self._install_package(tarball, name)
        details["installed"] = installed
        details["package"] = {key: manifest.get(key) for key in (
            "name", "built_utc", "eaq_commit", "eaq_run", "actquant", "cuda_arch", "toolkit", "options",
            "python", "glibc", "compiler")}
        problems = []
        if manifest.get("actquant", {}).get("commit") != ACTQUANT_COMMIT:
            problems.append(f"package is ActQuant {manifest.get('actquant', {}).get('commit')}, "
                            f"this script pins {ACTQUANT_COMMIT}")
        if self.arch and str(manifest.get("cuda_arch")) != str(self.arch):
            problems.append(f"package is built for sm_{manifest.get('cuda_arch')}, this GPU is sm_{self.arch}")
        if _version_tuple(manifest.get("toolkit", {}).get("version")) != _version_tuple(self.toolkit["version"]):
            problems.append(f"package was built with CUDA {manifest.get('toolkit', {}).get('version')}, "
                            f"the installed runtime is {self.toolkit['version']}")
        if problems:
            raise StageError("The pinned package does not match this host: " + "; ".join(problems) + ".", details)
        python = manifest.get("python")
        if python and _version_tuple(python) != _version_tuple(platform.python_version()):
            details["binding_note"] = (f"pi05.so was built for Python {python}; this is "
                                       f"{platform.python_version()}, so it will not import.")
        for index, text in enumerate(manifest.get("deviations", [])):
            if not text.startswith("CUDA toolkit"):  # the build host's; the toolkit stage records this host's
                self.deviation(f"package_{index}", text)
        self.deviation("prebuilt", f"Runtime not compiled here: release {pin['tag']} of {pin['repository']}, "
                                   "built by the EAQ GitHub Actions workflow.")
        return details

    def stage_download(self):
        if self.args.model in EXPORTS:
            return self._export_checkpoint()
        return self._download_release(self.checkpoint, CHECKPOINT_FILES)

    def _download_release(self, folder, files_wanted):
        try:
            from huggingface_hub import HfApi, hf_hub_download
        except ImportError as exc:
            raise StageError("huggingface_hub is not installed.") from exc
        folder.mkdir(parents=True, exist_ok=True)
        token = os.environ.get("HF_TOKEN") or None  # not required: the checkpoint is public
        files = {}
        for name, expected in files_wanted.items():
            target = folder / name
            if not (target.is_file() and (expected is None or self._sha256(target) == expected)):
                log(f"downloading {name}")
                started = time.monotonic()
                try:
                    path = hf_hub_download(CHECKPOINT_REPO, name, revision=CHECKPOINT_REVISION, token=token,
                                           local_dir=str(folder))
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
        return {"path": str(folder), "files": files, "revision_resolved": resolved}

    # ------------------------------------------------------------------ FP16 and Q8_0 references

    EXPORT_FILES = ("pi05.gguf", "tokenizer.model", "norm_stats.json", "eaq-export.json")

    def _export_valid(self, folder):
        """The export manifest of `folder` if its pi05.gguf matches it (and EXPORT_SHA256, if pinned)."""
        try:
            manifest = json.loads((folder / "eaq-export.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None
        if not all((folder / name).is_file() for name in self.EXPORT_FILES):
            return None
        if (manifest.get("export_flags", []) != EXPORTS[self.args.model]
                or manifest.get("config_overrides") != EXPORT_CONFIG):
            return None
        pinned = EXPORT_SHA256[self.args.model]
        if pinned and manifest.get("sha256") != pinned:
            return None
        return manifest if self._sha256(folder / "pi05.gguf") == manifest.get("sha256") else None

    def _export_checkpoint(self):
        """A reference export: already in the work folder, copied from --model-cache, or exported."""
        model = self.args.model
        label = {"fp16": "FP16", "q8": "Q8_0"}[model]
        cache = None
        if self.args.model_cache:
            cache = Path(self.args.model_cache) / f"{MODELS[model]}-{BASE_REVISION[:7]}-{ACTQUANT_COMMIT[:7]}"
        flags = " ".join(EXPORTS[model]) or "no quantization flags"
        self.deviation(f"model_{model}", f"Reference model: {label} export of {BASE_REPO}@{BASE_REVISION[:7]} by "
                                         f"ActQuant's tools/pi0.5/export_pi05.py ({flags}; config.json plus "
                                         f"{json.dumps(EXPORT_CONFIG)}), run with a lazy "
                                         "state-dict loader and a temp-file GGUF writer. norm_stats.json is the "
                                         "3-bit release's, so only the weights differ.")
        log(f"checking for a {label} export in the work folder")
        manifest = self._export_valid(self.checkpoint)
        if manifest:
            return {"source": "work folder", "path": str(self.checkpoint), "manifest": manifest}
        if cache and (cache / "eaq-export.json").is_file():
            log(f"copying the {label} export from {cache}")
            started = time.monotonic()
            self.checkpoint.mkdir(parents=True, exist_ok=True)
            for name in self.EXPORT_FILES:
                shutil.copyfile(cache / name, self.checkpoint / name)
            manifest = self._export_valid(self.checkpoint)
            if manifest:
                return {"source": "cache", "cache": str(cache), "path": str(self.checkpoint),
                        "copy_seconds": round(time.monotonic() - started, 1), "manifest": manifest}
            log("the cached copy does not match its manifest; exporting again")
        details = self._export(label)
        if cache:
            log(f"storing the {label} export in {cache}")
            partial = cache.with_name(cache.name + ".partial")
            shutil.rmtree(partial, ignore_errors=True)
            partial.mkdir(parents=True)
            for name in self.EXPORT_FILES:
                shutil.copyfile(self.checkpoint / name, partial / name)
            shutil.rmtree(cache, ignore_errors=True)
            partial.rename(cache)
            details["cached_to"] = str(cache)
        return details

    def _export(self, label):
        started = time.monotonic()
        model = self.args.model
        # The 3-bit release's tokenizer (the same 4,264,023-byte file as the gated
        # google/paligemma-3b-pt-224 tokenizer.model) and its quantile norm_stats.json.
        release = self.work / "checkpoints" / MODELS["actquant-3bpw"]
        small = {name: CHECKPOINT_FILES[name] for name in ("tokenizer.model", "norm_stats.json")}
        release_files = self._download_release(release, small)["files"]
        base = self._download_base()
        shutil.copyfile(release / "tokenizer.model", base["dir"] / "tokenizer.model")
        source = self.stage_source()
        python = self._export_venv()

        output = self.work / f"{model}-export"
        shutil.rmtree(output, ignore_errors=True)
        output.mkdir(parents=True)
        scratch = self.work / f"{model}-export-tmp"  # GGUFWriter's temp file: up to about 7 GB
        scratch.mkdir(exist_ok=True)
        # The base files, linked, with config.json extended by EXPORT_CONFIG.
        export_input = self.work / f"{model}-export-input"
        shutil.rmtree(export_input, ignore_errors=True)
        export_input.mkdir(parents=True)
        for item in base["dir"].iterdir():
            if item.is_file() and item.name != "config.json":
                (export_input / item.name).symlink_to(item.resolve())
        config = json.loads((base["dir"] / "config.json").read_text(encoding="utf-8"))
        (export_input / "config.json").write_text(json.dumps({**config, **EXPORT_CONFIG}, indent=2) + "\n",
                                                  encoding="utf-8")
        wrapper = self.work / "eaq_export_pi05.py"
        wrapper.write_text(EXPORT_WRAPPER, encoding="utf-8")
        env = {key: value for key, value in os.environ.items()
               if key not in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE", "VIRTUAL_ENV")}
        env.update(TMPDIR=str(scratch), PI05_OUTPUT_PRECISION="fp16", PYTHONUNBUFFERED="1")
        log(f"exporting the {label} model with export_pi05.py (CPU; several minutes)")
        result = _stream([python, wrapper, self.source, export_input, output, *EXPORTS[model]],
                         self.output / f"export_{model}.log",
                         timeout=3600, env=env, echo=False)
        shutil.rmtree(scratch, ignore_errors=True)
        if result["returncode"] != 0 or not (output / "pi05.gguf").is_file():
            raise StageError("export_pi05.py failed.", {"tail": result["tail"][-3000:]})

        # export_pi05.py derives norm_stats.json from lerobot's MEAN_STD normalizer files; the 3-bit
        # release ships quantile stats, which serve_policy.py prefers. Keep the release's, so the
        # two models differ only in their weights.
        exported_stats = json.loads((output / "norm_stats.json").read_text(encoding="utf-8"))
        shutil.move(output / "norm_stats.json", output / "norm_stats.export.json")
        shutil.copyfile(release / "norm_stats.json", output / "norm_stats.json")
        structure = self._compare_with_release(output / "pi05.gguf", label)
        digest = self._sha256(output / "pi05.gguf")
        pinned = EXPORT_SHA256[model]
        if pinned and digest != pinned:
            raise StageError(f"{label} export SHA-256 {digest} differs from the pinned {pinned}.",
                             {"structure": structure})
        manifest = {"sha256": digest, "bytes": (output / "pi05.gguf").stat().st_size,
                    "base": {"repo": BASE_REPO, "revision": BASE_REVISION},
                    "actquant_commit": ACTQUANT_COMMIT, "export": "tools/pi0.5/export_pi05.py",
                    "precision": model, "export_flags": EXPORTS[model], "config_overrides": EXPORT_CONFIG,
                    "tokenizer_sha256": release_files["tokenizer.model"]["sha256"],
                    "norm_stats": f"{CHECKPOINT_REPO}@{CHECKPOINT_REVISION[:7]}",
                    "exported_norm_stats_keys": sorted(exported_stats)}
        (output / "eaq-export.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        shutil.rmtree(self.checkpoint, ignore_errors=True)
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        output.rename(self.checkpoint)
        return {"source": "export", "path": str(self.checkpoint), "manifest": manifest, "structure": structure,
                "base": base["files"], "actquant_source": source.get("commit"),
                "export_seconds": result["seconds"], "seconds": round(time.monotonic() - started, 1)}

    def _download_base(self):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise StageError("huggingface_hub is not installed.") from exc
        folder = self.work / "base" / BASE_REPO.split("/")[-1]
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for name, expected in BASE_FILES.items():
            target = folder / name
            if not (target.is_file() and (expected is None or self._sha256(target) == expected)):
                log(f"downloading {BASE_REPO}/{name}")
                started = time.monotonic()
                try:
                    target = Path(hf_hub_download(BASE_REPO, name, revision=BASE_REVISION, local_dir=str(folder)))
                except Exception as exc:  # never log tokens or raw HTTP exceptions
                    code = getattr(getattr(exc, "response", None), "status_code", None)
                    raise StageError(f"Download of {BASE_REPO}/{name} failed ({type(exc).__name__}, HTTP {code}).")
                files[name] = {"download_seconds": round(time.monotonic() - started, 1)}
            digest = self._sha256(target)
            files.setdefault(name, {}).update(bytes=target.stat().st_size, sha256=digest)
            if expected is not None and digest != expected:
                raise StageError(f"{name} SHA-256 mismatch: expected {expected}, got {digest}.", {"files": files})
        return {"dir": folder, "files": files}

    def _export_venv(self):
        """A CPU venv with requirements/fp16-export.txt (torch CPU, safetensors, numpy, pyyaml)."""
        requirements = Path(self.args.export_requirements)
        if not requirements.is_file():
            raise StageError(f"{requirements} not found.")
        venv = self.work / "export-venv"
        python = venv / "bin" / "python"
        marker = venv / ".eaq-requirements"
        digest = self._sha256(requirements)
        if python.exists() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == digest:
            return python
        shutil.rmtree(venv, ignore_errors=True)
        uv = _which("uv")
        index = ["--extra-index-url", "https://download.pytorch.org/whl/cpu"]
        if uv:
            steps = [[uv, "venv", "--python", sys.executable, venv],
                     [uv, "pip", "install", "--python", python, "-r", requirements, *index,
                      "--index-strategy", "unsafe-best-match"]]
        else:
            steps = [[sys.executable, "-m", "venv", venv], [python, "-m", "pip", "install", "-r", requirements, *index]]
        log("installing the export environment (torch CPU, about 0.2 GB)")
        for command in steps:
            result = _run(command, timeout=1800)
            if result["returncode"] != 0:
                raise StageError("Installing the FP16 export environment failed.",
                                 {"command": [str(part) for part in command], "output": result["output"][-2500:]})
        marker.write_text(digest + "\n", encoding="utf-8")
        return python

    def _compare_with_release(self, path, label):
        """Tensor names, shapes and types of the export against the 3-bit release's header."""
        with open(path, "rb") as handle:
            our_metadata, ours = gguf_header(handle.read(8 * 2**20))
        url = f"https://huggingface.co/{CHECKPOINT_REPO}/resolve/{CHECKPOINT_REVISION}/pi05.gguf"
        try:  # the header only: the first megabytes, not the 2.4 GB file
            request = urllib.request.Request(url, headers={"Range": f"bytes=0-{8 * 2**20 - 1}"})
            with urllib.request.urlopen(request, timeout=120) as response:
                their_metadata, theirs = gguf_header(response.read())
        except Exception as exc:
            return {"compared": False, "reason": f"{type(exc).__name__}", "tensors": len(ours)}
        types = collections.defaultdict(collections.Counter)
        for name, (_, kind) in ours.items():
            types[name.split(".")[0]][kind] += 1
        missing, extra = sorted(set(theirs) - set(ours)), sorted(set(ours) - set(theirs))
        # The release pads some K-quant tensors to 256 columns; FP16 and Q8_0 tensors are not padded.
        reshaped = {name: {"export": ours[name][0], "release": theirs[name][0]}
                    for name in sorted(set(ours) & set(theirs)) if ours[name][0] != theirs[name][0]}
        metadata_differences = {
            key: {"export": our_metadata.get(key), "release": their_metadata.get(key)}
            for key in sorted(set(our_metadata) | set(their_metadata))
            if not key.startswith(("tokenizer.", "GGUF.")) and key not in METADATA_MAY_DIFFER
            and our_metadata.get(key) != their_metadata.get(key)}
        details = {"compared": True, "metadata_differences": metadata_differences,
                   "tensors": len(ours), "release_tensors": len(theirs),
                   "missing": missing[:20], "extra": extra[:20], "shape_differences": len(reshaped),
                   "shape_examples": dict(list(reshaped.items())[:5]),
                   "types": {group: dict(counts) for group, counts in types.items()}}
        # The release also carries quantile norm.*_q01/q99 tensors (its export read openpi's assets;
        # lerobot's checkpoint gives export_pi05.py mean/std only). The runtime does not read norm.*
        # tensors: serve_policy.py takes the stats from norm_stats.json, which is the release's here.
        if [name for name in missing if not name.startswith("norm.")] or extra or metadata_differences:
            raise StageError(f"The {label} export's tensors or metadata differ from the 3-bit release's.", details)
        return details

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
        # Pi05 default-constructs its vision encoder, projector and action expert (CPU contexts)
        # before replacing them with ones on the requested device, so the last line is the live one.
        backends = re.findall(r"create_backend: using (\S+) backend", output)
        parsed["backends"] = backends
        parsed["backend"] = backends[-1] if backends else None
        match = re.search(r"Output actions \((\d+) dim x (\d+) horizon", output)
        if match:
            parsed["action_dim"], parsed["action_horizon"] = int(match[1]), int(match[2])
        parsed["completed"] = "Inference completed successfully." in output
        return parsed

    def _cli(self, device, image, label):
        binary = self.runtime_bin / "pi05"
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
        binary = self.runtime_bin / "pi05"
        if not binary.is_file():
            raise StageError(f"{binary} does not exist; run the fetch (or build) stage.")
        device = self.args.infer_device
        missing = [name for name in CHECKPOINT_FILES if not (self.checkpoint / name).is_file()]
        if missing:
            raise StageError(f"Checkpoint files missing ({', '.join(missing)}); run the download stage.")
        image = self._smoke_image()
        self.deviation("smoke", "Smoke-test input is a synthetic 224x224 image and a fixed prompt, "
                                "not a LIBERO observation; it checks execution, not task behaviour.")
        details = {"image": image.name, "prompt": self.args.prompt, "runtime": str(self.runtime_bin),
                   "device": device}
        label = "cuda" if device.upper().startswith("CUDA") else device.lower()
        cuda = self._cli(device, image, label)
        # Stored as cli_cuda whatever the device, so readers of the report find it in one place.
        details["cli_cuda"] = cuda
        problems = []
        if cuda["returncode"] != 0 or not cuda["completed"]:
            problems.append(f"pi05 CLI exited with {cuda['returncode']} (see infer_{label}.log)")
        if cuda["backend"] != device:
            problems.append(f"pi05 ran on backend {cuda['backend']!r}, not {device}")
        if not self._finite(cuda.get("first_actions", [])) or not self._finite(list(cuda.get("stats", {}).values())):
            problems.append("actions missing or not finite")

        binding_file = self.runtime_bin / "pi05.so"
        binding_file = binding_file if binding_file.is_file() else None
        if binding_file is not None and not problems:
            details["binding"] = self._binding_run(binding_file, image)
            if details["binding"].get("first") and cuda.get("first_actions"):
                count = min(len(details["binding"]["first"]), len(cuda["first_actions"]))
                details["binding_vs_cli_max_abs_diff"] = max(
                    abs(a - b) for a, b in zip(details["binding"]["first"][:count], cuda["first_actions"][:count]))
        elif binding_file is None:
            details["binding"] = {"status": "not built"}

        if self.args.cpu_check and device != "CPU" and not problems:
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
                             device_name={self.args.infer_device!r}, n_threads=4, num_flow_steps=10)
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
    parser.add_argument("--no-gpu", action="store_true",
                        help="Host has no GPU (CI): needs --cuda-arch; the toolkit probe is built, not run.")
    parser.add_argument("--prebuilt-pin", type=Path,
                        default=Path(__file__).resolve().parents[1] / "requirements" / "actquant-prebuilt.json",
                        help="Release package pinned for the fetch stage.")
    parser.add_argument("--infer-device", default="CUDA0", help="ggml device for the infer stage (CUDA0 or CPU).")
    parser.add_argument("--jobs", type=int, help=f"Parallel build jobs (default: CPUs, limited by memory, at most {DEFAULT_MAX_JOBS}).")
    parser.add_argument("--no-vmm", action="store_true", help="Build with GGML_CUDA_NO_VMM=ON.")
    parser.add_argument("--no-flash-attn", action="store_true",
                        help="Build with GGML_CUDA_FA=OFF (FlashAttention kernels as stubs; shorter build).")
    parser.add_argument("--no-binding", action="store_true", help="Do not build the pi05.so Python binding.")
    parser.add_argument("--cpu-check", action="store_true", help="Also run the CLI on CPU and compare.")
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model", choices=list(MODELS), default="actquant-3bpw",
                        help="the released 3-bit checkpoint, or the FP16 or Q8_0 reference exported from its base")
    parser.add_argument("--model-cache", type=Path,
                        help="folder to reuse a reference export from, and to store a new one in (e.g. Google Drive)")
    parser.add_argument("--export-requirements", type=Path,
                        default=Path(__file__).resolve().parents[1] / "requirements" / "fp16-export.txt")
    parser.add_argument("--build-timeout", type=int, default=TIMEOUTS["build"])
    args = parser.parse_args()
    if args.no_gpu and not args.cuda_arch:
        parser.error("--no-gpu needs --cuda-arch")
    return Builder(args).run()


if __name__ == "__main__":
    sys.exit(main())
