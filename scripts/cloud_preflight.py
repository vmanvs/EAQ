"""Validate a Linux GPU runtime before downloading LaWAM weights.

This checks infrastructure, not model correctness or LIBERO task success.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
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

    repo, revision, config = MODELS[label]
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
                             if f.rfilename.endswith((".pt", ".safetensors"))]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--only", choices=["all", "runtime", "attention", "render", "access"], default="all")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[1],
                         capture_output=True, text=True, check=False)
    report = {"schema_version": 1, "scope": "preflight only; no policy inference or task success",
              "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
              "eaq_commit": git.stdout.strip() if git.returncode == 0 else None,
              "lawam_revision": LAWAM_REVISION, "python": platform.python_version(),
              "platform": platform.system(), "disk_free_gib": shutil.disk_usage(args.output).free / 2**30,
              "checks": {}}
    packages = sorted(f"{dist.metadata['Name']}=={dist.version}" for dist in importlib.metadata.distributions())
    (args.output / "packages.txt").write_text("\n".join(packages) + "\n", encoding="utf-8")
    checks = {"runtime": runtime_check, "attention": attention_check,
              "render": lambda: render_check(args.output)}
    if args.only in ("all", "access"):
        checks.update({f"access_{label}": lambda label=label: access_check(label, args.output) for label in MODELS})
    if args.only not in ("all", "access"):
        checks = {args.only: checks[args.only]}
    elif args.only == "access":
        checks = {key: value for key, value in checks.items() if key.startswith("access_")}
    for name, check in checks.items():
        try:
            report["checks"][name] = {"status": "passed", "details": check()}
        except Exception as exc:
            # Access errors are sanitized above; do not serialize arbitrary remote responses.
            message = str(exc) if name.startswith("access_") else f"{type(exc).__name__}: {exc}"
            report["checks"][name] = {"status": "failed", "error": message}
        report["passed"] = all(item["status"] == "passed" for item in report["checks"].values())
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"{name}: {report['checks'][name]['status']}", flush=True)
    print(f"Report: {args.output / 'report.json'}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
