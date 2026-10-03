"""Check which base checkpoint a released pi05.gguf was exported from.

ActQuant's 3-bit release keeps some tensors unquantized (the action expert and projector in F16, vision
biases in F32). Exported from a given lerobot checkpoint, they equal that checkpoint's weights cast to
the release's type. This script reads those tensors from the release and from a candidate
model.safetensors with HTTP range requests (a few MB each, no full download) and compares them.

    python scripts/checkpoint_provenance.py lerobot/pi05_libero_base a217bfd3b14673cf2ce597e69997ab21866438dd

Prints, per tensor, whether the cast candidate weights are identical to the release's, the largest
absolute difference, and the correlation. Needs numpy.
"""
from __future__ import annotations

import argparse
import json
import struct
import urllib.request

import numpy as np

RELEASE = ("https://huggingface.co/NU-World-Model-Embodied-AI/ActQuant-Pi05-LIBERO-3bpw/resolve/"
           "4d03f36f1ac019ad5e314dea584697ab29644171/pi05.gguf")
# Release tensor (stored unquantized) -> suffix of the matching lerobot safetensors key.
PAIRS = {
    "action.action_out.weight": "action_out_proj.weight",
    "action.blk.0.attn_q.weight": "gemma_expert.model.layers.0.self_attn.q_proj.weight",
    "action.blk.17.ffn_down.weight": "gemma_expert.model.layers.17.mlp.down_proj.weight",
    "v.blk.0.attn_q.bias": "vision_tower.vision_model.encoder.layers.0.self_attn.q_proj.bias",
    "mm.0.weight": "multi_modal_projector.linear.weight",
}
GGUF_FLOAT = {0: np.float32, 1: np.float16}
SCALAR_SIZE = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def fetch(url, start, end):
    request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    with urllib.request.urlopen(request, timeout=120) as response:
        return response.read()


def gguf_tensors(url, header_bytes=8 * 2**20):
    """{name: (count, type, absolute offset)} from a GGUF file's header."""
    data = fetch(url, 0, header_bytes - 1)
    pos = 8
    n_tensors, n_kv = struct.unpack_from("<QQ", data, pos)
    pos += 16
    alignment = 32

    def text():
        nonlocal pos
        (size,) = struct.unpack_from("<Q", data, pos)
        pos += 8 + size
        return data[pos - size:pos].decode("utf-8", "replace")

    def skip(kind):
        nonlocal pos
        if kind == 8:
            text()
        elif kind == 9:
            item, count = struct.unpack_from("<IQ", data, pos)
            pos += 12
            for _ in range(count):
                skip(item)
        else:
            pos += SCALAR_SIZE[kind]

    for _ in range(n_kv):
        key = text()
        (kind,) = struct.unpack_from("<I", data, pos)
        pos += 4
        if key == "general.alignment":
            (alignment,) = struct.unpack_from("<I", data, pos)
        skip(kind)
    infos = {}
    for _ in range(n_tensors):
        name = text()
        (ndim,) = struct.unpack_from("<I", data, pos)
        pos += 4
        shape = struct.unpack_from("<" + "Q" * ndim, data, pos)
        pos += 8 * ndim
        kind, offset = struct.unpack_from("<IQ", data, pos)
        pos += 12
        infos[name] = (int(np.prod(shape)), kind, offset)
    start = (pos + alignment - 1) // alignment * alignment
    return {name: (count, kind, start + offset) for name, (count, kind, offset) in infos.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("repo", help="candidate Hugging Face repository, e.g. lerobot/pi05_libero_base")
    parser.add_argument("revision", nargs="?", default="main")
    args = parser.parse_args()
    candidate = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/model.safetensors"

    release = gguf_tensors(RELEASE)
    (header_size,) = struct.unpack("<Q", fetch(candidate, 0, 7))
    header = json.loads(fetch(candidate, 8, 8 + header_size - 1))
    for name, suffix in PAIRS.items():
        count, kind, offset = release[name]
        ours = np.frombuffer(fetch(RELEASE, offset, offset + count * np.dtype(GGUF_FLOAT[kind]).itemsize - 1),
                             dtype=GGUF_FLOAT[kind]).astype(np.float32)
        key = next(k for k in header if k.endswith(suffix))
        begin, end = header[key]["data_offsets"]
        raw = fetch(candidate, 8 + header_size + begin, 8 + header_size + end - 1)
        if header[key]["dtype"] == "BF16":
            theirs = (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
        else:
            theirs = np.frombuffer(raw, dtype=np.float32)
        theirs = theirs.astype(GGUF_FLOAT[kind]).astype(np.float32)
        identical = bool(np.array_equal(ours, theirs))
        print(f"{name:32} {header[key]['dtype']:5} identical={identical!s:5} "
              f"max|diff|={float(np.max(np.abs(ours - theirs))):.6f} corr={float(np.corrcoef(ours, theirs)[0, 1]):.6f}")


if __name__ == "__main__":
    main()
