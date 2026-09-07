#!/usr/bin/env python3
"""Drive tests/gguf_cross_check.cpp against the real GGML loader.

Two modes:

  * No args: generate a synthetic but representative state_dict (f32/f16/bf16/
    f64/i32, 1-D..4-D), convert it, and validate — proves the GGUF *container*
    format (alignment, offsets, header) round-trips through ggml.

  * One or more .pt paths: load the real CosyVoice3 checkpoint(s), convert with
    tools/convert_weights.py, and validate every tensor's name, GGML type,
    dimension list (ne = reversed PyTorch shape), byte size and a CRC-32
    checksum against the actual state_dict — proves the *weight mapping*
    (naming, dim order, dtype) is correct.

Prereqs: a built ggml checkout at third_party/ggml-verify (see the cross-check
section in docs/WEIGHT_FORMAT.md). Run: python3 tests/cross_check.py [ckpt.pt ...]
"""

import os
import subprocess
import sys
import tempfile
import zlib

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import convert_weights  # noqa: E402

GGML_DIR = os.path.join(ROOT, "third_party", "ggml-verify")

_TYPE_NAME = {
    torch.float32: "f32",
    torch.float16: "f16",
    torch.bfloat16: "bf16",
    torch.float64: "f64",
    torch.int8: "i8",
    torch.int16: "i16",
    torch.int32: "i32",
    torch.int64: "i64",
}


def crc32(data: bytes) -> str:
    return f"{zlib.crc32(data) & 0xFFFFFFFF:08x}"


def make_state_dict():
    g = torch.Generator().manual_seed(1)
    return {
        "model.embed_tokens.weight": torch.randn(64, 16, generator=g),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 8, generator=g),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(16, generator=g),
        "flow.estimator.blocks.0.conv.weight": torch.randn(4, 4, 3, 3, generator=g),
        "flow.estimator.blocks.0.bias": torch.randn(4, generator=g).to(torch.bfloat16),
        "hift.conv.1.weight": torch.randn(8, 4, 5, generator=g),
        "hift.f16.weight": torch.randn(3, 5, generator=g).to(torch.float16),
        "hift.f64.weight": torch.randn(2, 3, generator=g).to(torch.float64),
        "hift.i32.weight": torch.randint(-100, 100, (7,), generator=g).to(torch.int32),
    }


def build_binary():
    bin_path = os.path.join(ROOT, "build", "gguf_cross_check")
    src = os.path.join(ROOT, "tests", "gguf_cross_check.cpp")
    inc = os.path.join(GGML_DIR, "include")
    lib = os.path.join(GGML_DIR, "build", "src")
    if not os.path.isdir(inc) or not os.path.isfile(os.path.join(lib, "libggml-base.so")):
        sys.exit(f"ggml checkout/build not found under {GGML_DIR}; "
                 "build it first (see docs/WEIGHT_FORMAT.md)")
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    cmd = ["c++", "-std=c++17", f"-I{inc}", src, f"-L{lib}", "-lggml-base",
           f"-Wl,-rpath,{lib}", "-o", bin_path]
    subprocess.run(cmd, check=True)
    return bin_path


def validate_state_dict(state_dict, out_path, bin_path):
    """Compare a .gguf (via the C++/ggml loader) against the source state_dict."""
    tensors = {k: v for k, v in state_dict.items() if torch.is_tensor(v)}
    proc = subprocess.run([bin_path, out_path], capture_output=True, text=True, check=True)
    lines = proc.stdout.splitlines()
    header = lines[0]
    tensor_lines = [ln for ln in lines[1:] if ln.startswith("TENSOR ")]

    n_tensors = int(header.split("n_tensors=")[1].split()[0])
    got = {}
    for ln in tensor_lines:
        name, rest = ln[len("TENSOR "):].split(" ", 1)
        got[name] = dict(kv.split("=") for kv in rest.split(" "))

    assert n_tensors == len(tensors), f"n_tensors {n_tensors} != {len(tensors)}"
    assert list(got.keys()) == list(tensors.keys()), (
        f"name/order mismatch:\n  ggml : {list(got.keys())}\n  torch: {list(tensors.keys())}")
    for name, t in tensors.items():
        r = got[name]
        assert r["type"] == _TYPE_NAME[t.dtype], f"{name}: type {r['type']} != {_TYPE_NAME[t.dtype]}"
        ne = convert_weights.ne_from_shape(tuple(t.shape))
        ne4 = ne + [1] * (4 - len(ne))
        assert r["ne"] == ",".join(str(d) for d in ne4), f"{name}: ne {r['ne']} != {ne4}"
        data = convert_weights.tensor_to_bytes(t)
        assert int(r["size"]) == len(data), f"{name}: size {r['size']} != {len(data)}"
        assert r["checksum"] == crc32(data), f"{name}: data checksum mismatch"
    return tensors


def run_one(checkpoint_path, out_path, bin_path, arch):
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = convert_weights.extract_state_dict(obj)
    convert_weights.convert_one(checkpoint_path, arch, out_path,
                                use_f16=False, strip_prefix="")
    tensors = validate_state_dict(state_dict, out_path, bin_path)
    n_elems = sum(t.numel() for t in tensors.values())
    print(f"cross-check OK [{os.path.basename(checkpoint_path)}]: "
          f"ggml loaded {len(tensors)} tensors; name/type/ne/size/data all match")
    print(f"  ({n_elems} elements total)")
    return tensors


def arch_for(path):
    base = os.path.basename(path)
    if base.startswith("llm"):
        return "cosyvoice3.llm"
    if base.startswith("flow"):
        return "cosyvoice3.flow"
    if base.startswith("hift"):
        return "cosyvoice3.hift"
    raise ValueError(f"cannot infer component from filename {base}")


def main():
    bin_path = build_binary()

    if len(sys.argv) == 1:
        # Synthetic: container-format regression.
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = os.path.join(tmp, "synthetic.pt")
            torch.save(make_state_dict(), ckpt)
            out = os.path.join(tmp, "synthetic.gguf")
            convert_weights.convert_one(ckpt, "cosyvoice3.flow", out,
                                        use_f16=False, strip_prefix="")
            tensors = validate_state_dict(make_state_dict(), out, bin_path)
            print(f"cross-check OK [synthetic]: {len(tensors)} tensors match")
        return

    # Real checkpoint(s): weight-mapping validation.
    with tempfile.TemporaryDirectory() as tmp:
        for path in sys.argv[1:]:
            arch = arch_for(path)
            out = os.path.join(tmp, f"{arch.rsplit('.', 1)[-1]}.gguf")
            run_one(path, out, bin_path, arch)


if __name__ == "__main__":
    main()
