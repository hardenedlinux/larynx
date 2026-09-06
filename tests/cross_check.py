#!/usr/bin/env python3
"""Drive tests/gguf_cross_check.cpp against the real GGML loader.

Because the real CosyVoice3 checkpoints are not present in the repo, this
generates a synthetic but representative state_dict (the same dtypes and shapes
the converter is designed for: f32/f16/bf16/f64/i32, 1-D..4-D), converts it with
tools/convert_weights.py, then loads the result with GGML's official
gguf_init_from_file() (via the throwaway third_party/ggml-verify build) and
compares every tensor's name, GGML type, dimension list, byte size and a
byte-level FNV-1a checksum against the source state_dict.

Prereqs: a built ggml checkout at third_party/ggml-verify (see the cross-check
README in docs/WEIGHT_FORMAT.md). Run: python3 tests/cross_check.py
"""

import os
import subprocess
import sys
import tempfile

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


def fnv1a64(data: bytes) -> int:
    h = 1469598103934665603
    for b in data:
        h ^= b
        h = (h * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def make_state_dict():
    g = torch.Generator().manual_seed(1)
    # dtypes and shapes representative of the three CosyVoice3 components.
    return {
        # llm: embedding + attention matmul + bias
        "model.embed_tokens.weight": torch.randn(64, 16, generator=g),
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 8, generator=g),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(16, generator=g),
        # flow: 4-D conv weight + a bf16 parameter
        "flow.estimator.blocks.0.conv.weight": torch.randn(4, 4, 3, 3, generator=g),
        "flow.estimator.blocks.0.bias": torch.randn(4, generator=g).to(torch.bfloat16),
        # hift: 3-D and misc dtypes
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
    if not os.path.isdir(inc) or not os.path.isfile(os.path.join(lib, "libggml.so")):
        sys.exit(f"ggml checkout/build not found under {GGML_DIR}; "
                 "build it first (see docs/WEIGHT_FORMAT.md)")
    os.makedirs(os.path.dirname(bin_path), exist_ok=True)
    cmd = ["c++", "-std=c++17", f"-I{inc}", src, f"-L{lib}", "-lggml-base",
           f"-Wl,-rpath,{lib}", "-o", bin_path]
    subprocess.run(cmd, check=True)
    return bin_path


def main():
    state_dict = make_state_dict()
    bin_path = build_binary()

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "synthetic.pt")
        torch.save(state_dict, ckpt)
        out = os.path.join(tmp, "flow.gguf")
        convert_weights.convert_one(ckpt, "cosyvoice3.flow", out,
                                    use_f16=False, strip_prefix="")

        proc = subprocess.run([bin_path, out], capture_output=True, text=True, check=True)
        lines = proc.stdout.splitlines()
        header = lines[0]
        tensor_lines = [ln for ln in lines[1:] if ln.startswith("TENSOR ")]

        n_tensors = int(header.split("n_tensors=")[1].split()[0])
        got = {}
        for ln in tensor_lines:
            # TENSOR <name> type=<t> ne=a,b,c,d size=<s> checksum=<hex>
            parts = ln[len("TENSOR "):].split(" ", 1)
            name = parts[0]
            rest = dict(kv.split("=") for kv in parts[1].split(" "))
            got[name] = rest

        # 1. tensor count
        assert n_tensors == len(state_dict), f"n_tensors {n_tensors} != {len(state_dict)}"
        # 2. names and order match the state_dict exactly
        assert list(got.keys()) == list(state_dict.keys()), (
            f"name/order mismatch:\n  ggml: {list(got.keys())}\n  torch: {list(state_dict.keys())}")
        # 3. per-tensor name/type/ne/size/data
        for name, t in state_dict.items():
            r = got[name]
            assert r["type"] == _TYPE_NAME[t.dtype], f"{name}: type {r['type']} != {_TYPE_NAME[t.dtype]}"
            ne = convert_weights.ne_from_shape(tuple(t.shape))
            ne4 = ne + [1] * (4 - len(ne))
            assert r["ne"] == ",".join(str(d) for d in ne4), f"{name}: ne {r['ne']} != {ne4}"
            data = convert_weights.tensor_to_bytes(t)
            assert int(r["size"]) == len(data), f"{name}: size {r['size']} != {len(data)}"
            assert r["checksum"] == f"{fnv1a64(data):016x}", f"{name}: data checksum mismatch"

        print(f"cross-check OK: ggml loaded {n_tensors} tensors; "
              f"name/type/ne/size/data all match the source state_dict")
        for name, t in state_dict.items():
            ne = convert_weights.ne_from_shape(tuple(t.shape))
            print(f"  {name:<42} {_TYPE_NAME[t.dtype]:>4} ne={ne}")


if __name__ == "__main__":
    main()
