#!/usr/bin/env python3
"""Numerical verification of the C++ CosyVoice3 LLM **single-step decode**.

Loads ``llm_decode_ref.npz`` (produced by tests/llm_decode_reference.py), feeds
the stored ``lm_input`` + ``next_token`` through the compiled
``larynx_llm_decode_dump`` utility, and compares:

  cache_k_{0..23}   prefill past_key_values key   (ROPED;   (KV_HEADS, L, HEAD_DIM))
  cache_v_{0..23}   prefill past_key_values value (raw;     (KV_HEADS, L, HEAD_DIM))
  decode_logits     llm_decoder of the single-token decode    (6761)
  decode_final_norm model.norm of the decode step             (896)

The cache comparison is the key addition over Checkpoint 3: it proves the C++
stores the rope'd key / raw value in the same layout transformers caches, and
that the decode step's single new key uses the correct absolute position
(cache_len). Any KV-layout or RoPE-position bug shows up here as an O(1) error.

Both sides run float32 (C++ forced to CPU via LARYNX_BACKEND=cpu).

Usage:
    python3 tests/verify_llm_decode.py
    LARYNX_LLM_DECODE_DUMP=./build/larynx_llm_decode_dump python3 tests/verify_llm_decode.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HEAD_DIM = 64
KV_HEADS = 2


def load_raw(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    n = int(np.prod(shape))
    assert data.size == n, f"{path}: got {data.size} floats, expected {n} ({shape})"
    return data.reshape(shape)


def compare(name, cpp, ref):
    cpp = cpp.astype(np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = np.abs(cpp - ref)
    max_err = float(diff.max())
    mean_err = float(diff.mean())
    scale = float(np.abs(ref).max())
    rel_err = max_err / scale if scale > 0 else max_err
    idx = np.unravel_index(int(diff.argmax()), diff.shape)
    return max_err, mean_err, rel_err, idx, \
        float(cpp.reshape(-1)[diff.argmax()]), float(ref.reshape(-1)[diff.argmax()])


def main():
    dump_bin = os.environ.get("LARYNX_LLM_DECODE_DUMP",
                              os.path.join(ROOT, "build", "larynx_llm_decode_dump"))
    gguf = os.environ.get("LLM_GGUF", os.path.join(ROOT, "build", "llm.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_decode_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_llm_decode_dump not found at {dump_bin}; build it first")
    if not os.path.exists(gguf):
        sys.exit(f"llm.gguf not found at {gguf}; run tools/convert_weights.py --llm llm.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/llm_decode_reference.py first")

    ref = np.load(ref_path)
    L = ref["lm_input"].shape[1]
    print(f"dump binary: {dump_bin}")
    print(f"llm.gguf   : {gguf}")
    print(f"reference  : {ref_path}  (L={L}, next_token={int(ref['next_token'][0])})\n")

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        ref["lm_input"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "lm_input.f32"))
        np.asarray(ref["next_token"], dtype=np.int32).tofile(os.path.join(indir, "next_token.i32"))

        subprocess.run([dump_bin, gguf, indir, outdir], check=True,
                       env={**os.environ, "LARYNX_BACKEND": "cpu"})

        rows = []

        # Cache tensors: C++ dumps the GGML tensor memory order for a tensor whose
        # ne = [HEAD_DIM, KV_HEADS, L] (ne[0] = d is the FASTEST-varying dim, then kv,
        # then seq). In numpy C-order the fastest axis is the LAST one, so the flat
        # buffer reshapes to [L, KV_HEADS, HEAD_DIM] = [t, kv, d]; transpose to
        # [kv, t, d] to match the reference (KV_HEADS, L, HEAD_DIM).
        for i in range(24):
            for kind, prefix in (("k", "cache_k"), ("v", "cache_v")):
                ref_t = ref[f"{prefix}_{i}"]                       # (KV_HEADS, L, HEAD_DIM)
                cpp = load_raw(os.path.join(outdir, f"{prefix}.{i}.f32"),
                               (L, KV_HEADS, HEAD_DIM))            # [t, kv, d]
                cpp_t = cpp.transpose(1, 0, 2)                     # (KV_HEADS, L, HEAD_DIM)
                mx, mean, rel, idx, cv, rv = compare(f"{prefix}_{i}", cpp_t, ref_t)
                rows.append((f"{prefix}_{i}", mx, mean, rel, idx, cv, rv))

        # Decode outputs.
        for name, fname in (("decode_logits", "decode_logits.f32"),
                            ("decode_final_norm", "decode_final_norm.f32")):
            ref_t = ref[name]
            cpp = load_raw(os.path.join(outdir, fname), ref_t.shape)
            mx, mean, rel, idx, cv, rv = compare(name, cpp, ref_t)
            rows.append((name, mx, mean, rel, idx, cv, rv))

    print(f"{'stage':<16} {'max abs':>12} {'mean abs':>12} {'rel err':>12} | worst @ index")
    print("-" * 100)
    for name, mx, mean, rel, idx, cv, rv in rows:
        print(f"{name:<16} {mx:>12.3e} {mean:>12.3e} {rel:>12.3e} | {str(idx)}")

    n_warn = sum(1 for r in rows if r[3] > 1e-4)
    n_fail = sum(1 for r in rows if r[3] > 1e-3)
    worst = max(rows, key=lambda r: r[3])
    print("\n" + "-" * 100)
    print(f"worst stage: {worst[0]} (rel err = {worst[3]:.3e}, max abs = {worst[1]:.3e})")
    print(f"{n_warn} stage(s) exceed 1e-4 relative; {n_fail} stage(s) exceed 1e-3 relative")
    if n_fail:
        print("FAIL: a decode/cache stage diverges beyond float32 accumulation noise.")
        sys.exit(1)
    print("PASS: decode step + KV cache within tolerance for a float32-vs-float32 comparison.")


if __name__ == "__main__":
    main()
