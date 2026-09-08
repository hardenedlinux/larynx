#!/usr/bin/env python3
"""Numerical verification of the C++ CosyVoice3 LLM **full autoregressive decode**.

Loads ``llm_decode_seq_ref.npz`` (produced by tests/llm_decode_seq_reference.py),
feeds the stored ``lm_input`` + sampled ``tokens`` through the compiled
``larynx_llm_decode_seq_dump`` utility, and compares the per-step ``llm_decoder``
logits.

The C++ does NOT re-implement the sampling RNG: it injects the reference's token
trajectory verbatim and emits only the deterministic per-step logits. This is the
"captured-token-sequence injection" strategy — it proves the KV-cache
autoregressive mechanics (absolute rope positions, causal mask, cache append)
are bit-correct over the whole trajectory. Any KV-layout, position-id, or
cache-append bug shows up as an O(1) error at the first step that follows it.

Both sides run float32 (C++ forced to CPU via LARYNX_BACKEND=cpu).

Usage:
    python3 tests/verify_llm_decode_seq.py
    LARYNX_LLM_DECODE_SEQ_DUMP=./build/larynx_llm_decode_seq_dump python3 tests/verify_llm_decode_seq.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEECH_VOCAB = 6761


def load_raw(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    n = int(np.prod(shape))
    assert data.size == n, f"{path}: got {data.size} floats, expected {n} ({shape})"
    return data.reshape(shape)


def main():
    dump_bin = os.environ.get("LARYNX_LLM_DECODE_SEQ_DUMP",
                              os.path.join(ROOT, "build", "larynx_llm_decode_seq_dump"))
    gguf = os.environ.get("LLM_GGUF", os.path.join(ROOT, "build", "llm.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_decode_seq_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_llm_decode_seq_dump not found at {dump_bin}; build it first")
    if not os.path.exists(gguf):
        sys.exit(f"llm.gguf not found at {gguf}; run tools/convert_weights.py --llm llm.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/llm_decode_seq_reference.py first")

    ref = np.load(ref_path)
    L = ref["lm_input"].shape[1]
    tokens = ref["tokens"]
    ref_logits = ref["logits"]  # (N, 6761)
    N = tokens.shape[0]
    print(f"dump binary: {dump_bin}")
    print(f"llm.gguf   : {gguf}")
    print(f"reference  : {ref_path}  (L={L}, N={N}, stop_token={int(ref['stop_token'][0])})")
    print(f"tokens[:8] : {tokens[:8].tolist()}\n")

    assert ref_logits.shape == (N, SPEECH_VOCAB), ref_logits.shape

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        ref["lm_input"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "lm_input.f32"))
        tokens.astype(np.int32).tofile(os.path.join(indir, "tokens.i32"))

        subprocess.run([dump_bin, gguf, indir, outdir], check=True,
                       env={**os.environ, "LARYNX_BACKEND": "cpu"})

        cpp_logits = load_raw(os.path.join(outdir, "seq_logits.f32"), (N, SPEECH_VOCAB))

    # Per-step comparison.
    cpp = cpp_logits.astype(np.float64)
    rf = ref_logits.astype(np.float64)
    diff = np.abs(cpp - rf)
    per_step_max = diff.max(axis=1)
    per_step_rel = per_step_max / np.abs(rf).max(axis=1).clip(min=1e-12)
    per_step_argmax = (cpp.argmax(axis=1) == rf.argmax(axis=1))

    print(f"{'step':>5} {'max abs':>12} {'rel err':>12} {'argmax match'}")
    print("-" * 60)
    for i in range(N):
        print(f"{i:>5} {per_step_max[i]:>12.3e} {per_step_rel[i]:>12.3e} "
              f"{str(bool(per_step_argmax[i])):>13}")

    worst = int(per_step_rel.argmax())
    n_argmax_mismatch = int((~per_step_argmax).sum())
    n_warn = int((per_step_rel > 1e-4).sum())
    n_fail = int((per_step_rel > 1e-3).sum())
    print("\n" + "-" * 60)
    print(f"worst step: {worst} (rel err = {per_step_rel[worst]:.3e}, "
          f"max abs = {per_step_max[worst]:.3e})")
    print(f"argmax mismatches: {n_argmax_mismatch}/{N}")
    print(f"{n_warn} step(s) exceed 1e-4 relative; {n_fail} step(s) exceed 1e-3 relative")
    if n_fail or n_argmax_mismatch:
        print("FAIL: a decode step diverges beyond float32 accumulation noise.")
        sys.exit(1)
    print("PASS: full decode sequence within tolerance for a float32-vs-float32 comparison.")


if __name__ == "__main__":
    main()
