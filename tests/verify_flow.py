#!/usr/bin/env python3
"""Numerical verification of the C++ Flow decoder against the PyTorch reference.

Loads ``flow_ref.npz`` (produced by tests/flow_reference.py), feeds the stored
inputs through the compiled ``larynx_flow_dump`` utility, and compares every
per-stage tensor the C++ decoder exposes against the PyTorch ground truth:

  spk, token_embed, prelookahead, mu, cond (pre-DiT stages)
  time_embed, input_embed, 22 x block, norm_out, dphi  (DiT at CFM step 1)
  feat (final mel)

For each stage it reports the max and mean absolute error, plus the exact
position and values of the worst offender. Both sides are float32, but the
reference runs PyTorch's optimized BLAS / SDPA kernels while the C++ runs GGML's
own float32 kernels, so the gap is floating-point accumulation, not a bug — the
expected magnitude is ~1e-4..1e-3 (see docs/DSP.md for the same analysis on the
DSP frontend).

Usage:
    python3 tests/verify_flow.py            # uses build/larynx_flow_dump
    LARYNX_FLOW_DUMP=./build/larynx_flow_dump python3 tests/verify_flow.py
    FLOW_GGUF=./build/flow.gguf python3 tests/verify_flow.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-stage (name, reference npz key, C++ raw-file, numpy shape).
# The C++ raw files are laid out identically to the PyTorch tensors (GGML's
# natural order == numpy row-major for these shapes), so we reshape in place.
STAGES = [
    ("spk",          "spk",                   "spk.f32",          (1, 80)),
    ("token_embed",  "token_embed",           "token_embed.f32",  (1, 12, 80)),
    ("prelookahead", "prelookahead",          "prelookahead.f32", (1, 12, 80)),
    ("mu",           "mu",                    "mu.f32",           (1, 80, 24)),
    ("cond",         "cond",                  "cond.f32",         (1, 80, 24)),
    ("time_embed",   "time_embed_step1",      "time_embed.f32",   (2, 1024)),
    ("input_proj",   "dit_input_proj_step1",  "input_proj.f32",   (2, 24, 1024)),
    ("conv_pos",     "dit_conv_pos_step1",    "conv_pos.f32",     (2, 24, 1024)),
    ("input_embed",  "dit_input_embed_step1", "input_embed.f32",  (2, 24, 1024)),
    ("norm_out",     "dit_norm_out_step1",    "norm_out.f32",     (2, 24, 1024)),
    ("dphi",         "dphi_step1",            "dphi.f32",         (2, 80, 24)),
    ("feat",         "feat",                  "feat.f32",         (1, 80, 16)),
]


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
    return max_err, mean_err, rel_err, idx, float(cpp.reshape(-1)[diff.argmax()]), float(ref.reshape(-1)[diff.argmax()])


def main():
    dump_bin = os.environ.get("LARYNX_FLOW_DUMP", os.path.join(ROOT, "build", "larynx_flow_dump"))
    gguf = os.environ.get("FLOW_GGUF", os.path.join(ROOT, "build", "flow.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "flow_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_flow_dump not found at {dump_bin}; build it first (cmake --build build)")
    if not os.path.exists(gguf):
        sys.exit(f"flow.gguf not found at {gguf}; run tools/convert_weights.py --flow flow.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/flow_reference.py first")

    ref = np.load(ref_path)
    print(f"dump binary: {dump_bin}")
    print(f"flow.gguf  : {gguf}")
    print(f"reference  : {ref_path}\n")

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        # Write inputs (tokens as int32; everything else as float32).
        ref["prompt_tokens"].astype(np.int32).reshape(-1).tofile(os.path.join(indir, "prompt_tokens.i32"))
        ref["tokens"].astype(np.int32).reshape(-1).tofile(os.path.join(indir, "tokens.i32"))
        ref["prompt_feat"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "prompt_feat.f32"))
        ref["spk_embedding"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "spk_embedding.f32"))
        ref["noise_z"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "noise_z.f32"))

        subprocess.run([dump_bin, gguf, indir, outdir], check=True)

        rows = []
        for name, npz_key, fname, shape in STAGES:
            cpp = load_raw(os.path.join(outdir, fname), shape)
            mx, mean, rel, idx, cpp_v, ref_v = compare(name, cpp, ref[npz_key])
            rows.append((name, mx, mean, rel, idx, cpp_v, ref_v))

        # Per-block rows (22 x (2, 24, 1024)).
        for i in range(22):
            name = f"block[{i:02d}]"
            shape = (2, 24, 1024)
            cpp = load_raw(os.path.join(outdir, f"block{i}.f32"), shape)
            mx, mean, rel, idx, cpp_v, ref_v = compare(name, cpp, ref[f"dit_block{i}"])
            rows.append((name, mx, mean, rel, idx, cpp_v, ref_v))

    # --- Deliverable table ---------------------------------------------------
    print(f"{'stage':<14} {'max abs':>12} {'mean abs':>12} {'rel err':>12} | worst @ index  cpp value        ref value")
    print("-" * 110)
    for name, mx, mean, rel, idx, cpp_v, ref_v in rows:
        print(f"{name:<14} {mx:>12.3e} {mean:>12.3e} {rel:>12.3e} | {str(idx):<14} {cpp_v:+.6e} {ref_v:+.6e}")

    # --- Verdict --------------------------------------------------------------
    # Exact-op float32: expect ~1e-4..1e-3. Intermediate DiT residuals are ~100x
    # the network I/O scale, so their *absolute* error inflates even when the
    # output is right; the scale-normalized error (max abs / max|ref|) is what
    # cleanly separates float32 accumulation (~1e-4..1e-3) from a real layout/op
    # bug (O(1)). Flag anything materially larger than accumulation as a divergence.
    worst = max(rows, key=lambda r: r[3])
    print("\n" + "-" * 110)
    print(f"worst stage (rel): {worst[0]} (rel err = {worst[3]:.3e}, max abs = {worst[1]:.3e})")
    n_warn = sum(1 for r in rows if r[3] > 1e-3)
    n_fail = sum(1 for r in rows if r[3] > 1e-2)
    print(f"{n_warn} stage(s) exceed 1e-3 relative; {n_fail} stage(s) exceed 1e-2 relative")
    if n_fail:
        print("FAIL: at least one stage diverges beyond float32 accumulation noise.")
        sys.exit(1)
    print("PASS: all stages within tolerance for a float32-vs-float32 comparison.")


if __name__ == "__main__":
    main()
