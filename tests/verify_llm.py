#!/usr/bin/env python3
"""Numerical verification of the C++ CosyVoice3 LLM (Qwen2) prefill.

Loads ``llm_ref.npz`` (produced by tests/llm_reference.py), feeds the stored
``lm_input`` through the compiled ``larynx_llm_dump`` utility, and compares every
per-stage tensor the C++ backbone exposes against the PyTorch ground truth:

  h{0..23}       per-transformer-layer post-residual hidden states (L*896 each)
  final_norm     model.norm applied to layer 23                       (L*896)
  logits         llm_decoder output (pre-softmax)                     (L*6761)

Both sides run float32. Unlike the Flow/HiFT networks (which are single forward
passes), the LLM stacks 24 layers of matmul + RoPE + GQA attention, so each
layer's ~1e-7 relative error compounds to ~1e-4..1e-3 at the output; the final
logits (6761-way) accumulate a little more. Tolerances are graded accordingly.

Usage:
    python3 tests/verify_llm.py            # uses build/larynx_llm_dump
    LARYNX_LLM_DUMP=./build/larynx_llm_dump python3 tests/verify_llm.py
    LLM_GGUF=./build/llm.gguf python3 tests/verify_llm.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (name, reference npz key, C++ raw-file, numpy shape, per-element rel threshold).
# All raw files are laid out identically to PyTorch row-major (GGML natural order).
STAGES = []
for i in range(24):
    STAGES.append((f"h{i}", f"h{i}", f"hidden_states.{i}.f32", None))
STAGES += [
    ("final_norm", "final_norm", "final_norm.f32", None),
    ("logits",     "logits",     "logits.f32",     None),
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
    return max_err, mean_err, rel_err, idx, \
        float(cpp.reshape(-1)[diff.argmax()]), float(ref.reshape(-1)[diff.argmax()])


def main():
    dump_bin = os.environ.get("LARYNX_LLM_DUMP", os.path.join(ROOT, "build", "larynx_llm_dump"))
    gguf = os.environ.get("LLM_GGUF", os.path.join(ROOT, "build", "llm.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_llm_dump not found at {dump_bin}; build it first (cmake --build build)")
    if not os.path.exists(gguf):
        sys.exit(f"llm.gguf not found at {gguf}; run tools/convert_weights.py --llm llm.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/llm_reference.py first")

    ref = np.load(ref_path)
    L = ref["lm_input"].shape[1]
    print(f"dump binary: {dump_bin}")
    print(f"llm.gguf   : {gguf}")
    print(f"reference  : {ref_path}  (L={L})\n")

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        ref["lm_input"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "lm_input.f32"))

        # Default: force CPU (deterministic ctest). LARYNX_VERIFY_BACKEND=cuda
        # unsets LARYNX_BACKEND so the dump picks ggml_backend_init_best (CUDA).
        env = dict(os.environ)
        if os.environ.get("LARYNX_VERIFY_BACKEND") == "cuda":
            env.pop("LARYNX_BACKEND", None)
        else:
            env["LARYNX_BACKEND"] = "cpu"
        subprocess.run([dump_bin, gguf, indir, outdir], check=True, env=env)

        rows = []
        for name, npz_key, fname, _ in STAGES:
            ref_t = ref[npz_key]
            cpp = load_raw(os.path.join(outdir, fname), ref_t.shape)
            mx, mean, rel, idx, cpp_v, ref_v = compare(name, cpp, ref_t)
            rows.append((name, mx, mean, rel, idx, cpp_v, ref_v))

    # --- Deliverable table ---------------------------------------------------
    print(f"{'stage':<12} {'max abs':>12} {'mean abs':>12} {'rel err':>12} | worst @ index        cpp value       ref value")
    print("-" * 110)
    for name, mx, mean, rel, idx, cpp_v, ref_v in rows:
        print(f"{name:<12} {mx:>12.3e} {mean:>12.3e} {rel:>12.3e} | {str(idx):<20} {cpp_v:+.6e} {ref_v:+.6e}")

    # --- Verdict --------------------------------------------------------------
    # Per-layer / final_norm: ~1e-4..1e-3. Logits: up to a few 1e-3. Flag anything
    # materially larger (thresholds match verify_flow.py / verify_hift.py).
    hidden = [r for r in rows if r[0].startswith("h")]
    worst_h = max(hidden, key=lambda r: r[3])
    print("\n" + "-" * 110)
    print(f"worst hidden-state stage (rel): {worst_h[0]} (rel err = {worst_h[3]:.3e}, "
          f"max abs = {worst_h[1]:.3e})")
    n_warn = sum(1 for r in rows if r[3] > 1e-3)
    n_fail = sum(1 for r in rows if r[3] > 1e-2)
    print(f"{n_warn} stage(s) exceed 1e-3 relative; {n_fail} stage(s) exceed 1e-2 relative")
    if n_fail:
        print("FAIL: at least one stage diverges beyond float32 accumulation noise.")
        sys.exit(1)
    print("PASS: all stages within tolerance for a float32-vs-float32 comparison.")


if __name__ == "__main__":
    main()
