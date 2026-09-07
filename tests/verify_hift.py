#!/usr/bin/env python3
"""Numerical verification of the C++ HiFT vocoder against the PyTorch reference.

Loads ``hift_ref.npz`` (produced by tests/hift_reference.py), feeds the stored
inputs (mel + the fixed SineGen2 buffers) through the compiled
``larynx_hift_dump`` utility, and compares every per-stage tensor the C++
vocoder exposes against the PyTorch ground truth:

  f0, sine_wavs, sine_merge, s_stft (source path)
  conv_pre, ups{0..2}[_lrelu], source_downs{0..2}, source_resblocks{0..2},
  fusion{0..2}, resblock{0..8}, post_resblocks{0..2}, reflection_pad,
  final_lrelu, conv_post, magnitude, phase, istft, speech (main network)

The f0 predictor runs in float64 on both sides (bit-faithful); the SineGen2 /
STFT / ISTFT run in float32 and are nearly bit-exact. Only the conv network runs
GGML's float32 kernels against PyTorch's own float32 kernels, so that part shows
~1e-4..1e-3 floating-point accumulation (see docs/DSP.md for the same analysis).

Usage:
    python3 tests/verify_hift.py            # uses build/larynx_hift_dump
    LARYNX_HIFT_DUMP=./build/larynx_hift_dump python3 tests/verify_hift.py
    HIFT_GGUF=./build/hift.gguf python3 tests/verify_hift.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Per-stage (name, reference npz key, C++ raw-file, numpy shape).
# C++ raw files are laid out identically to the PyTorch tensors (GGML's natural
# order == numpy row-major), so we reshape in place.  T = time, C = channels.
STAGES = [
    ("f0",               "f0",               "f0.f32",               (1, 30)),
    ("sine_wavs",        "sine_wavs",        "sine_wavs.f32",        (1, 14400, 9)),
    ("sine_merge",       "sine_merge",       "sine_merge.f32",       (1, 14400, 1)),
    ("s_stft",           "s_stft",           "s_stft.f32",           (1, 18, 3601)),
    ("conv_pre",         "conv_pre",         "conv_pre.f32",         (1, 512, 30)),
    ("ups0_lrelu",       "ups0_lrelu",       "ups0_lrelu.f32",       (1, 512, 30)),
    ("ups0",             "ups0",             "ups0.f32",             (1, 256, 240)),
    ("source_downs0",    "source_downs0",    "source_downs0.f32",    (1, 256, 240)),
    ("source_resblocks0","source_resblocks0","source_resblocks0.f32",(1, 256, 240)),
    ("fusion0",          "fusion0",          "fusion0.f32",          (1, 256, 240)),
    ("resblock0",        "resblock0",        "resblock0.f32",        (1, 256, 240)),
    ("resblock1",        "resblock1",        "resblock1.f32",        (1, 256, 240)),
    ("resblock2",        "resblock2",        "resblock2.f32",        (1, 256, 240)),
    ("post_resblocks0",  "post_resblocks0",  "post_resblocks0.f32",  (1, 256, 240)),
    ("ups1_lrelu",       "ups1_lrelu",       "ups1_lrelu.f32",       (1, 256, 240)),
    ("ups1",             "ups1",             "ups1.f32",             (1, 128, 1200)),
    ("source_downs1",    "source_downs1",    "source_downs1.f32",    (1, 128, 1200)),
    ("source_resblocks1","source_resblocks1","source_resblocks1.f32",(1, 128, 1200)),
    ("fusion1",          "fusion1",          "fusion1.f32",          (1, 128, 1200)),
    ("resblock3",        "resblock3",        "resblock3.f32",        (1, 128, 1200)),
    ("resblock4",        "resblock4",        "resblock4.f32",        (1, 128, 1200)),
    ("resblock5",        "resblock5",        "resblock5.f32",        (1, 128, 1200)),
    ("post_resblocks1",  "post_resblocks1",  "post_resblocks1.f32",  (1, 128, 1200)),
    ("ups2_lrelu",       "ups2_lrelu",       "ups2_lrelu.f32",       (1, 128, 1200)),
    ("ups2",             "ups2",             "ups2.f32",             (1, 64, 3600)),
    ("reflection_pad",   "reflection_pad",   "reflection_pad.f32",   (1, 64, 3601)),
    ("source_downs2",    "source_downs2",    "source_downs2.f32",    (1, 64, 3601)),
    ("source_resblocks2","source_resblocks2","source_resblocks2.f32",(1, 64, 3601)),
    ("fusion2",          "fusion2",          "fusion2.f32",          (1, 64, 3601)),
    ("resblock6",        "resblock6",        "resblock6.f32",        (1, 64, 3601)),
    ("resblock7",        "resblock7",        "resblock7.f32",        (1, 64, 3601)),
    ("resblock8",        "resblock8",        "resblock8.f32",        (1, 64, 3601)),
    ("post_resblocks2",  "post_resblocks2",  "post_resblocks2.f32",  (1, 64, 3601)),
    ("final_lrelu",      "final_lrelu",      "final_lrelu.f32",      (1, 64, 3601)),
    ("conv_post",        "conv_post",        "conv_post.f32",        (1, 18, 3601)),
    ("magnitude",        "magnitude",        "magnitude.f32",        (1, 9, 3601)),
    ("phase",            "phase",            "phase.f32",            (1, 9, 3601)),
    ("istft",            "istft",            "istft.f32",            (1, 14400)),
    ("speech",           "speech",           "speech.f32",           (1, 14400)),
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
    dump_bin = os.environ.get("LARYNX_HIFT_DUMP", os.path.join(ROOT, "build", "larynx_hift_dump"))
    gguf = os.environ.get("HIFT_GGUF", os.path.join(ROOT, "build", "hift.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hift_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_hift_dump not found at {dump_bin}; build it first (cmake --build build)")
    if not os.path.exists(gguf):
        sys.exit(f"hift.gguf not found at {gguf}; run tools/convert_weights.py --hift hift.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/hift_reference.py first")

    ref = np.load(ref_path)
    print(f"dump binary: {dump_bin}")
    print(f"hift.gguf  : {gguf}")
    print(f"reference  : {ref_path}\n")

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        # Write inputs (all float32; raw row-major == torch layout).
        ref["mel"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "mel.f32"))
        ref["rand_ini"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "rand_ini.f32"))
        ref["sine_waves"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "sine_waves.f32"))

        subprocess.run([dump_bin, gguf, indir, outdir], check=True)

        rows = []
        for name, npz_key, fname, shape in STAGES:
            cpp = load_raw(os.path.join(outdir, fname), shape)
            mx, mean, rel, idx, cpp_v, ref_v = compare(name, cpp, ref[npz_key])
            rows.append((name, mx, mean, rel, idx, cpp_v, ref_v))

    # --- Deliverable table ---------------------------------------------------
    print(f"{'stage':<18} {'max abs':>12} {'mean abs':>12} {'rel err':>12} | worst @ index  cpp value        ref value")
    print("-" * 110)
    for name, mx, mean, rel, idx, cpp_v, ref_v in rows:
        print(f"{name:<18} {mx:>12.3e} {mean:>12.3e} {rel:>12.3e} | {str(idx):<14} {cpp_v:+.6e} {ref_v:+.6e}")

    # --- Verdict --------------------------------------------------------------
    # f0/source/ISTFT are host float32/float64 and should be ~1e-7..1e-12; the
    # conv network is float32-vs-float32 (~1e-4..1e-3). Flag anything materially
    # larger than accumulation as a divergence (same threshold as verify_flow.py).
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
