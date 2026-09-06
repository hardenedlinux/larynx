#!/usr/bin/env python3
"""Numerical verification of the C++ DSP frontend against Python references.

For each test signal this script:
  1. writes the float32 PCM to a temp file,
  2. runs the compiled ``larynx_dump`` utility (path from $LARYNX_DUMP),
  3. loads the C++ log-mel / fbank / mean-normalized fbank outputs,
  4. computes the Python references
       whisper.log_mel_spectrogram(samples, n_mels=128)
       torchaudio.compliance.kaldi.fbank(samples, num_mel_bins=80, dither=0,
                                         sample_frequency=16000)
     (plus the per-bin mean-subtracted variant used by the Campplus frontend),
  5. reports max and mean absolute error for each feature.

A second section shows the C++ fbank against a float64 "true" fbank, and the
float32 reference against the same float64 truth, to make plain that the
observed C++-vs-float32 gap is the reference's own float32 FFT rounding, not a
bug in the C++.

This is pure-CPU and requires only numpy/torch/torchaudio/openai-whisper.

Usage:
    python3 tests/verify_dsp.py            # uses build/larynx_dump by default
    LARYNX_DUMP=./build/larynx_dump python3 tests/verify_dsp.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import whisper

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_floats(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    assert data.size == shape[0] * shape[1], (
        f"{path}: got {data.size} floats, expected {shape[0] * shape[1]}")
    return data.reshape(shape)


def run_dump(dump_bin, pcm_path, outdir):
    subprocess.run([dump_bin, pcm_path, outdir], check=True)


def ref_log_mel(samples):
    # samples: float32 numpy, 1-D, 16 kHz
    spec = whisper.log_mel_spectrogram(samples, n_mels=128)
    return spec.numpy()  # (128, n_frames)


def ref_fbank(samples):
    wav = torch.from_numpy(samples.astype(np.float32)).unsqueeze(0)  # (1, n)
    feat = kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000)
    return feat.numpy()  # (n_frames, 80)


def ref_fbank_norm(samples):
    feat = torch.from_numpy(ref_fbank(samples))
    return (feat - feat.mean(dim=0, keepdim=True)).numpy()


def ref_fbank_f64(samples):
    # float64 "true" fbank: isolates the float32 rounding of the reference.
    wav = torch.from_numpy(samples.astype(np.float64)).unsqueeze(0)
    return kaldi.fbank(wav, num_mel_bins=80, dither=0, sample_frequency=16000).numpy()


def stats(name, cpp, ref):
    assert cpp.shape == ref.shape, f"{name}: shape {cpp.shape} vs {ref.shape}"
    diff = np.abs(cpp.astype(np.float64) - ref.astype(np.float64))
    return diff.max(), diff.mean()


def make_signals():
    rng = np.random.default_rng(12345)
    sigs = {
        "noise_3s": (rng.standard_normal(48000) * 0.1).astype(np.float32),
        "sine_mix_1s": (
            0.5 * np.sin(2 * np.pi * 440 * np.arange(16000) / 16000)
            + 0.3 * np.sin(2 * np.pi * 1000 * np.arange(16000) / 16000)
            + 0.2 * np.sin(2 * np.pi * 3000 * np.arange(16000) / 16000)
        ).astype(np.float32),
        "chirp_2s": (
            0.4 * np.sin(2 * np.pi * np.cumsum(np.linspace(50, 6000, 32000)) / 16000)
        ).astype(np.float32),
        "short_500": (rng.standard_normal(500) * 0.05).astype(np.float32),
        "dc_offset_1s": (
            0.4 * np.sin(2 * np.pi * 220 * np.arange(16000) / 16000) + 0.35
        ).astype(np.float32),
    }
    return sigs


def main():
    dump_bin = os.environ.get("LARYNX_DUMP", os.path.join(ROOT, "build", "larynx_dump"))
    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_dump not found at {dump_bin}; build it first (cmake --build build)")

    print(f"whisper {whisper.__version__} | torch {torch.__version__} | torchaudio {torchaudio.__version__}")
    print(f"dump binary: {dump_bin}\n")

    sigs = make_signals()
    results = {}  # name -> dict of arrays/stats

    with tempfile.TemporaryDirectory() as tmp:
        for name, sig in sigs.items():
            pcm = os.path.join(tmp, name + ".f32")
            sig.tofile(pcm)
            outdir = os.path.join(tmp, name + "_out")
            os.makedirs(outdir)
            run_dump(dump_bin, pcm, outdir)

            n = len(sig)
            n_lm = n // 160
            n_fb = 1 + (n - 400) // 160

            cpp_lm = load_raw_floats(os.path.join(outdir, "logmel.f32"), (128, n_lm))
            cpp_fb = load_raw_floats(os.path.join(outdir, "fbank.f32"), (n_fb, 80))
            cpp_fbn = load_raw_floats(os.path.join(outdir, "fbank_norm.f32"), (n_fb, 80))

            results[name] = {
                "n": n,
                "cpp_lm": cpp_lm, "cpp_fb": cpp_fb, "cpp_fbn": cpp_fbn,
                "lm": stats("log-mel", cpp_lm, ref_log_mel(sig)),
                "fb": stats("fbank", cpp_fb, ref_fbank(sig)),
                "fbn": stats("fbank_norm", cpp_fbn, ref_fbank_norm(sig)),
                "fb_f64": stats("fbank_vs_f64", cpp_fb, ref_fbank_f64(sig)),
            }

    # --- Deliverable table: C++ vs float32 Python references ----------------
    print(f"{'signal':<14} {'n':>7} | {'log-mel max':>12} {'log-mel mean':>13} | "
          f"{'fbank max':>10} {'fbank mean':>11} | {'fbank_norm max':>14} {'fbank_norm mean':>15}")
    print("-" * 120)
    for name, r in results.items():
        lm_max, lm_mean = r["lm"]
        fb_max, fb_mean = r["fb"]
        fbn_max, fbn_mean = r["fbn"]
        print(f"{name:<14} {r['n']:>7} | {lm_max:>12.3e} {lm_mean:>13.3e} | "
              f"{fb_max:>10.3e} {fb_mean:>11.3e} | {fbn_max:>14.3e} {fbn_mean:>15.3e}")

    # --- Diagnostic: what is the reference's own precision? ------------------
    print("\nDiagnostic — fbank, C++ and float32 reference vs float64 'true' fbank:")
    print(f"{'signal':<14} | {'C++ vs f64 max':>15} {'C++ vs f64 mean':>15} | "
          f"{'f32-ref vs f64 max':>19} {'f32-ref vs f64 mean':>19}")
    print("-" * 96)
    for name, sig in sigs.items():
        fb64 = ref_fbank_f64(sig)
        fb32 = ref_fbank(sig)
        cpp_vs = results[name]["fb_f64"]
        ref_gap = np.abs(fb32.astype(np.float64) - fb64)
        print(f"{name:<14} | {cpp_vs[0]:>15.3e} {cpp_vs[1]:>15.3e} | "
              f"{ref_gap.max():>19.3e} {ref_gap.mean():>19.3e}")
    print("\nIf 'C++ vs f64' is far smaller than 'f32-ref vs f64', the C++-vs-float32 gap")
    print("is the reference's own float32 FFT rounding, not a C++ bug.")


if __name__ == "__main__":
    main()
