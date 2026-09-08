#!/usr/bin/env python3
"""Verify the HiFT SineGen2 fixed-source loading path (``load_source``).

The standalone C++ CLI must not borrow a live Python process for the two
SineGen2 buffers (``rand_ini`` + ``sine_waves``) — those are frozen once into
``build/hift_source.bin`` by tests/export_hift_source.py and loaded verbatim by
``HiftVocoder::load_source``. This script proves that loading path:

  1. **round-trip** — the buffers ``load_source`` reads back are byte-identical
     to the asset file (parse the header, then compare both buffers bit-exact);
  2. **self-consistency** — the production 3-arg ``vocode(mel)`` (which slices
     the full bank internally) is bit-identical to the 5-arg ``vocode`` fed the
     same slice explicitly (no off-by-one in the slicing/delegation).

The vocode *algorithm* itself (bit-exact vs PyTorch) is already proven by
tests/verify_hift.py; the two checks here complete the production path.

Usage:
    python3 tests/verify_hift_source.py
    LARYNX_HIFT_SOURCE_DUMP=./build/larynx_hift_source_dump python3 tests/verify_hift_source.py
"""

import os
import struct
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SINE_MAX_SAMPLES = 300 * 24000
HARMONIC_DIM = 9
MAGIC = 0x43525348
VERSION = 1
T_MEL = 30


def load_raw(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    n = int(np.prod(shape))
    assert data.size == n, f"{path}: got {data.size} floats, expected {n} ({shape})"
    return data.reshape(shape)


def read_source_bin(path):
    """Parse hift_source.bin -> (rand_ini (9,), sine_waves (SINE_MAX*9,))."""
    with open(path, "rb") as f:
        magic, version, harmonic_dim = struct.unpack("<III", f.read(12))
        (sine_max,) = struct.unpack("<Q", f.read(8))
        rand_ini = np.frombuffer(f.read(harmonic_dim * 4), dtype=np.float32)
        sine_waves = np.frombuffer(f.read(sine_max * harmonic_dim * 4), dtype=np.float32)
    assert magic == MAGIC, hex(magic)
    assert version == VERSION
    assert harmonic_dim == HARMONIC_DIM
    assert sine_max == SINE_MAX_SAMPLES
    return rand_ini, sine_waves


def main():
    dump_bin = os.environ.get("LARYNX_HIFT_SOURCE_DUMP",
                              os.path.join(ROOT, "build", "larynx_hift_source_dump"))
    gguf = os.environ.get("HIFT_GGUF", os.path.join(ROOT, "build", "hift.gguf"))
    source_bin = os.environ.get("HIFT_SOURCE_BIN",
                                os.path.join(ROOT, "build", "hift_source.bin"))

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_hift_source_dump not found at {dump_bin}; build it first")
    if not os.path.exists(gguf):
        sys.exit(f"hift.gguf not found at {gguf}; run tools/convert_weights.py --hift hift.pt --out-dir build/")
    if not os.path.exists(source_bin):
        sys.exit(f"hift_source.bin not found at {source_bin}; run tests/export_hift_source.py first")

    print(f"dump binary : {dump_bin}")
    print(f"hift.gguf   : {gguf}")
    print(f"source asset: {source_bin}  ({os.path.getsize(source_bin)/1e6:.1f} MB)\n")

    exp_rand_ini, exp_sine_waves = read_source_bin(source_bin)
    assert exp_rand_ini[0] == 0.0, "fundamental (col 0) phase must be zeroed"

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        # Any valid mel works (the checks are mel-agnostic); 80*T_MEL float32.
        np.random.RandomState(0).randn(80 * T_MEL).astype(np.float32) \
            .tofile(os.path.join(indir, "mel.f32"))

        env = dict(os.environ)
        env["LARYNX_BACKEND"] = "cpu"
        subprocess.run([dump_bin, gguf, source_bin, indir, outdir], check=True, env=env)

        got_rand_ini = load_raw(os.path.join(outdir, "rand_ini.f32"), (HARMONIC_DIM,))
        got_sine_waves = load_raw(os.path.join(outdir, "sine_waves.f32"),
                                  (SINE_MAX_SAMPLES * HARMONIC_DIM,))
        audio = load_raw(os.path.join(outdir, "audio.f32"), (T_MEL * 480,))
        audio_explicit = load_raw(os.path.join(outdir, "audio_explicit.f32"), (T_MEL * 480,))

    print(f"{'check':<22} {'max abs err':>14}")
    print("-" * 40)

    ok = True

    d = np.abs(got_rand_ini - exp_rand_ini).max()
    print(f"{'rand_ini (round-trip)':<22} {d:>14.6e}")
    ok &= d == 0.0

    d = np.abs(got_sine_waves - exp_sine_waves).max()
    print(f"{'sine_waves (round-trip)':<22} {d:>14.6e}")
    ok &= d == 0.0

    d = np.abs(audio - audio_explicit).max()
    print(f"{'audio vs explicit':<22} {d:>14.6e}")
    ok &= d == 0.0

    print("-" * 40)
    # Objective output statistics (no quality judgment): report, don't judge.
    print(f"audio: samples={audio.size}  max|a|={np.abs(audio).max():.6f}  "
          f"rms={np.sqrt(np.mean(audio.astype(np.float64) ** 2)):.6f}")

    if not ok:
        print("FAIL: load_source round-trip / production-vs-explicit is not bit-exact.")
        sys.exit(1)
    print("PASS: load_source round-trip bit-exact; production vocode == explicit vocode.")


if __name__ == "__main__":
    main()
