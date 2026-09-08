#!/usr/bin/env python3
"""Export the HiFT SineGen2 fixed source buffers as a self-contained binary asset.

The CausalHiFTGenerator's ``m_source.l_sin_gen`` (SineGen2, ``causal=True``)
samples two buffers once at construction time, from the *unseeded* global RNG:

  - ``rand_ini``    (1, 9)                initial phase offset (col 0 zeroed)
  - ``sine_waves``  (1, 300*24000, 9)     unvoiced-noise waveform bank (full 300 s)

They are not in ``hift.pt`` (plain tensor attributes, never saved), and PyTorch
re-draws them on every process start. A standalone C++ CLI therefore cannot rely
on a live Python process to hand them over — so this script freezes ONE snapshot
into a fixed asset the C++ side loads verbatim (``HiftVocoder::load_source``).

The values themselves are arbitrary (no fixed seed in the reference — they are
random noise/phases), so "whatever gets exported" is the ground truth, exactly
like the sampling candidate-set argument: correctness is *consuming the same
bytes*, not reproducing PyTorch's RNG. We still seed here so the export is
deterministic and the file is reproducible.

File format (little-endian):
    u32  magic             = 0x43525348  ("HSRC")
    u32  version           = 1
    u32  harmonic_dim      = 9            (NB_HARMONICS + 1)
    u64  sine_max_samples  = 300 * 24000  (7,200,000)
    f32[harmonic_dim]                    rand_ini
    f32[sine_max_samples * harmonic_dim] sine_waves   (row-major [t, h])

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap`` (same as tests/hift_reference.py).

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/export_hift_source.py [--checkpoint .../hift.pt] [--out .../hift_source.bin]
"""

import argparse
import hashlib
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")

SINE_MAX_SAMPLES = 300 * 24000
HARMONIC_DIM = 9          # NB_HARMONICS + 1
MAGIC = 0x43525348        # "HSRC"
VERSION = 1


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def build_model():
    from hyperpyyaml import load_hyperpyyaml
    with open(os.path.join(MODEL_DIR, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    return configs["hift"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "hift.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "hift_source.bin"))
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(0)  # deterministic export (values are arbitrary, but reproducible)

    model = build_model()
    model.eval()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    print(f"loaded {len(sd)} tensors from {args.checkpoint}")

    rand_ini = model.m_source.l_sin_gen.rand_ini.detach().cpu().numpy()          # (1, 9)
    sine_waves = model.m_source.l_sin_gen.sine_waves.detach().cpu().numpy()      # (1, 300*24000, 9)
    assert rand_ini.shape == (1, HARMONIC_DIM), rand_ini.shape
    assert sine_waves.shape == (1, SINE_MAX_SAMPLES, HARMONIC_DIM), sine_waves.shape
    assert abs(float(rand_ini[0, 0])) == 0.0, "fundamental (col 0) phase must be zeroed"

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", HARMONIC_DIM))
        f.write(struct.pack("<Q", SINE_MAX_SAMPLES))
        f.write(rand_ini.astype(np.float32).tobytes())
        f.write(sine_waves.astype(np.float32).tobytes())

    size = os.path.getsize(args.out)
    sha = hashlib.sha256()
    with open(args.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)

    print(f"wrote {args.out}")
    print(f"  rand_ini   : {rand_ini.shape}  col0={float(rand_ini[0, 0]):.1f}")
    print(f"  sine_waves : {sine_waves.shape}  (full 300 s bank)")
    print(f"  file size  : {size} bytes ({size / 1e6:.1f} MB)")
    print(f"  sha256     : {sha.hexdigest()}")


if __name__ == "__main__":
    main()
