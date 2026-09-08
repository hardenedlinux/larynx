#!/usr/bin/env python3
"""Export the Flow decoder's deterministic CFM seed noise as a fixed binary asset.

``CausalConditionalCFM.__init__`` runs ``set_all_random_seed(0)`` then samples

    rand_noise = torch.randn([1, 80, 50 * 300])   # (1, 80, 15000)

once at construction time (see cosyvoice/flow/flow_matching.py). Inference uses
``z = rand_noise[:, :, :mel_t]`` — deterministic, but it is a ``torch.randn``
Gaussian whose bytes the C++ side cannot reproduce without reimplementing
PyTorch's RNG. Same principle as the HiFT SineGen2 buffers: freeze the values
into a fixed asset and load them verbatim (``FlowDecoder::load_noise``), never
regenerate them.

File format (little-endian):
    u32  magic      = 0x45534E46  ("FNSE")
    u32  version    = 1
    u32  mel_dim    = 80
    u64  max_frames = 50 * 300   (15,000)
    f32[mel_dim * max_frames]   rand_noise  (row-major [c][t]: c*max_frames + t)

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap`` (same as tests/hift_reference.py).

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/export_flow_noise.py [--out .../flow_noise.bin]
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

MEL_DIM = 80
MAX_FRAMES = 50 * 300       # 15,000
MAGIC = 0x45534E46          # "FNSE"
VERSION = 1


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=os.path.join(ROOT, "build", "flow_noise.bin"))
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch
    from hyperpyyaml import load_hyperpyyaml

    torch.set_num_threads(1)
    with open(os.path.join(MODEL_DIR, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "hift": None})
    flow = configs["flow"]

    rand_noise = flow.decoder.rand_noise.detach().cpu().numpy()   # (1, 80, 15000)
    assert rand_noise.shape == (1, MEL_DIM, MAX_FRAMES), rand_noise.shape

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", MAGIC))
        f.write(struct.pack("<I", VERSION))
        f.write(struct.pack("<I", MEL_DIM))
        f.write(struct.pack("<Q", MAX_FRAMES))
        f.write(rand_noise.astype(np.float32).tobytes())

    size = os.path.getsize(args.out)
    sha = hashlib.sha256()
    with open(args.out, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            sha.update(chunk)

    print(f"wrote {args.out}")
    print(f"  rand_noise : {rand_noise.shape}  (deterministic, seed 0)")
    print(f"  file size  : {size} bytes ({size / 1e6:.2f} MB)")
    print(f"  sha256     : {sha.hexdigest()}")


if __name__ == "__main__":
    main()
