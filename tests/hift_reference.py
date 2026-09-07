#!/usr/bin/env python3
"""PyTorch numerical reference for the HiFT vocoder (``CausalHiFTGenerator``).

This is the *ground truth* for ``tests/verify_hift.py``. Unlike the Flow
reference (which had to inline the modules because the cosyvoice package would
not import on this box), the HiFT reference **imports the real package**: the
model is built by ``hyperpyyaml`` from ``cosyvoice3.yaml`` (``llm``/``flow``
overridden to ``None``), so every weight-norm parametrization, causal conv and
the ISTFT path are the actual runtime objects, not a reimplementation. The
orchestration (``inference``/``decode``) is then replayed step-by-step with
capture points, and the final waveform is cross-checked against a direct
``model.inference()`` call to prove the replay is faithful.

Scope (per the task): only ``speech_feat (mel) -> PCM``, non-streaming,
``finalize=True``. The streaming / incremental branch is not exercised.

The mel *input* is a fixed-seed ``torch.randn`` (it is an input, not a
model-internal random buffer). The three model-internal fixed buffers that the
vocoder samples at construction time are **exported verbatim** (never
regenerated) so the C++ side consumes the exact same values:

  - ``l_sin_gen.rand_ini``     (1, 9)         initial phase offset (col 0 zeroed)
  - ``l_sin_gen.sine_waves``   (1, L_s, 9)    unvoiced-noise waveform bank
  - ``m_source.uv``            (1, L_s, 1)    noise-branch gain bank (see note)

The third buffer is exported for completeness but is provably *not* on the
``finalize`` inference path: ``m_source.forward`` returns it as the discarded
``noise`` output (``inference`` does ``s, _, _ = self.m_source(s)``).

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/hift_reference.py [--checkpoint .../hift.pt] [--out .../hift_ref.npz]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")

# Validation-case constants (must match tests/verify_hift.py / hift_dump.cpp).
T_MEL = 30        # mel frames in
INPUT_SEED = 1234  # seed for the *mel input* only (buffers are fixed, no seed)

# cosyvoice3.yaml CausalHiFTGenerator parameters.
IN_CHANNELS = 80
BASE_CHANNELS = 512
NB_HARMONICS = 8
SAMPLING_RATE = 24000
NSF_ALPHA = 0.1
NSF_SIGMA = 0.003
NSF_VOICED_THRESHOLD = 10
UPSAMPLE_RATES = [8, 5, 3]
UPSAMPLE_KERNEL_SIZES = [16, 11, 7]
ISTFT_N_FFT = 16
ISTFT_HOP = 4
RESBLOCK_KERNEL_SIZES = [3, 7, 11]
RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
SOURCE_RESBLOCK_KERNEL_SIZES = [7, 7, 11]
SOURCE_RESBLOCK_DILATION_SIZES = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]
LRELU_SLOPE = 0.1
AUDIO_LIMIT = 0.99
CONV_PRE_LOOK_RIGHT = 4

# Derived.
TOTAL_SCALE = 8 * 5 * 3 * ISTFT_HOP                        # 480 (prod of upsample_rates)
N_BINS = ISTFT_N_FFT // 2 + 1                             # 9
OUT_CH = ISTFT_N_FFT + 2                                  # 18
L_S = TOTAL_SCALE * T_MEL                                 # excitation length (samples)


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def build_model():
    from hyperpyyaml import load_hyperpyyaml
    with open(os.path.join(MODEL_DIR, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={"llm": None, "flow": None})
    model = configs["hift"]
    return model


def run_inference(model, mel, cap):
    """Replay ``CausalHiFTGenerator.inference(finalize=True)`` with captures."""
    import torch

    # ---- mel -> f0 (float64, exactly as inference() does) ----
    model.f0_predictor.to(torch.float64)
    f0_f64 = model.f0_predictor(mel.to(torch.float64), finalize=True)
    cap["f0_f64"] = f0_f64
    f0 = f0_f64.to(mel.dtype)
    cap["f0"] = f0

    # ---- f0 -> source excitation ----
    s = model.f0_upsamp(f0[:, None]).transpose(1, 2)   # (1, L_s, 1)
    cap["f0_upsamp"] = s
    with torch.no_grad():
        sine_wavs, uv_out, _ = model.m_source.l_sin_gen(s)
    cap["sine_wavs"] = sine_wavs                       # (1, L_s, 9)
    cap["uv_out"] = uv_out                             # (1, L_s, 1)
    sine_merge = model.m_source.l_tanh(model.m_source.l_linear(sine_wavs))
    cap["sine_merge"] = sine_merge                     # (1, L_s, 1)
    s = sine_merge
    cap["s"] = s
    s = s.transpose(1, 2)                              # (1, 1, L_s)

    speech = run_decode(model, mel, s, cap, finalize=True)
    return speech


def run_decode(model, x, s, cap, finalize=True):
    """Replay ``CausalHiFTGenerator.decode(finalize=True)`` with captures."""
    import torch

    s_stft_real, s_stft_imag = model._stft(s.squeeze(1))
    cap["s_stft_real"] = s_stft_real                   # (1, 9, TT)
    cap["s_stft_imag"] = s_stft_imag                   # (1, 9, TT)
    s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
    cap["s_stft"] = s_stft                             # (1, 18, TT)

    x = model.conv_pre(x)
    cap["conv_pre"] = x                                # (1, 512, T_mel)

    for i in range(model.num_upsamples):
        x = torch.nn.functional.leaky_relu(x, model.lrelu_slope)
        cap[f"ups{i}_lrelu"] = x
        x = model.ups[i](x)
        cap[f"ups{i}"] = x
        if i == model.num_upsamples - 1:
            x = model.reflection_pad(x)
            cap["reflection_pad"] = x

        si = model.source_downs[i](s_stft)
        cap[f"source_downs{i}"] = si
        si = model.source_resblocks[i](si)
        cap[f"source_resblocks{i}"] = si
        x = x + si
        cap[f"fusion{i}"] = x

        xs = None
        for j in range(model.num_kernels):
            r = model.resblocks[i * model.num_kernels + j](x)
            cap[f"resblock{i * model.num_kernels + j}"] = r
            xs = r if xs is None else xs + r
        x = xs / model.num_kernels
        cap[f"post_resblocks{i}"] = x

    x = torch.nn.functional.leaky_relu(x)
    cap["final_lrelu"] = x
    x = model.conv_post(x)
    cap["conv_post"] = x                                # (1, 18, TT)
    magnitude = torch.exp(x[:, :N_BINS, :])
    cap["magnitude"] = magnitude
    phase = torch.sin(x[:, N_BINS:, :])
    cap["phase"] = phase

    x = model._istft(magnitude, phase)
    cap["istft"] = x
    x = torch.clamp(x, -model.audio_limit, model.audio_limit)
    cap["speech"] = x
    return x


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "hift.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "hift_ref.npz"))
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch

    torch.set_num_threads(1)
    torch.manual_seed(INPUT_SEED)

    model = build_model()
    model.eval()

    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)
    print(f"loaded {len(sd)} tensors from {args.checkpoint}")

    mel = torch.randn(1, IN_CHANNELS, T_MEL)
    assert mel.shape[2] == T_MEL

    cap = {}
    speech = run_inference(model, mel, cap)

    # Cross-check: the real inference() must agree with the manual replay.
    # NB: inference() returns the source as (1, 1, L_s) (transposed) — undo that
    # to compare against our (1, L_s, 1) sine_merge.
    speech_model, s_model = model.inference(mel, finalize=True)
    d = (speech - speech_model).abs().max().item()
    ds = (cap["sine_merge"] - s_model.transpose(1, 2)).abs().max().item()
    print(f"manual vs model.inference() max abs diff: speech={d:.3e} source={ds:.3e}")
    assert d < 1e-5 and ds < 1e-5, "manual replay diverged from model.inference()!"

    # Export the fixed buffers the C++ side must consume verbatim.
    rand_ini = model.m_source.l_sin_gen.rand_ini.detach().cpu()          # (1, 9)
    sine_waves = model.m_source.l_sin_gen.sine_waves[:, :L_S, :].detach().cpu()  # (1, L_s, 9)
    uv_buf = model.m_source.uv[:, :L_S].detach().cpu()                   # (1, L_s, 1)  (unused)
    assert sine_waves.shape == (1, L_S, NB_HARMONICS + 1)

    out = {
        # inputs + fixed buffers (C++ reads these)
        "mel": mel.detach().cpu().numpy(),                 # (1, 80, 30)
        "rand_ini": rand_ini.numpy(),                      # (1, 9)
        "sine_waves": sine_waves.numpy(),                  # (1, L_s, 9)
        "uv_buf": uv_buf.numpy(),                          # (1, L_s, 1)
        # f0 / source
        "f0": cap["f0"].detach().cpu().numpy(),            # (1, 30)
        "f0_f64": cap["f0_f64"].detach().cpu().numpy(),    # (1, 30) float64
        "f0_upsamp": cap["f0_upsamp"].detach().cpu().numpy(),
        "sine_wavs": cap["sine_wavs"].detach().cpu().numpy(),
        "uv_out": cap["uv_out"].detach().cpu().numpy(),
        "sine_merge": cap["sine_merge"].detach().cpu().numpy(),
        "s": cap["s"].detach().cpu().numpy(),              # (1, L_s, 1)
        # STFT of source
        "s_stft_real": cap["s_stft_real"].detach().cpu().numpy(),
        "s_stft_imag": cap["s_stft_imag"].detach().cpu().numpy(),
        "s_stft": cap["s_stft"].detach().cpu().numpy(),
        # main network
        "conv_pre": cap["conv_pre"].detach().cpu().numpy(),
        "reflection_pad": cap["reflection_pad"].detach().cpu().numpy(),
        "final_lrelu": cap["final_lrelu"].detach().cpu().numpy(),
        "conv_post": cap["conv_post"].detach().cpu().numpy(),
        "magnitude": cap["magnitude"].detach().cpu().numpy(),
        "phase": cap["phase"].detach().cpu().numpy(),
        "istft": cap["istft"].detach().cpu().numpy(),
        "speech": speech.detach().cpu().numpy(),
    }

    # Per-upsample-step + per-resblock stages.
    for i in range(len(UPSAMPLE_RATES)):
        for tag in ("ups", "source_downs", "source_resblocks", "fusion", "post_resblocks"):
            key = f"{tag}{i}"
            out[key] = cap[key].detach().cpu().numpy()
        out[f"ups{i}_lrelu"] = cap[f"ups{i}_lrelu"].detach().cpu().numpy()
    for j in range(len(UPSAMPLE_RATES) * len(RESBLOCK_KERNEL_SIZES)):
        out[f"resblock{j}"] = cap[f"resblock{j}"].detach().cpu().numpy()

    np.savez(args.out, **out)
    print(f"wrote {args.out}:")
    for k, v in out.items():
        print(f"  {k:16s} {str(v.dtype):8s} {str(v.shape)}")


if __name__ == "__main__":
    main()
