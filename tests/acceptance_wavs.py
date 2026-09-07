#!/usr/bin/env python3
"""Phase-2 acceptance gate: real-speech-token wav pair.

Produces the two audio deliverables that the numerical validation alone cannot
cover:

  - ``wav_reference.wav`` — the *full* official CosyVoice3 Python pipeline
    (frontend → LLM → PyTorch flow → HiFT) on one real text/prompt pair.
  - ``wav_ggml.wav`` — the GGML Flow decoder run on the **identical** flow
    inputs the reference flow consumed (same speech tokens, same prompt mel,
    same 192-dim speaker embedding, same deterministic CFM noise), vocoded by
    the **same** official Python HiFT.

Because the HiFT vocoder and the flow inputs are bit-identical on both sides,
any audible difference is attributable solely to the GGML Flow decoder vs. the
PyTorch reference flow.

It works by wrapping ``model.flow.inference`` once: the wrapper records the
tensors the reference flow is handed (and the mel it returns), then delegates to
the original. After the official run completes, those captured inputs are
written to a temp dir in the exact layout ``larynx_flow_dump`` expects, the C++
decoder is invoked, and its ``feat.f32`` mel is fed through ``model.hift``.

Usage (from anywhere):
    PYTHONPATH=... python3 tests/acceptance_wavs.py \
        --model-dir ~/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
        --flow-gguf build/flow.gguf --flow-dump build/larynx_flow_dump \
        --out-dir wavs

All CosyVoice paths default to the repo layout under ``~/Project/CosyVoice``.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")


def _bootstrap(args):
    """Put the CosyVoice package + deps on the path before importing it."""
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def main():
    ap = argparse.ArgumentParser(description="Generate wav_ggml.wav and wav_reference.wav from one real token stream.")
    ap.add_argument("--model-dir", default=os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B"))
    ap.add_argument("--flow-gguf", default=os.path.join(ROOT, "build", "flow.gguf"))
    ap.add_argument("--flow-dump", default=os.path.join(ROOT, "build", "larynx_flow_dump"))
    ap.add_argument("--prompt-wav", default=os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav"))
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "wavs"))
    ap.add_argument("--text", default="今天天气不错，我们一起去公园散步吧。")
    ap.add_argument("--instruct", default="You are a helpful assistant. 请用普通话表达。<|endofprompt|>")
    args = ap.parse_args()

    _bootstrap(args)

    import torch
    import torchaudio
    from cosyvoice.cli.cosyvoice import AutoModel

    os.makedirs(args.out_dir, exist_ok=True)
    ref_wav = os.path.join(args.out_dir, "wav_reference.wav")
    ggml_wav = os.path.join(args.out_dir, "wav_ggml.wav")

    t0 = time.time()
    print(f"[1/5] loading CosyVoice3 model from {args.model_dir}", flush=True)
    cosyvoice = AutoModel(model_dir=args.model_dir)
    print(f"      -> {type(cosyvoice).__name__}, sample_rate={cosyvoice.sample_rate} "
          f"({time.time() - t0:.1f}s)", flush=True)

    # Wrap flow.inference to capture the exact inputs the reference flow sees,
    # and the mel it produces. The wrapper delegates to the original so the
    # official pipeline runs unchanged.
    captured = {}
    orig_flow_inference = cosyvoice.model.flow.inference

    def wrapped_flow_inference(**kw):
        captured["token"] = kw["token"].detach().cpu()
        captured["prompt_token"] = kw["prompt_token"].detach().cpu()
        captured["prompt_feat"] = kw["prompt_feat"].detach().cpu()
        captured["embedding"] = kw["embedding"].detach().cpu()  # raw 192-dim, pre-normalize
        mel, aux = orig_flow_inference(**kw)
        captured["mel"] = mel.detach().cpu()
        return mel, aux

    cosyvoice.model.flow.inference = wrapped_flow_inference

    print(f"[2/5] running official inference: {args.text!r}", flush=True)
    ref_speech = None
    for out in cosyvoice.inference_instruct2(args.text, args.instruct, args.prompt_wav, stream=False):
        ref_speech = out["tts_speech"]
    assert ref_speech is not None, "inference_instruct2 yielded no speech"
    torchaudio.save(ref_wav, ref_speech, cosyvoice.sample_rate)
    print(f"      -> {ref_wav}  shape={tuple(ref_speech.shape)} "
          f"({ref_speech.shape[1] / cosyvoice.sample_rate:.2f}s)", flush=True)

    prompt_token = captured["prompt_token"]   # (1, P)
    token = captured["token"]                 # (1, Tt)
    prompt_feat = captured["prompt_feat"]     # (1, mel_len1, 80)
    embedding = captured["embedding"]         # (1, 192)
    ref_mel = captured["mel"]                 # (1, 80, OUT_FRAMES)

    P, Tt = prompt_token.shape[1], token.shape[1]
    mel_len1 = prompt_feat.shape[1]
    mel_t = (P + Tt) * 2                       # token_mel_ratio = 2
    out_frames = mel_t - mel_len1
    print(f"[3/5] flow inputs: prompt_tokens={P} tokens={Tt} mel_len1={mel_len1} "
          f"mel_t={mel_t} out_frames={out_frames}", flush=True)

    rand_noise = cosyvoice.model.flow.decoder.rand_noise  # (1, 80, 15000), cpu
    noise_z = rand_noise[:, :, :mel_t]                    # (1, 80, mel_t)

    # Persist the captured flow inputs so the C++ step can be re-run alone.
    in_dir = os.path.join(args.out_dir, "flow_inputs")
    os.makedirs(in_dir, exist_ok=True)
    prompt_token[0].numpy().astype(np.int32).tofile(os.path.join(in_dir, "prompt_tokens.i32"))
    token[0].numpy().astype(np.int32).tofile(os.path.join(in_dir, "tokens.i32"))
    prompt_feat[0].numpy().astype(np.float32).tofile(os.path.join(in_dir, "prompt_feat.f32"))
    embedding[0].numpy().astype(np.float32).tofile(os.path.join(in_dir, "spk_embedding.f32"))
    noise_z[0].numpy().astype(np.float32).tofile(os.path.join(in_dir, "noise_z.f32"))

    with tempfile.TemporaryDirectory() as tmp:
        outdir = os.path.join(tmp, "out")
        os.makedirs(outdir)

        print(f"[4/5] running GGML flow decoder ({args.flow_dump})", flush=True)
        subprocess.run([args.flow_dump, args.flow_gguf, in_dir, outdir], check=True)
        ggml_mel = np.fromfile(os.path.join(outdir, "feat.f32"), dtype=np.float32)

    assert ggml_mel.size == 80 * out_frames, \
        f"GGML mel size {ggml_mel.size} != 80*{out_frames}"
    ggml_mel = ggml_mel.reshape(1, 80, out_frames)

    mel_diff = np.abs(ggml_mel - ref_mel[0].numpy())
    print(f"      mel max abs err (GGML vs reference flow) = {mel_diff.max():.3e}  "
          f"mean = {mel_diff.mean():.3e}", flush=True)

    print("[5/5] vocoding GGML mel with official HiFT", flush=True)
    ggml_mel_t = torch.from_numpy(ggml_mel).to(cosyvoice.model.device)
    ggml_speech, _ = cosyvoice.model.hift.inference(speech_feat=ggml_mel_t, finalize=True)
    torchaudio.save(ggml_wav, ggml_speech.cpu(), cosyvoice.sample_rate)
    print(f"      -> {ggml_wav}  shape={tuple(ggml_speech.shape)} "
          f"({ggml_speech.shape[1] / cosyvoice.sample_rate:.2f}s)", flush=True)

    print("\nDone. Compare with:  python3 scripts/listen_compare.py "
          f"{ggml_wav} {ref_wav}")


if __name__ == "__main__":
    main()
