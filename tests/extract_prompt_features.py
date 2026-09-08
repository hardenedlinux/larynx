#!/usr/bin/env python3
"""Pre-extract the prompt-side features the C++ pipeline reads from disk.

The C++ CLI deliberately does NOT run the ONNX frontend (campplus + speech
tokenizer) or the matcha mel_spectrogram — those are deferred to standalone
tasks. This script is the "temporarily hand to Python" pre-extraction step: it
computes the three prompt-side tensors a prompt voice contributes and writes
them to the bundle layout ``Pipeline`` / the CLI expect:

    prompt_tokens.i32   flow_prompt_speech_token  (1, token_len)         int32
    prompt_feat.f32     prompt_speech_feat        (1, 2*token_len, 80)   float32
    spk_embedding.f32   flow_embedding            (1, 192)               float32

It builds ONLY ``CosyVoiceFrontEnd`` — never the LLM/Flow/HiFT — so it is cheap,
deterministic and does not compete with the decoders for GPU memory.

Usage (from anywhere):
    PYTHONPATH=... python3 tests/extract_prompt_features.py \
        --model-dir ~/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B \
        --prompt-wav ~/Project/CosyVoice/asset/zero_shot_prompt.wav \
        --out-dir wavs/flow_inputs
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)
    # Frontend is CPU-only and deterministic; keep it off the GPU so the ONNX
    # speech-tokenizer session doesn't contend with the decoders for VRAM.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")


def main():
    ap = argparse.ArgumentParser(description="Extract the prompt-side feature bundle.")
    ap.add_argument("--model-dir", default=os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B"))
    ap.add_argument("--prompt-wav", default=os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav"))
    ap.add_argument("--instruct", default="You are a helpful assistant. 请用普通话表达。<|endofprompt|>")
    ap.add_argument("--out-dir", default=os.path.join(ROOT, "wavs", "flow_inputs"))
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd
    from hyperpyyaml import load_hyperpyyaml

    yaml_path = os.path.join(args.model_dir, "cosyvoice3.yaml")
    with open(yaml_path, "r") as f:
        configs = load_hyperpyyaml(
            f, overrides={"qwen_pretrain_path": os.path.join(args.model_dir, "CosyVoice-BlankEN")})

    # Mirrors CosyVoice3.__init__'s frontend construction exactly, but stops
    # short of building the (lazy, !new:) LLM/Flow/HiFT.
    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"],
        configs["feat_extractor"],
        os.path.join(args.model_dir, "campplus.onnx"),
        os.path.join(args.model_dir, "speech_tokenizer_v3.onnx"),
        os.path.join(args.model_dir, "spk2info.pt"),
        configs["allowed_special"],
    )

    # tts_text='' is fine: the prompt-side tensors depend only on the prompt wav.
    model_input = frontend.frontend_zero_shot("", args.instruct, args.prompt_wav, 24000, "")
    prompt_tokens = model_input["flow_prompt_speech_token"]   # (1, token_len)
    prompt_feat = model_input["prompt_speech_feat"]           # (1, 2*token_len, 80)
    spk_embedding = model_input["flow_embedding"]             # (1, 192)

    token_len = prompt_tokens.shape[1]
    mel_len1 = prompt_feat.shape[1]
    assert mel_len1 == 2 * token_len, f"mel_len1={mel_len1} != 2*token_len={2*token_len}"
    assert spk_embedding.shape[1] == 192, f"spk_embedding dim {spk_embedding.shape[1]} != 192"

    os.makedirs(args.out_dir, exist_ok=True)
    prompt_tokens[0].numpy().astype(np.int32).tofile(os.path.join(args.out_dir, "prompt_tokens.i32"))
    prompt_feat[0].numpy().astype(np.float32).tofile(os.path.join(args.out_dir, "prompt_feat.f32"))
    spk_embedding[0].numpy().astype(np.float32).tofile(os.path.join(args.out_dir, "spk_embedding.f32"))

    print(f"wrote bundle to {args.out_dir}:")
    print(f"  prompt_tokens.i32  {token_len} tokens (int32)")
    print(f"  prompt_feat.f32    {mel_len1} frames x 80 (float32)")
    print(f"  spk_embedding.f32  {spk_embedding.shape[1]} dims (float32)")


if __name__ == "__main__":
    main()
