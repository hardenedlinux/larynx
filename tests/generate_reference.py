#!/usr/bin/env python3
"""Build the prefill ``lm_input`` + length schedule for several real TTS texts.

This is the *input preparation* half of the autonomous-generation verification
(tests/verify_generate.py drives the C++ half). It reuses the exact frontend
pipeline of CosyVoice3LM.inference — ``text_normalize`` -> ``_extract_text_token``
-> ``embed_tokens`` -> ``[sos; text; task_id]`` — so the resulting ``lm_input`` is
what the model actually sees, and computes the same ``min_len``/``max_len`` as
``inference_wrapper`` (``int(text_len * {2,20})``). The autoregressive DECODE loop
is *not* run here: that is what the C++ does on its own (the point of the test).

Runs under the CosyVoice python3.10 env, NOT the larynx ``.venv``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/generate_reference.py [--out .../generate_inputs.npz]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
QWEN_DIR = os.path.join(MODEL_DIR, "CosyVoice-BlankEN")

INSTRUCT = "You are a helpful assistant. 请用普通话表达。<|endofprompt|>"

MAX_TOKEN_TEXT_RATIO = 20
MIN_TOKEN_TEXT_RATIO = 2

# A spread of real, different-length Mandarin sentences (numbers included to also
# exercise text_normalize). The verify step checks each generated length lands in
# [min_len, max_len] with a normal stop (or reports an anomaly honestly).
TEXTS = [
    "你好。",
    "今天天气不错。",
    "今天天气不错，我们一起去公园散步吧。",
    "今天下午三点我们开会讨论项目进展。",
    "人工智能正在改变我们的生活方式，从语音助手到自动驾驶，从医疗诊断到金融分析。",
]


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR,
              os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "llm.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "generate_inputs.npz"))
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    torch.set_num_threads(1)

    with open(os.path.join(MODEL_DIR, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={
            "qwen_pretrain_path": QWEN_DIR,
            "flow": None, "hift": None, "hifigan": None,
        })
    llm = configs["llm"]
    llm.float()
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = llm.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors; missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, (missing, unexpected)
    llm.eval()

    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"], configs["feat_extractor"],
        os.path.join(MODEL_DIR, "campplus.onnx"),
        os.path.join(MODEL_DIR, "speech_tokenizer_v3.onnx"),
        os.path.join(MODEL_DIR, "spk2info.pt"),
        configs["allowed_special"])
    frontend.device = torch.device("cpu")

    prompt_token, _ = frontend._extract_text_token(INSTRUCT)

    data = {"texts": np.asarray(TEXTS, dtype=object), "n": np.asarray([len(TEXTS)])}
    with torch.no_grad():
        for i, text in enumerate(TEXTS):
            norm_text = frontend.text_normalize(text, split=True, text_frontend=True)[0]
            text_token, _ = frontend._extract_text_token(norm_text)
            text_all = torch.cat([prompt_token, text_token], dim=1)
            text_emb = llm.llm.model.model.embed_tokens(text_all)
            sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
            task_id_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
            lm_input = torch.cat([sos_emb, text_emb, task_id_emb], dim=1)

            L = lm_input.shape[1]
            text_len = text_token.shape[1]
            min_len = int(text_len * MIN_TOKEN_TEXT_RATIO)
            max_len = int(text_len * MAX_TOKEN_TEXT_RATIO)

            data[f"lm_input_{i}"] = lm_input.detach().cpu().numpy().astype(np.float32)  # (1,L,896)
            data[f"L_{i}"] = np.asarray([L], dtype=np.int64)
            data[f"text_len_{i}"] = np.asarray([text_len], dtype=np.int64)
            data[f"min_len_{i}"] = np.asarray([min_len], dtype=np.int64)
            data[f"max_len_{i}"] = np.asarray([max_len], dtype=np.int64)
            print(f"[{i}] text_len={text_len} L={L} min_len={min_len} max_len={max_len} :: {text}")

    np.savez(args.out, **data)
    print(f"wrote {args.out} ({len(TEXTS)} texts)")


if __name__ == "__main__":
    main()
