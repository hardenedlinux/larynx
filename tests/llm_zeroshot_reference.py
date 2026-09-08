#!/usr/bin/env python3
"""PyTorch reference for the CosyVoice3 LLM **zero-shot (voice-cloning) lm_input**.

The existing LLM references (llm_reference.py / llm_decode_reference.py /
generate_reference.py) build the lm_input for the *instruct2* branch, where
``prompt_speech_token`` is empty and the text is ``[instruct; <|endofprompt|>;
tts_text]``. This script is the complementary branch: ``inference_zero_shot``,
the voice-cloning path where the prompt audio's speech tokens ARE appended.

It is the ground truth for Phase-4 wrap-up 阶段1/2: prove that the LLM's real
input is assembled as

    lm_input = [sos(6561); embed_tokens([prompt_text; text]); task_id(6563);
                speech_embedding(prompt_speech_token)]

i.e. the *only* thing the zero-shot branch adds over instruct2 is the trailing
``speech_embedding(prompt_speech_token)`` block (the prompt audio's speech
tokens looked up in the 6761-row speech_embedding table). The 192-dim speaker
embedding is **not** part of the LLM input (``CosyVoice3LM.inference`` ignores
the ``embedding`` argument); it is consumed only by the Flow decoder.

Everything is read from the real pipeline (``inference_zero_shot`` ->
``frontend_zero_shot`` -> ``Qwen2LM.inference``) and the real assets
(``asset/zero_shot_prompt.wav`` + the ``speech_tokenizer_v3.onnx`` session), so
``prompt_speech_token`` here is a genuine speech-token sequence, not synthetic.

Dumped (float32 unless noted):

  ``prompt_text_tokens``   (1, Pt) int32  prompt_text token ids  (the "You are a
                                          helpful assistant.<|endofprompt|>希望…" line)
  ``text_tokens``          (1, T)  int32  tts_text token ids
  ``text_tokens_all``      (1, Pt+T) int32  concat([prompt_text, text])
  ``prompt_speech_token``  (1, P)  int32  speech_tokenizer_v3(prompt_wav)
  ``sos_emb``              (1, 896)        speech_embedding[6561]
  ``task_id_emb``          (1, 896)        speech_embedding[6563]
  ``text_emb``             (1, Pt+T, 896)  embed_tokens(concat text)
  ``prompt_speech_token_emb`` (1, P, 896)  speech_embedding(prompt_speech_token)
  ``lm_input``             (1, L, 896)     concat of the four blocks above

and the prefill replay (identical to llm_reference.py) so the same lm_input can
be fed straight through the C++ ``larynx_llm_dump`` and cross-checked:

  ``h{0..23}``  per-layer post-residual hidden states
  ``final_norm``  model.norm(layer23)
  ``logits``      llm_decoder(final_norm)   (1, L, 6761)

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/llm_zeroshot_reference.py [--checkpoint .../llm.pt] \
        [--out .../llm_zeroshot_ref.npz]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
QWEN_DIR = os.path.join(MODEL_DIR, "CosyVoice-BlankEN")
PROMPT_WAV = os.path.join(COSYVOICE_DIR, "asset", "zero_shot_prompt.wav")

# The official CosyVoice3 zero-shot pair (cosyvoice/example.py, cosyvoice3_example).
PROMPT_TEXT = "You are a helpful assistant.<|endofprompt|>希望你以后能够做的比我还好呦。"
TTS_TEXT = "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。"


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "llm.pt"))
    ap.add_argument("--prompt-wav", default=PROMPT_WAV)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "llm_zeroshot_ref.npz"))
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
    llm = configs["llm"]  # CosyVoice3LM
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

    # --- Replicate inference_zero_shot -> frontend_zero_shot tokenization. ------
    prompt_text_norm = frontend.text_normalize(PROMPT_TEXT, split=False, text_frontend=True)
    tts_text_norm = frontend.text_normalize(TTS_TEXT, split=True, text_frontend=True)
    tts_sentence = tts_text_norm[0]
    print(f"prompt_text (normalized): {prompt_text_norm!r}")
    print(f"tts_text   (normalized): {tts_sentence!r}")

    prompt_token, _ = frontend._extract_text_token(prompt_text_norm)   # (1, Pt)
    text_token, _ = frontend._extract_text_token(tts_sentence)         # (1, T)
    prompt_speech_token, _ = frontend._extract_speech_token(args.prompt_wav)  # (1, P)

    print(f"prompt_text tokens ({prompt_token.shape[1]}): {prompt_token[0].tolist()}")
    print(f"text        tokens ({text_token.shape[1]}): {text_token[0].tolist()}")
    print(f"prompt_speech_token ({prompt_speech_token.shape[1]}): "
          f"{prompt_speech_token[0].tolist()[:16]}{'...' if prompt_speech_token.shape[1] > 16 else ''}")

    # --- Build lm_input exactly as CosyVoice3LM.inference (Qwen2LM.inference). ---
    text_all = torch.cat([prompt_token, text_token], dim=1)            # (1, Pt+T)
    text_emb = llm.llm.model.model.embed_tokens(text_all)              # (1, Pt+T, 896)
    sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)   # (1, 1, 896)
    task_id_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    prompt_speech_token_emb = llm.speech_embedding(prompt_speech_token)  # (1, P, 896)
    lm_input = torch.cat([sos_emb, text_emb, task_id_emb, prompt_speech_token_emb], dim=1)

    L = lm_input.shape[1]
    Pt, T, P = prompt_token.shape[1], text_token.shape[1], prompt_speech_token.shape[1]
    assert L == 1 + (Pt + T) + 1 + P, (L, Pt, T, P)
    print(f"lm_input: L={L}  = 1 sos + ({Pt} prompt_text + {T} text) + 1 task_id + {P} prompt_speech_token")

    # --- Prefill replay (same as llm_reference.py), for C++ cross-check. --------
    masks = torch.ones(1, L, dtype=torch.bool)
    layer23 = {}
    def hook(m, i, o):
        layer23["x"] = o[0].detach().cpu().clone()
    h = llm.llm.model.model.layers[23].register_forward_hook(hook)
    try:
        with torch.no_grad():
            outs = llm.llm.model(inputs_embeds=lm_input, attention_mask=masks,
                                 output_hidden_states=True, return_dict=True)
    finally:
        h.remove()

    hs = outs.hidden_states
    assert len(hs) == 25, f"expected 25 hidden states, got {len(hs)}"
    logits = llm.llm_decoder(hs[-1])

    out = {
        "prompt_text_tokens": prompt_token.detach().cpu().numpy().astype(np.int32),      # (1, Pt)
        "text_tokens":        text_token.detach().cpu().numpy().astype(np.int32),        # (1, T)
        "text_tokens_all":    text_all.detach().cpu().numpy().astype(np.int32),          # (1, Pt+T)
        "prompt_speech_token": prompt_speech_token.detach().cpu().numpy().astype(np.int32),  # (1, P)
        "sos_emb":            sos_emb.detach().cpu().numpy(),                            # (1, 1, 896)
        "task_id_emb":        task_id_emb.detach().cpu().numpy(),                        # (1, 1, 896)
        "text_emb":           text_emb.detach().cpu().numpy(),                           # (1, Pt+T, 896)
        "prompt_speech_token_emb": prompt_speech_token_emb.detach().cpu().numpy(),       # (1, P, 896)
        "lm_input":           lm_input.detach().cpu().numpy(),                           # (1, L, 896)
        "final_norm":         hs[-1].detach().cpu().numpy(),                             # (1, L, 896)
        "logits":             logits.detach().cpu().numpy(),                             # (1, L, 6761)
    }
    for i in range(23):
        out[f"h{i}"] = hs[i + 1].detach().cpu().numpy()
    out["h23"] = layer23["x"].numpy()

    np.savez(args.out, **out)
    print(f"wrote {args.out}:")
    for k, v in out.items():
        print(f"  {k:24s} {str(v.dtype):8s} {str(v.shape)}")


if __name__ == "__main__":
    main()
