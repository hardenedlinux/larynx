#!/usr/bin/env python3
"""PyTorch numerical reference for the CosyVoice3 LLM (Qwen2) **prefill**.

This is the ground truth for ``tests/verify_llm.py``. Like the HiFT reference, it
imports the real package: the ``CosyVoice3LM`` is built by ``hyperpyyaml`` from
``cosyvoice3.yaml`` (``flow``/``hift``/``hifigan`` overridden to ``None``, and
``qwen_pretrain_path`` pointed at the ``CosyVoice-BlankEN`` HF checkpoint), then
the fine-tuned ``llm.pt`` state_dict is loaded **in float32** (the model is cast
with ``.float()`` *before* ``load_state_dict`` so the bf16 ``from_pretrained``
round-trip is overwritten by the exact float32 weights the GGUF conversion reads).

The lm_input is built exactly as ``CosyVoice3LM.inference`` does for
``inference_instruct2``:

    lm_input = [sos(6561); text_emb(instruct || tts_text); task_id(6563)]

(the prompt speech-token embedding is empty in instruct2 mode, and the speaker
embedding is *not* used by ``CosyVoice3LM.inference``). The prefill then replays
``Qwen2Encoder.forward``: ``self.model(inputs_embeds=lm_input,
attention_mask=all-True, output_hidden_states=True)``.

Captured per stage (GGML-layout byte-identical to the PyTorch row-major flat):

  ``h{0..22}``    layer 0..22 post-residual outputs  (hidden_states[1..23])
  ``h23``         layer 23 post-residual output      (forward hook; transformers
                  overwrites it with model.norm in ``hidden_states``)
  ``final_norm``  model.norm(layer23)                (hidden_states[-1])
  ``logits``      llm_decoder(final_norm)            (1, L, 6761)

The final RMSNorm (``model.norm``) **is** applied — verified against the pinned
transformers 4.51.3 ``modeling_qwen2.py``, whose ``all_hidden_states`` ends with
``self.norm(hidden_states)``.

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/llm_reference.py [--checkpoint .../llm.pt] [--out .../llm_ref.npz]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
QWEN_DIR = os.path.join(MODEL_DIR, "CosyVoice-BlankEN")

# The Phase-2/3 acceptance pair (tests/acceptance_wavs.py defaults).
TEXT = "今天天气不错，我们一起去公园散步吧。"
INSTRUCT = "You are a helpful assistant. 请用普通话表达。<|endofprompt|>"


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "llm.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_ref.npz"))
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    torch.set_num_threads(1)

    # --- Build the real CosyVoice3LM (+ tokenizer) from the yaml, no flow/hift. --
    with open(os.path.join(MODEL_DIR, "cosyvoice3.yaml")) as f:
        configs = load_hyperpyyaml(f, overrides={
            "qwen_pretrain_path": QWEN_DIR,
            "flow": None, "hift": None, "hifigan": None,
        })
    llm = configs["llm"]  # CosyVoice3LM
    llm.float()           # cast to f32 *before* loading, so f32 llm.pt is exact
    sd = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = llm.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors from {args.checkpoint}; "
          f"missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, (missing, unexpected)
    llm.eval()

    # Run on CPU: the C++ side is forced to the CPU backend (LARYNX_BACKEND=cpu)
    # for this verify, so both sides use float32 CPU kernels (deterministic, no
    # dependence on an idle GPU).
    frontend = CosyVoiceFrontEnd(
        configs["get_tokenizer"], configs["feat_extractor"],
        os.path.join(MODEL_DIR, "campplus.onnx"),
        os.path.join(MODEL_DIR, "speech_tokenizer_v3.onnx"),
        os.path.join(MODEL_DIR, "spk2info.pt"),
        configs["allowed_special"])
    frontend.device = torch.device("cpu")  # token tensors land on CPU

    # --- Tokenize exactly as inference_instruct2 -> frontend_instruct2 does. -----
    norm_text = frontend.text_normalize(TEXT, split=True, text_frontend=True)[0]
    text_token, _ = frontend._extract_text_token(norm_text)       # tts text tokens
    prompt_token, _ = frontend._extract_text_token(INSTRUCT)      # instruct tokens (raw)
    print(f"instruct tokens ({prompt_token.shape[1]}): {prompt_token[0].tolist()}")
    print(f"text     tokens ({text_token.shape[1]}): {text_token[0].tolist()}")

    # --- Build lm_input (CosyVoice3LM.inference semantics). ----------------------
    text_all = torch.cat([prompt_token, text_token], dim=1)          # (1, T)
    text_emb = llm.llm.model.model.embed_tokens(text_all)            # (1, T, 896)
    sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_id_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    lm_input = torch.cat([sos_emb, text_emb, task_id_emb], dim=1)    # (1, L, 896)
    L = lm_input.shape[1]
    print(f"lm_input: L={L}  (1 sos + {prompt_token.shape[1]} instruct + "
          f"{text_token.shape[1]} text + 1 task_id)")

    # --- Prefill (Qwen2Encoder.forward semantics). -------------------------------
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

    hs = outs.hidden_states                    # 25 entries: [embed, L0..L22, norm(L23)]
    assert len(hs) == 25, f"expected 25 hidden states, got {len(hs)}"
    logits = llm.llm_decoder(hs[-1])           # llm_decoder(final_norm)

    out = {
        "lm_input": lm_input.detach().cpu().numpy(),     # (1, L, 896)
        "text_tokens": text_all.detach().cpu().numpy(),  # (1, T) = instruct||text
        "final_norm": hs[-1].detach().cpu().numpy(),     # (1, L, 896)
        "logits": logits.detach().cpu().numpy(),         # (1, L, 6761)
    }
    for i in range(23):                                  # layers 0..22
        out[f"h{i}"] = hs[i + 1].detach().cpu().numpy()
    out["h23"] = layer23["x"].numpy()                    # layer 23 (raw, pre-norm)

    np.savez(args.out, **out)
    print(f"wrote {args.out}:")
    for k, v in out.items():
        print(f"  {k:12s} {str(v.dtype):8s} {str(v.shape)}")


if __name__ == "__main__":
    main()
