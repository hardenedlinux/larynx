#!/usr/bin/env python3
"""PyTorch numerical reference for the CosyVoice3 LLM **single-step decode**.

Checkpoint 4 of the Phase-4 LLM work. It extends the prefill reference
(tests/llm_reference.py) one step: run the Qwen2 prefill with ``use_cache=True``
(the same ``forward_one_step`` path ``CosyVoice3LM.inference_wrapper`` uses), then
feed one sampled token back through a single-token KV-cache decode step and dump
the resulting logits. This is the ground truth for ``tests/verify_llm_decode.py``.

Semantics (mirror ``Qwen2LM.inference_wrapper``, the non-vLLM path):

  1. prefill:  ``forward_one_step(lm_input, masks=tril(L))``  -> y_pred, cache
  2. logp    :  ``llm_decoder(y_pred[:, -1])``                 (position L-1)
  3. token   :  ``argmax(logp)``                               (deterministic)
  4. decode  :  ``forward_one_step(speech_emb[token], masks=tril(1), cache)``

Dumped (all float32; cache tensors are the ROPED key / raw value, batch dim
dropped, shape ``(KV_HEADS, seq, HEAD_DIM)`` = ``(2, 36, 64)``):

  ``lm_input``          (1, L, 896)   prefill embedding sequence (feeds the C++ tool)
  ``next_token``        int           argmax of prefill logits
  ``prefill_logits``    (1, 6761)     llm_decoder at position L-1 (cross-check vs Chk3)
  ``decode_logits``     (1, 6761)     llm_decoder of the single-step decode
  ``decode_final_norm`` (1, 896)      model.norm output of the decode step
  ``cache_k_{i}``       (2, L, 64)    past_key_values key   per layer 0..23
  ``cache_v_{i}``       (2, L, 64)    past_key_values value per layer 0..23

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/llm_decode_reference.py [--checkpoint .../llm.pt] [--out .../llm_decode_ref.npz]
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")
QWEN_DIR = os.path.join(MODEL_DIR, "CosyVoice-BlankEN")

TEXT = "今天天气不错，我们一起去公园散步吧。"
INSTRUCT = "You are a helpful assistant. 请用普通话表达。<|endofprompt|>"


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def unpack_cache(cache):
    """Return per-layer list of (key, value) tensors from a transformers cache."""
    if hasattr(cache, "key_cache"):  # DynamicCache (transformers >= 4.36)
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)  # legacy tuple-of-tuples


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "llm.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "llm_decode_ref.npz"))
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

    norm_text = frontend.text_normalize(TEXT, split=True, text_frontend=True)[0]
    text_token, _ = frontend._extract_text_token(norm_text)
    prompt_token, _ = frontend._extract_text_token(INSTRUCT)

    text_all = torch.cat([prompt_token, text_token], dim=1)
    text_emb = llm.llm.model.model.embed_tokens(text_all)
    sos_emb = llm.speech_embedding.weight[llm.sos].reshape(1, 1, -1)
    task_id_emb = llm.speech_embedding.weight[llm.task_id].reshape(1, 1, -1)
    lm_input = torch.cat([sos_emb, text_emb, task_id_emb], dim=1)
    L = lm_input.shape[1]
    print(f"lm_input: L={L}")

    # --- Prefill via forward_one_step (use_cache=True), same as inference_wrapper. ---
    masks = torch.tril(torch.ones((1, L, L), dtype=torch.bool))
    with torch.no_grad():
        y_pred, cache = llm.llm.forward_one_step(lm_input, masks=masks, cache=None)
    prefill_logits = llm.llm_decoder(y_pred[:, -1]).detach().cpu()  # (1, 6761)

    next_token = int(prefill_logits[0].argmax())
    print(f"next_token (argmax of prefill logits): {next_token}")

    # Capture the prefill KV cache BEFORE the decode step — transformers'
    # DynamicCache.update() mutates the cache object in place, so the returned
    # `cache` reference would otherwise report 37 tokens after the decode call.
    prefill_layers = unpack_cache(cache)
    prefill_cache = [
        (k[0].detach().cpu().numpy(), v[0].detach().cpu().numpy())  # (KV_HEADS, L, HEAD_DIM)
        for k, v in prefill_layers
    ]
    print(f"prefill cache layers: {len(prefill_cache)}; first key shape {prefill_cache[0][0].shape}")

    # --- Single-token decode with the KV cache. ---
    tok_emb = llm.speech_embedding.weight[next_token].reshape(1, 1, -1)
    dmasks = torch.tril(torch.ones((1, 1, 1), dtype=torch.bool))
    with torch.no_grad():
        y2, cache2 = llm.llm.forward_one_step(tok_emb, masks=dmasks, cache=cache)
    decode_logits = llm.llm_decoder(y2[:, -1]).detach().cpu()     # (1, 6761)
    decode_final_norm = y2[:, -1].detach().cpu()                  # (1, 896)

    out = {
        "lm_input": lm_input.detach().cpu().numpy(),               # (1, L, 896)
        "next_token": np.asarray([next_token], dtype=np.int64),
        "prefill_logits": prefill_logits.numpy(),                  # (1, 6761)
        "decode_logits": decode_logits.numpy(),                    # (1, 6761)
        "decode_final_norm": decode_final_norm.numpy(),            # (1, 896)
    }
    for i, (k, v) in enumerate(prefill_cache):
        out[f"cache_k_{i}"] = k
        out[f"cache_v_{i}"] = v

    np.savez(args.out, **out)
    print(f"wrote {args.out}:")
    for kk, vv in out.items():
        print(f"  {kk:18s} {str(vv.dtype):8s} {str(vv.shape)}")


if __name__ == "__main__":
    main()
