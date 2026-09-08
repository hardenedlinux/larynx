#!/usr/bin/env python3
"""PyTorch numerical reference for the CosyVoice3 LLM **full autoregressive decode**.

Checkpoint 5 of the Phase-4 LLM work. Extends the single-step reference
(tests/llm_decode_reference.py) to the *full* ``inference_wrapper`` loop (the
non-vLLM path): prefill, then repeatedly sample one speech token and feed its
embedding back through the KV cache until a stop token is produced (or
``max_len`` is reached). Dumps:

  ``lm_input``   (1, L, 896)   prefill embedding sequence (feeds the C++ tool)
  ``tokens``     (N,) int64    sampled speech-token ids, EXCLUDING the stop token
  ``stop_token`` int           the stop token (6561..6760) that ended the loop
  ``logits``     (N, 6761)     raw ``llm_decoder`` output at decode step 0..N-1

Sampling is ``ras_sampling`` (top_p=0.8, top_k=25 — confirmed from
cosyvoice3.yaml) which is stochastic (``torch.multinomial``). The reference seeds
torch (``manual_seed(0)``, single thread) so the captured ``tokens`` are a fixed,
reproducible trajectory. The C++ side does **not** re-implement the RNG: it
injects ``tokens`` verbatim and compares only the per-step ``logits`` — the
deterministic part of the loop. This "captured-token-sequence injection" strategy
proves the KV-cache autoregressive mechanics (absolute rope positions, causal
mask, cache append) are bit-correct over the whole trajectory without conflating
RNG differences, which would otherwise be impossible to compare exactly.

Runs under the CosyVoice python3.10 env (torch 2.3.1+cu121), NOT the larynx
``.venv`` — see ``_bootstrap``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/llm_decode_seq_reference.py [--checkpoint .../llm.pt] [--out .../llm_decode_seq_ref.npz]
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

SAMPLING = 25
MAX_TOKEN_TEXT_RATIO = 20
MIN_TOKEN_TEXT_RATIO = 2


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR, os.path.join(COSYVOICE_DIR, "third_party", "Matcha-TTS")):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    os.chdir(COSYVOICE_DIR)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=os.path.join(MODEL_DIR, "llm.pt"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "llm_decode_seq_ref.npz"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    _bootstrap()

    import numpy as np
    import torch
    from hyperpyyaml import load_hyperpyyaml
    from cosyvoice.cli.frontend import CosyVoiceFrontEnd

    torch.manual_seed(args.seed)
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

    # Same length math as CosyVoice3LM.inference: (text_len - prompt_text_len) is
    # the tts text token count (prompt_speech_token is empty here).
    text_len = text_token.shape[1]
    min_len = int(text_len * MIN_TOKEN_TEXT_RATIO)
    max_len = int(text_len * MAX_TOKEN_TEXT_RATIO)
    print(f"text_len={text_len} min_len={min_len} max_len={max_len}")

    # --- Full decode loop, mirroring CosyVoice3LM.inference_wrapper (non-vLLM). ---
    out_tokens = []
    logits_list = []
    cache = None
    stop_token = -1
    cur = lm_input
    with torch.no_grad():
        for i in range(max_len):
            masks = torch.tril(torch.ones((1, cur.shape[1], cur.shape[1]), dtype=torch.bool))
            y_pred, cache = llm.llm.forward_one_step(cur, masks=masks, cache=cache)
            raw = llm.llm_decoder(y_pred[:, -1])              # (1, 6761), pre-softmax
            logits_list.append(raw.detach().cpu().numpy())    # (1, 6761)
            logp = raw.log_softmax(dim=-1)
            top_ids = llm.sampling_ids(logp.squeeze(dim=0), out_tokens, SAMPLING,
                                       ignore_eos=(i < min_len))
            if top_ids in llm.stop_token_ids:
                stop_token = int(top_ids)
                break
            out_tokens.append(int(top_ids))
            cur = llm.speech_embedding.weight[top_ids].reshape(1, 1, -1)

    n = len(out_tokens)
    print(f"decoded {n} speech tokens, stop_token={stop_token} "
          f"(first 8: {out_tokens[:8]})")

    # logits_list has n+1 entries: [0] is the PREFILL logit (from which tokens[0]
    # was sampled), and [i+1] is the decode logit for tokens[i] (0..n-1). The C++
    # injects the n tokens and emits only decode logits, so dump logits_list[1:]
    # to keep `logits` aligned with `tokens`. (logits_list[0] duplicates Chk3's
    # prefill logits; logits_list[n] is the pre-stop logit for the stop token.)
    logits = np.concatenate(logits_list[1:], axis=0)  # (N, 6761)
    out = {
        "lm_input": lm_input.detach().cpu().numpy(),          # (1, L, 896)
        "tokens": np.asarray(out_tokens, dtype=np.int64),     # (N,)
        "stop_token": np.asarray([stop_token], dtype=np.int64),
        "logits": logits.astype(np.float32),                  # (N, 6761)
        "min_len": np.asarray([min_len], dtype=np.int64),     # steps with ignore_eos=True
        "max_len": np.asarray([max_len], dtype=np.int64),
    }

    np.savez(args.out, **out)
    print(f"wrote {args.out}:")
    for kk, vv in out.items():
        print(f"  {kk:12s} {str(vv.dtype):8s} {str(vv.shape)}")


if __name__ == "__main__":
    main()
