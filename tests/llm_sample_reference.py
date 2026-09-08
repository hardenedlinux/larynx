#!/usr/bin/env python3
"""PyTorch reference for the CosyVoice3 LLM **sampling policy** (Checkpoint 6,
sampling-only step).

Does NOT run the model: it loads the raw per-step logits captured by
tests/llm_decode_seq_reference.py (llm_decode_seq_ref.npz) and reproduces the
deterministic core of ``ras_sampling`` — ``nucleus_sampling`` (softmax -> sort
descending stable -> take while cum_prob < top_p and count < top_k) — producing
the candidate set at every decode step. The stochastic multinomial draw is
excluded (that is what the C++ samples with its own RNG); the candidate set is
fully deterministic and is the ground truth for tests/verify_llm_sampling.py.

Dumped (per decode step 0..N-1):

  ``weighted_scores`` (N, 6761)  log_softmax(logits[i]) with the ignore_eos mask
                                 (weighted_scores[6561] = -inf for i < min_len) —
                                 i.e. exactly what torch feeds ``sampling_ids``.
  ``candidates``     (N, top_k)  nucleus candidate token ids, padded with -1.
  ``cand_count``     (N,)        number of candidates per step.
  ``cand_probs``     (N, top_k)  candidate softmax probs, padded with 0.

Constants mirror cosyvoice3.yaml (ras_sampling top_p=0.8 top_k=25) and
CosyVoice3LM (speech_token_size=6561). Runs under the larynx .venv (torch only;
no CosyVoice import).

Usage:
    .venv/bin/python tests/llm_sample_reference.py [--ref llm_decode_seq_ref.npz] [--out llm_sample_ref.npz]
"""

import argparse
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TOP_P = 0.8
TOP_K = 25
SPEECH_TOKEN_SIZE = 6561


def nucleus_candidates(logp):
    """Return (indices, probs) for nucleus_sampling(logp, top_p, top_k), no RNG."""
    probs = logp.softmax(dim=0)
    sorted_val, sorted_idx = probs.sort(descending=True, stable=True)
    idx, prob = [], []
    cum_prob = 0.0
    for i in range(len(sorted_idx)):
        if cum_prob < TOP_P and len(prob) < TOP_K:
            cum_prob += sorted_val[i].item()
            idx.append(sorted_idx[i].item())
            prob.append(sorted_val[i].item())
        else:
            break
    return idx, prob


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "llm_decode_seq_ref.npz"))
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                  "llm_sample_ref.npz"))
    args = ap.parse_args()

    torch.set_num_threads(1)
    ref = np.load(args.ref)
    logits = torch.from_numpy(ref["logits"])        # (N, 6761) float32
    min_len = int(ref["min_len"][0])
    N = logits.shape[0]
    print(f"loaded {args.ref}: N={N}, min_len={min_len}")

    weighted_scores = np.empty((N, SPEECH_TOKEN_SIZE + 200), dtype=np.float32)
    candidates = np.full((N, TOP_K), -1, dtype=np.int64)
    cand_probs = np.zeros((N, TOP_K), dtype=np.float32)
    cand_count = np.zeros((N,), dtype=np.int64)

    for i in range(N):
        logp = logits[i].log_softmax(dim=-1)
        if i < min_len:
            logp[SPEECH_TOKEN_SIZE] = -float("inf")
        weighted_scores[i] = logp.numpy()
        idx, prob = nucleus_candidates(logp)
        k = len(idx)
        candidates[i, :k] = idx
        cand_probs[i, :k] = prob
        cand_count[i] = k

    np.savez(args.out,
             weighted_scores=weighted_scores,
             candidates=candidates,
             cand_count=cand_count,
             cand_probs=cand_probs)
    print(f"wrote {args.out}:")
    print(f"  weighted_scores  float32  {weighted_scores.shape}")
    print(f"  candidates       int64    {candidates.shape}")
    print(f"  cand_count       int64    {cand_count.shape}")
    print(f"  cand_probs       float32  {cand_probs.shape}")
    print(f"  cand_count range: {cand_count.min()}..{cand_count.max()}")


if __name__ == "__main__":
    main()
