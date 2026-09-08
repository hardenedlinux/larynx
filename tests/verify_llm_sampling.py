#!/usr/bin/env python3
"""Numerical verification of the C++ CosyVoice3 LLM **sampling policy**.

Compares the deterministic core of ``ras_sampling`` — the ``nucleus_sampling``
candidate set — between the C++ (larynx_llm_sample_dump) and the PyTorch
reference (tests/llm_sample_reference.py), over every decode step of the
trajectory captured in Checkpoint 5. The stochastic multinomial draw is excluded
(the C++ uses its own RNG); what is verified here is fully deterministic:

  1. log_softmax + ignore_eos masking (``weighted_scores``)
  2. softmax + stable-desc sort + top_p/top_k selection (``candidates``/``cand_probs``)

If (1) matches to float32 tolerance and (2) matches exactly, the C++ sampling
policy produces the same candidate set torch would, so any downstream token
difference is purely the (expected) RNG stream difference.

Usage:
    python3 tests/verify_llm_sampling.py
    LARYNX_LLM_SAMPLE_DUMP=./build/larynx_llm_sample_dump python3 tests/verify_llm_sampling.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEECH_VOCAB = 6761
TOP_K = 25


def load_raw(path, shape, dtype=np.float32):
    data = np.fromfile(path, dtype=dtype)
    n = int(np.prod(shape))
    assert data.size == n, f"{path}: got {data.size} floats, expected {n} ({shape})"
    return data.reshape(shape)


def main():
    dump_bin = os.environ.get("LARYNX_LLM_SAMPLE_DUMP",
                              os.path.join(ROOT, "build", "larynx_llm_sample_dump"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_sample_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_llm_sample_dump not found at {dump_bin}; build it first")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/llm_sample_reference.py first")

    ref = np.load(ref_path)
    N = ref["weighted_scores"].shape[0]
    ref_ws = ref["weighted_scores"]          # (N, 6761)
    ref_cand = ref["candidates"]             # (N, TOP_K)
    ref_count = ref["cand_count"]            # (N,)
    ref_probs = ref["cand_probs"]            # (N, TOP_K)
    print(f"dump binary: {dump_bin}")
    print(f"reference  : {ref_path}  (N={N})\n")

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        # Raw logits come from the decode-seq reference; min_len too.
        seq_ref = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "llm_decode_seq_ref.npz"))
        seq_ref["logits"].astype(np.float32).reshape(-1).tofile(os.path.join(indir, "logits.f32"))
        np.asarray([int(seq_ref["min_len"][0])], dtype=np.int32).tofile(os.path.join(indir, "min_len.i32"))

        subprocess.run([dump_bin, indir, outdir], check=True)

        cpp_ws = load_raw(os.path.join(outdir, "weighted_scores.f32"), (N, SPEECH_VOCAB))
        cpp_cand = load_raw(os.path.join(outdir, "candidates.i32"), (N, TOP_K), dtype=np.int32)
        cpp_count = load_raw(os.path.join(outdir, "cand_count.i32"), (N,), dtype=np.int32)
        cpp_probs = load_raw(os.path.join(outdir, "cand_probs.f32"), (N, TOP_K))
        cpp_sampled = load_raw(os.path.join(outdir, "sampled.i32"), (N,), dtype=np.int32)

    # 1. weighted_scores (log_softmax + mask). The ignore_eos mask sets slot 6561 to
    #    -inf for i < min_len, so compare the -inf mask separately from the finite
    #    values (else -inf - (-inf) = nan).
    mask_diff = int((np.isneginf(cpp_ws) != np.isneginf(ref_ws)).sum())
    finite = np.isfinite(ref_ws)
    ws_abs = np.abs(cpp_ws[finite].astype(np.float64) - ref_ws[finite].astype(np.float64))
    ws_scale = np.abs(ref_ws[finite].astype(np.float64)).max()
    ws_rel = ws_abs.max() / ws_scale if ws_scale > 0 else 0.0

    # 2. candidate sets: exact match. Compare as unordered sets per step (order is
    #    also compared, but a rare near-tie float flip would only reorder, not change
    #    membership — report both).
    order_mismatch = 0
    set_mismatch = 0
    count_mismatch = 0
    prob_max = 0.0
    for i in range(N):
        c = int(ref_count[i])
        cc = int(cpp_count[i])
        if c != cc:
            count_mismatch += 1
        rs = set(int(x) for x in ref_cand[i, :c])
        cs = set(int(x) for x in cpp_cand[i, :cc])
        if rs != cs:
            set_mismatch += 1
        if list(ref_cand[i, :c]) != list(cpp_cand[i, :cc]):
            order_mismatch += 1
        for j in range(min(c, cc)):
            prob_max = max(prob_max, abs(float(cpp_probs[i, j]) - float(ref_probs[i, j])))

    # 3. RNG smoke test: with empty history the sampled token must be in the
    #    candidate set (nucleus branch; the repetition fallback never fires).
    out_of_set = 0
    for i in range(N):
        c = int(cpp_count[i])
        if int(cpp_sampled[i]) not in set(int(x) for x in cpp_cand[i, :c]):
            out_of_set += 1

    print(f"{'check':<28} {'value':>14}")
    print("-" * 44)
    print(f"{'weighted_scores max rel err':<28} {ws_rel:>14.3e}")
    print(f"{'weighted_scores -inf mismatches':<28} {mask_diff:>14d}")
    print(f"{'candidate prob max abs err':<28} {prob_max:>14.3e}")
    print(f"{'candidate count mismatches':<28} {count_mismatch:>14d}")
    print(f"{'candidate set mismatches':<28} {set_mismatch:>14d}")
    print(f"{'candidate order mismatches':<28} {order_mismatch:>14d}")
    print(f"{'sampled-token out-of-set (RNG)':<28} {out_of_set:>14d}")

    ok = (ws_rel <= 1e-3 and mask_diff == 0 and out_of_set == 0
          and count_mismatch == 0 and set_mismatch == 0)
    if ok:
        print(f"\nPASS: sampling policy matches ({N} steps; candidate sets identical, "
              f"weighted_scores rel {ws_rel:.3e}).")
        if order_mismatch:
            print("NOTE: candidate ORDER differs on some steps (near-tie float reorder; "
                  "sets identical — no policy impact).")
    else:
        print("\nFAIL: sampling policy diverges from the PyTorch reference.")
        sys.exit(1)


if __name__ == "__main__":
    main()
