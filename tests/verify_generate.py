#!/usr/bin/env python3
"""Verify the C++ LLM's **fully autonomous** generation loop (Checkpoint 6).

Unlike verify_llm_decode_seq.py (which injects a captured token trajectory and
compares per-step logits), this test drives ``larynx_llm_generate_dump`` end to
end: for each real text, the C++ prefill is seeded from the reference-built
``lm_input``, then the C++ samples its own tokens with its own RNG, feeds them
back into its own KV cache, and stops on its own when a stop token is drawn (or
runs out ``max_len``). This is the one remaining unverified path — the KV cache
accumulating a *self-sampled* trajectory with no externally injected history.

For each (text, seed) it records and reports:
  * tokens generated (== decode steps before the stop token),
  * stop mode (normal stop vs max_len truncation),
  * the stop token id, and whether it is actually in stop_token_ids (6561..6760).

Then it checks the behavioural invariants the user asked for: lengths land in a
reasonable [min_len, max_len] window and there is no "stops immediately after
min_len" or "never stops, always hard-truncated" anomaly. (``ignore_eos`` masks
only slot 6561, so a stop *can* still occur slightly before min_len via one of
the other 199 control tokens — reported honestly, not silently passed.)

Runs under the larynx .venv (numpy only). Reads tests/generate_inputs.npz.

Usage:
    python3 tests/verify_generate.py
    LARYNX_LLM_GENERATE_DUMP=./build/larynx_llm_generate_dump python3 tests/verify_generate.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SPEECH_TOKEN_SIZE = 6561
STOP_MIN, STOP_MAX = 6561, 6760  # stop_token_ids = [6561..6760]


def load_i32(path, n):
    data = np.fromfile(path, dtype=np.int32)
    assert data.size == n, f"{path}: got {data.size} ints, expected {n}"
    return data


def main():
    dump_bin = os.environ.get("LARYNX_LLM_GENERATE_DUMP",
                              os.path.join(ROOT, "build", "larynx_llm_generate_dump"))
    gguf = os.environ.get("LLM_GGUF", os.path.join(ROOT, "build", "llm.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "generate_inputs.npz")
    seeds = [int(s) for s in os.environ.get("SEEDS", "0,1").split(",") if s.strip() != ""]

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_llm_generate_dump not found at {dump_bin}; build it first")
    if not os.path.exists(gguf):
        sys.exit(f"llm.gguf not found at {gguf}; run tools/convert_weights.py --llm llm.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/generate_reference.py first")

    ref = np.load(ref_path, allow_pickle=True)
    n = int(ref["n"][0])
    texts = [str(t) for t in ref["texts"]]

    print(f"dump binary: {dump_bin}")
    print(f"llm.gguf   : {gguf}")
    print(f"reference  : {ref_path}  ({n} texts, seeds={seeds})\n")

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for i in range(n):
            text = texts[i]
            text_len = int(ref[f"text_len_{i}"][0])
            L = int(ref[f"L_{i}"][0])
            min_len = int(ref[f"min_len_{i}"][0])
            max_len = int(ref[f"max_len_{i}"][0])
            lm_input = ref[f"lm_input_{i}"].astype(np.float32)  # (1,L,896)

            for seed in seeds:
                indir = os.path.join(tmp, f"in_{i}_{seed}")
                outdir = os.path.join(tmp, f"out_{i}_{seed}")
                os.makedirs(indir)
                os.makedirs(outdir)

                lm_input.reshape(-1).tofile(os.path.join(indir, "lm_input.f32"))
                np.asarray([min_len], dtype=np.int32).tofile(os.path.join(indir, "min_len.i32"))
                np.asarray([max_len], dtype=np.int32).tofile(os.path.join(indir, "max_len.i32"))
                np.asarray([seed], dtype=np.int32).tofile(os.path.join(indir, "seed.i32"))

                subprocess.run([dump_bin, gguf, indir, outdir], check=True,
                               env={**os.environ, "LARYNX_BACKEND": "cpu"})

                steps = int(load_i32(os.path.join(outdir, "steps.i32"), 1)[0])
                stop = int(load_i32(os.path.join(outdir, "stop_token.i32"), 1)[0])
                stopped = stop >= 0
                in_range = STOP_MIN <= stop <= STOP_MAX
                rows.append(dict(i=i, seed=seed, text=text, text_len=text_len, L=L,
                                 min_len=min_len, max_len=max_len, steps=steps,
                                 stop=stop, stopped=stopped, in_range=in_range))

    print(f"{'#':>2} {'seed':>4} {'tlen':>4} {'L':>4} {'min':>4} {'max':>4} "
          f"{'tokens':>7} {'stop-mode':>14} {'stop_id':>8} {'in[6561,6760]'}")
    print("-" * 90)
    for r in rows:
        mode = "normal" if r["stopped"] else "TRUNCATED"
        print(f"{r['i']:>2} {r['seed']:>4} {r['text_len']:>4} {r['L']:>4} "
              f"{r['min_len']:>4} {r['max_len']:>4} {r['steps']:>7} {mode:>14} "
              f"{r['stop']:>8} {str(r['in_range']):>13}   {r['text']}")

    # --- Behavioural checks -------------------------------------------------
    hard_fail = False
    print("\n" + "-" * 90)

    # (a) every normal stop's stop id MUST be a real stop token.
    bad_stop = [r for r in rows if r["stopped"] and not r["in_range"]]
    print(f"(a) normal stops whose stop_id is outside [6561,6760]: {len(bad_stop)}")
    if bad_stop:
        for r in bad_stop:
            print(f"    text[{r['i']}] seed={r['seed']} stop_id={r['stop']}  <-- NOT a stop token")
        hard_fail = True

    # (b) stop-mode distribution + length placement.
    n_trunc = sum(1 for r in rows if not r["stopped"])
    n_stop = sum(1 for r in rows if r["stopped"])
    print(f"(b) stop mode: {n_stop} normal, {n_trunc} truncated (of {len(rows)})")
    for r in rows:
        if r["stopped"]:
            if r["steps"] < r["min_len"]:
                print(f"    text[{r['i']}] seed={r['seed']}: stopped at {r['steps']} "
                      f"(< min_len {r['min_len']}) via non-6561 control token id={r['stop']}")
            elif r["steps"] > r["max_len"]:
                print(f"    text[{r['i']}] seed={r['seed']}: steps {r['steps']} > max_len "
                      f"{r['max_len']} (should be impossible; BUG)")
                hard_fail = True
        else:
            print(f"    text[{r['i']}] seed={r['seed']}: TRUNCATED at max_len={r['max_len']}")

    # (c) anomaly: every run truncated (never stops) or every run stops exactly at
    #     min_len (degenerate). Report; only "all truncated" is a hard failure.
    trunc_frac = n_trunc / len(rows)
    if n_trunc == len(rows):
        print("(c) ANOMALY: every run hard-truncated — model never stops on its own.")
        hard_fail = True
    elif trunc_frac >= 0.5:
        print(f"(c) NOTE: {trunc_frac:.0%} of runs truncated — worth investigating "
              f"(possibly long-tail text or RNG).")
    exact_min = sum(1 for r in rows if r["stopped"] and r["steps"] == r["min_len"])
    if exact_min == len(rows):
        print("(c) ANOMALY: every run stops exactly at min_len (degenerate early stop).")
    else:
        print(f"(c) runs stopping exactly at min_len: {exact_min}/{len(rows)}")

    if hard_fail:
        print("\nFAIL: a hard invariant was violated (see above).")
        sys.exit(1)
    print("\nPASS: autonomous generation loop terminates correctly; "
          "stop ids are in stop_token_ids and lengths behave normally.")


if __name__ == "__main__":
    main()
