#!/usr/bin/env python3
"""Verify the C++ CosyVoice3LM lm_input assembly (voice-cloning branch).

Feeds the token ids captured by tests/llm_zeroshot_reference.py (the concatenated
``text_tokens_all`` = prompt_text||text, plus ``prompt_speech_token``) through the
compiled ``larynx_llm_build_dump`` utility, which runs ``LLM::build_lm_input``, and
compares the result against the PyTorch reference embedding sequence:

  lm_input                  (1, L, 896)  [sos; text_emb; task_id; prompt_speech]
  sos_emb                   (896)        speech_embedding[6561]
  text_emb                  (PtT, 896)   embed_tokens(text_tokens_all)
  task_id_emb               (896)        speech_embedding[6563]
  prompt_speech_token_emb   (P, 896)     speech_embedding(prompt_speech_token)

The assembly is pure table lookup + concatenation (no floating-point arithmetic),
so — given the GGUF stores the same float32 weights as llm.pt — the C++ output is
expected to be **bit-exact** against the reference (max abs error == 0.0), not
merely within fp-noise. A nonzero error here is a real bug (wrong table, wrong
row, or wrong concat order), not accumulation noise.

Usage:
    python3 tests/verify_llm_build.py
    LARYNX_LLM_BUILD_DUMP=./build/larynx_llm_build_dump python3 tests/verify_llm_build.py
"""

import os
import subprocess
import sys
import tempfile

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw(path, shape):
    data = np.fromfile(path, dtype=np.float32)
    n = int(np.prod(shape))
    assert data.size == n, f"{path}: got {data.size} floats, expected {n} ({shape})"
    return data.reshape(shape)


def compare(name, cpp, ref):
    cpp = cpp.astype(np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    diff = np.abs(cpp - ref)
    max_err = float(diff.max())
    idx = np.unravel_index(int(diff.argmax()), diff.shape) if diff.size else ()
    return name, max_err, idx


def main():
    dump_bin = os.environ.get("LARYNX_LLM_BUILD_DUMP",
                              os.path.join(ROOT, "build", "larynx_llm_build_dump"))
    gguf = os.environ.get("LLM_GGUF", os.path.join(ROOT, "build", "llm.gguf"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llm_zeroshot_ref.npz")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_llm_build_dump not found at {dump_bin}; build it first")
    if not os.path.exists(gguf):
        sys.exit(f"llm.gguf not found at {gguf}; run tools/convert_weights.py --llm llm.pt --out-dir build/")
    if not os.path.exists(ref_path):
        sys.exit(f"reference not found at {ref_path}; run tests/llm_zeroshot_reference.py first")

    ref = np.load(ref_path)
    text_tokens = ref["text_tokens_all"]          # (1, PtT)
    prompt_speech_token = ref["prompt_speech_token"]  # (1, P)
    PtT = text_tokens.shape[1]
    P = prompt_speech_token.shape[1]
    L = 1 + PtT + 1 + P
    print(f"dump binary: {dump_bin}")
    print(f"llm.gguf   : {gguf}")
    print(f"reference  : {ref_path}  (PtT={PtT}, P={P}, L={L})\n")

    with tempfile.TemporaryDirectory() as tmp:
        indir = os.path.join(tmp, "in")
        outdir = os.path.join(tmp, "out")
        os.makedirs(indir)
        os.makedirs(outdir)

        text_tokens.astype(np.int32).reshape(-1).tofile(os.path.join(indir, "text_tokens.i32"))
        prompt_speech_token.astype(np.int32).reshape(-1).tofile(os.path.join(indir, "prompt_speech_token.i32"))

        env = dict(os.environ)
        if os.environ.get("LARYNX_VERIFY_BACKEND") == "cuda":
            env.pop("LARYNX_BACKEND", None)
        else:
            env["LARYNX_BACKEND"] = "cpu"
        subprocess.run([dump_bin, gguf, indir, outdir], check=True, env=env)

        # Each reference component is a slice of lm_input; compare all four plus
        # the whole sequence so a mismatch is localized to a table/row/order.
        rows = []
        rows.append(compare("lm_input", load_raw(os.path.join(outdir, "lm_input.f32"), (L, 896)),
                            ref["lm_input"].reshape(L, 896)))
        rows.append(compare("sos_emb", load_raw(os.path.join(outdir, "sos_emb.f32"), (896,)),
                            ref["sos_emb"].reshape(896)))
        rows.append(compare("text_emb", load_raw(os.path.join(outdir, "text_emb.f32"), (PtT, 896)),
                            ref["text_emb"].reshape(PtT, 896)))
        rows.append(compare("task_id_emb", load_raw(os.path.join(outdir, "task_id_emb.f32"), (896,)),
                            ref["task_id_emb"].reshape(896)))
        rows.append(compare("prompt_speech_token_emb",
                            load_raw(os.path.join(outdir, "prompt_speech_token_emb.f32"), (P, 896)),
                            ref["prompt_speech_token_emb"].reshape(P, 896)))

    print(f"{'stage':<24} {'max abs err':>14} | worst @ index")
    print("-" * 60)
    for name, max_err, idx in rows:
        print(f"{name:<24} {max_err:>14.6e} | {str(idx)}")

    worst = max(rows, key=lambda r: r[1])
    print("\n" + "-" * 60)
    print(f"worst: {worst[0]} (max abs err = {worst[1]:.6e})")
    if worst[1] != 0.0:
        print("FAIL: lm_input assembly is not bit-exact against the PyTorch reference.")
        sys.exit(1)
    print("PASS: lm_input assembly is bit-exact (embed_tokens + speech_embedding lookups + concat).")


if __name__ == "__main__":
    main()
