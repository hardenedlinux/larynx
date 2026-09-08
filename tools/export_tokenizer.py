#!/usr/bin/env python3
"""Export the CosyVoice3 text tokenizer's data for the C++ reimplementation.

The C++ tokenizer needs three data files (plus the fixed GPT-2 byte alphabet,
which is computed in C++):

  - ``vocab.tsv``       : Qwen2 base BPE vocab (151643 entries, token-string -> id).
  - ``merges.txt``      : the 134935 byte-pair merges, one "a b" per line, in
                          priority order (rank = line index).
  - ``added_tokens.tsv``: the 281 special tokens (id<TAB>content), ids 151643..151923.

This must run under the CosyVoice python3.10 environment (it imports
``transformers``/``tokenizers`` and the CosyVoice tokenizer wrapper), NOT the
repo ``.venv``. Example:

    PYTHONPATH=$HOME/Project/CosyVoice/.local/lib/python3.10/site-packages \
      ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
      tools/export_tokenizer.py --out-dir build/tokenizer
"""

import argparse
import json
import os
import shutil
import sys


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir",
                    default=os.path.expanduser(
                        "~/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"))
    ap.add_argument("--out-dir", default="build/tokenizer")
    args = ap.parse_args()

    cosyvoice_dir = os.path.expanduser("~/Project/CosyVoice")
    if cosyvoice_dir not in sys.path:
        sys.path.insert(0, cosyvoice_dir)

    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

    token_path = os.path.join(args.model_dir, "CosyVoice-BlankEN")
    tok = get_qwen_tokenizer(token_path=token_path,
                             skip_special_tokens=False,
                             version="cosyvoice3")
    bt = tok.tokenizer.backend_tokenizer
    cfg = json.loads(bt.to_str())

    os.makedirs(args.out_dir, exist_ok=True)

    # 1. vocab.tsv — "token<TAB>id" for the 151643 base BPE tokens. Byte-level
    #    tokens never contain TAB/newline (bytes are mapped into the GPT-2 byte
    #    alphabet), so TAB is an unambiguous separator and the C++ side can parse
    #    it with plain string ops (no JSON parser needed).
    vocab = cfg["model"]["vocab"]
    with open(os.path.join(args.out_dir, "vocab.tsv"), "w", encoding="utf-8") as f:
        for token, vid in sorted(vocab.items(), key=lambda kv: kv[1]):
            assert "\t" not in token and "\n" not in token, repr(token)
            f.write(f"{token}\t{vid}\n")

    # 2. merges.txt — byte-pair merges in priority order (rank = line index).
    merges = cfg["model"]["merges"]
    with open(os.path.join(args.out_dir, "merges.txt"), "w", encoding="utf-8") as f:
        for a, b in merges:
            f.write(f"{a} {b}\n")

    # 3. added_tokens.tsv — the special tokens with their fixed ids.
    added = cfg["added_tokens"]
    with open(os.path.join(args.out_dir, "added_tokens.tsv"), "w", encoding="utf-8") as f:
        for t in sorted(added, key=lambda x: x["id"]):
            assert t["special"] is True, f"non-special added token: {t}"
            f.write(f'{t["id"]}\t{t["content"]}\n')

    print(f"vocab:        {len(vocab)} entries")
    print(f"merges:       {len(merges)} entries")
    print(f"added_tokens: {len(added)} entries (ids {added[0]['id']}..{added[-1]['id']})")
    print(f"wrote -> {os.path.abspath(args.out_dir)}")


if __name__ == "__main__":
    main()
