#!/usr/bin/env python3
"""Ground-truth text-tokenizer IDs for the C++ ``Qwen2Tokenizer``.

Calls the *real* ``CosyVoice3Tokenizer.encode()`` (==
``Qwen2TokenizerFast([text])["input_ids"]``) for a fixed corpus and writes a
length-prefixed binary file that ``tests/tokenizer_dump.cpp`` reads back (the
text) and ``tests/verify_tokenizer.py`` compares against (the ids).

The corpus covers the three checkpoint-2 requirements plus edge cases:

  - the baseline prompt ending in ``<|endofprompt|>``
  - a real Pronunciation-Inpainting sentence (``[j]`` / ``[ǐ]`` bracket tokens)
  - ``<strong>…</strong>`` stress markers
  - English contractions, mixed CJK/ASCII/digits, leading/trailing whitespace,
    newlines/tabs/CRLF, chat special tokens, sound/laughter markers,
    NFC (decomposed ``e + U+0301`` -> ``é``), and overlapping special tokens
    (``[i]``/``[in]``/``[ing]``/``[ian]``) that must longest-match.

Binary format (little-endian)::

    uint32 N
    per case:
      uint32 text_len;  uint8 text[text_len]      (UTF-8)
      uint32 id_len;    int32 ids[id_len]

Runs under the CosyVoice python3.10 env, NOT the larynx ``.venv``.

Usage:
    ~/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10 \
        tests/tokenizer_reference.py [--out tests/tokenizer_cases.bin]
"""

import argparse
import os
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COSYVOICE_DIR = os.path.expanduser("~/Project/CosyVoice")
SITE_PACKAGES = os.path.join(COSYVOICE_DIR, ".local", "lib", "python3.10", "site-packages")
MODEL_DIR = os.path.join(COSYVOICE_DIR, "pretrained_models", "Fun-CosyVoice3-0.5B")

# Fixed corpus. Each string is tokenized verbatim (CosyVoice3Tokenizer does not
# add BOS/EOS; its bos_token is None and the encode() wrapper emits only the
# content ids).
CORPUS = [
    # 1. Baseline prompt (checkpoint-2 requirement).
    "请用普通话表达。<|endofprompt|>",
    # 1a. Phase-2/3 acceptance inputs (tests/acceptance_wavs.py --text and
    #     --instruct defaults): the exact strings the end-to-end gate feeds the
    #     tokenizer.
    "今天天气不错，我们一起去公园散步吧。",
    "You are a helpful assistant. 请用普通话表达。<|endofprompt|>",
    # 2. Real Pronunciation-Inpainting sentence (example.py line 94).
    "高管也通过电话、短信、微信等方式对报道[j][ǐ]予好评。",
    # 2b. The PI bracket tokens in isolation (minimal repro).
    "报道[j][ǐ]予好评",
    # 3. <strong> stress markers (example.py line 31).
    "在面对挑战时，他展现了非凡的<strong>勇气</strong>与<strong>智慧</strong>。",
    # 4. English contractions (case-insensitive branch of the GPT-2 regex).
    "I'm can't you've don't he'll she'd it's",
    # 5. Mixed CJK + ASCII + digits + punctuation.
    "Hello world 你好世界 price 3.14 元，折扣 20%！",
    # 6. Leading / trailing whitespace (ByteLevel add_prefix_space=false).
    "  leading and trailing  ",
    # 7. Newlines, tabs, CRLF.
    "line1\nline2\tindented\r\nnext",
    # 8. Chat special tokens.
    "<|im_start|>user<|im_end|>",
    # 9. Sound / laughter markers.
    "[laughter][breath][cough]<laughter>哈哈</laughter>",
    # 10. NFC: decomposed e + combining acute normalizes to é (U+00E9).
    "café = café",
    # 11. Overlapping special tokens: longest match must win.
    "[i][in][ing][ian][u][un][uo][uang]",
    # 12. Tone-marked pinyin bracket tokens.
    "[uǐ][iǎng][ǚ][ǜ][iè]",
    # 13. Whitespace stress: multiple spaces / tabs / newlines / CRLF.
    "a  b   c\t d\n\n e\r\nf",
    # 14. Runs of whitespace only.
    "   ",
    # 15. Contractions vs lone apostrophes.
    "don't isn't it's 'twas o'clock",
    # 16. Case-insensitive contractions (mixed case).
    "IT'S it'S It's DON'T don'T YOU'RE you'RE",
    # 17. Unicode whitespace: ideographic space U+3000 + NBSP U+00A0.
    "全角　空格　半角 空格",
    # 18. Special tokens adjacent to CJK with no separator.
    "你好[breath]世界[mn]<laughter>哈哈",
    # 19. Special-token prefixes without the closing bracket (must NOT match).
    "<|im_start <|endofprompt|> [i [ǐ",
    # 20. Numbers and CJK punctuation.
    "第2章 3.14 1,000 2024年 100%",
    # 21. Decomposed vs precomposed (both must NFC to the same token).
    "é é",
]


def _bootstrap():
    for p in (SITE_PACKAGES, COSYVOICE_DIR):
        if p and p not in sys.path:
            sys.path.insert(0, p)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out",
                    default=os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         "tokenizer_cases.bin"))
    args = ap.parse_args()

    _bootstrap()

    from cosyvoice.tokenizer.tokenizer import get_qwen_tokenizer

    tok = get_qwen_tokenizer(token_path=os.path.join(MODEL_DIR, "CosyVoice-BlankEN"),
                             skip_special_tokens=False,
                             version="cosyvoice3")

    with open(args.out, "wb") as f:
        f.write(struct.pack("<I", len(CORPUS)))
        for text in CORPUS:
            ids = tok.encode(text)
            assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
            data = text.encode("utf-8")
            f.write(struct.pack("<I", len(data)))
            f.write(data)
            f.write(struct.pack("<I", len(ids)))
            f.write(struct.pack("<%di" % len(ids), *ids))
            print(f"{len(ids):4d} ids  {text!r}")

    print(f"wrote {args.out} ({len(CORPUS)} cases)")


if __name__ == "__main__":
    main()
