#!/usr/bin/env python3
"""Bit-exact comparison of the C++ Qwen2Tokenizer against CosyVoice3Tokenizer.

Loads ``tests/tokenizer_cases.bin`` (produced by tests/tokenizer_reference.py:
text + ground-truth ids), feeds the text through the compiled
``larynx_tokenizer_dump`` utility, and asserts every case is *identical* — this
is a tokenizer, so there is no floating-point tolerance: any id difference in
any position is a FAIL.

Usage:
    python3 tests/verify_tokenizer.py            # uses build/larynx_tokenizer_dump
    LARYNX_TOKENIZER_DUMP=./build/larynx_tokenizer_dump python3 tests/verify_tokenizer.py
"""

import os
import struct
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def read_cases(path):
    cases = []
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        for _ in range(n):
            tl = struct.unpack("<I", f.read(4))[0]
            text = f.read(tl).decode("utf-8")
            il = struct.unpack("<I", f.read(4))[0]
            ids = list(struct.unpack("<%di" % il, f.read(4 * il)))
            cases.append((text, ids))
    return cases


def read_cpp(path):
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        ids_list = []
        for _ in range(n):
            il = struct.unpack("<I", f.read(4))[0]
            ids_list.append(list(struct.unpack("<%di" % il, f.read(4 * il))))
    return ids_list


def main():
    dump_bin = os.environ.get("LARYNX_TOKENIZER_DUMP",
                              os.path.join(ROOT, "build", "larynx_tokenizer_dump"))
    data_dir = os.path.join(ROOT, "build", "tokenizer")
    cases_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "tokenizer_cases.bin")

    if not os.path.exists(dump_bin):
        sys.exit(f"larynx_tokenizer_dump not found at {dump_bin}; build it first "
                 f"(cmake --build build)")
    if not os.path.exists(os.path.join(data_dir, "vocab.tsv")):
        sys.exit(f"tokenizer data not found in {data_dir}; run tools/export_tokenizer.py")
    if not os.path.exists(cases_path):
        sys.exit(f"reference not found at {cases_path}; run tests/tokenizer_reference.py")

    ref = read_cases(cases_path)

    with tempfile.TemporaryDirectory() as tmp:
        out_bin = os.path.join(tmp, "cpp.bin")
        subprocess.run([dump_bin, data_dir, cases_path, out_bin], check=True)
        cpp = read_cpp(out_bin)

    n_fail = 0
    for i, ((text, ref_ids), cpp_ids) in enumerate(zip(ref, cpp)):
        if ref_ids == cpp_ids:
            print(f"case {i:2d}: PASS  ({len(ref_ids)} tokens)  {text!r}")
            continue
        n_fail += 1
        print(f"case {i:2d}: FAIL  text={text!r}")
        print(f"   ref {len(ref_ids):3d}: {ref_ids}")
        print(f"   cpp {len(cpp_ids):3d}: {cpp_ids}")
        m = min(len(ref_ids), len(cpp_ids))
        for j in range(m):
            if ref_ids[j] != cpp_ids[j]:
                print(f"   first diff @ {j}: ref={ref_ids[j]}  cpp={cpp_ids[j]}")
                break

    print("-" * 70)
    total = len(ref)
    print(f"{total - n_fail}/{total} cases bit-exact")
    if n_fail:
        print("FAIL: tokenizer is not bit-exact vs CosyVoice3Tokenizer.")
        sys.exit(1)
    print("PASS: all cases bit-exact vs CosyVoice3Tokenizer.")


if __name__ == "__main__":
    main()
