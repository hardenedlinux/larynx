#!/usr/bin/env python3
"""Round-trip test for the GGUF writer + checkpoint converter.

Builds a synthetic state_dict (matmul weight, bias, conv weight, an embedding,
and a bf16 tensor), converts it via tools/convert_weights.py, then reads the
GGUF back with tools/gguf.py and asserts names, GGML types, dimensions (reversed
shape) and raw bytes are all preserved. Requires torch.

Usage: python3 tests/test_convert.py
"""

import os
import sys
import tempfile

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import convert_weights  # noqa: E402
import gguf  # noqa: E402


def main():
    rng = torch.Generator().manual_seed(0)
    state_dict = {
        "model.layers.0.self_attn.q_proj.weight": torch.randn(16, 8, generator=rng),
        "model.layers.0.self_attn.q_proj.bias": torch.randn(16, generator=rng),
        "model.embed_tokens.weight": torch.randn(64, 8, generator=rng),
        "flow.estimator.blocks.0.conv.weight": torch.randn(4, 4, 3, 3, generator=rng),
        "hift.bias": torch.randn(10, generator=rng),
        "some.bfloat16.weight": torch.randn(4, 4, generator=rng).to(torch.bfloat16),
    }

    with tempfile.TemporaryDirectory() as tmp:
        ckpt = os.path.join(tmp, "flow.pt")
        torch.save(state_dict, ckpt)
        out = os.path.join(tmp, "flow.gguf")
        convert_weights.convert_one(ckpt, "cosyvoice3.flow", out, use_f16=False, strip_prefix="")

        metadata, tensors = gguf.read_gguf(out)
        got = {name: (gtype, ne, data) for name, gtype, ne, data in tensors}

        assert len(got) == len(state_dict), f"{len(got)} tensors != {len(state_dict)}"

        for key, t in state_dict.items():
            assert key in got, f"missing tensor {key}"
            gtype, ne, data = got[key]
            expected_ne = convert_weights.ne_from_shape(tuple(t.shape))
            assert ne == expected_ne, f"{key}: ne {ne} != {expected_ne}"

            if t.dtype == torch.bfloat16:
                expected_bytes = t.detach().cpu().contiguous().view(torch.uint16).numpy().tobytes()
                assert gtype == gguf.GGML_TYPE_BF16, f"{key}: type {gtype} != BF16"
            else:
                expected_bytes = t.detach().cpu().contiguous().numpy().tobytes()
                assert gtype == convert_weights._TORCH_TO_GGML[t.dtype]
            assert data == expected_bytes, f"{key}: data mismatch"

        # metadata sanity
        md = dict((k, v) for k, t, v in metadata)
        assert md["general.architecture"] == "cosyvoice3.flow"
        assert md["general.file_type"] == 0

        print(f"round-trip OK: {len(state_dict)} tensors preserved byte-for-byte")
        for name, (gtype, ne, _data) in sorted(got.items()):
            print(f"  {name:<45} type={gtype:>2} ne={ne}")


if __name__ == "__main__":
    main()
