#!/usr/bin/env python3
"""Convert CosyVoice3 PyTorch checkpoints to GGUF (GGML-loadable) files.

Reads the offline ``flow.pt`` / ``hift.pt`` / ``llm.pt`` (or ``llm.rl.pt``)
checkpoints and rewrites each one as a GGUF v3 file whose tensors are byte-for-
byte the PyTorch state_dict entries (no quantization, no inference). This is an
offline build/prep tool (docs/ARCHITECTURE.md section 5); it never ships in the
deployed binary and does not reintroduce a runtime Python dependency.

GGUF tensor naming is the identity of the state_dict key (optionally with a
prefix stripped). The dimension list stored in GGUF is the reversed PyTorch
shape (GGML's `ne`, fastest-varying dimension first); the raw data is written in
PyTorch's contiguous (row-major) order, which is GGML's natural layout. See
docs/WEIGHT_FORMAT.md for the full field-by-field mapping.

Usage:
    python3 tools/convert_weights.py --flow flow.pt --hift hift.pt \
        --llm llm.pt --out-dir ggml/
    python3 tools/convert_weights.py --llm llm.rl.pt --out-dir ggml/

Options:
    --f16          store float32 weights as float16 (optional precision
                   reduction; NOT a quantization scheme). Default: preserve.
    --strip-prefix remove this string prefix from every state_dict key.
"""

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gguf  # noqa: E402

_TORCH_TO_GGML = {
    torch.float32: gguf.GGML_TYPE_F32,
    torch.float16: gguf.GGML_TYPE_F16,
    torch.bfloat16: gguf.GGML_TYPE_BF16,
    torch.float64: gguf.GGML_TYPE_F64,
    torch.int8: gguf.GGML_TYPE_I8,
    torch.int16: gguf.GGML_TYPE_I16,
    torch.int32: gguf.GGML_TYPE_I32,
    torch.int64: gguf.GGML_TYPE_I64,
}

# Per-checkpoint architecture tag (informational metadata only).
_ARCH = {
    "llm": "cosyvoice3.llm",
    "flow": "cosyvoice3.flow",
    "hift": "cosyvoice3.hift",
}


def extract_state_dict(obj):
    """Unwrap common checkpoint wrappers and return a flat name->Tensor dict."""
    if isinstance(obj, dict):
        # A plain state_dict: every value is a Tensor / Parameter.
        if obj and all(torch.is_tensor(v) for v in obj.values()):
            return dict(obj)
        # Nested wrappers sometimes used by training frameworks.
        for key in ("state_dict", "model", "module", "model_state_dict"):
            if isinstance(obj.get(key), dict):
                return extract_state_dict(obj[key])
        # A combined checkpoint ({'flow': {...}, 'hift': {...}}) — only reached
        # if the caller pointed us at one; report clearly.
        sub = {k: v for k, v in obj.items() if isinstance(v, dict)}
        raise ValueError(
            f"checkpoint has multiple nested dicts ({sorted(sub)}); pass each "
            "component checkpoint separately (--flow / --hift / --llm)")
    raise ValueError(f"unexpected checkpoint type {type(obj)} (expected a state_dict)")


def ne_from_shape(shape):
    """GGML dimension list = reversed PyTorch shape, trailing 1s trimmed
    (mirrors ggml_n_dims semantics)."""
    ne = list(reversed(shape))
    while len(ne) > 1 and ne[-1] == 1:
        ne.pop()
    return ne


def tensor_to_bytes(t):
    t = t.detach().cpu()
    if t.dtype == torch.bfloat16:
        # numpy has no bfloat16; reinterpret as the raw 16-bit pattern.
        return t.contiguous().view(torch.uint16).numpy().tobytes()
    return t.contiguous().numpy().tobytes()


def merge_weight_norm(state_dict):
    """Fold torch ``weight_norm`` parametrizations into a single effective weight.

    ``torch.nn.utils.parametrizations.weight_norm`` stores two parameters per
    wrapped module — ``.parametrizations.weight.original0`` (the magnitude ``g``,
    shape ``(OC, 1, 1)``) and ``.parametrizations.weight.original1`` (the
    direction ``v``, shape ``(OC, IC, K)``). The effective weight the forward
    pass actually uses is ``w = v * g / ||v||_2`` (norm over every non-output
    dimension), so we bake that in here and drop the parametrization entries,
    yielding a plain ``<module>.weight`` (what ``remove_weight_norm`` would
    produce). Checkpoints without parametrizations (flow/llm) pass through
    unchanged.
    """
    g_suffix = ".parametrizations.weight.original0"
    v_suffix = ".parametrizations.weight.original1"

    prefixes = set()
    for name in state_dict:
        if name.endswith(g_suffix):
            prefixes.add(name[: -len(g_suffix)])

    if not prefixes:
        return state_dict

    out = {name: t for name, t in state_dict.items()
           if not name.endswith(g_suffix) and not name.endswith(v_suffix)}
    for prefix in prefixes:
        g = state_dict[prefix + g_suffix]   # (OC, 1, 1)
        v = state_dict[prefix + v_suffix]   # (OC, IC, K)
        norm = torch.sqrt(torch.sum(v * v, dim=tuple(range(1, v.dim())), keepdim=True))
        out[prefix + ".weight"] = (v * (g / norm)).contiguous()
    return out


def convert_one(checkpoint_path, arch, out_path, use_f16, strip_prefix):
    print(f"  {checkpoint_path} -> {out_path}")
    obj = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = merge_weight_norm(extract_state_dict(obj))

    tensors = []
    n_f16 = 0
    n_total = 0
    for name, t in state_dict.items():
        if not torch.is_tensor(t):
            continue
        if t.requires_grad:
            t = t.detach()
        ggml_type = _TORCH_TO_GGML[t.dtype]
        if use_f16 and t.dtype == torch.float32:
            t = t.half()
            ggml_type = gguf.GGML_TYPE_F16
            n_f16 += 1
        gguf_name = name[len(strip_prefix):] if strip_prefix and name.startswith(strip_prefix) else name
        ne = ne_from_shape(tuple(t.shape))
        data = tensor_to_bytes(t)
        tensors.append((gguf_name, ggml_type, ne, data))
        n_total += 1

    file_type = 1 if any(g == gguf.GGML_TYPE_F16 for _, g, _, _ in tensors) else 0
    metadata = [
        ("general.architecture", gguf.GGUF_TYPE_STRING, arch),
        ("general.name", gguf.GGUF_TYPE_STRING, os.path.basename(checkpoint_path)),
        ("general.file_type", gguf.GGUF_TYPE_UINT32, file_type),
    ]

    gguf.write_gguf(out_path, tensors, metadata)
    print(f"    wrote {n_total} tensors"
          + (f" ({n_f16} cast to f16)" if n_f16 else "")
          + f", file_type={file_type}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--llm", help="path to llm.pt or llm.rl.pt")
    p.add_argument("--flow", help="path to flow.pt")
    p.add_argument("--hift", help="path to hift.pt")
    p.add_argument("--out-dir", required=True, help="output directory for .gguf files")
    p.add_argument("--f16", action="store_true",
                   help="store float32 weights as float16 (precision reduction, not quantization)")
    p.add_argument("--strip-prefix", default="",
                   help="strip this prefix from every tensor name")
    args = p.parse_args()

    jobs = [("llm", args.llm), ("flow", args.flow), ("hift", args.hift)]
    jobs = [(k, v) for k, v in jobs if v]
    if not jobs:
        p.error("at least one of --llm/--flow/--hift is required")

    os.makedirs(args.out_dir, exist_ok=True)
    for kind, path in jobs:
        out_path = os.path.join(args.out_dir, kind + ".gguf")
        convert_one(path, _ARCH[kind], out_path, args.f16, args.strip_prefix)
    print("done.")


if __name__ == "__main__":
    main()
