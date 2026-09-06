# GGUF weight format

CosyVoice3 checkpoints are converted offline (Python, build/prep time only) into
GGUF v3 files that the GGML runtime loads directly. This document specifies the
exact on-disk layout and the mapping from a PyTorch `state_dict` to GGUF
tensors. The writer/reader is `tools/gguf.py`; the checkpoint converter is
`tools/convert_weights.py`.

## 1. File layout (GGUF v3, little-endian)

The layout below is transcribed from ggml's `include/gguf.h` and
`src/gguf.cpp::gguf_write_to_file`, and the writer is validated by a byte-for-byte
round-trip test (`tests/test_convert.py`). All integers are little-endian.

```
offset 0    "GGUF"                          (4 bytes, magic)
            version                         (u32 = 3)
            n_tensors                       (i64)
            n_kv                            (i64)

    -- metadata KV section, n_kv entries --
    per KV:  key                            (string: u64 length + UTF-8 bytes)
             type                           (i32, gguf_type)
             value                          (scalar, or array below)

    -- tensor info section, n_tensors entries --
    per tensor:
             name                           (string: u64 length + UTF-8 bytes)
             n_dims                         (u32)
             dims                           (i64 × n_dims, the `ne` list)
             type                           (i32, ggml_type)
             offset                         (u64, bytes from start of data blob)

    [zero padding to `alignment`]

    -- data blob --
    tensor 0 bytes (padded to `alignment`)
    tensor 1 bytes (padded to `alignment`)
    ...
```

- **Alignment** defaults to 32 bytes and is honoured by padding the end of the
  metadata section and the end of every tensor. (ggml reads `general.alignment`
  from the KV list when present; we do not emit it and rely on the 32-byte
  default.)
- **Tensor offsets** are relative to the start of the *data blob*, not to the
  start of the file.
- **Strings** are a `u64` byte-length followed by the raw UTF-8 bytes, no NUL
  terminator.

## 2. Type enums

`gguf_type` (metadata value types), from `gguf.h`:

| value | type |
|---|---|
| 0 | uint8 |
| 1 | int8 |
| 2 | uint16 |
| 3 | int16 |
| 4 | uint32 |
| 5 | int32 |
| 6 | float32 |
| 7 | bool (stored as i8) |
| 8 | string |
| 9 | array (`elem_type:i32`, `count:u64`, then elements) |
| 10 | uint64 |
| 11 | int64 |
| 12 | float64 |

`ggml_type` (tensor data types), from `ggml.h` — the subset relevant here:

| value | type | bytes |
|---|---|---|
| 0 | F32 | 4 |
| 1 | F16 | 2 |
| 24 | I8 | 1 |
| 25 | I16 | 2 |
| 26 | I32 | 4 |
| 27 | I64 | 8 |
| 28 | F64 | 8 |
| 30 | BF16 | 2 |

## 3. Tensor organisation

For each `state_dict` entry, one GGUF tensor is written with:

- **`name`** = the state_dict key, verbatim (optionally with `--strip-prefix`
  removed). The converter deliberately keeps the PyTorch keys — e.g.
  `model.layers.0.self_attn.q_proj.weight` — so the later GGML modeling code
  resolves each key unambiguously; there is no rename table to drift.
- **`dims` / `ne`** = the PyTorch shape **reversed**, with trailing `1`s trimmed
  (mirroring `ggml_n_dims`). PyTorch shapes are row-major with the *last* dim
  contiguous, which is exactly GGML's "fastest-varying dimension first" `ne`
  ordering — so reversing is correct and the data needs no transpose.

  Examples:

  | PyTorch shape | `ne` stored |
  |---|---|
  | `[out, in]` (Linear weight) | `[in, out]` |
  | `[out]` (bias) | `[out]` |
  | `[vocab, hidden]` (embedding) | `[hidden, vocab]` |
  | `[out_c, in_c, kh, kw]` (conv) | `[kw, kh, in_c, out_c]` |
  | `[seq, batch, hidden]` | `[hidden, batch, seq]` |

- **`type`** = the dtype mapping table below.
- **`data`** = the tensor's contiguous bytes in PyTorch (row-major) order,
  written as-is. BF16 is reinterpreted as `uint16` (numpy has no bf16) so the
  16-bit pattern is preserved exactly.

### dtype mapping

| PyTorch dtype | ggml_type |
|---|---|
| float32 | F32 (0) |
| float16 | F16 (1) |
| bfloat16 | BF16 (30) |
| float64 | F64 (28) |
| int8 | I8 (24) |
| int16 | I16 (25) |
| int32 | I32 (26) |
| int64 | I64 (27) |

### What the converter does *not* do

- No quantization. `--f16` is an optional float32→float16 *precision* cast only;
  the default preserves every dtype bit-for-bit.
- No numerical transformation of any kind (no transposes, reshapes, merges,
  splits, or de-quantization). A round-trip through GGUF yields the identical
  state_dict bytes.

## 4. Metadata

Each file carries three KV entries for identification only:

| key | type | value |
|---|---|---|
| `general.architecture` | string | `cosyvoice3.llm` / `.flow` / `.hift` |
| `general.name` | string | source checkpoint filename |
| `general.file_type` | uint32 | `0` = all-original, `1` = contains f16 |

## 5. Checkpoint components

| component | source | architecture tag | consumer (future phase) |
|---|---|---|---|
| LLM | `llm.pt` / `llm.rl.pt` | `cosyvoice3.llm` | Qwen2 autoregressive decoder |
| Flow | `flow.pt` | `cosyvoice3.flow` | DiT flow-matching estimator |
| HiFT | `hift.pt` | `cosyvoice3.hift` | HiFi-GAN-style vocoder |

Note the LLM checkpoint keys may live under a prefix (e.g. a wrapped
`state_dict`); `extract_state_dict()` unwraps `state_dict`/`model`/`module`
wrappers automatically, and `--strip-prefix` handles any remaining namespace.

## 6. External validation

The round-trip test (`tests/test_convert.py`) proves the writer and reader in
`tools/gguf.py` are self-consistent, but both could share the same wrong
assumption. To close that gap, the format was validated against the **real GGML
library's** loader.

**Method.** A throwaway checkout of ggml was built and `tests/gguf_cross_check.cpp`
was linked against `libggml-base.so`. The program calls the official
`gguf_init_from_file()` on a `.gguf` produced by `tools/convert_weights.py` and,
for every tensor, reports the name, `ggml_type`, `ne` dimension list, byte size,
and a byte-level FNV-1a 64-bit checksum of the raw data read back from the file.
`tests/cross_check.py` generates the source `state_dict`, converts it, runs the
C++ binary, and compares each field against the source tensor.

**ggml used.** commit `e91ded1` (`e91ded11bdcd78c42f9c8d3978ff6686eb4c1226`),
"ggml : bump version to 0.23.0 (#1618)", version `0.23.0`.

**Result.** PASS. `gguf_init_from_file` returned a valid context (no error); the
reported tensor count (9) matched the source; and every tensor's name (and
order), type, `ne` (= reversed PyTorch shape), byte size, and FNV-1a checksum
matched the source `state_dict` exactly. This includes the implicit checks the
round-trip test cannot exercise:

- ggml's reader **asserts each tensor's on-disk offset equals the running padded
  sum** of the preceding tensors' sizes; loading succeeded, so the
  alignment/offset arithmetic (alignment 32, offsets relative to the data blob,
  each tensor padded) is byte-correct.
- `ne` ordering and the dtype→`ggml_type` mapping are confirmed against ggml's
  own `ggml_type_name()` / `gguf_get_tensor_ne()`.

**Caveat (reported as-is).** The real `flow.pt` / `hift.pt` / `llm.pt` checkpoints
were not present in the repo, so the validation used a synthetic but
representative `state_dict` covering the dtypes and shapes the converter is
designed for — f32/f16/bf16/f64/i32, 1-D through 4-D (including a 4-D conv
weight and a 3-D tensor). The GGUF container treats all dtypes and dimensions
uniformly, so this exercises the full read/write path; no format issue was
found to report or fix.

**Reproduce** (from the repo root):

```sh
git clone --depth 1 https://github.com/ggml-org/ggml.git third_party/ggml-verify
cmake -S third_party/ggml-verify -B third_party/ggml-verify/build \
      -DGGML_BUILD_EXAMPLES=OFF -DGGML_BUILD_TESTS=OFF
cmake --build third_party/ggml-verify/build --target ggml -j
.venv/bin/python tests/cross_check.py
```

`third_party/ggml-verify/` is git-ignored: it is a one-off validation build, kept
separate from the GGML dependency the Phase 2 modeling code will actually link.
