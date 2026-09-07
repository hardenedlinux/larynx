# ADR-0002: Disable GGML CUDA TF32 tensor-op math via vendor patch + static link

**Status:** Accepted
**Context date:** 2026-09-08

## Decision

We patch the vendored GGML checkout so its CUDA backend uses
`CUBLAS_DEFAULT_MATH` instead of `CUBLAS_TF32_TENSOR_OP_MATH` for cuBLAS GEMMs,
and we static-link GGML into `larynx` and its tools. The patch is committed as
`patches/0001-ggml-cuda-disable-tf32-tensor-op-math.patch`, applied idempotently
by CMake at configure time, and GGML is built as static archives
(`BUILD_SHARED_LIBS=OFF`). We do **not** rely on `ggml_mul_mat_set_prec`,
which is a no-op for this problem.

## Context

The Flow decoder and HiFT vocoder were validated on CPU at ~1e-4 relative error
against their PyTorch F32 references. When the CUDA backend was enabled, the
same verification regressed by two orders of magnitude:

| Stage | CPU (rel err) | CUDA, TF32 on (rel err) | CUDA, TF32 off (rel err) |
|---|---|---|---|
| Flow `norm_out` | 7.3e-4 | 3.230e-2 | 5.326e-4 |
| HiFT `speech`/`sine_wavs` | ~4.7e-4 | 1.168e-2 | 4.718e-4 |

The root cause is TF32 tensor cores. GGML's CUDA backend sets
`cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH)` at the cuBLAS handle
level (`third_party/ggml/src/ggml-cuda/common.cuh:1505`). This applies to **all**
F32 cuBLAS GEMMs on that handle — including `ggml_mul_mat` over our F32 weights —
reducing the mantissa from 24 to 10 bits and producing the ~3e-2 relative error
above. It is a precision floor, not a bug in our graph: flipping that one
constant to `CUBLAS_DEFAULT_MATH` (full IEEE F32) collapsed both stages back to
the CPU-level ~1e-4, confirming TF32 as the sole cause.

## Why `ggml_mul_mat_set_prec` does not fix this

The obvious public-API fix is `ggml_mul_mat_set_prec(t, GGML_PREC_F32)`. It does
not help here, because it never reaches the cuBLAS math mode:

- `ggml_mul_mat_set_prec` only stores `GGML_PREC_F32` into `dst->op_params[0]`.
- In `ggml-cuda.cu` (~1632) that op-param is consulted only to pick a
  `compute_type`:
  ```cpp
  ggml_type compute_type = src0->type;   // = GGML_TYPE_F32 for our weights
  ...
  if (dst->op_params[0] == GGML_PREC_F32) { compute_type = GGML_TYPE_F32; }
  ```
  Our weights are already F32, so this branch is a no-op — it does not change
  the cuBLAS math mode, which is a property of the *handle*, set once in
  `common.cuh` and not exposed through the GGML graph API.

We confirmed this empirically: adding `ggml_mul_mat_set_prec(..., GGML_PREC_F32)`
at every `ggml_mul_mat` call site produced byte-identical output to the
TF32-on baseline. The 5 placeholder calls were removed and the approach
abandoned.

## Why vendor patch + static link

Two concerns are addressed together:

- **Correctness/isolation.** The only way to flip the math mode is to change
  the vendored source line. Because we keep that change as a versioned patch
  applied at build time, the vendored checkout in the working tree stays a
  faithful copy of the pinned upstream commit until configure runs, and no
  stray `libggml-cuda.so` is ever dropped onto a system path — the fix travels
  inside the `larynx` binary itself.
- **Maintainability.** Patches live in one directory (`patches/`), are applied
  idempotently (skip if already applied), and **fail loudly** — never silently —
  if upstream moves the target code so the patch no longer applies
  (message: "patch 应用失败，可能是 GGML 版本升级导致代码变化，需要手动检查更新 patch 内容").
  On a future GGML submodule bump, the build stops and forces a human to
  re-examine the patch rather than quietly regressing to TF32.

Static linking (`BUILD_SHARED_LIBS=OFF`) is the companion to the patch: it means
the only way to run the wrong (TF32) code is to rebuild from a GGML checkout
that both reverted the patch *and* was re-linked — there is no shared library a
user could substitute at runtime. This is distinct from `GGML_STATIC`, which adds
a full `-static` system link and is deliberately **not** used.

## Consequences

- `ldd build/larynx` shows no `libggml-base.so` / `libggml-cpu.so` /
  `libggml-cuda.so`; NVIDIA's `libcuda.so` / `libcudart.so` / `libcublas.so`
  remain dynamically linked (unavoidable, and out of scope).
- The `patches/` directory and the CMake apply-step are part of the build
  contract; any future GGML upgrade must re-validate `0001`.
- CUDA numerical parity now matches the CPU baseline (~1e-4), restoring the
  stagewise verification methodology from ADR-0001 as the acceptance gate for
  both backends.
