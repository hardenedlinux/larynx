# ADR-0001: Drop ONNX Runtime for compute; use GGML + CUDA instead

**Status:** Accepted
**Context date:** 2026-09-05

## Decision

The Flow decoder, HiFT vocoder, and LLM backbone will be implemented using
GGML with its CUDA backend, calling into GGML's own kernels rather than
ONNX Runtime. ONNX Runtime C++ is retained only for the two small,
pre-shipped audio frontend networks (`campplus.onnx`,
`speech_tokenizer_v3.onnx`), which showed no numerical issues during
prototyping.

## Context

A working ONNX export of the Flow decoder estimator was produced using the
official `cosyvoice/bin/export_onnx.py` script. Stagewise numerical
comparison against the PyTorch eager-mode reference (same seq_len, same
seed) showed:

- Blocks 0-2: normal FP32 noise (~0.07% relative error at large-magnitude
  positions).
- Block 3 onward: token-localized errors growing from ~1-16% relative error
  to full sign flips and order-of-magnitude divergence by blocks 17-19.
- Aggregate max/mean error at the final output stayed misleadingly small
  (~0.16 absolute, ~0.4% relative) because residual connections diluted the
  localized errors  this was only visible via per-position, per-block
  diagnosis, not via aggregate metrics.

Two hypotheses were investigated:

1. **Chunk mask tracing artifact**  `cosyvoice/utils/mask.py`'s
   `add_optional_chunk_mask` used `.sum().item() != 0` to conditionally
   repair all-false mask rows; this Python-level branch gets frozen at
   trace time (`TracerWarning` observed). Patched to use tensor-level
   `torch.where` (later `Or`/`Not`, since this ONNX Runtime build had no
   registered kernel for `Where` on bool tensors). **Confirmed necessary
   but not sufficient**: re-running the identical diagnostic
   (seq_len=123, seed=0) after the patch showed the previously-recorded
   anomalies (block3 (0,93,411), block17 (1,0,297), block18 (0,22,674),
   block19 (0,41,749)) essentially unchanged, meaning this test case never
   triggered the branch this patch fixed.
2. **Attention backend / precision divergence** (not yet empirically
   confirmed, investigation stopped here)  suspected that PyTorch eager
   SDPA dispatches to a fused kernel (flash attention or similar) while
   `torch.onnx.export` tracing decomposes SDPA into explicit
   matmulsoftmaxmatmul subgraph nodes, and/or that PyTorch's default TF32
   matmul (enabled by default on Ampere+ GPUs) diverges from ONNX Runtime's
   CUDA EP default (full FP32). Either would explain the block-by-block
   growth pattern via nonlinear (softmax) amplification of an initially
   tiny divergence.

## Why the project didn't chase hypothesis 2 to ground

Two independent points converged during discussion, before further
debugging time was spent:

- If the true cause is attention-backend/precision divergence between two
  independently-implemented runtimes, that is not a bug fixable by patching
  CosyVoice source  it's an inherent cost of using two different numerical
  engines for the same computation.
- A pre-existing, GNU-style architectural principle (C core with
  language-independent bindings, e.g. Emacs/Elisp, GDB/Python) applies here
  more directly than initially framed: PyTorch's own core (ATen, autograd)
  is already substantially language-independent  LibTorch proves this by
  running TorchScript models in pure C++ with zero Python runtime
  dependency. The real fix for "two implementations disagree" is not
  picking a third implementation (ONNX Runtime) and debugging the diff 
  it's using the *same* implementation end-to-end. GGML, driven directly
  (not through llama.cpp's higher-level, text-LLM-shaped assumptions),
  gives that: one CUDA-backed tensor library, one code path, nothing to
  diverge from ONNX Runtime's kernels because ONNX Runtime is no longer in
  the loop for compute.

## Consequences

- The ONNX export/verification/diagnostic scripts and the `mask.py` patch
  produced during the ONNX Runtime prototype are retained only as reference
  material (they document real, confirmed source-level facts about
  CosyVoice3  architecture, tensor shapes, batch/CFG semantics  that
  remain valid regardless of runtime choice). They are not part of the
  Larynx build.
- The LLM backbone's custom heads (`speech_embedding`, `llm_decoder`) rule
  out llama.cpp as a shortcut (see main ARCHITECTURE.md 1); GGML is used
  directly.
- Numerical validation methodology (stagewise, per-position, not just
  aggregate) carries forward unchanged  it is what surfaced this problem
  in the first place and remains the right way to validate the GGML
  implementation too.
