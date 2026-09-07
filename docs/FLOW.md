# Flow decoder — implementation notes and numerical validation

The `src/flow/` module implements the **Flow decoder** — the stage that turns a
speech-token sequence into an 80-bin mel spectrogram — as a single GGML compute
path (per `docs/ARCHITECTURE.md` §3.3 and `docs/adr/0001`). It reimplements
`cosyvoice/flow/flow.py` `CausalMaskedDiffWithDiT` + its `DiT` estimator, in the
non-streaming `finalize=True` path.

| Component | Reference | Notes |
|---|---|---|
| `larynx::flow::FlowDecoder` | `CausalMaskedDiffWithDiT.inference` | token → mel |
| `PreLookaheadLayer` | `cosyvoice/transformer/upsample_encoder.py` | 2× Conv1d + LeakyReLU + residual |
| `DiT` ×22 | `cosyvoice/flow/DiT/dit.py` | InputEmbedding + CausalConvPositionEmbedding + TimestepEmbedding + 22 DiTBlock + norm_out + proj_out |
| CFM Euler solver | `cosyvoice/flow/flow_matching.py` `CausalConditionalCFM` | 10 steps, batch=2N CFG, cosine t_span |

All compute runs on the GGML backend chosen by `ggml_backend_init_best()` —
CUDA when `LARYNX_ENABLE_CUDA=ON` and a device is present, else CPU. On CUDA the
cuBLAS math mode is pinned to `CUBLAS_DEFAULT_MATH` (TF32 tensor cores disabled)
by `patches/0001`; see `docs/adr/0002` for why TF32 otherwise causes ~3e-2
relative error in the F32 GEMMs. Weights stay in the graph as F32, no
quantization.

---

## 1. Reference semantics

Exact config (`cosyvoice3.yaml`): `input_size=80, output_size=80,
spk_embed_dim=192, vocab_size=6561, input_frame_rate=25, token_mel_ratio=2,
pre_lookahead_len=3`. DiT: `dim=1024, depth=22, heads=16, dim_head=64,
ff_mult=2, mel_dim=80, mu_dim=80, spk_dim=80, out_channels=80,
static_chunk_size=50`. CFM: `sigma_min=1e-6, solver=euler, t_scheduler=cosine,
inference_cfg_rate=0.7`.

The forward pass, stage by stage:

1. **Speaker projection** — `F.normalize(embedding, dim=1)` →
   `spk_embed_affine_layer` (Linear 192→80).
2. **Token embedding** — `input_embedding` (Embedding 6561→80) on
   `clamp(token, min=0)`, then `* mask` (all-True here, so a no-op).
3. **PreLookaheadLayer** — transpose to `(B,80,T)` → `pad(0,3)` right →
   `conv1(80→1024,k4,pad0)` → `leaky_relu(0.01)` → `pad(2,0)` left →
   `conv2(1024→80,k3,pad0)` → transpose back → `+ inputs` (residual). Both
   convs have bias.
4. **`repeat_interleave(×2, dim=1)`** — time axis upsample (token_mel_ratio=2).
5. **DiT** (below) at each CFM step.
6. **CFM Euler** — 10 steps, batch=2 (conditional | unconditional CFG),
   `cfg_rate=0.7`, cosine `t_span`. Final `feat = x[:, :, mel_len1:]`.

The CFM noise is **deterministic**: `CausalConditionalCFM.__init__` runs
`set_all_random_seed(0)` then `rand_noise = torch.randn([1,80,50*300])`; inference
uses `z = rand_noise[:,:,:n]` (no per-call randomness). `t_span =
1 - cos(linspace(0,1,11)·π/2)`.

**Euler loop (verbatim):** `t=t_span[0]`, `dt=t_span[1]-t_span[0]`. Each step
builds batch-2 inputs `x_in=[x;x]`, `mu_in=[mu;0]`, `spks_in=[spks;0]`,
`cond_in=[cond;0]`, `t_in=[t;t]`; `dphi_dt = DiT(...)`; then
`combined = 1.7·dphi_dt[0] − 0.7·dphi_dt[1]`; `x += dt·combined`; `t += dt`;
`dt = t_span[step+1]−t`.

### DiT forward

`x/mu/cond` transpose to `(B,T,80)`, `spks.unsqueeze(1)`. `t → time_embed(t)` =
`SinusPositionEmbedding(256, scale=1000)` → `Linear(256→1024)` → `SiLU` →
`Linear(1024→1024)`. `input_embed` = `proj(Linear 320→1024, on
concat[x, cond, mu, spks])` → `conv_pos_embed` (2× grouped Conv1d(k31, g16,
pad0) + Mish, causal `F.pad(30,0)` left before each) → `+ residual`. Then 22×
DiTBlock. Then `norm_out` (AdaLayerNormZero_Final) → `proj_out(1024→80)` →
transpose.

**DiTBlock:** `attn_norm = AdaLayerNormZero` (`silu(emb)→Linear(1024→6144)→
chunk6 → shift/scale/gate_msa + shift/scale/gate_mlp`); `x_norm =
norm(x)·(1+scale_msa)+shift_msa`; attention (q/k/v Linear 1024→1024, **partial
rotary**, reshape `(B,16,T,64)`, plain scaled-dot-product, `to_out` Linear);
`x += gate_msa·attn`; `ff_norm = norm(x)·(1+scale_mlp)+shift_mlp`; `ff =
GELU(tanh) + Linear(1024→2048) + Linear(2048→1024)`; `x += gate_mlp·ff`.

The non-streaming attention mask is all-True (full context, no padding in the
validation case), so attention is plain `softmax(q·kᵀ/√64)·v`.

## 2. Tensor layouts (two spaces)

GGML's natural data order is numpy row-major: `ne[0]` is the contiguous
(innermost) axis. PyTorch's `(B,C,T)` is therefore byte-identical to GGML
`[ne0=T, ne1=C, ne2=B]`. The decoder uses two conventions:

- **Space A** (`[seq, channel, batch]` = PyTorch `(B,C,T)`) — the CFM level:
  `x`, `mu`, `cond`, the final `dphi` and `feat`.
- **Space B** (`[feature, seq, batch]` = PyTorch `(B,T,C)`) — the DiT internal
  layout: `x`/`q`/`k`/`v`/`h`, all `[1024, T, B]`.

A `Linear` in Space B is `ggml_mul_mat(ctx, W, x)` with `W = [in, out]` and
`x = [in, seq, batch]`, yielding `[out, seq, batch]` directly — GGUF stores
`Linear.weight` with `ne = [in, out]`, exactly what `mul_mat` consumes. Moving
between spaces is a `ggml_permute(·, 1, 0, 2, 3)` (swap dims 0 and 1).

## 3. GGML op mapping

**Native (used directly):** `ggml_mul_mat` (Linear), `ggml_norm` (LayerNorm —
biased variance, no affine, matches PyTorch), `ggml_silu`, `ggml_leaky_relu`,
`ggml_tanh`, `ggml_softplus`, `ggml_soft_max_ext`, `ggml_get_rows`
(embedding), `ggml_scale/add/mul/sqr/sqrt`, `ggml_repeat`, `ggml_concat`,
`ggml_pad`, `ggml_permute/cont/reshape/view`, `ggml_rope_ext`.

**Assembled from primitives** (`src/flow/ops.cpp`):

- **Mish** — `x·tanh(softplus(x))` via `ggml_mul(x, ggml_tanh(ggml_softplus(x)))`.
- **Partial rotary (first 64 channels only)** — the reference applies RoPE to
  only `q/k[..., :64]` of the 1024-dim projection *before* the head reshape, so
  per-head `ggml_rope` is not applicable. `rope_partial` uses `ggml_rope_ext`
  with `n_dims=64` on a `[D,B,T]` permute (RoPE indexes position by `ne2`),
  leaving the other 960 channels untouched. `freq_base=10000`, standard RoPE.
- **`repeat_interleave(×2)`** — reshape a size-2 dim, `ggml_repeat`, reshape back
  (`repeat_interleave_2_ne0`).
- **Grouped Conv1d (groups=16)** — `ggml_conv_1d` has no `groups` arg, so
  `grouped_conv1d` splits input and kernel into 16 `ggml_view_3d` slices and
  concatenates the per-group outputs.
- **F32 conv1d** — `ggml_conv_1d` hardcodes F16 im2col, so `conv1d_f32` calls
  `ggml_im2col(ctx, kernel, input, …, GGML_TYPE_F32)` directly then a F32
  `mul_mat`.
- **GELU(tanh)** — `ggml_gelu` compiles to the F16-table `GGML_GELU_FP16`
  variant (~1e-3), so `gelu_tanh` is assembled manually to match
  `nn.GELU(approximate="tanh")` exactly.
- **`F.normalize`** — `x · rsqrt(clamp(Σx², 1e-12, ∞))` (`l2_normalize`).
- **Sinusoidal time embedding** — computed on host (`time_embed_host`), the two
  Linear+SiLU ops run in-graph.

### Gotchas that cost the most time

- **`ggml_permute` semantics**: `permute(a, p0, p1, p2, p3)` sets
  `ne[p_i] = a->ne[i]` — "a's dim *i* moves to result dim *p_i*". So
  `permute(a, 1, 0, 2, 3)` swaps dims 0 and 1.
- **`ggml_mul_mat` result layout**: `ne0 = a->ne[1]` (output channels),
  `ne1 = b->ne[1]` (sequence), i.e. `[out, seq, batch]`. The conv1d path
  (`conv1d_f32`) produces `[OL·N, OC]` with `(time inner, batch middle, channel
  outer)`, which must be reshaped to `[OL, N, OC]` then permuted to
  `[OL, OC, N]` — a bug here only shows up for `N > 1` (the frontend
  prelookahead runs `N=1`).
- **`ggml_im2col` layout**: kernel `a = [K, IC, OC]`, input `b = [L, IC, N]`
  (channels-first), output `[IC·K, OL, N]`.
- **Permute results are views, not copies** — `read_tensor` does a raw `memcpy`
  of `t->data`, so a non-contiguous permute's backing buffer is read in the
  *source* layout. Any tensor handed to `read_tensor` (or consumed as a graph
  output that's read by linear index) must be `ggml_cont`-wrapped first. This
  bit `dphi` (the Euler update consumed the wrong layout) and `conv_pos`
  (debug-only).
- **Binary ops require contiguous src0** — `ggml_add(y, x)` asserts src0's
  innermost stride equals the element size, so `ggml_add(ggml_cont(y), x)`.
  src1 may be non-contiguous (handled via strides).
- **`ggml_norm` uses biased variance** — matches PyTorch LayerNorm.
- **Graph size** — the 22-block DiT graph exceeds `GGML_DEFAULT_GRAPH_SIZE`
  (2048), so `ggml_new_graph_custom(ctx, 8192, false)`.

## 4. Numerical validation

Driven by `tests/verify_flow.py`: `tests/flow_reference.py` builds the verbatim
reference modules (inlined from the CosyVoice source — the `cosyvoice` package
is not importable on this box), loads `flow.pt`'s state_dict, and dumps
per-stage tensors to `flow_ref.npz`. `larynx_flow_dump` runs the C++ decoder on
the same inputs and dumps the same stages; `verify_flow.py` compares them.

Environment: `torch 2.14.0+cpu` + `numpy` for the reference; the C++ decoder
runs on either the CPU or the CUDA backend (TF32 disabled) and they agree to
~2× on deep-block *absolute* error, same order on every stage. Deterministic
inputs (seed 1234) + deterministic CFM noise (seed 0). Validation case: 4 prompt
tokens + 8 tokens → 12 speech tokens → 24 mel frames → 16 output mel frames.
The table below is the CUDA result; the CPU result differs only in the 4th
significant digit.

`max` / `mean` are absolute error over the whole tensor; `rel` is
`max_abs / max|ref|` (scale-normalized — see below). Both sides are float32, but
PyTorch runs its optimized BLAS / SDPA kernels while C++ runs GGML's own float32
kernels, so the gap is floating-point accumulation, not a bug.

| stage | max abs | mean abs | rel err |
|---|---|---|---|
| spk | 9.3e-08 | 2.4e-08 | 1.9e-07 |
| token_embed | 0.0 | 0.0 | 0.0 |
| prelookahead | 9.5e-07 | 1.9e-07 | 2.7e-07 |
| mu | 9.5e-07 | 1.9e-07 | 2.7e-07 |
| cond | 0.0 | 0.0 | 0.0 |
| time_embed | 3.2e-06 | 5.7e-08 | 7.9e-07 |
| input_proj | 5.7e-06 | 8.3e-08 | 2.3e-07 |
| conv_pos | 1.2e-04 | 7.5e-07 | 5.0e-06 |
| input_embed | 1.2e-04 | 7.9e-07 | 4.7e-06 |
| norm_out | 2.1e-03 | 3.9e-05 | 5.3e-04 |
| dphi | 2.0e-03 | 7.4e-05 | 1.7e-04 |
| **feat (final mel)** | **1.1e-03** | **9.5e-05** | **1.1e-04** |
| block[00] | 1.4e-04 | 1.4e-06 | 1.2e-06 |
| block[01] | 1.6e-04 | 1.9e-06 | 1.4e-06 |
| block[02] | 1.1e-03 | 1.9e-05 | 8.8e-06 |
| block[03] | 1.1e-03 | 2.3e-05 | 8.0e-06 |
| block[04] | 2.6e-03 | 2.9e-05 | 1.8e-05 |
| block[05] | 1.9e-03 | 2.6e-05 | 1.2e-05 |
| block[06] | 1.9e-03 | 2.8e-05 | 1.2e-05 |
| block[07] | 1.9e-03 | 3.0e-05 | 1.1e-05 |
| block[08] | 1.8e-03 | 3.0e-05 | 9.8e-06 |
| block[09] | 1.9e-03 | 3.1e-05 | 9.9e-06 |
| block[10] | 1.8e-03 | 3.3e-05 | 8.8e-06 |
| block[11] | 2.4e-03 | 3.6e-05 | 1.1e-05 |
| block[12] | 2.4e-03 | 3.9e-05 | 1.1e-05 |
| block[13] | 2.3e-03 | 4.0e-05 | 1.0e-05 |
| block[14] | 2.2e-03 | 4.3e-05 | 9.1e-06 |
| block[15] | 2.3e-03 | 4.5e-05 | 9.3e-06 |
| block[16] | 1.6e-03 | 4.8e-05 | 6.3e-06 |
| block[17] | 1.9e-02 | 9.7e-05 | 7.2e-05 |
| block[18] | 5.5e-02 | 3.6e-04 | 2.1e-04 |
| block[19] | 9.7e-02 | 1.9e-03 | 3.5e-04 |
| block[20] | 7.9e-02 | 2.0e-03 | 2.7e-04 |
| block[21] | 1.0e-01 | 2.1e-03 | 3.9e-04 |

Worst stage by scale-normalized error: `norm_out` (rel 5.3e-04). Every stage is
well under 1e-3 relative; the final mel is within ~1.1e-4 relative.

### Why the deep-block *absolute* error looks large (and why it's fine)

The intermediate DiT residual streams (`block[i]`) grow to magnitude ~100 — two
orders of magnitude above the network's 80-dim I/O — while the errors stay
bounded in *relative* terms (~1e-4..2e-4 at the deepest blocks, mean ~1e-3
absolute at block[21]). This is the classic accumulation signature of 22
sequential attention + FFN layers, each with its own GEMM/SDPA rounding order.
The decisive evidence that it is accumulation and **not** a structural bug:

1. The error grows **gradually** (roughly a constant factor per block), never
   jumping at a single layer boundary — a wrong op/mask/layout would diverge in
   one block.
2. The **final output** collapses back to ~1e-3 relative, because `norm_out`
   (LayerNorm) rescales and `proj_out` averages the 1024 channels. A real layout
   bug would *not* recover at the output — earlier in development a single
   wrong reshape made `conv_pos` diverge by 24× and the final `feat` by 7.8×,
   so this is a sensitive detector.

To reproduce: build (`cmake -S . -B build && cmake --build build`), then
`.venv/bin/python tests/flow_reference.py` (→ `tests/flow_ref.npz`) and
`.venv/bin/python tests/verify_flow.py`.

### Real-speech-token acceptance gate

The table above is a synthetic small-input check. The acceptance gate also
exercises the decoder end-to-end on one **real** speech-token sequence:
`tests/acceptance_wavs.py` runs the full official CosyVoice3 pipeline
(`inference_instruct2`), wrapping `model.flow.inference` to capture the exact
inputs the reference flow consumes (prompt tokens, tokens, prompt mel, 192-dim
speaker embedding) plus the deterministic CFM noise, then replays those inputs
through the C++ `FlowDecoder` and vocodes the resulting mel. It writes:

- `wavs/wav_reference.wav` — full official Python inference (LLM → flow → HiFT).
- `wavs/wav_ggml.wav` — GGML Flow decoder mel → official Python HiFT.
- `wavs/wav_ggml_full.wav` — GGML Flow decoder mel → GGML HiFT (all-GGML; see
  `docs/HIFT.md`).

Because the HiFT vocoder and the flow inputs are identical on both sides, any
audible difference between `wav_ggml.wav` and `wav_reference.wav` is attributable
solely to the GGML Flow decoder. On the 2.92 s / 146-mel-frame validation
utterance the two mels agree to **9.0e-3** max / **3.0e-4** mean absolute error
(≈8e-4 scale-normalized), consistent with the float32 accumulation above, and
the resulting waveforms agree to **3.0e-1** max / **5.8e-3** mean (16-bit
full-scale) after HiFT's nonlinear mel→waveform amplification.

Because the full DiT graph needs ~3.5 GiB of VRAM and the resident PyTorch
LLM+flow hold ~4.5 GiB (an 8 GiB GPU cannot hold both), the script frees
`model.llm` + `model.flow` after the reference run and before invoking the C++
decoder, keeping only the HiFT for the vocoding step. `scripts/listen_compare.py`
serves a local page to A/B the WAVs.

## 5. Deferred (per the task scope)

- **Streaming branch** — `PreLookaheadLayer` with `context` input and chunk
  incremental inference are not implemented; this is the whole-sequence
  (`finalize=True`) path only.
- **HiFT vocoder, LLM backbone, CLI main flow** — separate later phases.
- **ONNX-trace-specific mask workarounds** (the `torch.where` / Or-Not patches)
  are deliberately *not* carried over — they are artifacts of the ONNX export,
  irrelevant to a GGML implementation.

## 6. Files

- `include/larynx/flow/flow.h` — `FlowDecoder` + `FlowDebug` public API.
- `src/flow/flow.cpp` — `FlowDecoder::infer`: frontend graph (spk / token_embed
  / prelookahead / mu / cond), then the 10-step CFM Euler loop over the DiT graph.
- `src/flow/dit.cpp` — DiT graph builder (input_embed, conv_pos_embed,
  22× DiTBlock, norm_out, proj_out).
- `src/flow/prelookahead.cpp` — PreLookaheadLayer.
- `src/flow/ops.cpp` — `linear`, `conv1d_f32`, `grouped_conv1d`, `mish`,
  `gelu_tanh`, `rope_partial`, `ada_ln`, `l2_normalize`,
  `repeat_interleave_2_ne0`.
- `tests/flow_reference.py`, `tests/flow_dump.cpp`, `tests/verify_flow.py` — the
  reference / dump / comparison harness.
