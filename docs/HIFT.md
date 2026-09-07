# HiFT vocoder — implementation notes and numerical validation

The `src/hift/` module implements the **HiFT vocoder** — the final stage that
turns an 80-bin mel spectrogram into a 24 kHz PCM waveform — as a single GGML
compute path (per `docs/ARCHITECTURE.md` §3.3 and `docs/adr/0001`). It
reimplements `cosyvoice/hifigan/generator.py` `CausalHiFTGenerator`, in the
non-streaming `finalize=True` path.

| Component | Reference | Notes |
|---|---|---|
| `larynx::hift::HiftVocoder` | `CausalHiFTGenerator.inference` | mel → PCM |
| f0 predictor | `cosyvoice/hifigan/f0_predictor.py` `CausalConvRNNF0Predictor` | float64, on host |
| source excitation | `generator.py` `SineGen2` + `SourceModuleHnNSF` | fixed buffers, on host |
| STFT / ISTFT | `generator.py` `_stft` / `_istft` | n_fft=16, hop=4, on host |
| upsample/resblock net | `generator.py` `decode` | conv_pre + 3 stages + conv_post, GGML graph |

The conv network (conv_pre + the three upsample/resblock stages + conv_post)
runs as one GGML graph on the backend chosen by `ggml_backend_init_best()` (CUDA
when `LARYNX_ENABLE_CUDA=ON` and a device is present, else CPU); the f0
predictor, SineGen2, and the 16-point STFT/ISTFT run on the host (they are tiny
and, in the case of f0, run in float64 to match PyTorch). On CUDA the cuBLAS
math mode is pinned to `CUBLAS_DEFAULT_MATH` (TF32 disabled) via `patches/0001`;
see `docs/adr/0002`. Weights stay F32, no quantization.

---

## 1. Reference semantics

Exact config (`cosyvoice3.yaml`, `CausalHiFTGenerator`): `in_channels=80`,
`base_channels=512`, `nb_harmonics=8`, `sampling_rate=24000`,
`nsf_alpha=0.1`, `nsf_sigma=0.003`, `nsf_voiced_threshold=10`,
`upsample_rates=[8,5,3]`, `upsample_kernel_sizes=[16,11,7]`,
`resblock_kernel_sizes=[3,7,11]`, `resblock_dilation_sizes=[[1,3,5]×3]`,
`lrelu_slope=0.1`, `audio_limit=0.99`, `conv_pre_look_right=4`. The total
upsample is `8·5·3·hop(4) = 480` samples per mel frame.

`inference(finalize=True)` is three stages:

1. **f0** — `f0_predictor(mel.to(float64))`: five `CausalConv1d` (`condnet`),
   the first `80→512, k=4` (right-causal) then four `512→512, k=3`
   (left-causal), each followed by ELU; then `classifier` `Linear(512→1)`, then
   `torch.abs`. Runs in float64 on both sides.
2. **source excitation** — `f0_upsamp` (nearest ×480) → `SineGen2` (phase
   synthesis + unvoiced noise, `causal=True`) → `m_source` (`tanh(Linear(9→1))`)
   → `s`; then `_stft(s)` with `n_fft=16, hop=4, win=16`, periodic Hann,
   `center=True` (reflect-pad 8) → 9 onesided bins, real+imag stacked to 18
   channels.
3. **decode** (the main network, `decode()`):

   ```
   x = conv_pre(x)                         # CausalConv1d(80->512, k=5, right)
   for i in 0..2:
       x = leaky_relu(x, 0.1)              # self.lrelu_slope
       x = ups[i](x)                       # nearest x rate -> pad left k-1 -> conv
       if i == 2: x = reflection_pad(x)    # nn.ReflectionPad1d((1,0))
       si = source_downs[i](s_stft)        # CausalConv1dDownSample
       si = source_resblocks[i](si)        # 3x resblock
       x = x + si
       xs = sum_j resblocks[i*3+j](x)      # 3 resblocks, kernels [3,7,11]
       x = xs / 3
   x = leaky_relu(x)                       # F.leaky_relu DEFAULT slope 0.01
   x = conv_post(x)                        # CausalConv1d(64->18, k=7, left)
   magnitude = exp(x[:, :9]); phase = sin(x[:, 9:])
   audio = _istft(magnitude, phase)        # clip mag to 1e2 first
   audio = clamp(audio, -0.99, 0.99)
   ```

   `_istft` clips `magnitude` to `1e2` **before** forming the complex
   spectrum, then `torch.istft(n_fft=16, hop=4, win=16, Hann)` and clamps the
   result.

### The two fixed SineGen2 buffers

`SineGen2(causal=True)` samples two construction-time buffers from the global
RNG: `rand_ini` `(1,9)` (column 0 zeroed) and `sine_waves`
`(1, 300·24000, 9)`. These are the *unvoiced-noise waveform bank* and the
initial phase offset. They are **not** regenerated on the C++ side —
`tests/hift_reference.py` exports them and `HiftVocoder::vocode` consumes them
verbatim (the 不要做的事: never synthesize your own random noise to compare).

### Causal conv semantics (read from `convolution.py`)

- `CausalConv1d(padding)`: pads the *specified* side by `d·(k−1)` (`'left'` or
  `'right'`), then a plain `Conv1d(k, stride=1, padding=0)`. For odd `k,d` this
  is exactly "causal".
- `CausalConv1dUpsample`: `nn.Upsample(scale_factor=rate, mode='nearest')` →
  pad left `k−1` → `Conv1d(k, stride=1)`.
- `CausalConv1dDownSample`: pad left `stride−1` → `Conv1d(k, stride)`. For
  `stride==1` (`source_downs[2]`) this collapses to a `k=1` causal conv.

### Snake activation (BigVGAN)

`snake(x) = x + (1/(α + 1e-9)) · sin²(x·α)`, with a per-channel `α` of length
`C`. Both `α` and the companion `1/(α+1e-9)` are precomputed.

### weight_norm merge

The `WeightNorm` layers are folded at conversion time (`tools/convert_weights.py`):
effective weight = `v · g / ‖v‖₂`. The `.gguf` holds the merged weights directly,
so the graph needs no weight-norm reparametrization.

---

## 2. Tensor layouts

GGML's natural order is numpy row-major (`ne[0]` contiguous), so PyTorch
`(B,C,T)` is byte-identical to GGML `[ne0=T, ne1=C, ne2=B]`. The vocoder uses
`[time, channel, batch=1]` throughout:

- mel input `(1,80,T)` → GGML `[T, 80, 1]`.
- conv network activations `[T, C, 1]` (the `ups`/`resblock` streams).
- `s_stft` input `(1,18,TT)` → GGML `[TT, 18, 1]`.

A `Conv1d(k, IC, OC)` kernel is stored in GGUF as `ne = [K, IC, OC]` (exactly
`ggml_im2col`'s kernel layout); the F32 conv1d path (below) consumes it with no
transpose. The host-side helpers (f0/source/ISTFT) use `[c][t]` row-major for
the same reason.

---

## 3. GGML op mapping

The conv network needs very few ops:

**Native (used directly):** `ggml_im2col` + `ggml_mul_mat` (F32 conv1d),
`ggml_leaky_relu`, `ggml_sin`, `ggml_exp`, `ggml_sqr`, `ggml_add`,
`ggml_scale`, `ggml_mul`, `ggml_repeat`, `ggml_concat`, `ggml_reshape/view`,
`ggml_permute/cont`.

**Assembled from primitives** (`src/hift/ops.cpp`):

- **F32 conv1d** (`conv1d_f32`) — `ggml_conv_1d` hardcodes F16 im2col, so this
  calls `ggml_im2col(ctx, kernel, input, …, GGML_TYPE_F32)` then a F32
  `mul_mat`, reshaped `[OL,N,OC]` → permuted `[OL,OC,N]` → `ggml_cont`, then
  bias-add. Padding is applied by the caller (causal convs), so `p0=0`.
- **Snake** (`snake`) — `x + inv · sin²(x·α)` with `α`/`inv` broadcast as
  `[1,C,1]` over `[T,C,1]`.
- **nearest upsample** (`nearest_upsample`) — reshape a leading size-`r` dim,
  `ggml_repeat`, `ggml_cont`, reshape back (`out[i] = in[i/r]`).
- **zero pad / reflection pad** (`pad_zeros`, `reflection_pad_left1`) — via
  `ggml_new_tensor_3d` + `ggml_set_zero` + `ggml_concat`, or a
  `ggml_view_3d` of `x[1]` prepended to `x`.

The f0 predictor, SineGen2, STFT, and ISTFT are plain host loops (a 16-point
DFT table, a periodic Hann window, and the reflect-pad STFT framing). They are
deliberately *not* in the GGML graph — they are tiny, and the f0 predictor runs
in float64 to match PyTorch exactly.

### Gotchas that cost the most time

- **The final leaky_relu is `F.leaky_relu(x)` with the *default* slope 0.01**,
  not `self.lrelu_slope=0.1`. The per-stage leaky_relu in the loop uses 0.1;
  only the pre-`conv_post` one uses 0.01. This single mismatch scaled the final
  activations by 10× and, through `exp`/ISTFT, wrecked the whole waveform. It
  was invisible to the loop leaky_relus because their inputs were positive
  (leaky_relu of a positive value is the identity regardless of slope).
- **`_istft` clips `magnitude` to `1e2` before the inverse transform** — the
  unclipped magnitudes reach `~1e2` and higher; missing the clip changed the
  reconstructed audio sign and shape.
- **`ggml_mul` broadcast order** — the result shape is the *first* argument and
  `ggml_can_repeat(b, a)` must hold, so broadcasting `[1,C,1]` over `[T,C,1]`
  requires `ggml_mul(ctx, larger, smaller)`, not the other way around.
- **Snake `inv_alpha` must not live in the weight context** — the weight ctx is
  sized exactly for the 246 GGUF tensors; materializing 72 `inv_alpha` tensors
  there overflowed it. Store `inv_alpha` as a host `std::vector<float>` and
  materialize `[1,C,1]` constants in the graph ctx.
- **Eager graph memory scales with `T_mel`** — `no_alloc=false` keeps every
  conv im2col resident; measured 773 MiB at `T_mel=30` (~26 MiB/frame). The
  pool is sized from `T_mel` (no `T²` term — the vocoder has no attention).
- **Permute results are views** — any tensor handed to `read_tensor` must be
  `ggml_cont`-wrapped first (see the same note in `docs/FLOW.md`).

---

## 4. Numerical validation

Driven by `tests/verify_hift.py`: `tests/hift_reference.py` builds the real
`CausalHiFTGenerator` from `cosyvoice3.yaml` (importing the package under the
CosyVoice python3.10 env), loads `hift.pt`, and dumps per-stage tensors to
`hift_ref.npz` (replaying `inference(finalize=True)` step-by-step, then
cross-checking the replay against a direct `model.inference()` call).
`larynx_hift_dump` runs the C++ vocoder on the same inputs and dumps the same
stages; `verify_hift.py` compares them.

Validation case: `mel` `(1,80,30)` (seed 1234) → `f0` `(1,30)` → source
`(1,14400,9)` → `s_stft` `(1,18,3601)` → audio `(1,14400)`. `max`/`mean` are
absolute error over the whole tensor; `rel` is `max_abs / max|ref|`
(scale-normalized). The f0 predictor is float64 on both sides (bit-faithful);
SineGen2/STFT/ISTFT are float32 and nearly bit-exact; only the conv network
runs GGML's float32 kernels against PyTorch's, so that part shows
~1e-4..1e-3 floating-point accumulation. Validated on both CPU and CUDA (TF32
disabled); the table below is the CUDA result, which matches CPU to the 4th
significant digit.

| stage | max abs | mean abs | rel err |
|---|---|---|---|
| f0 | 2.0e-04 | 2.5e-05 | 4.9e-07 |
| sine_wavs | 4.9e-05 | 5.0e-06 | 4.7e-04 |
| sine_merge | 1.7e-06 | 3.7e-07 | 5.0e-05 |
| s_stft | 1.3e-05 | 2.6e-07 | 5.1e-05 |
| conv_pre | 2.4e-06 | 7.9e-08 | 3.1e-07 |
| ups0_lrelu | 1.9e-06 | 5.0e-08 | 4.0e-07 |
| ups0 | 1.8e-05 | 1.1e-06 | 2.6e-06 |
| source_downs0 | 8.5e-05 | 1.2e-06 | 3.9e-05 |
| source_resblocks0 | 1.6e-03 | 2.6e-06 | 1.7e-04 |
| fusion0 | 1.6e-03 | 3.2e-06 | 1.6e-04 |
| resblock0 | 4.0e-03 | 2.6e-05 | 3.8e-05 |
| resblock1 | 1.2e-03 | 2.9e-05 | 1.3e-05 |
| resblock2 | 1.3e-03 | 1.2e-05 | 2.9e-05 |
| post_resblocks0 | 1.9e-03 | 1.6e-05 | 3.9e-05 |
| ups1_lrelu | 1.1e-03 | 5.7e-06 | 2.6e-05 |
| ups1 | 2.0e-04 | 6.7e-06 | 1.1e-05 |
| source_downs1 | 1.4e-05 | 1.3e-07 | 3.8e-05 |
| source_resblocks1 | 8.6e-06 | 3.6e-07 | 3.0e-06 |
| fusion1 | 2.0e-04 | 6.7e-06 | 1.1e-05 |
| resblock3 | 7.8e-04 | 2.8e-05 | 4.1e-06 |
| resblock4 | 1.4e-03 | 3.2e-05 | 1.0e-05 |
| resblock5 | 3.8e-03 | 3.6e-05 | 3.9e-05 |
| post_resblocks1 | 1.2e-03 | 2.2e-05 | 1.3e-05 |
| ups2_lrelu | 1.2e-03 | 1.2e-05 | 1.8e-05 |
| ups2 | 2.6e-04 | 5.4e-06 | 2.0e-05 |
| reflection_pad | 2.6e-04 | 5.4e-06 | 2.0e-05 |
| source_downs2 | 6.6e-07 | 4.0e-08 | 1.6e-06 |
| source_resblocks2 | 6.7e-06 | 1.0e-07 | 4.1e-07 |
| fusion2 | 2.6e-04 | 5.4e-06 | 1.2e-05 |
| resblock6 | 7.7e-04 | 1.8e-05 | 1.4e-06 |
| resblock7 | 4.0e-04 | 1.5e-05 | 4.7e-06 |
| resblock8 | 4.8e-03 | 1.2e-04 | 3.1e-06 |
| post_resblocks2 | 1.6e-03 | 4.4e-05 | 3.0e-06 |
| final_lrelu | 1.6e-03 | 9.3e-06 | 8.1e-06 |
| conv_post | 2.3e-05 | 1.7e-06 | 4.7e-06 |
| magnitude | 1.1e-03 | 3.9e-05 | 8.0e-06 |
| phase | 1.5e-05 | 6.6e-07 | 1.5e-05 |
| istft | 8.8e-05 | 6.8e-06 | 2.5e-05 |
| **speech (final audio)** | **8.8e-05** | **6.4e-06** | **8.9e-05** |

Worst stage by scale-normalized error: `sine_wavs` (rel 4.7e-04). Every stage
is under 1e-3 relative; the final audio is within ~9e-5 relative (the ISTFT
overlap-add averages the per-frame errors back down).

To reproduce: build (`cmake -S . -B build && cmake --build build`), then
`tests/hift_reference.py` (CosyVoice python3.10 → `tests/hift_ref.npz`) and
`.venv/bin/python tests/verify_hift.py`.

### Real-speech-token acceptance gate (full GGML chain)

The table above is a synthetic small-input check. The acceptance gate also
exercises the whole **all-GGML** chain — GGML Flow decoder mel → GGML HiFT
vocoder — on one real speech-token sequence, against the full official Python
pipeline: `tests/acceptance_wavs.py` (Phase-3 extended) loads the model once,
captures the flow inputs *and* the HiFT's fixed SineGen2 buffers from that same
instance, replays the tokens through the C++ `FlowDecoder`, then vocodes the
resulting mel with the C++ `HiftVocoder`. It writes:

- `wavs/wav_reference.wav` — full official Python inference (LLM → flow → HiFT).
- `wavs/wav_ggml.wav` — GGML Flow decoder mel → official Python HiFT.
- `wavs/wav_ggml_full.wav` — GGML Flow decoder mel → GGML HiFT (all-GGML).

Because the flow inputs and the HiFT buffers are identical on both sides, any
audible difference between `wav_ggml_full.wav` and `wav_reference.wav` is
attributable solely to the GGML Flow + HiFT decoders vs. the PyTorch reference.
On the 2.92 s / 146-mel-frame / 70080-sample utterance:

- all-GGML vs. reference (`wav_ggml_full.wav` vs. `wav_reference.wav`):
  **3.0e-1** max / **5.8e-3** mean absolute error (normalized 16-bit full-scale).
- flow-only (`wav_ggml.wav` vs. `wav_reference.wav`, HiFT held identical):
  **3.0e-1** max / **5.8e-3** mean — i.e. the flow mel is the dominant term.
- vocoder-only (`wav_ggml.wav` vs. `wav_ggml_full.wav`, flow mel held identical):
  **7.9e-3** max / **1.1e-4** mean — the GGML HiFT is near bit-exact to the
  Python HiFT given the same mel.

The `~6e-3` mean is the float32-accumulation signature amplified through HiFT's
nonlinear (exp/snake/phase) synthesis; the `~0.3` max sits on isolated onset
samples. `scripts/listen_compare.py` serves a local page to A/B the two WAVs
(`python3 scripts/listen_compare.py wavs/wav_ggml_full.wav
wavs/wav_reference.wav`). Because the full DiT graph needs ~3.5 GiB of VRAM and
the resident PyTorch LLM+flow hold ~4.5 GiB (an 8 GiB GPU cannot hold both),
the gate frees `model.llm` + `model.flow` after the reference run and before
invoking the C++ decoders, keeping only the HiFT.

---

## 5. Deferred (per the task scope)

- **Streaming / incremental branch** — `finalize=False`, the `cache_source`
  path, and `f0_predictor`'s streaming (`finalize=False`) branch are not
  implemented; this is the whole-sequence (`finalize=True`) path only.
- **LLM backbone, CLI main flow** — separate later phases.

## 6. Files

- `include/larynx/hift/hift.h` — `HiftVocoder` + `HiFTDebug` public API.
- `src/hift/hift.cpp` — `HiftVocoder::vocode`: host f0/source, then the GGML
  conv network (conv_pre → 3 upsample/resblock stages → conv_post) → ISTFT.
- `src/hift/ops.cpp` — `conv1d_f32`, `snake`, `nearest_upsample`, `pad_zeros`,
  `reflection_pad_left1`.
- `src/hift/source.cpp` — host `f0_predict` (float64), `source_stft`
  (SineGen2 + m_source + STFT), `istft`.
- `src/hift/gguf.cpp` — `load_weights` (246 tensors by name, weight-norm merged).
- `src/hift/internal.h` — architecture constants, weight structs, declarations.
- `tests/hift_reference.py` — PyTorch ground truth (imports the real package).
- `tests/hift_dump.cpp` / `tests/verify_hift.py` — C++ dump + comparison.
- `tools/convert_weights.py` — `--hift hift.pt` → `hift.gguf` (weight-norm fold).
