# Larynx  Architecture Reference

**Status:** Design locked after ONNX Runtime prototype revealed unresolved
numerical divergence in the Flow decoder (see `docs/adr/0001-drop-onnx-runtime-for-compute.md`).
**Target:** Zero Python dependency at deployment time. A single native
binary, `larynx`, that takes text (+ voice script) and produces PCM/WAV.

---

## 1. Why this architecture

Three failed/rejected alternatives, kept here so the reasoning isn't lost:

1. **Python + PyTorch runtime**  rejected from the start (project's entire
   purpose is to remove this).
2. **C++ + ONNX Runtime + CUDA EP**  prototyped, produced a working export
   for the Flow decoder estimator, but stagewise numerical diagnosis found
   token-localized errors starting at block 3 of 22, growing to full sign
   flips by block 17-19. Root cause suspected to be a combination of:
   attention backend divergence (PyTorch eager SDPA likely dispatches to a
   fused kernel; ONNX export decomposes SDPA into explicit matmul+softmax
   subgraphs) and TF32 default mismatch (PyTorch enables TF32 matmul by
   default on Ampere+, ONNX Runtime's CUDA EP does not). Never fully
   confirmed because the project pivoted before that experiment was run.
3. **llama.cpp**  considered for the LLM stage, rejected: CosyVoice3's
   `speech_embedding` and `llm_decoder` are non-standard heads (not a
   vocabulary embedding / not a standard LM head), and llama.cpp's model
   loading, tokenizer binding, and sampling are built around the "this is a
   normal text LLM" assumption. Retrofitting would cost more than building
   directly on GGML.

**Chosen: GGML (the tensor library, not llama.cpp) + CUDA, for everything
except the two small audio frontend networks, which stay on ONNX Runtime
C++ (no Python involved, already numerically validated, no reason to
rewrite).**

The reasoning that generalizes beyond this project: PyTorch and ONNX
Runtime are two independently-implemented numerical engines. Even given
identical math, they are not guaranteed to produce identical floating-point
results, because floating-point addition is not associative  different
kernel implementations (fused vs. decomposed attention, different
accumulation order, different default precision modes) produce different,
individually deterministic, but mutually divergent results. This divergence
compounds across a deep network via nonlinear operations like softmax. Using
GGML instead of ONNX Runtime does not eliminate this class of problem in
general  it eliminates it *for this specific pipeline* by ensuring there is
only one computation path (GGML's own CUDA kernels), so there is nothing to
diverge from.

---

## 2. Pipeline

```
text + voice script
        
        

   DSP frontend (C++)    Whisper-style log-mel (128-bin), for speech tokenizer input
   no framework dep      Kaldi-style fbank (80-bin, mean-normalized), for campplus input

        
   
            
 
 speech   campplus    ONNX Runtime C++ API. NOT reimplemented  already
tokenizer (speaker     ships as ONNX in the official checkpoint, and no
 (ONNX)  embedding)   numerical issue was ever found here.
 
        
        

  LLM backbone (GGML + CUDA)    Qwen2ForCausalLM backbone (GGML has native
                                 Qwen2 kernel support) wrapped with:
  - speech_embedding             - custom embedding: nn.Embedding(speech_token_size+200, llm_input_size)
  - llm_decoder                   - custom output head: nn.Linear(llm_output_size, speech_token_size+200)
  - ras_sampling (C++)            - control token offsets: sos/eos/task_id/fill_token
  - prefill/decode split          - sampling reimplemented from cosyvoice/utils/common.py::ras_sampling

          speech token sequence
        

  Flow decoder (GGML + CUDA) 
  - PreLookaheadLayer             2x Conv1d + LeakyReLU + residual (see cosyvoice/transformer/upsample_encoder.py)
  - DiT  22 blocks               AttnProcessor (plain SDPA, no JointAttnProcessor), LayerNorm, FFN
  - CFM Euler solver (C++ loop)   10-step, batch=2N (cond+uncond CFG doubling), calls DiT estimator per step

          mel spectrogram
        

  HiFT vocoder (GGML + CUDA) 
  - upsampling convs           
  - F0 predictor                
  - ISTFT via matmul+conv1d       NOT torch.istft  explicit DFT-matrix / overlap-add,
    (no raw cuFFT call)            using only ops GGML already supports

        
        
      PCM/WAV
```

---

## 3. Component notes (from actual source inspection, not guesses)

### 3.1 DSP frontend
- Speech tokenizer input: `whisper.log_mel_spectrogram(speech, n_mels=128)`,
  shape `(1,128,T)`, audio fixed at 16kHz, 30s.
- Campplus input: `kaldi.fbank(speech, num_mel_bins=80, dither=0,
  sample_frequency=16000)` minus per-utterance mean, shape `(1,T,80)`.
- Neither is raw waveform. Both must be bit-compatible (or numerically close
  enough not to shift token IDs) with the Python reference implementations
  (`whisper` package, `torchaudio.compliance.kaldi`).
- `whisper.cpp` has an existing, reusable C++ implementation of the 128-bin
  log-mel extraction  use as reference, don't reimplement blind.
- No known lightweight C++ reference exists for Kaldi-style fbank with
  dither=0 + mean normalization  this one has to be written from Kaldi's
  algorithm definition and validated numerically from scratch.

### 3.2 LLM backbone
- Base: `Qwen2ForCausalLM` (confirmed from `cosyvoice/llm/llm.py`).
- `CosyVoice3LM.inference_wrapper` is a Python generator with `while`/`if`
  control flow and dynamic `ras_sampling` calls  this cannot be traced or
  scripted as-is; it must be reimplemented as an explicit C++ loop:
  prefill (full prompt  initial KV cache) then decode (one token at a time
   updated KV cache), matching the semantics of `Qwen2Encoder.forward_one_step`.
- `ras_sampling` is defined in `cosyvoice/utils/common.py`  read this file
  for the exact sampling algorithm before implementing; do not assume
  standard top-k/top-p without checking.

### 3.3 Flow decoder
- Estimator inputs: `x, mask, mu, t, spks, cond`. Real inference calls this
  with batch = `2  request_count` (CFG: conditional + unconditional rows
  concatenated), never batch = request_count. Any C++ implementation must
  preserve this convention  see `CausalConditionalCFM.solve_euler` in
  `cosyvoice/flow/flow_matching.py`.
- Attention processor is `AttnProcessor` (plain SDPA), not
  `JointAttnProcessor`  confirmed from `cosyvoice/flow/DiT/modules.py`.
- Chunk masking goes through `add_optional_chunk_mask` /
  `subsequent_chunk_mask` in `cosyvoice/utils/mask.py`. A GGML
  reimplementation does not need to replicate the ONNX-export-specific
  workarounds developed during the ONNX Runtime prototype (those existed
  only to route around tracing limitations that don't apply here)  but the
  underlying masking *semantics* (including the all-false-row repair) must
  still be preserved.

### 3.4 HiFT vocoder
- `CausalHiFTGenerator._stft` / `_istft` call `torch.stft` / `torch.istft`
  directly  these must be replaced with an explicit, GGML-expressible
  formulation (DFT matrix multiply + overlap-add via Conv1d), not a raw
  cuFFT call, to keep the whole pipeline inside one tensor graph / backend.

---

## 4. Explicit non-goals

- No custom GPU kernels  GGML's existing CUDA kernels (matmul, conv1d,
  attention, elementwise) are used as-is. Writing new low-level CUDA kernels
  is out of scope.
- No llama.cpp dependency.
- No vLLM  its scheduling loop runs in Python; incompatible with the
  zero-Python-at-runtime goal even if isolated as a "service."
- No attempt to reproduce bit-exact PyTorch output. The correctness bar is
  numerical closeness sufficient for acceptable audio quality, validated the
  same way the ONNX prototype was validated: stagewise comparison against
  the PyTorch reference, with explicit per-position error inspection, not
  just aggregate max/mean error.
- ONNX Runtime is retained *only* for `campplus.onnx` / `speech_tokenizer_v3.onnx`.
  It is not used anywhere else in the pipeline, and this is a deliberate,
  narrow exception, not a partial reversal of the "no ONNX Runtime" decision.

---

## 5. Weight conversion

`.pt` checkpoints (`llm.pt` or `llm.rl.pt`, `flow.pt`, `hift.pt`) are
converted offline to a GGML-loadable format by a Python script
(`tools/convert_weights.py`). This script runs once, at build/prep time, on
a developer machine  it is not part of the deployed binary and does not
violate the zero-Python-at-runtime goal, the same way PyTorch's own build
process needs Python without the compiled `libtorch.so` needing it at
runtime.

---

## 6. Validation methodology (carried over from the ONNX prototype, proven useful)

For each component, in this order:
1. Static feasibility check against real source (already mostly done for
   Flow/HiFT, see 3).
2. Implement in isolation.
3. Compare against PyTorch eager-mode reference: shape, dtype, max/mean
   absolute and relative error  **and**, critically, per-position
   inspection of the worst offenders, not just aggregate stats. The ONNX
   prototype's most important lesson: aggregate error metrics can hide
   token-localized structural errors that residual connections dilute into
   invisibility at the output layer.
4. Only after per-stage numerical validation passes, wire into the full
   pipeline and validate end-to-end audio quality.
