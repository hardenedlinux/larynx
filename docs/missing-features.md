# Missing features vs. upstream CosyVoice3

This document records the functional gaps between Larynx and the upstream
[CosyVoice3](https://github.com/QwenAudio/CosyVoice) reference. It is a parity
checklist: what the reference pipeline does at runtime that Larynx does not yet
cover.

The gaps fall into four buckets — text frontend, acoustic frontend, inference
modes, and runtime capabilities — plus two items that are *not* functional gaps
but implementation-route differences (acceleration backend) or deployment
surfaces.

| Category | Missing feature | Upstream CosyVoice3 | Larynx status |
|---|---|---|---|
| Text frontend | `text_normalize` (digit / date / punctuation / polyphone normalization; ttsfrd or WeText) | normalizes `tts_text` and `prompt_text` before every inference | Not implemented; tokenizes raw text. Fine for plain text, diverges from the reference on digits / English / dates. |
| Acoustic frontend | campplus speaker embedding + speech_tokenizer_v3 (ONNX) | runs the ONNX models at inference time | Deferred; pre-extracted by Python into `spk_embedding.f32` / `prompt_tokens.i32` |
| Acoustic frontend | matcha 80-bin `mel_spectrogram` (the Flow `prompt_feat`) | computed at runtime | Deferred; pre-extracted by Python into `prompt_feat.f32` |
| Acoustic frontend | `spk2info` pre-registered speakers (`spk_id` → cached embedding, no prompt wav) | supports `zero_shot_spk_id != ''` | Not wired |
| Inference mode | zero-shot voice cloning (`inference_zero_shot`; `prompt_speech_token` fed to the LLM) | one of six modes | `lm_input` path verified (Phase 1); no CLI mode |
| Inference mode | cross-lingual (`inference_cross_lingual`; drops the LLM prompt text) | one of six modes | Not implemented |
| Inference mode | instruct / SFT (`inference_instruct` / `inference_sft`, via `spk_id`) | one of six modes | Not implemented |
| Inference mode | voice conversion (`inference_vc`, `source_wav` → prompt) | one of six modes | Not implemented |
| Runtime | streaming inference `stream=True` (chunked Flow + mel overlap + HiFT cache) | incremental `token2wav(finalize=False)` | Whole-sequence `finalize=True` only |
| Runtime | speech-rate control `speed` (`F.interpolate`) | `token2wav(speed=...)` | Not implemented |
| Runtime | streaming text input (`Generator` / `inference_bistream`) | supported | Not implemented |
| Acceleration | vLLM / TensorRT / fp16 autocast | optional (`load_vllm` / `load_trt` / `fp16`) | Superseded by GGML (vLLM/TRT N/A; fp16 optional at conversion via `--f16`) |
| Deployment | WebUI / FastAPI demo | shipped demo server | Not provided |

## Notes

- **Implemented**: the `inference_instruct2` mode (instruction-following with
  `<|endofprompt|>`) — the community mainline. The other five `inference_*`
  modes listed above are gaps.
- The **acceleration** row is a *design difference*, not a functional gap: GGML
  replaces the vLLM / TensorRT / fp16 backends. Keep it distinct from the real
  feature gaps above when presenting this list.
