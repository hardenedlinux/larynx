# Larynx

A native reimplementation of the CosyVoice3 text-to-speech pipeline with **zero
Python at runtime**. The neural networks (LLM / Flow / HiFT) run on
[GGML](https://github.com/ggml-org/ggml) with CUDA acceleration; the acoustic
frontend (campplus / speech_tokenizer) uses ONNX Runtime only for its dedicated
purpose. Weight conversion happens once, offline, in Python.

> **Status: Phase 1.** Repository skeleton, DSP frontend, and the weight
> conversion tool are in place. The Qwen2 / DiT / HiFT models themselves are
> *not* implemented yet — they are later phases. No ONNX Runtime integration and
> no quantization have been introduced.

## Layout

| dir | purpose |
|---|---|
| `src/dsp/` | Whisper 128-bin log-mel + Kaldi 80-bin fbank (verified) |
| `src/frontend/` | campplus / speech_tokenizer (placeholder) |
| `src/llm/` | Qwen2 autoregressive decoder (placeholder) |
| `src/flow/` | DiT flow-matching estimator (placeholder) |
| `src/hift/` | HiFi-GAN vocoder (placeholder) |
| `src/pipeline/` | orchestration (placeholder) |
| `src/cli/` | `larynx` entry point (skeleton) |
| `tools/` | offline prep: `convert_weights.py`, `gguf.py`, `gen_mel_filters.py` |
| `tests/` | `verify_dsp.py` (numerical), `test_convert.py` (GGUF round-trip) |
| `docs/` | `ARCHITECTURE.md`, ADRs, `DSP.md`, `WEIGHT_FORMAT.md` |

## Build

```sh
cmake -S . -B build            # enables CUDA if a toolkit is detected
cmake --build build -j
```

This produces `larynx` and `larynx_dump` (a feature-dump utility used by the
DSP verification). CUDA is optional — the DSP and skeleton build and run on CPU.

## Verify the DSP frontend

```sh
.venv/bin/python tests/verify_dsp.py
```

compares `larynx_dump` output against `whisper.log_mel_spectrogram` and
`torchaudio.compliance.kaldi.fbank`. See `docs/DSP.md` for the reference
implementation, the deviations, and the measured error.

## Convert weights

```sh
.venv/bin/python tools/convert_weights.py --flow flow.pt --hift hift.pt --llm llm.pt --out-dir ggml/
```

writes `llm.gguf` / `flow.gguf` / `hift.gguf` (format-only conversion, no
quantization). See `docs/WEIGHT_FORMAT.md` for the tensor layout.

## Docs

- `docs/ARCHITECTURE.md` — authoritative design.
- `docs/adr/0001-drop-onnx-runtime-for-compute.md` — why ONNX Runtime is frontend-only.
- `docs/DSP.md` — DSP implementation + validation numbers.
- `docs/WEIGHT_FORMAT.md` — GGUF tensor organisation.
