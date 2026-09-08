# Larynx

A native reimplementation of the CosyVoice3 text-to-speech pipeline with **zero
Python at runtime**. The neural networks (LLM / Flow / HiFT) run on
[GGML](https://github.com/ggml-org/ggml) with CUDA acceleration. Weight
conversion and the acoustic frontend (campplus / speech tokenizer / matcha mel)
are computed **once, offline, in Python** and frozen into files the C++
binary loads verbatim.

**It's designed for better deployment in product as a single executable binary file.**

This project is Human architectured and co-authored by AI.
- LLM: deepseek-v4-pro
- Coding Assistant: Claude Code

## Deps in runtime

```bash
ldd larynx
	linux-vdso.so.1 (0x00007ffceb3fd000)
	libicuuc.so.74 => /lib/x86_64-linux-gnu/libicuuc.so.74 (0x00007aab5e400000)
	libgomp.so.1 => /lib/x86_64-linux-gnu/libgomp.so.1 (0x00007aab66b91000)
	libcudart.so.12 => /lib/x86_64-linux-gnu/libcudart.so.12 (0x00007aab5e000000)
	libcublas.so.12 => /lib/x86_64-linux-gnu/libcublas.so.12 (0x00007aab57600000)
	libcuda.so.1 => /lib/x86_64-linux-gnu/libcuda.so.1 (0x00007aab51e00000)
	libstdc++.so.6 => /lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007aab51a00000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007aab5e717000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007aab66b61000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007aab51600000)
	libicudata.so.74 => /lib/x86_64-linux-gnu/libicudata.so.74 (0x00007aab4f800000)
	/lib64/ld-linux-x86-64.so.2 (0x00007aab66c0a000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007aab66b5a000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007aab66b55000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007aab66b50000)
	libcublasLt.so.12 => /lib/x86_64-linux-gnu/libcublasLt.so.12 (0x00007aab2e800000)
```

## Status

All four phases are implemented and verified numerically against the PyTorch
reference:

1. DSP frontend (Whisper 128-bin log-mel + Kaldi 80-bin fbank) — `verify_dsp.py`
2. Flow decoder (PreLookaheadLayer + DiT ×22 + CFM Euler) — `verify_flow.py`
3. HiFT vocoder — `verify_hift.py`
4. LLM backbone (Qwen2-0.5B + CosyVoice3LM heads + ras_sampling) — `verify_llm*.py`

The end-to-end CLI (LLM → Flow → HiFT) is wired and cross-checked by
`tests/verify_e2e.py`. Still **deferred** (pre-extracted by Python): the ONNX
frontend (campplus + speech tokenizer) and the matcha 80-bin mel — the CLI reads
their outputs as files instead of running them.

## Layout

| dir | purpose |
|---|---|
| `src/dsp/` | Whisper 128-bin log-mel + Kaldi 80-bin fbank (verified) |
| `src/llm/` | Qwen2 autoregressive decoder + CosyVoice3LM heads + ras_sampling (verified) |
| `src/flow/` | DiT flow-matching estimator (verified) |
| `src/hift/` | Causal HiFi-GAN vocoder (verified) |
| `src/pipeline/` | orchestration: tokenizer → LLM → Flow → HiFT |
| `src/cli/` | `larynx` end-to-end entry point |
| `src/frontend/` | reserved for the deferred ONNX frontend |
| `tools/` | offline prep: `convert_weights.py`, `export_tokenizer.py`, `gguf.py`, `gen_mel_filters.py` |
| `tests/` | `verify_*.py` numerical checks + `export_*.py` / `extract_prompt_features.py` asset producers |
| `docs/` | `ARCHITECTURE.md`, ADRs, `DSP.md`, `FLOW.md`, `HIFT.md`, `LLM.md`, `WEIGHT_FORMAT.md` |

## Build

```sh
cmake -S . -B build            # enables CUDA if a toolkit is detected
cmake --build build -j
```

This produces `larynx` plus the `larynx_*_dump` verification utilities. CUDA is
auto-detected: `ggml`'s CUDA backend is compiled when `LARYNX_ENABLE_CUDA=ON`
(default) *and* a CUDA toolchain is found; otherwise it builds CPU-only. Both
backends are linked into `larynx` — at runtime it picks CUDA when a device is
present and falls back to CPU. Force the CPU backend with `LARYNX_BACKEND=cpu`
(used by the numerical verify scripts so they don't depend on an idle GPU).

> Note: the LLM is large. `llm.gguf` is ~2.6 GB, so a CUDA run needs that much
> free VRAM (plus the Flow graph). If `cudaMalloc` reports out-of-memory, either
> free the GPU or prefix the run with `LARYNX_BACKEND=cpu`.

## Prepare models & assets (offline, one-time)

Two Python environments are used:

- **`.venv`** — repo-local, `torch` + `numpy`, for weight conversion.
- **CosyVoice python3.10** — the reference environment that can import
  `cosyvoice`/`transformers`, for tokenizer/asset/prompt extraction.

```sh
# paths used below
MODEL="$HOME/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B"
PY310="$HOME/.local/share/uv/python/cpython-3.10-linux-x86_64-gnu/bin/python3.10"
PYTHONPATH="$HOME/Project/CosyVoice/.local/lib/python3.10/site-packages"
```

**1. Convert weights** (`.venv`):

```sh
.venv/bin/python tools/convert_weights.py \
  --llm "$MODEL/llm.pt" --flow "$MODEL/flow.pt" --hift "$MODEL/hift.pt" \
  --out-dir build/
```

writes `build/llm.gguf` / `build/flow.gguf` / `build/hift.gguf` (format-only
conversion, no quantization). Use `--llm "$MODEL/llm.rl.pt"` for the RL-tuned
checkpoint.

**2. Export the text tokenizer** (python3.10):

```sh
PYTHONPATH="$PYTHONPATH" "$PY310" tools/export_tokenizer.py --out-dir build/tokenizer
```

writes `vocab.tsv` / `merges.txt` / `added_tokens.tsv`.

**3. Export the fixed RNG buffers** (python3.10):

```sh
"$PY310" tests/export_hift_source.py    # -> build/hift_source.bin  (HiFT SineGen2 rand_ini + sine_waves)
"$PY310" tests/export_flow_noise.py     # -> build/flow_noise.bin   (Flow CFM seed noise)
```

These are the model-internal buffers the reference samples once from PyTorch's
RNG at construction; the C++ side loads the frozen values instead of
reimplementing the RNG.

**4. Extract the prompt-voice bundle** (python3.10):

```sh
"$PY310" tests/extract_prompt_features.py --out-dir wavs/flow_inputs
```

runs campplus + speech tokenizer + matcha mel on the prompt wav and writes
`prompt_tokens.i32` / `prompt_feat.f32` / `spk_embedding.f32` (the deferred
frontend, "temporarily handed to Python"). Pass `--prompt-wav <wav>` to use a
different voice.

## Run

```sh
./build/larynx \
  --text "今天天气不错，我们一起去公园散步吧。" \
  --prompt-dir wavs/flow_inputs \
  --out wavs/hello.wav
```

Model/asset paths default to `build/llm.gguf`, `build/flow.gguf`,
`build/hift.gguf`, `build/hift_source.bin`, `build/flow_noise.bin` and
`build/tokenizer`. `--text` is required; `--instruct` defaults to
`"You are a helpful assistant. 请用普通话表达。<|endofprompt|>"` and must contain
`<|endofprompt|>`. Optional dumps:

```sh
./build/larynx --text ... --prompt-dir wavs/flow_inputs --out wavs/hello.wav \
  --seed 0 \
  --dump-tokens wavs/hello.tokens.i32 \
  --dump-mel    wavs/hello.mel.f32 \
  --dump-audio  wavs/hello.audio.f32
```

`--seed` drives the LLM sampling RNG; the speech-token sequence is stochastic,
so different seeds (or no `--seed`) give different audio. `LARYNX_BACKEND=cpu`
forces CPU.

## Verify

```sh
ctest --test-dir build            # DSP / flow / hift / tokenizer / llm numerical checks
"$PY310" tests/verify_e2e.py      # end-to-end CLI vs PyTorch (CPU, slower)
```

The `ctest` suite needs the reference `.npz` dumps, which are regenerated by the
matching `tests/*_reference.py` scripts (see their docstrings). See
`docs/DSP.md`, `docs/FLOW.md`, `docs/HIFT.md`, `docs/LLM.md` for the measured
error numbers.

## Docs

- `docs/ARCHITECTURE.md` — authoritative design.
- `docs/adr/0001-drop-onnx-runtime-for-compute.md` — why ONNX Runtime is frontend-only.
- `docs/DSP.md` / `docs/FLOW.md` / `docs/HIFT.md` / `docs/LLM.md` — per-stage reference + validation numbers.
- `docs/WEIGHT_FORMAT.md` — GGUF tensor organisation.
