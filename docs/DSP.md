# DSP frontend — implementation notes and numerical validation

The `src/dsp/` module provides the two audio feature extractors the pipeline
needs before any network runs:

| Function | Reference | Shape |
|---|---|---|
| `larynx::dsp::log_mel_spectrogram` | `whisper.log_mel_spectrogram(speech, n_mels=128)` | `(128, n/160)` |
| `larynx::dsp::fbank` | `torchaudio.compliance.kaldi.fbank(speech, num_mel_bins=80, dither=0, sample_frequency=16000)` | `(1+(n-400)/160, 80)` |

Both are pure C++ (no framework, no GPU). All arithmetic is accumulated in
`double` and rounded to `float32` at the boundary, so the C++ is *more*
accurate than the float32 Python reference; the residual difference is the
reference's own float32 FFT rounding (demonstrated below).

---

## 1. Whisper 128-bin log-mel

**Reference:** `ggml-org/whisper.cpp`, `log_mel_spectrogram` /
`log_mel_spectrogram_worker_thread` (src/whisper.cpp), which itself mirrors
`openai/whisper` `whisper/audio.py::log_mel_spectrogram`. The transform
pipeline — windowed FFT → power spectrum → mel filterbank → `log10` → clamp to
`max - 8.0` → `(x+4)/4` — is shared verbatim.

**Changes vs. whisper.cpp (and why):**

1. **Frame count / padding semantics.** whisper.cpp pads to a fixed 30 s and
   applies its own reflect-pad convention (and drops the last frame via
   `(len - n_fft)/hop`). We instead match `torch.stft(audio, 400, 160,
   window=hann(400), return_complex=True)` exactly: `center=True`,
   `pad_mode="reflect"`, then `stft[..., :-1]`. Note `stft[..., :-1]` drops the
   last **time frame** (not the last frequency bin), so the frame count is
   `n / 160`, and the reflect pad is 200 samples on each side *without*
   repeating the edge sample (`P[i] = x[200-i]` on the left, `P[n+200+j] =
   x[n-2-j]` on the right). This makes the output byte-for-byte shaped like the
   Python reference, which whisper.cpp does not.
2. **Mel filterbank source.** whisper.cpp reads `filters` from the GGML model
   file. We embed the exact `mel_128` (128×201, float32) matrix from
   `openai/whisper` `whisper/assets/mel_filters.npz`
   (`librosa.filters.mel(sr=16000, n_fft=400, n_mels=128)`) as a generated
   header `src/dsp/mel_filters_128.h` (see `tools/gen_mel_filters.py`). This
   avoids reimplementing librosa's mel construction and any parameter drift.
3. **FFT.** `n_fft = 400` is not a power of two. whisper.cpp uses a recursive
   Cooley–Tukey with a DFT fallback for odd sizes; we compute the 201 one-sided
   bins directly by a table-driven DFT (twiddles precomputed once). Same result,
   simpler and exact.

Constants: `n_fft = 400`, `hop = 160`, periodic Hann window
`0.5 - 0.5·cos(2πi/400)`, 201 one-sided bins (Nyquist bin dropped by `[:-1]`),
`log10(max(·, 1e-10))`, global clamp to `max - 8.0`, rescale `(x+4)/4`.

## 2. Kaldi 80-bin fbank

**No known C++ reference exists** for this exact variant; it was implemented
from the Kaldi algorithm definition as expressed by
`torchaudio.compliance.kaldi.fbank` (torchaudio v2.11 source), with every
parameter at its default except the three the task fixes. Exact defaults read
from the source (not guessed):

| Parameter | Value |
|---|---|
| window_type | `povey` = `hann(400, periodic=False) ** 0.85` |
| frame_length / frame_shift | 25 ms / 10 ms → 400 / 160 samples |
| round_to_power_of_two | True → FFT size 512 |
| preemphasis_coefficient | 0.97 |
| remove_dc_offset | True (subtract per-frame mean) |
| snip_edges | True → `n_frames = 1 + (n-400)//160` |
| use_log_fbank / use_power | True / True |
| low_freq / high_freq | 20 Hz / 8000 Hz (Nyquist) |
| dither | 0 (explicit) |
| energy_floor / use_energy / raw_energy | 1.0 / False (energy not appended) |
| log floor | `torch.finfo(float).eps` = 1.1920928955078125e-07 |

Steps: frame → remove DC → preemphasis (`y[0]=0.03·x[0]`,
`y[j]=x[j]-0.97·x[j-1]`) → povey window → zero-pad to 512 → `rfft` → `|·|²` →
mel filterbank (triangular, Sniley mel, 80×257) → `ln(max(·, eps))`.

Per-utterance mean normalization (`subtract_bin_mean`) is applied *after* fbank,
matching the CosyVoice/Campplus frontend `feat - feat.mean(dim=0, keepdim=True)`.

## 3. Numerical validation

Driven by `tests/verify_dsp.py`, which writes float32 PCM, runs the compiled
`larynx_dump` utility, and compares against the Python references. Environment:
`whisper 20250625`, `torch 2.14.0+cpu`, `torchaudio 2.11.0+cpu`, CPU-only, fixed
seed 12345. `max`/`mean` are absolute error over the whole feature tensor.

### C++ vs. float32 Python reference (the deliverable comparison)

| signal | n | log-mel max / mean | fbank max / mean | fbank_norm max / mean |
|---|---|---|---|---|
| noise_3s | 48000 | 3.1e-06 / 2.5e-08 | 5.8e-04 / 3.5e-06 | 5.8e-04 / 3.1e-06 |
| sine_mix_1s | 16000 | 1.2e-05 / 6.9e-08 | 8.8e-04 / 4.3e-05 | 5.1e-04 / 2.8e-05 |
| chirp_2s | 32000 | 3.9e-05 / 3.0e-07 | 1.1e-02 / 1.2e-04 | 1.1e-02 / 1.3e-04 |
| short_500 | 500 | 3.6e-07 / 2.9e-08 | 2.0e-05 / 3.5e-06 | 0.0 / 0.0 (single frame) |
| dc_offset_1s | 16000 | 3.0e-05 / 8.4e-07 | 4.4e-04 / 1.0e-05 | 3.7e-04 / 8.7e-06 |

### Why the fbank max is ~1e-2 — the reference's own float32 rounding

The C++ fbank was also compared against a float64 "true" fbank (same kaldi
pipeline run on float64 input), and the float32 reference was compared against
the same truth:

| signal | C++ vs f64 max / mean | float32-ref vs f64 max / mean |
|---|---|---|
| noise_3s | 1.8e-04 / 3.0e-06 | 5.8e-04 / 1.0e-06 |
| sine_mix_1s | 1.3e-04 / 4.5e-06 | 8.8e-04 / 4.0e-05 |
| chirp_2s | 3.7e-04 / 2.8e-06 | 1.1e-02 / 1.2e-04 |
| short_500 | 2.0e-05 / 3.1e-06 | 1.1e-05 / 8.2e-07 |
| dc_offset_1s | 8.1e-05 / 2.3e-06 | 4.4e-04 / 9.0e-06 |

The C++ sits within ~4e-4 max / ~3e-6 mean of the *true* value, while the
float32 reference deviates from truth by up to ~1e-2. The C++-vs-float32 gap is
therefore the reference's own float32 FFT rounding (largest in the chirp's
near-floor high-frequency bins), **not** a bug. This is the same class of
issue documented in `docs/adr/0001` — and why the architecture prefers one
numerically-honest code path over matching a float32 reference bit-for-bit.

To reproduce: build (`cmake -S . -B build && cmake --build build`) then
`.venv/bin/python tests/verify_dsp.py` (the venv has torch/torchaudio/numpy).
