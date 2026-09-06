// Whisper-style log-mel spectrogram.
//
// Reference: ggml-org/whisper.cpp `log_mel_spectrogram` /
// `log_mel_spectrogram_worker_thread` (src/whisper.cpp), which itself mirrors
// openai/whisper `whisper/audio.py::log_mel_spectrogram`.  The transform
// pipeline (windowed FFT -> power spectrum -> mel filterbank -> log10 -> clamp
// to max-8 -> (x+4)/4) is shared; the differences are documented in docs/DSP.md
// and are: (1) whisper.cpp pads to a fixed 30 s and uses its own reflect-pad
// convention, whereas we match torch.stft(center=True, pad_mode="reflect")
// exactly so the output is bit-for-bit shaped like the Python reference; and
// (2) we embed the exact `mel_128` filterbank from whisper's mel_filters.npz
// rather than reading it from a GGML model file.

#include "larynx/dsp/log_mel.h"

#include "mel_filters_128.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace larynx::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr int kNFFT = 400;        // n_fft
constexpr int kHop = 160;         // hop_length
constexpr int kPad = kNFFT / 2;   // center pad (200)
constexpr int kNFFTBins = 201;    // one-sided bins 0..200 (Nyquist bin dropped)

// Precomputed DFT twiddles: cos/sin(2*pi*k*m/N) for k in [0, 201), m in [0, 400).
// N=400 is not a power of two; a direct DFT of the 201 one-sided bins is exact
// and simple, and the tables are computed once.
struct DftTwiddles {
  std::vector<double> cos_table;  // [k * kNFFT + m]
  std::vector<double> sin_table;

  DftTwiddles() : cos_table(kNFFTBins * kNFFT), sin_table(kNFFTBins * kNFFT) {
    for (int k = 0; k < kNFFTBins; ++k) {
      for (int m = 0; m < kNFFT; ++m) {
        double theta = 2.0 * kPi * k * m / (double)kNFFT;
        cos_table[k * kNFFT + m] = std::cos(theta);
        sin_table[k * kNFFT + m] = std::sin(theta);
      }
    }
  }
};

const DftTwiddles &twiddles() {
  static const DftTwiddles t;
  return t;
}

}  // namespace

std::vector<float> log_mel_spectrogram(const float *samples, size_t n, int n_mels) {
  if (n_mels != kWhisperNMels) {
    return {};  // only the 128-bin tokenizer frontend is needed
  }
  if (n < kPad + 1) {
    return {};  // reflect padding of width 200 needs at least 201 samples
  }

  // torch.stft(center=True, pad_mode="reflect"): reflect-pad kPad samples on
  // each side, without repeating the edge sample.
  const size_t padded_len = n + 2 * kPad;
  std::vector<float> padded(padded_len);
  for (int i = 0; i < kPad; ++i) {
    padded[i] = samples[kPad - i];          // left: samples[200], ..., samples[1]
  }
  std::copy(samples, samples + n, padded.begin() + kPad);
  for (int j = 0; j < kPad; ++j) {
    padded[n + kPad + j] = samples[n - 2 - j];  // right: samples[n-2], ..., samples[n-201]
  }

  // periodic Hann window (torch.hann_window(400, periodic=True)).
  std::vector<float> hann(kNFFT);
  for (int i = 0; i < kNFFT; ++i) {
    hann[i] = (float)(0.5 - 0.5 * std::cos(2.0 * kPi * i / (double)kNFFT));
  }

  // whisper drops the last STFT frame via `stft[..., :-1]`: n_frames = n / hop.
  const size_t n_frames = n / kHop;
  const DftTwiddles &tw = twiddles();

  std::vector<float> mel((size_t)n_mels * n_frames, 0.0f);

  for (size_t t = 0; t < n_frames; ++t) {
    const float *win = padded.data() + t * kHop;

    // power spectrum (201 bins) via direct DFT of the windowed frame.
    std::vector<double> mag2(kNFFTBins, 0.0);
    for (int k = 0; k < kNFFTBins; ++k) {
      const double *ct = &tw.cos_table[k * kNFFT];
      const double *st = &tw.sin_table[k * kNFFT];
      double re = 0.0, im = 0.0;
      for (int m = 0; m < kNFFT; ++m) {
        double s = (double)hann[m] * win[m];
        re += s * ct[m];
        im -= s * st[m];
      }
      mag2[k] = re * re + im * im;
    }

    // mel filterbank projection (128 x 201) @ (201) -> (128), then log10.
    for (int j = 0; j < n_mels; ++j) {
      const float *filt = kWhisperMelFilters + j * kNFFTBins;
      double sum = 0.0;
      for (int k = 0; k < kNFFTBins; ++k) {
        sum += filt[k] * mag2[k];
      }
      double logv = std::log10(std::max(sum, 1e-10));
      mel[j * n_frames + t] = (float)logv;
    }
  }

  // clamp to (global max - 8.0), then rescale (x + 4.0)/4.0.
  double mmax = -1e20;
  for (float v : mel) {
    mmax = std::max(mmax, (double)v);
  }
  mmax -= 8.0;
  for (float &v : mel) {
    v = (float)((std::max((double)v, mmax) + 4.0) / 4.0);
  }

  return mel;
}

}  // namespace larynx::dsp
