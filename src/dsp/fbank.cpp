// Kaldi-style log filterbank (Campplus input).
//
// Implemented from the Kaldi algorithm definition as expressed by
// torchaudio.compliance.kaldi.fbank (torchaudio v2.11) with all parameters at
// their defaults except num_mel_bins=80, dither=0, sample_frequency=16000.
// There is no known lightweight C++ reference for this exact variant, so the
// steps below are transcribed from the torchaudio source (see docs/DSP.md):
//   frame (snip_edges) -> remove DC -> preemphasis(0.97) -> povey window ->
//   zero-pad to 512 -> rfft -> |.|^2 -> mel filterbank -> ln(floor at eps).

#include "larynx/dsp/fbank.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace larynx::dsp {
namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kSampleRate = 16000.0;
constexpr int kWindowSize = 400;     // 25 ms @ 16 kHz
constexpr int kWindowShift = 160;    // 10 ms @ 16 kHz
constexpr int kPaddedSize = 512;     // round_to_power_of_two(400)
constexpr int kNumFFTBins = 256;     // padded_size / 2
constexpr double kFloatEps = 1.1920928955078125e-07;  // torch.finfo(float).eps

// In-place radix-2 Cooley-Tukey FFT (N = kPaddedSize, a power of two).
// Forward convention X[k] = sum_n x[n] exp(-i 2 pi k n / N), matching torch.fft.rfft.
void fft_radix2(std::vector<double> &re, std::vector<double> &im) {
  const int n = (int)re.size();
  for (int i = 1, j = 0; i < n; ++i) {
    int bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      std::swap(re[i], re[j]);
      std::swap(im[i], im[j]);
    }
  }
  for (int len = 2; len <= n; len <<= 1) {
    const double ang = -2.0 * kPi / len;
    const double w_re = std::cos(ang), w_im = std::sin(ang);
    const int half = len >> 1;
    for (int i = 0; i < n; i += len) {
      double c_re = 1.0, c_im = 0.0;
      for (int k = 0; k < half; ++k) {
        const double u_re = re[i + k], u_im = im[i + k];
        const double v_re = re[i + k + half] * c_re - im[i + k + half] * c_im;
        const double v_im = re[i + k + half] * c_im + im[i + k + half] * c_re;
        re[i + k] = u_re + v_re;
        im[i + k] = u_im + v_im;
        re[i + k + half] = u_re - v_re;
        im[i + k + half] = u_im - v_im;
        const double nw_re = c_re * w_re - c_im * w_im;
        c_im = c_re * w_im + c_im * w_re;
        c_re = nw_re;
      }
    }
  }
}

// Mel filterbank (num_bins x (kNumFFTBins+1)), last column zero (the padded
// bin). Matches torchaudio get_mel_banks with vtln_warp_factor == 1.0.
std::vector<double> build_mel_banks(int num_bins) {
  const double fft_bin_width = kSampleRate / kPaddedSize;  // 31.25 Hz
  const double mel_low = 1127.0 * std::log(1.0 + 20.0 / 700.0);
  const double mel_high = 1127.0 * std::log(1.0 + 8000.0 / 700.0);
  const double delta = (mel_high - mel_low) / (num_bins + 1.0);

  std::vector<double> banks((size_t)num_bins * (kNumFFTBins + 1), 0.0);
  for (int b = 0; b < num_bins; ++b) {
    const double left = mel_low + b * delta;
    const double center = mel_low + (b + 1.0) * delta;
    const double right = mel_low + (b + 2.0) * delta;
    for (int k = 0; k < kNumFFTBins; ++k) {
      const double mel_k = 1127.0 * std::log(1.0 + (fft_bin_width * k) / 700.0);
      const double up = (mel_k - left) / (center - left);
      const double down = (right - mel_k) / (right - center);
      banks[b * (kNumFFTBins + 1) + k] = std::max(0.0, std::min(up, down));
    }
  }
  return banks;
}

const std::vector<double> &mel_banks(int num_bins) {
  static const std::vector<double> banks = build_mel_banks(80);
  (void)num_bins;
  return banks;
}

}  // namespace

std::vector<float> fbank(const float *samples, size_t n, int num_mel_bins) {
  if (num_mel_bins != 80) {
    return {};  // only the 80-bin Campplus frontend is needed
  }
  if (n < kWindowSize) {
    return {};  // snip_edges: no frame fits entirely within the signal
  }

  const size_t n_frames = 1 + (n - kWindowSize) / kWindowShift;

  // povey window = hann(400, periodic=False) ** 0.85
  std::vector<double> povey(kWindowSize);
  for (int i = 0; i < kWindowSize; ++i) {
    const double h = 0.5 - 0.5 * std::cos(2.0 * kPi * i / (double)(kWindowSize - 1));
    povey[i] = std::pow(h, 0.85);
  }

  const std::vector<double> &banks = mel_banks(num_mel_bins);

  std::vector<float> feat(n_frames * (size_t)num_mel_bins);
  std::vector<double> dc(kWindowSize);
  std::vector<double> frame(kPaddedSize, 0.0);
  std::vector<double> re(kPaddedSize), im(kPaddedSize);
  std::vector<double> power(kNumFFTBins + 1);

  for (size_t i = 0; i < n_frames; ++i) {
    const float *seg = samples + i * kWindowShift;

    // remove DC offset: subtract this frame's mean.
    double mean = 0.0;
    for (int j = 0; j < kWindowSize; ++j) mean += seg[j];
    mean /= kWindowSize;
    for (int j = 0; j < kWindowSize; ++j) dc[j] = (double)seg[j] - mean;

    // preemphasis: frame[0] = 0.03*dc[0]; frame[j] = dc[j] - 0.97*dc[j-1].
    frame[0] = 0.03 * dc[0];
    for (int j = 1; j < kWindowSize; ++j) frame[j] = dc[j] - 0.97 * dc[j - 1];

    // povey window, then zero-pad to kPaddedSize.
    for (int j = 0; j < kWindowSize; ++j) frame[j] *= povey[j];
    for (int j = kWindowSize; j < kPaddedSize; ++j) frame[j] = 0.0;

    // rfft -> power spectrum (|X|^2), bins 0..256.
    for (int j = 0; j < kPaddedSize; ++j) {
      re[j] = frame[j];
      im[j] = 0.0;
    }
    fft_radix2(re, im);
    for (int k = 0; k <= kNumFFTBins; ++k) {
      power[k] = re[k] * re[k] + im[k] * im[k];
    }

    // mel filterbank projection, then natural log with an eps floor.
    for (int b = 0; b < num_mel_bins; ++b) {
      const double *bank = banks.data() + b * (kNumFFTBins + 1);
      double sum = 0.0;
      for (int k = 0; k <= kNumFFTBins; ++k) {
        sum += bank[k] * power[k];
      }
      feat[i * (size_t)num_mel_bins + b] = (float)std::log(std::max(sum, kFloatEps));
    }
  }

  return feat;
}

void subtract_bin_mean(std::vector<float> &feat, size_t n_frames, size_t n_bins) {
  for (size_t b = 0; b < n_bins; ++b) {
    double mean = 0.0;
    for (size_t i = 0; i < n_frames; ++i) mean += feat[i * n_bins + b];
    mean /= n_frames;
    for (size_t i = 0; i < n_frames; ++i) {
      feat[i * n_bins + b] = (float)((double)feat[i * n_bins + b] - mean);
    }
  }
}

}  // namespace larynx::dsp
