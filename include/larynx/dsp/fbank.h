#pragma once

#include <cstddef>
#include <vector>

namespace larynx::dsp {

// Kaldi-style log filterbank (80 bins), the Campplus speaker-embedding input.
//
// Matches the Python reference
//   torchaudio.compliance.kaldi.fbank(samples, num_mel_bins=80, dither=0,
//                                     sample_frequency=16000)
// with every other parameter left at its torchaudio default: povey window,
// frame_length=25ms, frame_shift=10ms, preemphasis_coefficient=0.97,
// remove_dc_offset=True, snip_edges=True, round_to_power_of_two=True (512-point
// FFT), use_log_fbank=True, use_power=True, low_freq=20 Hz, high_freq=8000 Hz,
// use_energy=False.
//
// Input:  `samples` is float32 mono PCM at 16 kHz, `n` samples. Requires n >= 400.
// Output: row-major float32, shape (n_frames x num_mel_bins),
//         n_frames = 1 + (n - 400)/160.
std::vector<float> fbank(const float *samples, size_t n, int num_mel_bins = 80);

// Per-utterance mean normalization applied by the CosyVoice/Campplus frontend:
// subtract each mel bin's mean across frames (feat -= mean over the frame axis).
// Matches `feat - feat.mean(dim=0, keepdim=True)`. Operates in place.
void subtract_bin_mean(std::vector<float> &feat, size_t n_frames, size_t n_bins);

}  // namespace larynx::dsp
