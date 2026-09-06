#pragma once

#include <cstddef>
#include <vector>

namespace larynx::dsp {

// Whisper-style log-mel spectrogram (128 bins).
//
// Numerically matches the Python reference
//   whisper.log_mel_spectrogram(samples, n_mels=128)
// i.e. torch.stft with n_fft=400, hop_length=160, a *periodic* Hann window,
// center=True, pad_mode="reflect"; power spectrum (201 one-sided bins);
// projection through the precomputed mel filterbank; log10 with a 1e-10 floor;
// global clamp to (max - 8.0); rescale (x + 4)/4. The last STFT frame is
// dropped (`stft[..., :-1]`), matching the Python reference.
//
// Input:  `samples` is float32 mono PCM in [-1, 1] at 16 kHz, `n` samples.
//         Requires n >= 201 (torch.stft's reflect padding).
// Output: row-major float32, shape (n_mels x n_frames), n_frames = n/160.
//         Only n_mels == 128 is supported (the speech tokenizer's input).
std::vector<float> log_mel_spectrogram(const float *samples, size_t n,
                                       int n_mels = 128);

}  // namespace larynx::dsp
