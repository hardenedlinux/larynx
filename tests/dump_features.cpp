// Test utility: reads a raw float32 mono PCM file (16 kHz), computes the two
// DSP frontend features, and writes them back as raw float32 so
// tests/verify_dsp.py can compare against the Python references.
//
// Usage: larynx_dump <input.f32> <outdir>
// Writes: <outdir>/logmel.f32      (128 x n_frames_lm,  row-major)
//         <outdir>/fbank.f32       (n_frames_fb x 80,   row-major)
//         <outdir>/fbank_norm.f32  (n_frames_fb x 80,   per-bin mean subtracted)
//         <outdir>/shapes.txt

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "larynx/dsp/fbank.h"
#include "larynx/dsp/log_mel.h"

namespace {

bool read_all_floats(const std::string &path, std::vector<float> &out) {
  FILE *f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0 || size % 4 != 0) {
    std::fprintf(stderr, "input is not raw float32\n");
    std::fclose(f);
    return false;
  }
  out.resize(size / 4);
  if (std::fread(out.data(), 4, out.size(), f) != out.size()) {
    std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  return true;
}

bool write_all_floats(const std::string &path, const std::vector<float> &v) {
  FILE *f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return false;
  }
  std::fwrite(v.data(), 4, v.size(), f);
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char **argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <input.f32> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string outdir = argv[2];

  std::vector<float> samples;
  if (!read_all_floats(argv[1], samples)) return 1;

  const size_t n = samples.size();

  std::vector<float> lm = larynx::dsp::log_mel_spectrogram(samples.data(), n, 128);
  const size_t n_frames_lm = n / 160;
  if (lm.size() != 128 * n_frames_lm) {
    std::fprintf(stderr, "log-mel shape mismatch: got %zu, expected %zu\n",
                 lm.size(), 128 * n_frames_lm);
    return 1;
  }

  std::vector<float> fb = larynx::dsp::fbank(samples.data(), n, 80);
  const size_t n_frames_fb = 1 + (n - 400) / 160;
  if (fb.size() != n_frames_fb * 80) {
    std::fprintf(stderr, "fbank shape mismatch: got %zu, expected %zu\n",
                 fb.size(), n_frames_fb * 80);
    return 1;
  }

  std::vector<float> fb_norm = fb;
  larynx::dsp::subtract_bin_mean(fb_norm, n_frames_fb, 80);

  write_all_floats(outdir + "/logmel.f32", lm);
  write_all_floats(outdir + "/fbank.f32", fb);
  write_all_floats(outdir + "/fbank_norm.f32", fb_norm);

  FILE *s = std::fopen((outdir + "/shapes.txt").c_str(), "w");
  if (s) {
    std::fprintf(s, "n_samples %zu\n", n);
    std::fprintf(s, "logmel %d %zu\n", 128, n_frames_lm);
    std::fprintf(s, "fbank %zu %d\n", n_frames_fb, 80);
    std::fclose(s);
  }
  return 0;
}
