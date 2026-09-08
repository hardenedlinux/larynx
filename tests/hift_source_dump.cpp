// Test utility: verifies the HiFT SineGen2 fixed-source loading path
// (HiftVocoder::load_source) against the exported asset (tests/export_hift_source.py
// -> build/hift_source.bin).
//
// This is the C++ half of tests/verify_hift_source.py. It
//   1. loads hift.gguf + hift_source.bin,
//   2. dumps the buffers read back by load_source (rand_ini + the full sine_waves
//      bank) so the verify step can confirm they are byte-identical to the file,
//   3. runs the *production* 3-arg vocode() (which slices the bank internally)
//      and, as a self-consistency check, the 5-arg vocode() fed the same slice
//      explicitly — both audios must be bit-identical.
//
// Usage: larynx_hift_source_dump <hift.gguf> <hift_source.bin> <indir> <outdir>
// Reads : <indir>/mel.f32   (80*T_mel float32)
// Writes: <outdir>/rand_ini.f32        (9)         loaded phase offsets
//         <outdir>/sine_waves.f32      (7200000*9) loaded full bank
//         <outdir>/audio.f32           (T_mel*480) production vocode()
//         <outdir>/audio_explicit.f32  (T_mel*480) 5-arg vocode() with same slice

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "larynx/hift/hift.h"

namespace {

bool read_f32(const std::string& path, std::vector<float>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0 || size % 4 != 0) {
    std::fprintf(stderr, "not raw float32: %s\n", path.c_str());
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

bool write_f32(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return false;
  }
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: %s <hift.gguf> <hift_source.bin> <indir> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string gguf = argv[1];
  const std::string source_bin = argv[2];
  const std::string indir = argv[3];
  const std::string outdir = argv[4];

  std::vector<float> mel;
  if (!read_f32(indir + "/mel.f32", mel)) return 1;

  larynx::hift::HiftVocoder vocoder;
  if (!vocoder.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }
  if (!vocoder.load_source(source_bin)) {
    std::fprintf(stderr, "failed to load source asset %s\n", source_bin.c_str());
    return 1;
  }

  const std::vector<float>& rand_ini = vocoder.source_rand_ini();
  const std::vector<float>& sine_waves = vocoder.source_sine_waves();
  if (!write_f32(outdir + "/rand_ini.f32", rand_ini)) return 1;
  if (!write_f32(outdir + "/sine_waves.f32", sine_waves)) return 1;

  // Production path (slices the bank internally).
  std::vector<float> audio;
  if (!vocoder.vocode(mel, audio)) {
    std::fprintf(stderr, "production vocode failed\n");
    return 1;
  }
  if (!write_f32(outdir + "/audio.f32", audio)) return 1;

  // Explicit path with the same slice, for the self-consistency check.
  const int T_MEL = (int)(mel.size() / 80);
  const int L_S = T_MEL * 480;
  const size_t need = (size_t)L_S * 9;
  std::vector<float> sw(sine_waves.begin(), sine_waves.begin() + need);
  std::vector<float> audio_explicit;
  if (!vocoder.vocode(mel, rand_ini, sw, audio_explicit, nullptr)) {
    std::fprintf(stderr, "explicit vocode failed\n");
    return 1;
  }
  if (!write_f32(outdir + "/audio_explicit.f32", audio_explicit)) return 1;

  std::fprintf(stderr, "source dump done: rand_ini=%zu sine_waves=%zu audio=%zu (T_MEL=%d)\n",
               rand_ini.size(), sine_waves.size(), audio.size(), T_MEL);
  return 0;
}
