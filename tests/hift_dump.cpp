// Test utility: reads the validation-case inputs (written by tests/verify_hift.py
// from hift_ref.npz), runs the C++ HiFT vocoder, and dumps every per-stage tensor
// as raw float32 for comparison against the PyTorch reference.
//
// Usage: larynx_hift_dump <hift.gguf> <indir> <outdir>
// Reads : <indir>/mel.f32        (80*30 float32)
//         <indir>/rand_ini.f32   (9     float32)
//         <indir>/sine_waves.f32 (14400*9 float32)
// Writes: <outdir>/*.f32  (f0, sine_wavs, sine_merge, s_stft, conv_pre,
//                          ups{0..2}[_lrelu], source_downs{0..2},
//                          source_resblocks{0..2}, fusion{0..2}, resblock{0..8},
//                          post_resblocks{0..2}, reflection_pad, final_lrelu,
//                          conv_post, magnitude, phase, istft, speech)

#include <cstdio>
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
  std::fwrite(v.data(), 4, v.size(), f);
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <hift.gguf> <indir> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string gguf = argv[1];
  const std::string indir = argv[2];
  const std::string outdir = argv[3];

  std::vector<float> mel, rand_ini, sine_waves;
  if (!read_f32(indir + "/mel.f32", mel)) return 1;
  if (!read_f32(indir + "/rand_ini.f32", rand_ini)) return 1;
  if (!read_f32(indir + "/sine_waves.f32", sine_waves)) return 1;

  larynx::hift::HiftVocoder vocoder;
  if (!vocoder.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  larynx::hift::HiFTDebug dbg;
  std::vector<float> audio;
  if (!vocoder.vocode(mel, rand_ini, sine_waves, audio, &dbg)) {
    std::fprintf(stderr, "vocode failed\n");
    return 1;
  }

  write_f32(outdir + "/f0.f32", dbg.f0);
  write_f32(outdir + "/sine_wavs.f32", dbg.sine_wavs);
  write_f32(outdir + "/sine_merge.f32", dbg.sine_merge);
  write_f32(outdir + "/s_stft.f32", dbg.s_stft);
  write_f32(outdir + "/conv_pre.f32", dbg.conv_pre);

  char name[64];
  for (int i = 0; i < 3; i++) {
    std::snprintf(name, sizeof name, "/ups%d_lrelu.f32", i);
    write_f32(outdir + name, dbg.ups_lrelu[i]);
    std::snprintf(name, sizeof name, "/ups%d.f32", i);
    write_f32(outdir + name, dbg.ups[i]);
    std::snprintf(name, sizeof name, "/source_downs%d.f32", i);
    write_f32(outdir + name, dbg.source_downs[i]);
    std::snprintf(name, sizeof name, "/source_resblocks%d.f32", i);
    write_f32(outdir + name, dbg.source_resblocks[i]);
    std::snprintf(name, sizeof name, "/fusion%d.f32", i);
    write_f32(outdir + name, dbg.fusion[i]);
    std::snprintf(name, sizeof name, "/post_resblocks%d.f32", i);
    write_f32(outdir + name, dbg.post_resblocks[i]);
  }
  for (int i = 0; i < 9; i++) {
    std::snprintf(name, sizeof name, "/resblock%d.f32", i);
    write_f32(outdir + name, dbg.resblocks[i]);
  }

  write_f32(outdir + "/reflection_pad.f32", dbg.reflection_pad);
  write_f32(outdir + "/final_lrelu.f32", dbg.final_lrelu);
  write_f32(outdir + "/conv_post.f32", dbg.conv_post);
  write_f32(outdir + "/magnitude.f32", dbg.magnitude);
  write_f32(outdir + "/phase.f32", dbg.phase);
  write_f32(outdir + "/istft.f32", dbg.istft);
  write_f32(outdir + "/speech.f32", audio);

  return 0;
}
