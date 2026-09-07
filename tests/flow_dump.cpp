// Test utility: reads the validation-case inputs (written by
// tests/verify_flow.py from flow_ref.npz), runs the C++ Flow decoder, and dumps
// every per-stage tensor as raw float32 for comparison against the PyTorch
// reference.
//
// Usage: larynx_flow_dump <flow.gguf> <indir> <outdir>
// Reads : <indir>/prompt_tokens.i32 (4 x int32)
//         <indir>/tokens.i32        (8 x int32)
//         <indir>/prompt_feat.f32   (8*80 float32)
//         <indir>/spk_embedding.f32 (192  float32)
//         <indir>/noise_z.f32       (24*80 float32)
// Writes: <outdir>/spk.f32 token_embed.f32 prelookahead.f32 mu.f32 cond.f32
//                time_embed.f32 input_embed.f32 norm_out.f32 dphi.f32
//                block{0..21}.f32 feat.f32

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "larynx/flow/flow.h"

namespace {

bool read_i32(const std::string& path, std::vector<int32_t>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    return false;
  }
  std::fseek(f, 0, SEEK_END);
  long size = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (size < 0 || size % 4 != 0) {
    std::fprintf(stderr, "not raw int32: %s\n", path.c_str());
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
    std::fprintf(stderr, "usage: %s <flow.gguf> <indir> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string gguf = argv[1];
  const std::string indir = argv[2];
  const std::string outdir = argv[3];

  std::vector<int32_t> prompt_tokens, tokens;
  std::vector<float> prompt_feat, spk_embedding, noise_z;
  if (!read_i32(indir + "/prompt_tokens.i32", prompt_tokens)) return 1;
  if (!read_i32(indir + "/tokens.i32", tokens)) return 1;
  if (!read_f32(indir + "/prompt_feat.f32", prompt_feat)) return 1;
  if (!read_f32(indir + "/spk_embedding.f32", spk_embedding)) return 1;
  if (!read_f32(indir + "/noise_z.f32", noise_z)) return 1;

  larynx::flow::FlowDecoder decoder;
  if (!decoder.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  larynx::flow::FlowDebug dbg;
  std::vector<float> mel;
  if (!decoder.infer(prompt_tokens, tokens, prompt_feat, spk_embedding, noise_z,
                     mel, &dbg)) {
    std::fprintf(stderr, "infer failed\n");
    return 1;
  }

  write_f32(outdir + "/spk.f32", dbg.spk);
  write_f32(outdir + "/token_embed.f32", dbg.token_embed);
  write_f32(outdir + "/prelookahead.f32", dbg.prelookahead);
  write_f32(outdir + "/mu.f32", dbg.mu);
  write_f32(outdir + "/cond.f32", dbg.cond);
  write_f32(outdir + "/time_embed.f32", dbg.time_embed);
  write_f32(outdir + "/input_proj.f32", dbg.input_proj);
  write_f32(outdir + "/conv_pos.f32", dbg.conv_pos);
  write_f32(outdir + "/input_embed.f32", dbg.input_embed);
  write_f32(outdir + "/norm_out.f32", dbg.norm_out);
  write_f32(outdir + "/dphi.f32", dbg.dphi);
  for (size_t i = 0; i < dbg.blocks.size(); i++) {
    char name[64];
    std::snprintf(name, sizeof name, "/block%zu.f32", i);
    write_f32(outdir + name, dbg.blocks[i]);
  }
  write_f32(outdir + "/feat.f32", mel);

  return 0;
}
