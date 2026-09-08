// Test utility: assembles the CosyVoice3LM prefill lm_input from raw token ids
// (the voice-cloning branch) and dumps it for comparison against the PyTorch
// reference (tests/llm_zeroshot_reference.py -> llm_zeroshot_ref.npz).
//
// This is the C++ half of the Phase-4 wrap-up 阶段2 check: it proves
// build_lm_input() reproduces, bit-exactly, the embedding sequence
// CosyVoice3LM.inference constructs —
//
//   [sos(6561); embed_tokens([prompt_text; text]); task_id(6563);
//    speech_embedding(prompt_speech_token)]
//
// The dump also slices out the three middle/trailing blocks so the verify step
// can localize a mismatch to a specific embedding table or index (they are pure
// slices of lm_input, not independently computed).
//
// Usage: larynx_llm_build_dump <llm.gguf> <indir> <outdir>
// Reads : <indir>/text_tokens.i32          (Pt+T int32 token ids)
//         <indir>/prompt_speech_token.i32  (P int32 speech-token ids; may be empty)
// Writes: <outdir>/lm_input.f32            (L*896 float32, L = 1+PtT+1+P)
//         <outdir>/sos_emb.f32             (896)
//         <outdir>/text_emb.f32            (PtT*896)
//         <outdir>/task_id_emb.f32         (896)
//         <outdir>/prompt_speech_token_emb.f32 (P*896)

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "larynx/llm/llm.h"

namespace {

constexpr int64_t HIDDEN = 896;

bool read_i32(const std::string& path, std::vector<int>& out) {
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
  std::vector<int32_t> raw(size / 4);
  if (std::fread(raw.data(), 4, raw.size(), f) != raw.size()) {
    std::fprintf(stderr, "short read on %s\n", path.c_str());
    std::fclose(f);
    return false;
  }
  std::fclose(f);
  out.assign(raw.begin(), raw.end());
  return true;
}

bool write_f32(const std::string& path, const float* data, size_t n) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) {
    std::fprintf(stderr, "cannot write %s\n", path.c_str());
    return false;
  }
  bool ok = std::fwrite(data, 4, n, f) == n;
  std::fclose(f);
  return ok;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <llm.gguf> <indir> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string gguf = argv[1];
  const std::string indir = argv[2];
  const std::string outdir = argv[3];

  std::vector<int> text_tokens, prompt_speech_token;
  if (!read_i32(indir + "/text_tokens.i32", text_tokens)) return 1;
  if (!read_i32(indir + "/prompt_speech_token.i32", prompt_speech_token)) return 1;
  const size_t PtT = text_tokens.size(), P = prompt_speech_token.size();

  larynx::llm::LLM llm;
  if (!llm.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  std::vector<float> lm_input;
  int L = 0;
  if (!llm.build_lm_input(text_tokens, prompt_speech_token, lm_input, &L)) {
    std::fprintf(stderr, "build_lm_input failed\n");
    return 1;
  }
  if ((size_t)L != 1 + PtT + 1 + P || lm_input.size() != (size_t)L * HIDDEN) {
    std::fprintf(stderr, "bad L=%d vs expected %zu (PtT=%zu P=%zu)\n", L, 1 + PtT + 1 + P, PtT, P);
    return 1;
  }

  const float* p = lm_input.data();
  if (!write_f32(outdir + "/lm_input.f32", p, (size_t)L * HIDDEN)) return 1;
  if (!write_f32(outdir + "/sos_emb.f32", p, HIDDEN)) return 1;
  p += HIDDEN;
  if (!write_f32(outdir + "/text_emb.f32", p, PtT * HIDDEN)) return 1;
  p += PtT * HIDDEN;
  if (!write_f32(outdir + "/task_id_emb.f32", p, HIDDEN)) return 1;
  p += HIDDEN;
  if (!write_f32(outdir + "/prompt_speech_token_emb.f32", p, P * HIDDEN)) return 1;

  std::fprintf(stderr, "build_lm_input done: L=%d (1 + %zu text + 1 + %zu prompt_speech)\n",
               L, PtT, P);
  return 0;
}
