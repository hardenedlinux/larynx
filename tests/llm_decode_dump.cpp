// Test utility: verifies the C++ Qwen2 **single-step decode** against the
// PyTorch reference (tests/llm_decode_reference.py, compared by
// tests/verify_llm_decode.py).
//
// Reads the prefill lm_input and the already-sampled next token, runs
// prefill() (which populates the KV cache) then one decode() step, and dumps:
//   decode_logits.f32     (SPEECH_VOCAB=6761 floats)   llm_decoder output
//   decode_final_norm.f32 (HIDDEN=896 floats)          model.norm output
//   cache_k.{0..23}.f32   (HEAD_DIM*KV_HEADS*L floats) roped key, GGML layout
//   cache_v.{0..23}.f32   (HEAD_DIM*KV_HEADS*L floats) raw value
//
// Usage: larynx_llm_decode_dump <llm.gguf> <indir> <outdir>
// Reads : <indir>/lm_input.f32  (L*896 float32; L inferred from size)
//         <indir>/next_token.i32 (single int32 token id)
// Writes: the files above.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "larynx/llm/llm.h"

namespace {

constexpr int64_t HIDDEN = 896;

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
    std::fprintf(stderr, "usage: %s <llm.gguf> <indir> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string gguf = argv[1];
  const std::string indir = argv[2];
  const std::string outdir = argv[3];

  std::vector<float> lm_input;
  if (!read_f32(indir + "/lm_input.f32", lm_input)) return 1;
  if (lm_input.size() % HIDDEN != 0) {
    std::fprintf(stderr, "lm_input size %zu not a multiple of %lld\n", lm_input.size(),
                 (long long)HIDDEN);
    return 1;
  }
  const int L = (int)(lm_input.size() / HIDDEN);

  FILE* tf = std::fopen((indir + "/next_token.i32").c_str(), "rb");
  if (!tf) {
    std::fprintf(stderr, "cannot open %s/next_token.i32\n", indir.c_str());
    return 1;
  }
  int32_t next_token = -1;
  if (std::fread(&next_token, 4, 1, tf) != 1) {
    std::fprintf(stderr, "short read on next_token.i32\n");
    std::fclose(tf);
    return 1;
  }
  std::fclose(tf);

  larynx::llm::LLM llm;
  if (!llm.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  if (!llm.prefill(lm_input, L)) {
    std::fprintf(stderr, "prefill failed\n");
    return 1;
  }

  // Dump the prefill KV cache (36 tokens) before the decode step mutates it, so
  // it can be compared directly against the PyTorch past_key_values.
  char name[64];
  for (int i = 0; i < 24; i++) {
    std::snprintf(name, sizeof name, "/cache_k.%d.f32", i);
    write_f32(outdir + name, llm.cache_key(i));
    std::snprintf(name, sizeof name, "/cache_v.%d.f32", i);
    write_f32(outdir + name, llm.cache_value(i));
  }

  std::vector<float> logits, final_norm;
  if (!llm.decode(next_token, logits, &final_norm)) {
    std::fprintf(stderr, "decode failed\n");
    return 1;
  }

  write_f32(outdir + "/decode_logits.f32", logits);
  write_f32(outdir + "/decode_final_norm.f32", final_norm);

  std::fprintf(stderr,
               "decode done: prefill L=%d, token=%d, cache_len=%d, wrote 24x2 cache "
               "tensors + logits + final_norm\n",
               L, next_token, llm.cache_len());
  return 0;
}
