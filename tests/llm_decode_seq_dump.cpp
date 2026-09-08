// Test utility: verifies the C++ Qwen2 **full autoregressive decode** against
// the PyTorch reference (tests/llm_decode_seq_reference.py, compared by
// tests/verify_llm_decode_seq.py).
//
// Reads the prefill lm_input and the sampled token trajectory (captured from a
// seeded PyTorch run), runs prefill() then one decode() step per injected token,
// and dumps the per-step llm_decoder logits. The C++ does NOT re-implement the
// sampling RNG — it injects the reference's tokens verbatim and only emits the
// deterministic per-step logits, which is what the verify script compares.
//
//   seq_logits.f32   (N*6761 floats)   llm_decoder output for steps 0..N-1,
//                                      concatenated (step-major)
//
// Usage: larynx_llm_decode_seq_dump <llm.gguf> <indir> <outdir>
// Reads : <indir>/lm_input.f32  (L*896 float32; L inferred from size)
//         <indir>/tokens.i32    (N int32 speech token ids)
// Writes: the file above.

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "larynx/llm/llm.h"

namespace {

constexpr int64_t HIDDEN = 896;
constexpr int64_t SPEECH_VOCAB = 6761;

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

  std::vector<int32_t> tokens;
  if (!read_i32(indir + "/tokens.i32", tokens)) return 1;

  larynx::llm::LLM llm;
  if (!llm.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  if (!llm.prefill(lm_input, L)) {
    std::fprintf(stderr, "prefill failed\n");
    return 1;
  }

  std::vector<float> seq_logits;
  seq_logits.reserve(tokens.size() * SPEECH_VOCAB);
  std::vector<float> logits;
  for (size_t i = 0; i < tokens.size(); i++) {
    logits.clear();
    if (!llm.decode(tokens[i], logits)) {
      std::fprintf(stderr, "decode step %zu (token %d) failed\n", i, tokens[i]);
      return 1;
    }
    if ((int64_t)logits.size() != SPEECH_VOCAB) {
      std::fprintf(stderr, "decode step %zu returned %zu logits, expected %lld\n",
                   i, logits.size(), (long long)SPEECH_VOCAB);
      return 1;
    }
    seq_logits.insert(seq_logits.end(), logits.begin(), logits.end());
  }

  write_f32(outdir + "/seq_logits.f32", seq_logits);

  std::fprintf(stderr,
               "decode seq done: prefill L=%d, %zu steps, cache_len=%d, wrote seq_logits.f32\n",
               L, tokens.size(), llm.cache_len());
  return 0;
}
