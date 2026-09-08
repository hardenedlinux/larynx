// Test utility: runs the C++ LLM's fully autonomous generation loop for a single
// prefill input, and dumps the resulting trajectory + stop statistics. Driven by
// tests/verify_generate.py, which feeds several real texts (via the reference's
// pre-built lm_input) and checks the stop behavior.
//
// This is the decode-loop-autonomy half of Checkpoint 6: unlike
// larynx_llm_decode_seq_dump (which injects a captured token trajectory), this
// tool lets the C++ sample its own tokens with its own RNG, feed them back into
// its own KV cache, and stop on its own when a stop token (>=6561) is drawn.
//
//   tokens.i32       (N int32)  sampled speech tokens, EXCLUDING the stop token
//   stop_token.i32   (1 int32)  stop token id (6561..6760), or -1 if truncated
//   steps.i32        (1 int32)  decode steps executed == N
//
// Usage: larynx_llm_generate_dump <llm.gguf> <indir> <outdir>
// Reads : <indir>/lm_input.f32 (L*896 float32; L inferred from size)
//         <indir>/min_len.i32, <indir>/max_len.i32, <indir>/seed.i32 (single int32 each)
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

bool read_i32_one(const std::string& path, int32_t& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f || std::fread(&out, 4, 1, f) != 1) {
    if (f) std::fclose(f);
    std::fprintf(stderr, "cannot read %s\n", path.c_str());
    return false;
  }
  std::fclose(f);
  return true;
}

bool write_i32(const std::string& path, const std::vector<int32_t>& v) {
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

  int32_t min_len = 0, max_len = 0, seed = 0;
  if (!read_i32_one(indir + "/min_len.i32", min_len)) return 1;
  if (!read_i32_one(indir + "/max_len.i32", max_len)) return 1;
  if (!read_i32_one(indir + "/seed.i32", seed)) return 1;

  larynx::llm::LLM llm;
  if (!llm.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  larynx::llm::GenerationResult res;
  if (!llm.generate(lm_input, L, (int)min_len, (int)max_len, (unsigned)seed, &res)) {
    std::fprintf(stderr, "generate failed\n");
    return 1;
  }

  std::vector<int32_t> tokens;
  tokens.reserve(res.tokens.size());
  for (int t : res.tokens) tokens.push_back((int32_t)t);
  write_i32(outdir + "/tokens.i32", tokens);
  write_i32(outdir + "/stop_token.i32", {(int32_t)res.stop_token});
  write_i32(outdir + "/steps.i32", {(int32_t)res.steps});

  std::fprintf(stderr,
               "generate done: L=%d min_len=%d max_len=%d seed=%d -> %zu tokens, "
               "stop_token=%d (%s), cache_len=%d\n",
               L, (int)min_len, (int)max_len, (int)seed, res.tokens.size(), res.stop_token,
               res.stopped ? "normal-stop" : (res.truncated ? "max_len-truncated" : "?"),
               llm.cache_len());
  return 0;
}
