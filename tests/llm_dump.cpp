// Test utility: reads the prefill lm_input (written by tests/verify_llm.py from
// llm_ref.npz), runs the C++ Qwen2 backbone prefill, and dumps the per-layer
// hidden states, the final RMSNorm output, and the llm_decoder logits as raw
// float32 for comparison against the PyTorch reference.
//
// Usage: larynx_llm_dump <llm.gguf> <indir> <outdir>
// Reads : <indir>/lm_input.f32  (L*896 float32; L inferred from size)
// Writes: <outdir>/hidden_states.{0..23}.f32  (each L*896 float32)
//         <outdir>/final_norm.f32             (L*896 float32)
//         <outdir>/logits.f32                 (L*6761 float32)

#include <cstdio>
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

  larynx::llm::LLM llm;
  if (!llm.load(gguf)) {
    std::fprintf(stderr, "failed to load %s\n", gguf.c_str());
    return 1;
  }

  larynx::llm::LLMDebug dbg;
  if (!llm.prefill(lm_input, L, &dbg)) {
    std::fprintf(stderr, "prefill failed\n");
    return 1;
  }

  char name[64];
  for (int i = 0; i < (int)dbg.hidden_states.size(); i++) {
    std::snprintf(name, sizeof name, "/hidden_states.%d.f32", i);
    write_f32(outdir + name, dbg.hidden_states[i]);
  }
  write_f32(outdir + "/final_norm.f32", dbg.final_norm);
  write_f32(outdir + "/logits.f32", dbg.logits);

  std::fprintf(stderr, "prefill done: L=%d, wrote %d hidden states + final_norm + logits\n",
               L, (int)dbg.hidden_states.size());
  return 0;
}
