// Test utility: reproduces the deterministic nucleus candidate set from the
// per-step raw logits, for comparison against tests/llm_sample_reference.py
// (compared by tests/verify_llm_sampling.py).
//
// This is the sampling-only sub-step of Checkpoint 6. It does NOT run the model:
// it reads the raw llm_decoder logits captured by tests/llm_decode_seq_reference.py,
// computes log_softmax -> ignore_eos mask -> softmax -> stable-desc sort -> the
// top_p/top_k nucleus selection, and dumps the result. The multinomial draw is
// exercised separately (it is RNG-dependent and cannot bit-match torch).
//
//   weighted_scores.f32  (N*6761)  log_softmax with ignore_eos mask applied
//   candidates.i32       (N*top_k) candidate token ids, padded with -1
//   cand_count.i32       (N)       candidate count per step
//   cand_probs.f32       (N*top_k) candidate softmax probs, padded with 0
//
// Usage: larynx_llm_sample_dump <indir> <outdir>
// Reads : <indir>/logits.f32  (N*6761 float32)
//         <indir>/min_len.i32  (single int32)
// Writes: the files above.

#include <cstdio>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>

#include "larynx/llm/sampling.h"

namespace {

constexpr int SPEECH_VOCAB = 6761;
constexpr int TOP_K = 25;

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
  if (!f) return false;
  std::fwrite(v.data(), 4, v.size(), f);
  std::fclose(f);
  return true;
}

bool write_i32(const std::string& path, const std::vector<int32_t>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  std::fwrite(v.data(), 4, v.size(), f);
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    std::fprintf(stderr, "usage: %s <indir> <outdir>\n", argv[0]);
    return 2;
  }
  const std::string indir = argv[1];
  const std::string outdir = argv[2];

  std::vector<float> logits;
  if (!read_f32(indir + "/logits.f32", logits)) return 1;
  if (logits.size() % SPEECH_VOCAB != 0) {
    std::fprintf(stderr, "logits size %zu not a multiple of %d\n", logits.size(), SPEECH_VOCAB);
    return 1;
  }
  const int N = (int)(logits.size() / SPEECH_VOCAB);

  int32_t min_len = 0;
  FILE* tf = std::fopen((indir + "/min_len.i32").c_str(), "rb");
  if (!tf || std::fread(&min_len, 4, 1, tf) != 1) {
    std::fprintf(stderr, "cannot read %s/min_len.i32\n", indir.c_str());
    if (tf) std::fclose(tf);
    return 1;
  }
  std::fclose(tf);

  std::vector<float> weighted_scores((size_t)N * SPEECH_VOCAB);
  std::vector<int32_t> candidates((size_t)N * TOP_K, -1);
  std::vector<int32_t> cand_count((size_t)N, 0);
  std::vector<float> cand_probs((size_t)N * TOP_K, 0.0f);

  std::vector<float> logp(SPEECH_VOCAB);
  std::vector<int> idx(SPEECH_VOCAB);
  std::vector<float> prob(SPEECH_VOCAB);
  std::vector<int32_t> sampled((size_t)N);
  for (int i = 0; i < N; i++) {
    const float* raw = logits.data() + (size_t)i * SPEECH_VOCAB;
    larynx::llm::log_softmax(raw, SPEECH_VOCAB, logp.data());
    if (i < min_len) logp[larynx::llm::SPEECH_TOKEN_SIZE] = -INFINITY;

    std::copy(logp.begin(), logp.end(), weighted_scores.begin() + (size_t)i * SPEECH_VOCAB);

    int c = larynx::llm::nucleus_candidates(logp.data(), SPEECH_VOCAB, 0.8f, TOP_K,
                                            idx.data(), prob.data());
    cand_count[i] = c;
    for (int j = 0; j < c; j++) {
      candidates[(size_t)i * TOP_K + j] = idx[j];
      cand_probs[(size_t)i * TOP_K + j] = prob[j];
    }

    // Smoke-test the RNG draw: empty history -> no repetition -> nucleus branch,
    // so the sampled token MUST be inside the candidate set.
    larynx::llm::SamplingParams sp;
    sp.seed = 0;
    sampled[i] = larynx::llm::ras_sample(logp.data(), SPEECH_VOCAB, {}, sp);
  }

  write_f32(outdir + "/weighted_scores.f32", weighted_scores);
  write_i32(outdir + "/candidates.i32", candidates);
  write_i32(outdir + "/cand_count.i32", cand_count);
  write_f32(outdir + "/cand_probs.f32", cand_probs);
  write_i32(outdir + "/sampled.i32", sampled);

  std::fprintf(stderr, "sample dump done: N=%d, min_len=%d, wrote weighted_scores + candidates\n",
               N, min_len);
  return 0;
}
