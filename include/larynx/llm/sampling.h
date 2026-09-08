#pragma once

#include <cstdint>
#include <vector>

namespace larynx::llm {

// CosyVoice speech-token vocabulary: real speech tokens are [0, SPEECH_TOKEN_SIZE);
// ids >= SPEECH_TOKEN_SIZE are the stop/fill/task/sos control tokens.
constexpr int SPEECH_TOKEN_SIZE = 6561;

// Sampling hyperparameters (cosyvoice3.yaml: ras_sampling top_p=0.8 top_k=25).
struct SamplingParams {
  float top_p = 0.8f;
  int top_k = 25;
  int win_size = 10;
  float tau_r = 0.1f;
  unsigned seed = 0;
};

// Stable argsort of `x` descending: idx[0] is the index of the largest value,
// ties broken by smaller index first (matches torch sort(descending, stable)).
void argsort_desc(const float* x, int n, int* idx);

// Host float32 log_softmax / softmax over x[0..n), written to out[0..n).
void log_softmax(const float* x, int n, float* out);
void softmax(const float* x, int n, float* out);

// Deterministic nucleus candidate set — the exact CosyVoice ``nucleus_sampling``
// (top_p + top_k) policy, minus the multinomial draw. `logp` are the LOG-SOFTMAX
// scores (ignore_eos mask already applied, i.e. masked slot is -inf). Fills
// out_idx/out_prob (each sized n) with the selected tokens (in descending-prob
// order) and their softmax probs; returns the count. Deterministic: no RNG.
int nucleus_candidates(const float* logp, int n, float top_p, int top_k,
                       int* out_idx, float* out_prob);

// Repetition-aware sampling (CosyVoice ``ras_sampling``). `logp` are LOG-SOFTMAX
// scores (ignore_eos already applied); `decoded_tokens` is the history used for
// the repetition check. Returns a sampled token id. The multinomial draw uses a
// std::mt19937 seeded by p.seed — NOT torch's RNG — so the chosen token differs
// from a torch run, but it is drawn from the same candidate set / distribution.
int ras_sample(const float* logp, int n, const std::vector<int>& decoded_tokens,
               const SamplingParams& p);

}  // namespace larynx::llm
