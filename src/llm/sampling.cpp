#include "larynx/llm/sampling.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>

namespace larynx::llm {

namespace {

// -inf for masking (torch uses float('-inf') in sampling_ids).
constexpr float kNegInf = -INFINITY;

float max_of(const float* x, int n) {
  float m = -INFINITY;
  for (int i = 0; i < n; i++) m = std::max(m, x[i]);
  return m;
}

// Stable multinomial draw over unnormalized weights w[0..n) (weights >= 0).
// Returns the drawn index. Uses a local mt19937 so it is reproducible given
// `seed` but independent of torch's RNG.
int multinomial(const float* w, int n, unsigned seed) {
  std::mt19937 gen(seed);
  double total = 0.0;
  for (int i = 0; i < n; i++) total += (double)w[i];
  if (total <= 0.0) return 0;  // all-zero (shouldn't happen with valid logits)
  std::uniform_real_distribution<double> dist(0.0, total);
  double r = dist(gen);
  double cum = 0.0;
  for (int i = 0; i < n; i++) {
    cum += (double)w[i];
    if (r < cum) return i;
  }
  return n - 1;
}

}  // namespace

void argsort_desc(const float* x, int n, int* idx) {
  for (int i = 0; i < n; i++) idx[i] = i;
  std::stable_sort(idx, idx + n, [&](int a, int b) {
    // descending value; ties broken by smaller index (torch stable sort).
    if (x[a] != x[b]) return x[a] > x[b];
    return a < b;
  });
}

void log_softmax(const float* x, int n, float* out) {
  float m = max_of(x, n);
  double sum = 0.0;
  for (int i = 0; i < n; i++) sum += std::exp((double)(x[i] - m));
  float logsum = m + (float)std::log(sum);
  for (int i = 0; i < n; i++) out[i] = x[i] - logsum;
}

void softmax(const float* x, int n, float* out) {
  float m = max_of(x, n);
  double sum = 0.0;
  for (int i = 0; i < n; i++) {
    out[i] = std::exp(x[i] - m);
    sum += out[i];
  }
  float inv = (float)(1.0 / sum);
  for (int i = 0; i < n; i++) out[i] *= inv;
}

int nucleus_candidates(const float* logp, int n, float top_p, int top_k,
                       int* out_idx, float* out_prob) {
  // mirror CosyVoice nucleus_sampling: softmax -> sort desc stable -> take while
  // cum_prob < top_p and count < top_k.
  std::vector<float> probs((size_t)n);
  softmax(logp, n, probs.data());

  std::vector<int> sorted_idx((size_t)n);
  argsort_desc(probs.data(), n, sorted_idx.data());

  int count = 0;
  float cum_prob = 0.0f;
  for (int i = 0; i < n; i++) {
    if (cum_prob < top_p && count < top_k) {
      cum_prob += probs[sorted_idx[i]];
      out_idx[count] = sorted_idx[i];
      out_prob[count] = probs[sorted_idx[i]];
      count++;
    } else {
      break;
    }
  }
  return count;
}

int ras_sample(const float* logp, int n, const std::vector<int>& decoded_tokens,
               const SamplingParams& p) {
  // nucleus draw.
  std::vector<int> idx((size_t)n);
  std::vector<float> prob((size_t)n);
  int count = nucleus_candidates(logp, n, p.top_p, p.top_k, idx.data(), prob.data());
  int top_ids = idx[multinomial(prob.data(), count, p.seed)];

  // repetition-aware fallback (ras_sampling): if the drawn token repeats too much
  // in the recent window, mask it and re-sample from the full softmax.
  int rep_num = 0;
  int win = std::min<int>(p.win_size, (int)decoded_tokens.size());
  for (int i = (int)decoded_tokens.size() - win; i < (int)decoded_tokens.size(); i++) {
    if (decoded_tokens[i] == top_ids) rep_num++;
  }
  if ((float)rep_num >= (float)p.win_size * p.tau_r) {
    std::vector<float> masked((size_t)n);
    std::copy(logp, logp + n, masked.begin());
    masked[top_ids] = kNegInf;
    // random_sampling: full softmax multinomial (masked slot -> prob 0).
    std::vector<float> full((size_t)n);
    softmax(masked.data(), n, full.data());
    top_ids = multinomial(full.data(), n, p.seed + 1);
  }
  return top_ids;
}

}  // namespace larynx::llm
