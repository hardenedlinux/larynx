#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace larynx::llm {

// Qwen2 backbone (GGML + CUDA) with the CosyVoice3 speech_embedding / llm_decoder
// heads. Phase 4 checkpoint 3: prefill numerical verification only.
std::string module_name();

// Per-stage capture for tests/verify_llm.py. All buffers are raw float32 in
// GGML layout (feature-major): a (1, L, D) PyTorch tensor is stored flat as
// [t*D + d], which is byte-identical to ggml's [D, L, 1] order [d + t*D].
struct LLMDebug {
  // One buffer per transformer layer (24), each L*896 floats: the raw post-residual
  // output of each layer (before the final model.norm). Note the reference does not
  // expose layer 23's raw output (transformers applies model.norm in place); it is
  // checked via `final_norm` below.
  std::vector<std::vector<float>> hidden_states;
  // model.norm (final RMSNorm) applied to layer 23: L*896 floats. This is what
  // Qwen2Encoder.forward returns as `hidden_states[-1]` and feeds to llm_decoder.
  std::vector<float> final_norm;
  // llm_decoder output (logits, pre-softmax): L*6761 floats.
  std::vector<float> logits;
};

struct GenerationResult;

class LLM {
 public:
  LLM();
  ~LLM();

  LLM(const LLM&) = delete;
  LLM& operator=(const LLM&) = delete;

  // Load weights from an llm.gguf produced by tools/convert_weights.py.
  bool load(const std::string& path);

  // Run the Qwen2 prefill (causal) over `lm_input` (1, L, 896) row-major.
  // `lm_input` is the already-concatenated embedding sequence
  // [sos; text; task_id; prompt_speech] that CosyVoice3LM.inference builds.
  // Populates the internal KV cache so a following decode() continues
  // autoregressively from position L. Returns false on error. When `debug` is
  // non-null, captures the 24 per-layer hidden states and the llm_decoder logits.
  bool prefill(const std::vector<float>& lm_input, int L, LLMDebug* debug = nullptr);

  // Single-token autoregressive decode step using the KV cache populated by
  // prefill()/previous decode() calls. `token` is a speech token id in
  // [0, SPEECH_VOCAB); its embedding is looked up from speech_embedding. The
  // new token's logits (SPEECH_VOCAB floats) are written to `logits`; if
  // `final_norm` is non-null, the model.norm output (HIDDEN floats) is written
  // there. Appends the new key/value to the cache. Returns false on error.
  bool decode(int token, std::vector<float>& logits,
              std::vector<float>* final_norm = nullptr);

    // Number of tokens currently held in the KV cache (0 until prefill runs).
  int cache_len() const;

  // For verification: the ROPED key / raw value cache of `layer`, flat float32
  // in GGML layout [HEAD_DIM, KV_HEADS, cache_len] (d fastest, then kv, then
  // seq). Valid only after prefill() has populated the cache.
  const std::vector<float>& cache_key(int layer) const;
  const std::vector<float>& cache_value(int layer) const;

  // Fully autonomous autoregressive generation: prefill `lm_input`, then sample
  // each token with the C++ ras_sampling RNG, feed it back into the KV cache via
  // decode(), and continue until a stop token (>= SPEECH_TOKEN_SIZE=6561) is
  // sampled or `max_len` steps elapse. `min_len` steps use ignore_eos=True (the
  // same gate as CosyVoice3LM.inference_wrapper). Results are written to `out`
  // (must be non-null); returns false on error.
  bool generate(const std::vector<float>& lm_input, int L,
                int min_len, int max_len, unsigned seed, GenerationResult* out);

 private:
  struct Impl;
  Impl* impl_;
};

// Outcome of a generate() call (all fields valid on success).
struct GenerationResult {
  std::vector<int> tokens;        // sampled speech tokens, excludes the stop token
  int stop_token = -1;            // stop token id (6561..6760) if stopped, else -1
  bool stopped = false;           // true if ended by a stop token
  bool truncated = false;         // true if max_len was reached without a stop token
  int steps = 0;                  // decode steps executed (== tokens.size() unless truncated)
};

}  // namespace larynx::llm
