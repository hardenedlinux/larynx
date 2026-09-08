#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "larynx/backend.h"

namespace larynx::llm {

// ---------------------------------------------------------------------------
// Qwen2 architecture constants (CosyVoice-BlankEN/config.json).
// ---------------------------------------------------------------------------
constexpr int HIDDEN        = 896;     // hidden_size
constexpr int INTERMEDIATE  = 4864;    // intermediate_size (SwiGLU inner)
constexpr int N_HEADS       = 14;      // num_attention_heads
constexpr int KV_HEADS      = 2;       // num_key_value_heads (GQA 7:1)
constexpr int HEAD_DIM      = 64;      // HIDDEN / N_HEADS
constexpr int KV_DIM        = 128;     // KV_HEADS * HEAD_DIM
constexpr int N_LAYERS      = 24;      // num_hidden_layers
constexpr int SPEECH_VOCAB  = 6761;    // speech_embedding rows / llm_decoder cols
constexpr int TEXT_VOCAB    = 151936;  // embed_tokens rows (Qwen2 text vocab)
constexpr float RMS_EPS     = 1e-6f;   // rms_norm_eps
constexpr float ROPE_THETA  = 1000000.0f;  // rope_theta

// CosyVoice3LM special-token ids, all looked up in speech_embedding (not
// llm_embedding — that is the legacy TransformerLM path). SPEECH_TOKEN_SIZE is
// defined in sampling.h; kept literal here so internal.h does not depend on it.
constexpr int SOS_TOKEN     = 6561;    // = speech_token_size + 0  (CosyVoice3LM.sos)
constexpr int TASK_ID_TOKEN = 6563;    // = speech_token_size + 2  (CosyVoice3LM.task_id)

// ---------------------------------------------------------------------------
// Loaded weights (resolved by name from llm.gguf; ne = reversed torch shape).
// ---------------------------------------------------------------------------
struct LLMWeights {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;  // device buffer the tensors live in

  ggml_tensor* llm_decoder = nullptr;  // [HIDDEN, SPEECH_VOCAB] (Linear 896->6761, no bias)
  ggml_tensor* final_norm  = nullptr;  // [HIDDEN]  model.model.norm.weight (applied before llm_decoder)
  ggml_tensor* speech_embedding = nullptr;  // [HIDDEN, SPEECH_VOCAB] Embedding(6761, 896) row lookup for decode
  ggml_tensor* embed_tokens = nullptr;      // [HIDDEN, TEXT_VOCAB]  Embedding(151936, 896) for the text (prompt_text||text)

  struct Layer {
    ggml_tensor* q_w = nullptr;      // [HIDDEN, HIDDEN]
    ggml_tensor* q_b = nullptr;      // [HIDDEN]
    ggml_tensor* k_w = nullptr;      // [HIDDEN, KV_DIM]
    ggml_tensor* k_b = nullptr;      // [KV_DIM]
    ggml_tensor* v_w = nullptr;      // [HIDDEN, KV_DIM]
    ggml_tensor* v_b = nullptr;      // [KV_DIM]
    ggml_tensor* o_w = nullptr;      // [HIDDEN, HIDDEN]  (no bias)
    ggml_tensor* gate_w = nullptr;   // [HIDDEN, INTERMEDIATE]
    ggml_tensor* up_w = nullptr;     // [HIDDEN, INTERMEDIATE]
    ggml_tensor* down_w = nullptr;   // [INTERMEDIATE, HIDDEN]
    ggml_tensor* in_norm = nullptr;  // [HIDDEN]  input_layernorm.weight
    ggml_tensor* post_norm = nullptr;// [HIDDEN]  post_attention_layernorm.weight
  };
  Layer layer[N_LAYERS];
};

// ---------------------------------------------------------------------------
// ops.cpp — Qwen2 primitives (all layouts are ggml-native [feature, seq, batch]).
// ---------------------------------------------------------------------------
// RMSNorm: x / sqrt(mean(x^2) + eps) * w.  x = [D, T, B], w = [D].
ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w);
// Linear: y = x @ W^T + b.  w = [in, out], b = [out] (may be null).
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x);
// Full NEOX rotary on q = [heads*HEAD_DIM, T, B] reshaped to [HEAD_DIM, heads, T, B].
// pos = [T] i32 positions. Returns [HEAD_DIM, heads, T, B].
ggml_tensor* rope_full(ggml_context* ctx, ggml_tensor* q, ggml_tensor* pos, int heads);
// GQA repeat: k = [d, kv, T, B] -> [d, kv*n_rep, T, B] (consecutive, matching HF repeat_kv).
ggml_tensor* repeat_kv(ggml_context* ctx, ggml_tensor* k, int n_rep);
// Scaled-dot-product attention with a causal mask. q/k/v = [d, heads, T, B]
// (already roped + GQA-repeated); mask = [T, T, 1, 1] (0 / -inf). Returns [heads*d, T, B].
ggml_tensor* attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                       ggml_tensor* mask);

// ---------------------------------------------------------------------------
// gguf.cpp
// ---------------------------------------------------------------------------
bool load_weights(const std::string& path, ggml_backend_t backend, LLMWeights* out);

}  // namespace larynx::llm
