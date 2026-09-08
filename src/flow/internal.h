#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "larynx/backend.h"

namespace larynx::flow {

// ---------------------------------------------------------------------------
// Architecture constants (cosyvoice3.yaml + tests/flow_reference.py)
// ---------------------------------------------------------------------------
constexpr int MEL_DIM         = 80;   // mel bins
constexpr int SPK_EMBED_DIM   = 192;  // speaker embedding input dim
constexpr int SPK_DIM         = 80;   // projected speaker dim
constexpr int VOCAB           = 6561;
constexpr int DIM             = 1024; // DiT hidden dim
constexpr int HEADS           = 16;
constexpr int HEAD_DIM        = 64;
constexpr int ROT_DIM         = 64;   // partial rotary dims (== head_dim)
constexpr int FF_INNER        = 2048;
constexpr int DEPTH           = 22;
constexpr int PRE_LOOKAHEAD   = 3;    // conv1 kernel = 4, conv2 kernel = 3
constexpr int TOKEN_MEL_RATIO = 2;
constexpr int N_TIMESTEPS     = 10;
constexpr float CFG_RATE      = 0.7f;
constexpr int CONV_POS_KERNEL = 31;
constexpr int CONV_POS_GROUPS = 16;
constexpr float LN_EPS        = 1e-6f;
constexpr float ROPE_FREQ_BASE = 10000.0f;

// CausalConditionalCFM builds one deterministic seed-noise bank at construction
// time (set_all_random_seed(0) + torch.randn([1, 80, 50*300])). The C++ decoder
// loads the exported bank verbatim (FlowDecoder::load_noise) and slices the
// first MEL_T = Tseq*TOKEN_MEL_RATIO columns per call — never regenerates it.
constexpr int NOISE_MAX_FRAMES = 50 * 300;   // 15,000

// ---------------------------------------------------------------------------
// Loaded weights (resolved by name from flow.gguf; ne = reversed torch shape)
// ---------------------------------------------------------------------------
struct FlowWeights {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;  // device buffer the tensors live in

  ggml_tensor* input_embedding = nullptr;  // [80, 6561]

  ggml_tensor* spk_affine_w = nullptr;     // [192, 80]
  ggml_tensor* spk_affine_b = nullptr;     // [80]

  ggml_tensor* pre_conv1_w = nullptr;      // [4, 80, 1024]
  ggml_tensor* pre_conv1_b = nullptr;      // [1024]
  ggml_tensor* pre_conv2_w = nullptr;      // [3, 1024, 80]
  ggml_tensor* pre_conv2_b = nullptr;      // [80]

  ggml_tensor* time_mlp_0_w = nullptr;     // [256, 1024]
  ggml_tensor* time_mlp_0_b = nullptr;     // [1024]
  ggml_tensor* time_mlp_2_w = nullptr;     // [1024, 1024]
  ggml_tensor* time_mlp_2_b = nullptr;     // [1024]

  ggml_tensor* input_embed_proj_w = nullptr; // [320, 1024]
  ggml_tensor* input_embed_proj_b = nullptr; // [1024]
  ggml_tensor* conv_pos_1_w = nullptr;       // [31, 64, 1024]
  ggml_tensor* conv_pos_1_b = nullptr;       // [1024]
  ggml_tensor* conv_pos_2_w = nullptr;       // [31, 64, 1024]
  ggml_tensor* conv_pos_2_b = nullptr;       // [1024]

  struct Block {
    ggml_tensor* attn_norm_w = nullptr;  // [1024, 6144]
    ggml_tensor* attn_norm_b = nullptr;  // [6144]
    ggml_tensor* to_q_w = nullptr;       // [1024, 1024]
    ggml_tensor* to_q_b = nullptr;
    ggml_tensor* to_k_w = nullptr;
    ggml_tensor* to_k_b = nullptr;
    ggml_tensor* to_v_w = nullptr;
    ggml_tensor* to_v_b = nullptr;
    ggml_tensor* to_out_w = nullptr;     // [1024, 1024]
    ggml_tensor* to_out_b = nullptr;
    ggml_tensor* ff_0_w = nullptr;       // [1024, 2048]
    ggml_tensor* ff_0_b = nullptr;
    ggml_tensor* ff_2_w = nullptr;       // [2048, 1024]
    ggml_tensor* ff_2_b = nullptr;
  };
  Block block[DEPTH];

  ggml_tensor* norm_out_w = nullptr;      // [1024, 2048]
  ggml_tensor* norm_out_b = nullptr;      // [2048]
  ggml_tensor* proj_out_w = nullptr;      // [1024, 80]
  ggml_tensor* proj_out_b = nullptr;      // [80]
};

// Host copy of the time-embedding MLP. It is computed outside the DiT graph so
// the DiT graph stays static across CFM steps (only the sinus input changes).
struct TimeMlpHost {
  std::vector<float> w0, b0;   // 256 -> 1024
  std::vector<float> w2, b2;   // 1024 -> 1024
};

// Result of building the DiT graph (once, reused across the 10 CFM steps).
struct DiTGraph {
  ggml_cgraph* gf = nullptr;
  ggml_tensor* x_in = nullptr;      // [T, 80, B]
  ggml_tensor* mu_in = nullptr;     // [T, 80, B]
  ggml_tensor* cond_in = nullptr;   // [T, 80, B]
  ggml_tensor* spks_in = nullptr;   // [80, B]
  ggml_tensor* t_emb_in = nullptr;  // [1024, B]
  ggml_tensor* pos = nullptr;       // [T] i32 position ids (rotary)
  ggml_tensor* dphi = nullptr;      // [T, 80, B]

  // capture references (valid after a compute)
  ggml_tensor* input_embed = nullptr;  // [1024, T, B]
  ggml_tensor* input_proj  = nullptr;  // [1024, T, B]  (proj before conv_pos_embed)
  ggml_tensor* conv_pos    = nullptr;  // [1024, T, B]  (conv_pos_embed output)
  ggml_tensor* blocks[DEPTH];          // [1024, T, B]
  ggml_tensor* norm_out = nullptr;     // [1024, T, B]
};

// ---------------------------------------------------------------------------
// ops.cpp — assembled primitives (all layouts are ggml-native)
// ---------------------------------------------------------------------------
// Linear: y = x @ W^T + b.  w = [in, out], b = [out].
ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x);
// Conv1d (F32, stride 1, no pad, no dilation) via ggml_im2col + F32 mul_mat.
// kernel = [K, IC, OC], input = [L, IC, N], bias = [OC] (may be null).
ggml_tensor* conv1d_f32(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* bias);
// Grouped conv1d (groups-way channel split then concat along ne1).
ggml_tensor* grouped_conv1d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input,
                            ggml_tensor* bias, int groups);
// Mish: x * tanh(softplus(x)).
ggml_tensor* mish(ggml_context* ctx, ggml_tensor* x);
// GELU(tanh) — assembled from primitives to avoid ggml_gelu's FP16 lookup table
// (which is only ~1e-3 accurate and would dominate the numerical error).
ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x, TensorInit* init);
// Partial rotary (first n_dims channels only), position indexed by ne2.
// q = [D, T, B]; pos = [T] i32.
ggml_tensor* rope_partial(ggml_context* ctx, ggml_tensor* q, ggml_tensor* pos, int n_dims);
// AdaLayerNorm modulation: norm(x) * (1 + scale) + shift.
// x = [D, T, B]; scale/shift = [D, B].
ggml_tensor* ada_ln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, ggml_tensor* shift,
                    TensorInit* init);
// L2 normalize over ne0 (F.normalize(dim=1)): x / max(sqrt(sum(x^2)), 1e-12).
ggml_tensor* l2_normalize(ggml_context* ctx, ggml_tensor* x);
// repeat_interleave(x, 2, dim=0) — x = [T, C, N] -> [2T, C, N].
ggml_tensor* repeat_interleave_2_ne0(ggml_context* ctx, ggml_tensor* x);

// ---------------------------------------------------------------------------
// gguf.cpp
// ---------------------------------------------------------------------------
bool load_weights(const std::string& path, ggml_backend_t backend,
                  FlowWeights* out, TimeMlpHost* time_mlp);

// ---------------------------------------------------------------------------
// prelookahead.cpp
// ---------------------------------------------------------------------------
// token_embed = [80, T, 1] (== (B=1, T, 80)) -> [80, T, 1].
ggml_tensor* build_prelookahead(ggml_context* ctx, const FlowWeights& w, ggml_tensor* token_embed);

// ---------------------------------------------------------------------------
// dit.cpp
// ---------------------------------------------------------------------------
DiTGraph build_dit(ggml_context* ctx, const FlowWeights& w, int T, int B, TensorInit* init);

// ---------------------------------------------------------------------------
// flow_matching.cpp
// ---------------------------------------------------------------------------
// SinusPositionEmbedding(256, scale=1000) + time_mlp, for a scalar timestep.
std::vector<float> time_embed_host(float t, const TimeMlpHost& mlp);

}  // namespace larynx::flow
