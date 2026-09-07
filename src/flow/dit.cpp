#include "internal.h"

#include <cmath>
#include <cstring>

namespace larynx::flow {

namespace {

ggml_tensor* new_input3(ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1, int64_t ne2,
                        const char* name) {
  ggml_tensor* t = ggml_new_tensor_3d(ctx, type, ne0, ne1, ne2);
  ggml_set_name(t, name);
  ggml_set_input(t);
  return t;
}

ggml_tensor* new_input2(ggml_context* ctx, ggml_type type, int64_t ne0, int64_t ne1,
                        const char* name) {
  ggml_tensor* t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
  ggml_set_name(t, name);
  ggml_set_input(t);
  return t;
}

// View the c-th 1024-channel chunk of a [6*DIM, B] modulation tensor.
// Returned contiguous so callers can ggml_reshape_3d it to [DIM, 1, B].
ggml_tensor* view_chunk(ggml_context* ctx, ggml_tensor* emb, int c) {
  const size_t es = ggml_type_size(emb->type);
  ggml_tensor* v = ggml_view_2d(ctx, emb, DIM, emb->ne[1], emb->ne[0] * es, c * DIM * es);
  return ggml_cont(ctx, v);
}

ggml_tensor* attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                       int T, int B) {
  // q/k/v = [DIM, T, B] -> [head_dim, heads, T, B]
  ggml_tensor* q4 = ggml_reshape_4d(ctx, q, HEAD_DIM, HEADS, T, B);
  ggml_tensor* k4 = ggml_reshape_4d(ctx, k, HEAD_DIM, HEADS, T, B);
  ggml_tensor* v4 = ggml_reshape_4d(ctx, v, HEAD_DIM, HEADS, T, B);

  ggml_tensor* Q = ggml_permute(ctx, q4, 0, 2, 1, 3);   // [d, T, heads, B]
  ggml_tensor* K = ggml_permute(ctx, k4, 0, 2, 1, 3);   // [d, T, heads, B]
  ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);            // [T, T, heads, B]
  KQ = ggml_soft_max_ext(ctx, KQ, nullptr, 1.0f / std::sqrt((float)HEAD_DIM), 0.0f);

  ggml_tensor* V = ggml_cont(ctx, ggml_permute(ctx, v4, 1, 2, 0, 3));   // [T, d, heads, B] (cont for mul_mat)
  ggml_tensor* KQV = ggml_mul_mat(ctx, V, KQ);          // [d, T, heads, B]
  ggml_tensor* o = ggml_permute(ctx, KQV, 0, 2, 1, 3);  // [d, heads, T, B]
  o = ggml_cont(ctx, o);
  return ggml_reshape_3d(ctx, o, DIM, T, B);            // [DIM, T, B]
}

ggml_tensor* dit_block(ggml_context* ctx, const FlowWeights::Block& blk, ggml_tensor* x,
                       ggml_tensor* t, ggml_tensor* pos, int T, int B) {
  // AdaLayerNormZero: emb = linear(silu(t)) -> [6144, B], chunk into 6 x [1024, B].
  ggml_tensor* emb = linear(ctx, blk.attn_norm_w, blk.attn_norm_b, ggml_silu(ctx, t));
  ggml_tensor* shift_msa = view_chunk(ctx, emb, 0);
  ggml_tensor* scale_msa = view_chunk(ctx, emb, 1);
  ggml_tensor* gate_msa  = view_chunk(ctx, emb, 2);
  ggml_tensor* shift_mlp = view_chunk(ctx, emb, 3);
  ggml_tensor* scale_mlp = view_chunk(ctx, emb, 4);
  ggml_tensor* gate_mlp  = view_chunk(ctx, emb, 5);

  ggml_tensor* norm = ada_ln(ctx, x, scale_msa, shift_msa);

  ggml_tensor* q = rope_partial(ctx, linear(ctx, blk.to_q_w, blk.to_q_b, norm), pos, ROT_DIM);
  ggml_tensor* k = rope_partial(ctx, linear(ctx, blk.to_k_w, blk.to_k_b, norm), pos, ROT_DIM);
  ggml_tensor* v = linear(ctx, blk.to_v_w, blk.to_v_b, norm);
  ggml_tensor* attn = linear(ctx, blk.to_out_w, blk.to_out_b, attention(ctx, q, k, v, T, B));

  x = ggml_add(ctx, x, ggml_mul(ctx, attn, ggml_reshape_3d(ctx, gate_msa, DIM, 1, B)));

  ggml_tensor* ff_norm = ada_ln(ctx, x, scale_mlp, shift_mlp);
  ggml_tensor* f1 = gelu_tanh(ctx, linear(ctx, blk.ff_0_w, blk.ff_0_b, ff_norm));
  ggml_tensor* f2 = linear(ctx, blk.ff_2_w, blk.ff_2_b, f1);

  x = ggml_add(ctx, x, ggml_mul(ctx, f2, ggml_reshape_3d(ctx, gate_mlp, DIM, 1, B)));
  return x;
}

}  // namespace

DiTGraph build_dit(ggml_context* ctx, const FlowWeights& w, int T, int B) {
  DiTGraph g;

  // Inputs (Space A = [seq, channel, batch] = PyTorch (B, C, T)).
  g.x_in     = new_input3(ctx, GGML_TYPE_F32, T, MEL_DIM, B, "x");
  g.mu_in    = new_input3(ctx, GGML_TYPE_F32, T, MEL_DIM, B, "mu");
  g.cond_in  = new_input3(ctx, GGML_TYPE_F32, T, MEL_DIM, B, "cond");
  g.spks_in  = new_input2(ctx, GGML_TYPE_F32, MEL_DIM, B, "spks");
  g.t_emb_in = new_input2(ctx, GGML_TYPE_F32, DIM, B, "t_emb");

  // Position ids [0..T-1].
  ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, T);
  for (int i = 0; i < T; i++) ((int32_t*)pos->data)[i] = i;

  // Space A -> Space B ([feature, seq, batch] = PyTorch (B, T, C)).
  ggml_tensor* x    = ggml_permute(ctx, g.x_in, 1, 0, 2, 3);
  ggml_tensor* mu   = ggml_permute(ctx, g.mu_in, 1, 0, 2, 3);
  ggml_tensor* cond = ggml_permute(ctx, g.cond_in, 1, 0, 2, 3);

  // spks [80, B] -> [80, T, B] (broadcast over the sequence axis).
  ggml_tensor* spks3 = ggml_reshape_3d(ctx, g.spks_in, MEL_DIM, 1, B);
  ggml_tensor* spks_tgt = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, MEL_DIM, T, B);
  ggml_tensor* spks = ggml_repeat(ctx, spks3, spks_tgt);

  // InputEmbedding: proj(concat[x, cond, mu, spks]) + conv_pos_embed.
  ggml_tensor* cat = ggml_concat(ctx, ggml_concat(ctx, ggml_concat(ctx, x, cond, 0), mu, 0), spks, 0);
  ggml_tensor* xe = linear(ctx, w.input_embed_proj_w, w.input_embed_proj_b, cat);  // [1024, T, B]
  g.input_proj = xe;

  ggml_tensor* xcf = ggml_permute(ctx, xe, 1, 0, 2, 3);              // [T, 1024, B]
  xcf = ggml_pad_ext(ctx, xcf, CONV_POS_KERNEL - 1, 0, 0, 0, 0, 0, 0, 0);
  ggml_tensor* c1 = grouped_conv1d(ctx, w.conv_pos_1_w, xcf, w.conv_pos_1_b, CONV_POS_GROUPS);
  c1 = mish(ctx, c1);
  c1 = ggml_pad_ext(ctx, c1, CONV_POS_KERNEL - 1, 0, 0, 0, 0, 0, 0, 0);
  ggml_tensor* c2 = grouped_conv1d(ctx, w.conv_pos_2_w, c1, w.conv_pos_2_b, CONV_POS_GROUPS);
  c2 = mish(ctx, c2);
  ggml_tensor* pos_embed = ggml_cont(ctx, ggml_permute(ctx, c2, 1, 0, 2, 3));  // [1024, T, B]
  g.conv_pos = pos_embed;
  ggml_tensor* h = ggml_add(ctx, xe, pos_embed);                     // residual
  g.input_embed = h;

  ggml_tensor* t = g.t_emb_in;  // host-computed time embedding, [1024, B]

  for (int i = 0; i < DEPTH; i++) {
    h = dit_block(ctx, w.block[i], h, t, pos, T, B);
    g.blocks[i] = h;
  }

  // norm_out: AdaLayerNormZero_Final.
  ggml_tensor* emb = linear(ctx, w.norm_out_w, w.norm_out_b, ggml_silu(ctx, t));  // [2048, B]
  ggml_tensor* scale = ggml_cont(ctx, ggml_view_2d(ctx, emb, DIM, B, emb->ne[0] * 4, 0));
  ggml_tensor* shift = ggml_cont(ctx, ggml_view_2d(ctx, emb, DIM, B, emb->ne[0] * 4, DIM * 4));
  g.norm_out = ada_ln(ctx, h, scale, shift);

  // proj_out -> Space A output.
  ggml_tensor* proj = linear(ctx, w.proj_out_w, w.proj_out_b, g.norm_out);  // [80, T, B]
  g.dphi = ggml_cont(ctx, ggml_permute(ctx, proj, 1, 0, 2, 3));             // [T, 80, B] (cont for read)

  g.gf = ggml_new_graph_custom(ctx, 8192, false);
  ggml_build_forward_expand(g.gf, g.dphi);
  return g;
}

}  // namespace larynx::flow
