#include "internal.h"

#include <cstdio>
#include <cstring>

#include "gguf.h"

#include "larynx/backend.h"

namespace larynx::flow {

namespace {

ggml_tensor* get_tensor(ggml_context* ctx, const char* name) {
  ggml_tensor* t = ggml_get_tensor(ctx, name);
  if (t == nullptr) {
    fprintf(stderr, "flow: missing tensor '%s' in flow.gguf\n", name);
  }
  return t;
}

// Build a per-block name suffix (e.g. ".22") into `buf`, returning buf.
const char* block_name(char* buf, size_t cap, int i, const char* tail) {
  snprintf(buf, cap, "decoder.estimator.transformer_blocks.%d.%s", i, tail);
  return buf;
}

}  // namespace

bool load_weights(const std::string& path, ggml_backend_t backend,
                  FlowWeights* out, TimeMlpHost* time_mlp) {
  ggml_context* wctx = nullptr;
  struct gguf_init_params params = {
      /*.no_alloc =*/ true,  // tensors get their data in the backend buffer below
      /*.ctx      =*/ &wctx,
  };
  struct gguf_context* gctx = gguf_init_from_file(path.c_str(), params);
  if (gctx == nullptr) {
    fprintf(stderr, "flow: failed to load '%s'\n", path.c_str());
    return false;
  }

  out->ctx = wctx;
  auto* w = out;

  w->input_embedding = get_tensor(wctx, "input_embedding.weight");

  w->spk_affine_w = get_tensor(wctx, "spk_embed_affine_layer.weight");
  w->spk_affine_b = get_tensor(wctx, "spk_embed_affine_layer.bias");

  w->pre_conv1_w = get_tensor(wctx, "pre_lookahead_layer.conv1.weight");
  w->pre_conv1_b = get_tensor(wctx, "pre_lookahead_layer.conv1.bias");
  w->pre_conv2_w = get_tensor(wctx, "pre_lookahead_layer.conv2.weight");
  w->pre_conv2_b = get_tensor(wctx, "pre_lookahead_layer.conv2.bias");

  w->time_mlp_0_w = get_tensor(wctx, "decoder.estimator.time_embed.time_mlp.0.weight");
  w->time_mlp_0_b = get_tensor(wctx, "decoder.estimator.time_embed.time_mlp.0.bias");
  w->time_mlp_2_w = get_tensor(wctx, "decoder.estimator.time_embed.time_mlp.2.weight");
  w->time_mlp_2_b = get_tensor(wctx, "decoder.estimator.time_embed.time_mlp.2.bias");

  w->input_embed_proj_w = get_tensor(wctx, "decoder.estimator.input_embed.proj.weight");
  w->input_embed_proj_b = get_tensor(wctx, "decoder.estimator.input_embed.proj.bias");
  w->conv_pos_1_w = get_tensor(wctx, "decoder.estimator.input_embed.conv_pos_embed.conv1.0.weight");
  w->conv_pos_1_b = get_tensor(wctx, "decoder.estimator.input_embed.conv_pos_embed.conv1.0.bias");
  w->conv_pos_2_w = get_tensor(wctx, "decoder.estimator.input_embed.conv_pos_embed.conv2.0.weight");
  w->conv_pos_2_b = get_tensor(wctx, "decoder.estimator.input_embed.conv_pos_embed.conv2.0.bias");

  char buf[256];
  for (int i = 0; i < DEPTH; i++) {
    auto& b = w->block[i];
    b.attn_norm_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn_norm.linear.weight"));
    b.attn_norm_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn_norm.linear.bias"));
    b.to_q_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_q.weight"));
    b.to_q_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_q.bias"));
    b.to_k_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_k.weight"));
    b.to_k_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_k.bias"));
    b.to_v_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_v.weight"));
    b.to_v_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_v.bias"));
    b.to_out_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_out.0.weight"));
    b.to_out_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "attn.to_out.0.bias"));
    b.ff_0_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "ff.ff.0.0.weight"));
    b.ff_0_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "ff.ff.0.0.bias"));
    b.ff_2_w = get_tensor(wctx, block_name(buf, sizeof buf, i, "ff.ff.2.weight"));
    b.ff_2_b = get_tensor(wctx, block_name(buf, sizeof buf, i, "ff.ff.2.bias"));
  }

  w->norm_out_w = get_tensor(wctx, "decoder.estimator.norm_out.linear.weight");
  w->norm_out_b = get_tensor(wctx, "decoder.estimator.norm_out.linear.bias");
  w->proj_out_w = get_tensor(wctx, "decoder.estimator.proj_out.weight");
  w->proj_out_b = get_tensor(wctx, "decoder.estimator.proj_out.bias");

  // Upload the weights into the backend buffer (CPU keeps them in a host buffer,
  // CUDA in device memory). Must happen before any host copy below, which reads
  // the tensors back through the backend.
  w->buffer = upload_gguf_weights(gctx, wctx, backend, path);
  gguf_free(gctx);
  if (!w->buffer) {
    ggml_free(wctx);
    out->ctx = nullptr;
    return false;
  }

  // Copy the time-embedding MLP to host (it runs outside the DiT graph).
  if (time_mlp != nullptr) {
    auto copy = [](ggml_tensor* t, std::vector<float>* dst) {
      dst->resize(ggml_nelements(t));
      ggml_backend_tensor_get(t, dst->data(), 0, ggml_nbytes(t));
    };
    copy(w->time_mlp_0_w, &time_mlp->w0);
    copy(w->time_mlp_0_b, &time_mlp->b0);
    copy(w->time_mlp_2_w, &time_mlp->w2);
    copy(w->time_mlp_2_b, &time_mlp->b2);
  }

  // Verify none of the lookups failed.
  bool ok = true;
  auto chk = [&ok](ggml_tensor* t) { if (t == nullptr) ok = false; };
  chk(w->input_embedding);
  chk(w->spk_affine_w); chk(w->spk_affine_b);
  chk(w->pre_conv1_w); chk(w->pre_conv1_b);
  chk(w->pre_conv2_w); chk(w->pre_conv2_b);
  chk(w->time_mlp_0_w); chk(w->time_mlp_0_b); chk(w->time_mlp_2_w); chk(w->time_mlp_2_b);
  chk(w->input_embed_proj_w); chk(w->input_embed_proj_b);
  chk(w->conv_pos_1_w); chk(w->conv_pos_1_b);
  chk(w->conv_pos_2_w); chk(w->conv_pos_2_b);
  chk(w->norm_out_w); chk(w->norm_out_b);
  chk(w->proj_out_w); chk(w->proj_out_b);
  for (int i = 0; i < DEPTH; i++) {
    auto& b = w->block[i];
    chk(b.attn_norm_w); chk(b.attn_norm_b);
    chk(b.to_q_w); chk(b.to_q_b); chk(b.to_k_w); chk(b.to_k_b); chk(b.to_v_w); chk(b.to_v_b);
    chk(b.to_out_w); chk(b.to_out_b);
    chk(b.ff_0_w); chk(b.ff_0_b); chk(b.ff_2_w); chk(b.ff_2_b);
  }
  return ok;
}

}  // namespace larynx::flow
