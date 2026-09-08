#include "internal.h"

#include <cstdio>
#include <cstring>

#include "gguf.h"

#include "larynx/backend.h"

namespace larynx::llm {

namespace {

ggml_tensor* get_tensor(ggml_context* ctx, const char* name) {
  ggml_tensor* t = ggml_get_tensor(ctx, name);
  if (t == nullptr) {
    fprintf(stderr, "llm: missing tensor '%s' in llm.gguf\n", name);
  }
  return t;
}

// Build a per-layer name prefix (e.g. "llm.model.model.layers.0.") into `buf`.
const char* layer_prefix(char* buf, size_t cap, int i) {
  snprintf(buf, cap, "llm.model.model.layers.%d.", i);
  return buf;
}

}  // namespace

bool load_weights(const std::string& path, ggml_backend_t backend, LLMWeights* out) {
  ggml_context* wctx = nullptr;
  struct gguf_init_params params = {
      /*.no_alloc =*/ true,  // tensors get their data in the backend buffer below
      /*.ctx      =*/ &wctx,
  };
  struct gguf_context* gctx = gguf_init_from_file(path.c_str(), params);
  if (gctx == nullptr) {
    fprintf(stderr, "llm: failed to load '%s'\n", path.c_str());
    return false;
  }

  out->ctx = wctx;

  out->llm_decoder = get_tensor(wctx, "llm_decoder.weight");
  out->final_norm  = get_tensor(wctx, "llm.model.model.norm.weight");
  out->speech_embedding = get_tensor(wctx, "speech_embedding.weight");

  char pfx[256];
  char name[320];
  for (int i = 0; i < N_LAYERS; i++) {
    layer_prefix(pfx, sizeof pfx, i);
    auto& l = out->layer[i];

    snprintf(name, sizeof name, "%sself_attn.q_proj.weight", pfx);
    l.q_w = get_tensor(wctx, name);
    snprintf(name, sizeof name, "%sself_attn.q_proj.bias", pfx);
    l.q_b = get_tensor(wctx, name);

    snprintf(name, sizeof name, "%sself_attn.k_proj.weight", pfx);
    l.k_w = get_tensor(wctx, name);
    snprintf(name, sizeof name, "%sself_attn.k_proj.bias", pfx);
    l.k_b = get_tensor(wctx, name);

    snprintf(name, sizeof name, "%sself_attn.v_proj.weight", pfx);
    l.v_w = get_tensor(wctx, name);
    snprintf(name, sizeof name, "%sself_attn.v_proj.bias", pfx);
    l.v_b = get_tensor(wctx, name);

    snprintf(name, sizeof name, "%sself_attn.o_proj.weight", pfx);
    l.o_w = get_tensor(wctx, name);

    snprintf(name, sizeof name, "%smlp.gate_proj.weight", pfx);
    l.gate_w = get_tensor(wctx, name);
    snprintf(name, sizeof name, "%smlp.up_proj.weight", pfx);
    l.up_w = get_tensor(wctx, name);
    snprintf(name, sizeof name, "%smlp.down_proj.weight", pfx);
    l.down_w = get_tensor(wctx, name);

    snprintf(name, sizeof name, "%sinput_layernorm.weight", pfx);
    l.in_norm = get_tensor(wctx, name);
    snprintf(name, sizeof name, "%spost_attention_layernorm.weight", pfx);
    l.post_norm = get_tensor(wctx, name);
  }

  // Upload the weights into the backend buffer (CPU keeps them in a host buffer,
  // CUDA in device memory).
  out->buffer = upload_gguf_weights(gctx, wctx, backend, path);
  gguf_free(gctx);
  if (!out->buffer) {
    ggml_free(wctx);
    out->ctx = nullptr;
    return false;
  }

  // Verify none of the lookups failed.
  bool ok = out->llm_decoder != nullptr && out->final_norm != nullptr &&
            out->speech_embedding != nullptr;
  for (int i = 0; i < N_LAYERS && ok; i++) {
    const auto& l = out->layer[i];
    if (!l.q_w || !l.q_b || !l.k_w || !l.k_b || !l.v_w || !l.v_b || !l.o_w ||
        !l.gate_w || !l.up_w || !l.down_w || !l.in_norm || !l.post_norm) {
      ok = false;
    }
  }
  return ok;
}

}  // namespace larynx::llm
