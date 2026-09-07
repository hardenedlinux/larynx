#include "internal.h"

#include <cstdio>
#include <cstring>

#include "gguf.h"

#include "larynx/backend.h"

namespace larynx::hift {

namespace {

ggml_tensor* get_tensor(ggml_context* ctx, const char* name) {
  ggml_tensor* t = ggml_get_tensor(ctx, name);
  if (t == nullptr) {
    fprintf(stderr, "hift: missing tensor '%s' in hift.gguf\n", name);
  }
  return t;
}

// Compute the companion inv_alpha = 1/(alpha + 1e-9) on the host. (Reads the
// tensor back through the backend, since weights may live in device memory.)
void make_inv_alpha(ggml_tensor* alpha, std::vector<float>* out) {
  const int64_t n = alpha->ne[0];
  out->resize(n);
  std::vector<float> a(n);
  ggml_backend_tensor_get(alpha, a.data(), 0, sizeof(float) * n);
  for (int64_t i = 0; i < n; i++) {
    out->at(i) = 1.0f / (a[i] + SNAKE_EPS);
  }
}

void copy_to_host(ggml_tensor* t, std::vector<float>* dst) {
  dst->resize(ggml_nelements(t));
  ggml_backend_tensor_get(t, dst->data(), 0, ggml_nbytes(t));
}

void copy_to_host_double(ggml_tensor* t, std::vector<double>* dst) {
  const int64_t n = ggml_nelements(t);
  dst->resize(n);
  std::vector<float> src(n);
  ggml_backend_tensor_get(t, src.data(), 0, sizeof(float) * n);
  for (int64_t i = 0; i < n; i++) dst->at(i) = (double)src[i];
}

void load_conv(ggml_context* ctx, const char* name, ConvW* out) {
  char buf[256];
  snprintf(buf, sizeof buf, "%s.weight", name);
  out->w = get_tensor(ctx, buf);
  snprintf(buf, sizeof buf, "%s.bias", name);
  out->b = get_tensor(ctx, buf);
}

void load_resblock(ggml_context* ctx, const char* name, ResBlockW* out) {
  char buf[256];
  for (int j = 0; j < 3; j++) {
    snprintf(buf, sizeof buf, "%s.convs1.%d.weight", name, j);
    out->conv1[j].w = get_tensor(ctx, buf);
    snprintf(buf, sizeof buf, "%s.convs1.%d.bias", name, j);
    out->conv1[j].b = get_tensor(ctx, buf);
    snprintf(buf, sizeof buf, "%s.convs2.%d.weight", name, j);
    out->conv2[j].w = get_tensor(ctx, buf);
    snprintf(buf, sizeof buf, "%s.convs2.%d.bias", name, j);
    out->conv2[j].b = get_tensor(ctx, buf);
    snprintf(buf, sizeof buf, "%s.activations1.%d.alpha", name, j);
    out->act1[j].alpha = get_tensor(ctx, buf);
    make_inv_alpha(out->act1[j].alpha, &out->act1[j].inv_alpha);
    snprintf(buf, sizeof buf, "%s.activations2.%d.alpha", name, j);
    out->act2[j].alpha = get_tensor(ctx, buf);
    make_inv_alpha(out->act2[j].alpha, &out->act2[j].inv_alpha);
  }
}

}  // namespace

bool load_weights(const std::string& path, ggml_backend_t backend, HiFTWeights* out) {
  ggml_context* wctx = nullptr;
  struct gguf_init_params params = {
      /*.no_alloc =*/ true,  // tensors get their data in the backend buffer below
      /*.ctx      =*/ &wctx,
  };
  struct gguf_context* gctx = gguf_init_from_file(path.c_str(), params);
  if (gctx == nullptr) {
    fprintf(stderr, "hift: failed to load '%s'\n", path.c_str());
    return false;
  }

  out->ctx = wctx;
  auto* w = out;

  // Upload the weights into the backend buffer before anything reads them: the
  // host-side copies below (inv_alpha, f0/m_source weights) use
  // ggml_backend_tensor_get, which needs the data already in the buffer.
  w->buffer = upload_gguf_weights(gctx, wctx, backend, path);
  gguf_free(gctx);
  if (!w->buffer) {
    ggml_free(wctx);
    out->ctx = nullptr;
    return false;
  }

  load_conv(wctx, "conv_pre", &w->conv_pre);
  load_conv(wctx, "conv_post", &w->conv_post);

  char buf[256];
  for (int i = 0; i < NUM_UPS; i++) {
    snprintf(buf, sizeof buf, "ups.%d", i);
    load_conv(wctx, buf, &w->ups[i]);
    snprintf(buf, sizeof buf, "source_downs.%d", i);
    load_conv(wctx, buf, &w->source_downs[i]);
    snprintf(buf, sizeof buf, "source_resblocks.%d", i);
    load_resblock(wctx, buf, &w->source_resblocks[i]);
  }
  for (int i = 0; i < NUM_UPS * NUM_KERNELS; i++) {
    snprintf(buf, sizeof buf, "resblocks.%d", i);
    load_resblock(wctx, buf, &w->resblocks[i]);
  }

  // Host-side copies.
  copy_to_host(get_tensor(wctx, "m_source.l_linear.weight"), &w->m_source_w);
  copy_to_host(get_tensor(wctx, "m_source.l_linear.bias"), &w->m_source_b);

  for (int i = 0; i < 5; i++) {
    int idx = (i == 0) ? 0 : 2 * i;  // 0, 2, 4, 6, 8
    snprintf(buf, sizeof buf, "f0_predictor.condnet.%d.weight", idx);
    copy_to_host_double(get_tensor(wctx, buf), &w->f0_w[i]);
    snprintf(buf, sizeof buf, "f0_predictor.condnet.%d.bias", idx);
    copy_to_host_double(get_tensor(wctx, buf), &w->f0_b[i]);
  }
  copy_to_host_double(get_tensor(wctx, "f0_predictor.classifier.weight"), &w->f0_cls_w);
  copy_to_host_double(get_tensor(wctx, "f0_predictor.classifier.bias"), &w->f0_cls_b);

  // Verify none of the lookups failed.
  bool ok = true;
  auto chk = [&ok](ggml_tensor* t) { if (t == nullptr) ok = false; };
  chk(w->conv_pre.w); chk(w->conv_pre.b);
  chk(w->conv_post.w); chk(w->conv_post.b);
  for (int i = 0; i < NUM_UPS; i++) {
    chk(w->ups[i].w); chk(w->ups[i].b);
    chk(w->source_downs[i].w); chk(w->source_downs[i].b);
    for (int j = 0; j < 3; j++) {
      chk(w->source_resblocks[i].conv1[j].w); chk(w->source_resblocks[i].conv1[j].b);
      chk(w->source_resblocks[i].conv2[j].w); chk(w->source_resblocks[i].conv2[j].b);
      chk(w->source_resblocks[i].act1[j].alpha);
      chk(w->source_resblocks[i].act2[j].alpha);
    }
  }
  for (int i = 0; i < NUM_UPS * NUM_KERNELS; i++) {
    for (int j = 0; j < 3; j++) {
      chk(w->resblocks[i].conv1[j].w); chk(w->resblocks[i].conv1[j].b);
      chk(w->resblocks[i].conv2[j].w); chk(w->resblocks[i].conv2[j].b);
      chk(w->resblocks[i].act1[j].alpha);
      chk(w->resblocks[i].act2[j].alpha);
    }
  }
  return ok;
}

}  // namespace larynx::hift
