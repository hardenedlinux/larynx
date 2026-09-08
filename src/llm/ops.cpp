#include "internal.h"

#include <cmath>

namespace larynx::llm {

ggml_tensor* rms_norm(ggml_context* ctx, ggml_tensor* x, ggml_tensor* w) {
  ggml_tensor* n = ggml_rms_norm(ctx, x, RMS_EPS);          // [D, T, B]
  ggml_tensor* w3 = ggml_reshape_3d(ctx, w, w->ne[0], 1, 1); // [D, 1, 1]
  return ggml_mul(ctx, n, w3);                              // broadcast weight
}

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
  ggml_tensor* y = ggml_mul_mat(ctx, w, x);
  if (b != nullptr) {
    y = ggml_add(ctx, y, b);  // b = [out] broadcasts over the trailing dims
  }
  return y;
}

ggml_tensor* rope_full(ggml_context* ctx, ggml_tensor* q, ggml_tensor* pos, int heads) {
  // q = [heads*HEAD_DIM, T, B] -> [HEAD_DIM, heads, T, B]; ggml_rope_ext indexes
  // position by ne2, which is T here, so no permute is needed.
  const int64_t T = q->ne[1];
  const int64_t B = q->ne[2];
  ggml_tensor* q4 = ggml_reshape_4d(ctx, q, HEAD_DIM, heads, T, B);
  return ggml_rope_ext(ctx, q4, pos, nullptr, HEAD_DIM,
                       GGML_ROPE_TYPE_NEOX, 0,
                       ROPE_THETA, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
}

ggml_tensor* repeat_kv(ggml_context* ctx, ggml_tensor* k, int n_rep) {
  if (n_rep == 1) return k;
  const int64_t d  = k->ne[0];  // HEAD_DIM
  const int64_t kv = k->ne[1];  // KV_HEADS
  const int64_t T  = k->ne[2];
  const int64_t B  = k->ne[3];
  ggml_tensor* kc = ggml_cont(ctx, k);
  // [d, kv, T, B] -> [d, 1, kv, T*B]: a size-1 dim at ne1 lets ggml_repeat expand
  // it n_rep times; the kv head moves to ne2, seq+batch to ne3.
  ggml_tensor* src = ggml_reshape_4d(ctx, kc, d, 1, kv, T * B);
  ggml_tensor* tgt = ggml_new_tensor_4d(ctx, k->type, d, n_rep, kv, T * B);
  ggml_tensor* rep = ggml_repeat(ctx, src, tgt);  // [d, n_rep, kv, T*B]
  ggml_tensor* cont = ggml_cont(ctx, rep);
  // Merge ne1(n_rep) x ne2(kv) into a single head dim and split ne3 back out.
  return ggml_reshape_4d(ctx, cont, d, kv * n_rep, T, B);  // [d, q_heads, T, B]
}

ggml_tensor* attention(ggml_context* ctx, ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                       ggml_tensor* mask) {
  // q/k/v = [HEAD_DIM, heads, T, B]. Mirror the Flow module's attention but with a
  // causal mask. KQ[key, query] so softmax runs over the key axis (ne0).
  const int64_t T = q->ne[2];
  const int64_t B = q->ne[3];
  ggml_tensor* Q = ggml_permute(ctx, q, 0, 2, 1, 3);   // [d, T, heads, B]
  ggml_tensor* K = ggml_permute(ctx, k, 0, 2, 1, 3);   // [d, T, heads, B]
  ggml_tensor* KQ = ggml_mul_mat(ctx, K, Q);           // [T, T, heads, B]
  KQ = ggml_soft_max_ext(ctx, KQ, mask, 1.0f / std::sqrt((float)HEAD_DIM), 0.0f);

  ggml_tensor* V = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));  // [T, d, heads, B]
  ggml_tensor* KQV = ggml_mul_mat(ctx, V, KQ);          // [d, T, heads, B]
  ggml_tensor* o = ggml_permute(ctx, KQV, 0, 2, 1, 3);  // [d, heads, T, B]
  o = ggml_cont(ctx, o);
  return ggml_reshape_3d(ctx, o, HIDDEN, T, B);  // [HIDDEN, T, B]
}

}  // namespace larynx::llm
