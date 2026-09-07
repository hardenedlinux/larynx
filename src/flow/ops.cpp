#include "internal.h"

#include <cmath>

namespace larynx::flow {

namespace {

// Fresh [1]-shaped scalar constant = 1.0f. Created in the graph context with
// no_alloc=true, so its data is NULL here; the value is registered in `init`
// and written once the tensors are placed in the backend buffer.
ggml_tensor* make_one(ggml_context* ctx, TensorInit* init) {
  ggml_tensor* t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
  init->fill_scalar(t, 1.0f);
  return t;
}

}  // namespace

ggml_tensor* linear(ggml_context* ctx, ggml_tensor* w, ggml_tensor* b, ggml_tensor* x) {
  ggml_tensor* y = ggml_mul_mat(ctx, w, x);
  if (b != nullptr) {
    y = ggml_add(ctx, y, b);  // b = [out] broadcasts over the trailing dims
  }
  return y;
}

ggml_tensor* conv1d_f32(ggml_context* ctx, ggml_tensor* a, ggml_tensor* b, ggml_tensor* bias) {
  // Mirrors ggml_conv_1d but forces F32 im2col (ggml_conv_1d hardcodes F16).
  ggml_tensor* im2col = ggml_im2col(ctx, a, b, 1, 0, 0, 0, 1, 0, false, GGML_TYPE_F32);
  ggml_tensor* result = ggml_mul_mat(ctx,
      ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
      ggml_reshape_2d(ctx, a, a->ne[0] * a->ne[1], a->ne[2]));
  // mul_mat yields [OL*N, OC] with (time inner, batch middle, channel outer).
  // Reinterpret as [OL, N, OC] then transpose to [OL, OC, N] (== PyTorch (N,OC,OL)).
  result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], a->ne[2]);  // [OL, N, OC]
  result = ggml_permute(ctx, result, 0, 2, 1, 3);                                 // [OL, OC, N]
  result = ggml_cont(ctx, result);
  if (bias != nullptr) {
    result = ggml_add(ctx, result, ggml_reshape_3d(ctx, bias, 1, a->ne[2], 1));  // per-channel
  }
  return result;
}

ggml_tensor* grouped_conv1d(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input,
                            ggml_tensor* bias, int groups) {
  const int64_t K   = kernel->ne[0];
  const int64_t ICg = kernel->ne[1];   // input channels per group (GGUF stores IC/groups)
  const int64_t OC  = kernel->ne[2];   // total output channels
  const int64_t OCg = OC / groups;     // output channels per group
  const int64_t L   = input->ne[0];
  const int64_t N   = input->ne[2];

  ggml_tensor* out = nullptr;
  for (int g = 0; g < groups; g++) {
    // kernel output-channel slice [K, ICg, OCg]
    ggml_tensor* kg = ggml_view_3d(ctx, kernel, K, ICg, OCg,
                                   kernel->nb[1], kernel->nb[2], g * OCg * kernel->nb[2]);
    // input-channel slice [L, ICg, N]
    ggml_tensor* ig = ggml_view_3d(ctx, input, L, ICg, N,
                                   input->nb[1], input->nb[2], g * ICg * input->nb[1]);
    ggml_tensor* bg = ggml_view_1d(ctx, bias, OCg, g * OCg * ggml_type_size(bias->type));
    ggml_tensor* cg = conv1d_f32(ctx, kg, ig, bg);  // [OL, OCg, N]
    out = (out == nullptr) ? cg : ggml_concat(ctx, out, cg, 1);  // concat channels
  }
  return out;
}

ggml_tensor* mish(ggml_context* ctx, ggml_tensor* x) {
  return ggml_mul(ctx, x, ggml_tanh(ctx, ggml_softplus(ctx, x)));
}

ggml_tensor* gelu_tanh(ggml_context* ctx, ggml_tensor* x, TensorInit* init) {
  // 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))) — matches nn.GELU("tanh").
  const float c0 = 0.044715f;
  const float c1 = 0.7978845608028654f;  // sqrt(2/pi)
  ggml_tensor* x2 = ggml_mul(ctx, x, x);
  ggml_tensor* x3 = ggml_mul(ctx, x2, x);
  ggml_tensor* inner = ggml_add(ctx, x, ggml_scale(ctx, x3, c0));  // x + 0.044715*x^3
  inner = ggml_scale(ctx, inner, c1);                              // * sqrt(2/pi)
  ggml_tensor* t = ggml_add(ctx, ggml_tanh(ctx, inner), make_one(ctx, init));  // tanh + 1
  return ggml_mul(ctx, ggml_scale(ctx, x, 0.5f), t);
}

ggml_tensor* rope_partial(ggml_context* ctx, ggml_tensor* q, ggml_tensor* pos, int n_dims) {
  // ggml_rope_ext indexes position by ne2, so q [D, T, B] -> [D, B, T].
  ggml_tensor* qp = ggml_permute(ctx, q, 0, 2, 1, 3);
  ggml_tensor* r = ggml_rope_ext(ctx, qp, pos, nullptr, n_dims,
                                 GGML_ROPE_TYPE_NORMAL, 0,
                                 ROPE_FREQ_BASE, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
  return ggml_cont(ctx, ggml_permute(ctx, r, 0, 2, 1, 3));  // cont for downstream reshape
}

ggml_tensor* ada_ln(ggml_context* ctx, ggml_tensor* x, ggml_tensor* scale, ggml_tensor* shift,
                    TensorInit* init) {
  const int64_t D = x->ne[0];
  const int64_t B = x->ne[2];
  ggml_tensor* n = ggml_norm(ctx, x, LN_EPS);                 // [D, T, B]
  ggml_tensor* s = ggml_reshape_3d(ctx, scale, D, 1, B);      // [D, 1, B]
  ggml_tensor* one = make_one(ctx, init);
  ggml_tensor* s1 = ggml_add(ctx, s, one);                    // 1 + scale
  ggml_tensor* y = ggml_mul(ctx, n, s1);                      // norm * (1 + scale)
  return ggml_add(ctx, y, ggml_reshape_3d(ctx, shift, D, 1, B));
}

ggml_tensor* l2_normalize(ggml_context* ctx, ggml_tensor* x) {
  ggml_tensor* ssq = ggml_sqr(ctx, x);
  ggml_tensor* s = ggml_sum_rows(ctx, ssq);                   // [1, B]
  s = ggml_clamp(ctx, s, 1e-12f, INFINITY);
  ggml_tensor* denom = ggml_sqrt(ctx, s);                     // [1, B]
  return ggml_div(ctx, x, denom);                             // x / ||x||
}

ggml_tensor* repeat_interleave_2_ne0(ggml_context* ctx, ggml_tensor* x) {
  const int64_t T = x->ne[0], C = x->ne[1], N = x->ne[2];
  ggml_tensor* xc = ggml_cont(ctx, x);                        // reshape needs contiguity
  ggml_tensor* x4 = ggml_reshape_4d(ctx, xc, 1, T, C, N);     // [1, T, C, N]
  ggml_tensor* tgt = ggml_new_tensor_4d(ctx, x->type, 2, T, C, N);
  ggml_tensor* rep = ggml_repeat(ctx, x4, tgt);               // [2, T, C, N]
  ggml_tensor* cont = ggml_cont(ctx, rep);
  return ggml_reshape_3d(ctx, cont, 2 * T, C, N);             // [2T, C, N]
}

}  // namespace larynx::flow
