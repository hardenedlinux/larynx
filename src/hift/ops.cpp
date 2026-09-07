#include "internal.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace larynx::hift {

ggml_tensor* conv1d_f32(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input,
                        ggml_tensor* bias, int s0, int d0) {
  // Mirrors ggml_conv_1d but forces F32 im2col (ggml_conv_1d hardcodes F16).
  // Padding is applied manually by the caller (causal convs), so p0 = 0.
  ggml_tensor* im2col = ggml_im2col(ctx, kernel, input, s0, 0, 0, 0, d0, 0, false,
                                    GGML_TYPE_F32);
  ggml_tensor* result = ggml_mul_mat(ctx,
      ggml_reshape_2d(ctx, im2col, im2col->ne[0], im2col->ne[2] * im2col->ne[1]),
      ggml_reshape_2d(ctx, kernel, kernel->ne[0] * kernel->ne[1], kernel->ne[2]));
  // mul_mat yields [OL*N, OC]; reinterpret [OL, N, OC] then transpose to [OL, OC, N].
  result = ggml_reshape_3d(ctx, result, im2col->ne[1], im2col->ne[2], kernel->ne[2]);
  result = ggml_permute(ctx, result, 0, 2, 1, 3);
  result = ggml_cont(ctx, result);
  if (bias != nullptr) {
    result = ggml_add(ctx, result, ggml_reshape_3d(ctx, bias, 1, kernel->ne[2], 1));
  }
  return result;
}

ggml_tensor* snake(ggml_context* ctx, ggml_tensor* x, const SnakeW& sw, TensorInit* init) {
  // x: [T, C, N]. alpha/inv_alpha: [C] -> [1, C, 1].
  const int64_t C = sw.alpha->ne[0];
  ggml_tensor* a = ggml_reshape_3d(ctx, sw.alpha, 1, C, 1);
  // inv_alpha is host data; materialize it as a [1, C, 1] constant (written once
  // the tensors are placed in the backend buffer).
  ggml_tensor* inv = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, C, 1);
  init->fill(inv, sw.inv_alpha);
  ggml_tensor* xa = ggml_mul(ctx, x, a);                 // x * alpha
  ggml_tensor* s = ggml_sqr(ctx, ggml_sin(ctx, xa));     // sin^2(x*alpha)
  return ggml_add(ctx, x, ggml_mul(ctx, s, inv));        // x + (1/(a+eps))*sin^2
}

ggml_tensor* nearest_upsample(ggml_context* ctx, ggml_tensor* x, int r) {
  const int64_t T = x->ne[0], C = x->ne[1], N = x->ne[2];
  ggml_tensor* xc = ggml_cont(ctx, x);                          // reshape needs contiguity
  ggml_tensor* x4 = ggml_reshape_4d(ctx, xc, 1, T, C, N);       // [1, T, C, N]
  ggml_tensor* tgt = ggml_new_tensor_4d(ctx, x->type, r, T, C, N);
  ggml_tensor* rep = ggml_repeat(ctx, x4, tgt);                 // [r, T, C, N]
  ggml_tensor* cont = ggml_cont(ctx, rep);
  return ggml_reshape_3d(ctx, cont, r * T, C, N);               // [r*T, C, N]
}

ggml_tensor* pad_zeros(ggml_context* ctx, ggml_tensor* x, int left, int right,
                       TensorInit* init) {
  ggml_tensor* out = x;
  if (left > 0) {
    ggml_tensor* z = ggml_new_tensor_3d(ctx, x->type, left, x->ne[1], x->ne[2]);
    init->zero(z);
    out = ggml_concat(ctx, z, out, 0);
  }
  if (right > 0) {
    ggml_tensor* z = ggml_new_tensor_3d(ctx, x->type, right, x->ne[1], x->ne[2]);
    init->zero(z);
    out = ggml_concat(ctx, out, z, 0);
  }
  return out;
}

ggml_tensor* reflection_pad_left1(ggml_context* ctx, ggml_tensor* x) {
  // nn.ReflectionPad1d((1, 0)): out[0] = x[1], out[t] = x[t-1] for t >= 1.
  ggml_tensor* xc = ggml_cont(ctx, x);
  ggml_tensor* first = ggml_view_3d(ctx, xc, 1, xc->ne[1], xc->ne[2],
                                    xc->nb[1], xc->nb[2], 1 * xc->nb[0]);
  return ggml_concat(ctx, first, xc, 0);  // [1 + T, C, N]
}

}  // namespace larynx::hift
