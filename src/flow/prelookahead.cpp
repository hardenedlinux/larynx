#include "internal.h"

namespace larynx::flow {

// PreLookaheadLayer (upsample_encoder.py). token_embed = [80, T, 1] (== (1,T,80)).
// inputs -> transpose -> pad(0,3) -> conv1(80->1024,k4) -> leaky_relu
//        -> pad(2,0) -> conv2(1024->80,k3) -> transpose -> + inputs
ggml_tensor* build_prelookahead(ggml_context* ctx, const FlowWeights& w, ggml_tensor* token_embed) {
  ggml_tensor* x = ggml_permute(ctx, token_embed, 1, 0, 2, 3);   // [T, 80, 1] channels-first
  x = ggml_pad(ctx, x, PRE_LOOKAHEAD, 0, 0, 0);                   // pad right 3 -> [T+3, 80, 1]

  ggml_tensor* y = conv1d_f32(ctx, w.pre_conv1_w, x, w.pre_conv1_b);  // [T, 1024, 1]
  y = ggml_leaky_relu(ctx, y, 0.01f, false);

  y = ggml_pad_ext(ctx, y, 2, 0, 0, 0, 0, 0, 0, 0);               // pad left 2 -> [T+2, 1024, 1]
  y = conv1d_f32(ctx, w.pre_conv2_w, y, w.pre_conv2_b);           // [T, 80, 1]

  y = ggml_permute(ctx, y, 1, 0, 2, 3);                           // [80, T, 1]
  return ggml_add(ctx, ggml_cont(ctx, y), token_embed);           // residual (src0 must be contiguous)
}

}  // namespace larynx::flow
