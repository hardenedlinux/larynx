#include "internal.h"

#include <cmath>

namespace larynx::flow {

// TimestepEmbedding: SinusPositionEmbedding(256, scale=1000) + 2-layer MLP.
// Computed on the host in float32 (matching the reference) so the DiT graph
// stays static across CFM steps.
std::vector<float> time_embed_host(float t, const TimeMlpHost& mlp) {
  // SinusPositionEmbedding: emb[j] = sin(1000*t*coeff[j]) / cos(...), coeff[j] =
  // exp(-j * ln(10000)/127). half_dim = 128.
  std::vector<float> emb(256);
  const float log10000_127 = (float)(std::log(10000.0) / 127.0);
  for (int j = 0; j < 128; j++) {
    const float coeff = expf(-(float)j * log10000_127);
    const float ang = 1000.0f * t * coeff;
    emb[j] = sinf(ang);
    emb[j + 128] = cosf(ang);
  }

  // Linear(256 -> 1024). w0 = [in=256, out=1024] row-major (== torch (1024, 256)).
  std::vector<float> h(1024);
  for (int o = 0; o < 1024; o++) {
    float acc = mlp.b0[o];
    for (int i = 0; i < 256; i++) acc += mlp.w0[o * 256 + i] * emb[i];
    h[o] = acc;
  }
  // SiLU.
  for (int o = 0; o < 1024; o++) {
    const float x = h[o];
    h[o] = x / (1.0f + expf(-x));
  }
  // Linear(1024 -> 1024).
  std::vector<float> out(1024);
  for (int o = 0; o < 1024; o++) {
    float acc = mlp.b2[o];
    for (int i = 0; i < 1024; i++) acc += mlp.w2[o * 1024 + i] * h[i];
    out[o] = acc;
  }
  return out;
}

}  // namespace larynx::flow
