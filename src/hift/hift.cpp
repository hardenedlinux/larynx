#include "larynx/hift/hift.h"

#include "internal.h"
#include "larynx/backend.h"

#include <cmath>
#include <cstdio>
#include <cstring>

namespace larynx::hift {

std::string module_name() {
  return "hift (CausalHiFTGenerator: f0/SineGen2 + upsample/resblock net + ISTFT; GGML CPU, implemented)";
}

namespace {

void read_tensor(ggml_tensor* t, std::vector<float>& dst) {
  dst.resize(ggml_nelements(t));
  ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
}

// A causal ResBlock: 3x (snake1 -> conv1(dilation) -> snake2 -> conv2(dil=1)),
// each with a residual add. `k` is the kernel size (same for all 6 convs); the
// conv1 dilations are RESBLOCK_DILATIONS == {1, 3, 5}.
ggml_tensor* build_resblock(ggml_context* ctx, const ResBlockW& rb,
                            ggml_tensor* x, int k, TensorInit* init) {
  for (int j = 0; j < 3; j++) {
    ggml_tensor* xt = snake(ctx, x, rb.act1[j], init);
    xt = conv1d_f32(ctx, rb.conv1[j].w,
                    pad_zeros(ctx, xt, RESBLOCK_DILATIONS[j] * (k - 1), 0, init),
                    rb.conv1[j].b, 1, RESBLOCK_DILATIONS[j]);
    xt = snake(ctx, xt, rb.act2[j], init);
    xt = conv1d_f32(ctx, rb.conv2[j].w,
                    pad_zeros(ctx, xt, k - 1, 0, init),
                    rb.conv2[j].b, 1, 1);
    x = ggml_add(ctx, xt, x);
  }
  return x;
}

}  // namespace

struct HiftVocoder::Impl {
  ggml_backend_t backend = nullptr;
  HiFTWeights w;
  bool loaded = false;

  // SineGen2 fixed source buffers (from load_source): the full 300 s noise bank,
  // sliced per call. Empty until load_source succeeds.
  std::vector<float> rand_ini;      // (NB_HARM + 1)
  std::vector<float> sine_waves;    // (SINE_MAX_SAMPLES * (NB_HARM + 1)) row-major
  bool source_loaded = false;
};

HiftVocoder::HiftVocoder() : impl_(new Impl()) {}

HiftVocoder::~HiftVocoder() {
  if (impl_) {
    if (impl_->w.buffer) ggml_backend_buffer_free(impl_->w.buffer);
    if (impl_->w.ctx) ggml_free(impl_->w.ctx);
    if (impl_->backend) ggml_backend_free(impl_->backend);
    delete impl_;
  }
}

bool HiftVocoder::load(const std::string& path) {
  if (impl_->loaded) return true;
  impl_->backend = backend_init_best();
  if (!impl_->backend) return false;
  if (!load_weights(path, impl_->backend, &impl_->w)) {
    ggml_backend_free(impl_->backend);
    impl_->backend = nullptr;
    return false;
  }
  impl_->loaded = true;
  return true;
}

namespace {

// On-disk layout of the asset written by tests/export_hift_source.py.
constexpr uint32_t HSRC_MAGIC = 0x43525348;  // "HSRC"
constexpr uint32_t HSRC_VERSION = 1;

}  // namespace

bool HiftVocoder::load_source(const std::string& path) {
  if (!impl_) return false;
  if (impl_->source_loaded) return true;

  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) {
    std::fprintf(stderr, "hift: cannot open source asset %s\n", path.c_str());
    return false;
  }

  uint32_t magic = 0, version = 0, harmonic_dim = 0;
  uint64_t sine_max = 0;
  bool ok = std::fread(&magic, 4, 1, f) == 1 &&
            std::fread(&version, 4, 1, f) == 1 &&
            std::fread(&harmonic_dim, 4, 1, f) == 1 &&
            std::fread(&sine_max, 8, 1, f) == 1;
  if (!ok || magic != HSRC_MAGIC || version != HSRC_VERSION ||
      harmonic_dim != (uint32_t)(NB_HARM + 1) || sine_max != (uint64_t)SINE_MAX_SAMPLES) {
    std::fprintf(stderr,
                 "hift: bad source asset header in %s (magic=%08x ver=%u dim=%u max=%llu)\n",
                 path.c_str(), magic, version, harmonic_dim,
                 (unsigned long long)sine_max);
    std::fclose(f);
    return false;
  }

  const size_t rand_n = (size_t)harmonic_dim;
  const size_t wave_n = (size_t)sine_max * harmonic_dim;
  impl_->rand_ini.resize(rand_n);
  impl_->sine_waves.resize(wave_n);
  ok = std::fread(impl_->rand_ini.data(), 4, rand_n, f) == rand_n &&
       std::fread(impl_->sine_waves.data(), 4, wave_n, f) == wave_n;
  std::fclose(f);
  if (!ok) {
    std::fprintf(stderr, "hift: short read on source asset %s\n", path.c_str());
    impl_->rand_ini.clear();
    impl_->sine_waves.clear();
    return false;
  }

  impl_->source_loaded = true;
  return true;
}

const std::vector<float>& HiftVocoder::source_rand_ini() const {
  return impl_->rand_ini;
}

const std::vector<float>& HiftVocoder::source_sine_waves() const {
  return impl_->sine_waves;
}

bool HiftVocoder::vocode(const std::vector<float>& mel,
                        const std::vector<float>& rand_ini,
                        const std::vector<float>& sine_waves_buf,
                        std::vector<float>& audio,
                        HiFTDebug* debug) {
  if (!impl_ || !impl_->loaded) return false;
  if (mel.size() % IN_CH != 0) return false;
  const int T_MEL = (int)(mel.size() / IN_CH);          // mel frames
  const int L_S = T_MEL * TOTAL_SCALE;                  // excitation samples
  const int N_FRAMES = L_S / ISTFT_HOP + 1;             // STFT frames
  if (rand_ini.size() != (size_t)(NB_HARM + 1)) return false;
  if (sine_waves_buf.size() != (size_t)L_S * (NB_HARM + 1)) return false;

  // ---- host: mel -> f0 (float64 predictor), cast back to float32 ----
  std::vector<double> mel_d((size_t)IN_CH * T_MEL);
  for (int c = 0; c < IN_CH; c++)
    for (int t = 0; t < T_MEL; t++)
      mel_d[(size_t)c * T_MEL + t] = (double)mel[(size_t)c * T_MEL + t];

  std::vector<double> f0_f64;
  f0_predict(f0_f64, mel_d, impl_->w);

  std::vector<float> f0(T_MEL);
  for (int t = 0; t < T_MEL; t++) f0[t] = (float)f0_f64[t];

  // ---- host: f0 -> source excitation -> s_stft ----
  std::vector<float> s_stft, sine_wavs, sine_merge;
  source_stft(f0, rand_ini, sine_waves_buf, impl_->w, s_stft, &sine_wavs, &sine_merge);

  // ---- GGML graph: conv_pre + upsample/resblock net + conv_post ----
  // no_alloc=true: the ctx holds only tensor/graph metadata; the actual data
  // (mostly the 72 conv im2col tensors) lives in the backend buffer allocated
  // after the graph is built.
  const size_t graph_mem =
      ggml_tensor_overhead() * (size_t)65536 + ggml_graph_overhead();
  ggml_init_params gp = {.mem_size = graph_mem, .mem_buffer = nullptr, .no_alloc = true};
  ggml_context* gctx = ggml_init(gp);
  if (!gctx) {
    fprintf(stderr, "hift: ggml_init failed for graph_mem=%zu (T_MEL=%d)\n",
            graph_mem, T_MEL);
    return false;
  }

  HiFTGraph g;
  g.mel_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, T_MEL, IN_CH, 1);
  ggml_set_input(g.mel_in);

  g.s_stft_in = ggml_new_tensor_3d(gctx, GGML_TYPE_F32, N_FRAMES, OUT_CH, 1);
  ggml_set_input(g.s_stft_in);

  TensorInit init;

  const HiFTWeights& w = impl_->w;

  // conv_pre: CausalConv1d(80 -> 512, k=5, right) -> pad right 4.
  ggml_tensor* x = conv1d_f32(gctx, w.conv_pre.w,
                              pad_zeros(gctx, g.mel_in, 0, CONV_PRE_LOOK_RIGHT, &init),
                              w.conv_pre.b, 1, 1);
  g.conv_pre = x;  // [T_MEL, 512, 1]

  for (int i = 0; i < NUM_UPS; i++) {
    x = ggml_leaky_relu(gctx, x, LRELU_SLOPE, false);
    g.ups_lrelu[i] = x;

    // CausalConv1dUpsample: nearest xrate -> pad left (k-1) -> conv(k, stride=1).
    ggml_tensor* u = nearest_upsample(gctx, x, UPSAMPLE_RATES[i]);
    u = conv1d_f32(gctx, w.ups[i].w,
                   pad_zeros(gctx, u, UPSAMPLE_KERNELS[i] - 1, 0, &init),
                   w.ups[i].b, 1, 1);
    x = u;
    g.ups[i] = x;

    if (i == NUM_UPS - 1) {
      x = reflection_pad_left1(gctx, x);
      g.reflection_pad = x;
    }

    // source: CausalConv1dDownSample (u==1 -> plain CausalConv1d k=1).
    ggml_tensor* si;
    if (SRC_DOWN_STRIDES[i] == 1) {
      si = conv1d_f32(gctx, w.source_downs[i].w, g.s_stft_in, w.source_downs[i].b, 1, 1);
    } else {
      si = conv1d_f32(gctx, w.source_downs[i].w,
                      pad_zeros(gctx, g.s_stft_in, SRC_DOWN_STRIDES[i] - 1, 0, &init),
                      w.source_downs[i].b, SRC_DOWN_STRIDES[i], 1);
    }
    g.source_downs[i] = si;
    si = build_resblock(gctx, w.source_resblocks[i], si, SRC_RB_KERNELS[i], &init);
    g.source_resblocks[i] = si;

    x = ggml_add(gctx, x, si);
    g.fusion[i] = x;

    // num_kernels ResBlocks, summed, then / num_kernels.
    ggml_tensor* xs = nullptr;
    for (int j = 0; j < NUM_KERNELS; j++) {
      ggml_tensor* r = build_resblock(gctx, w.resblocks[i * NUM_KERNELS + j], x,
                                      RESBLOCK_KERNELS[j], &init);
      g.resblock_out[i * NUM_KERNELS + j] = r;
      xs = (xs == nullptr) ? r : ggml_add(gctx, xs, r);
    }
    x = ggml_scale(gctx, xs, 1.0f / (float)NUM_KERNELS);
    g.post_resblocks[i] = x;
  }

  // final: leaky_relu -> conv_post (CausalConv1d(64 -> 18, k=7, left) -> pad left 6).
  // NB: the final leaky_relu uses F.leaky_relu's *default* slope 0.01, unlike the
  // per-stage leaky_relu which uses self.lrelu_slope = 0.1 (see generator.py).
  x = ggml_leaky_relu(gctx, x, 0.01f, false);
  g.final_lrelu = x;
  x = conv1d_f32(gctx, w.conv_post.w, pad_zeros(gctx, x, 6, 0, &init), w.conv_post.b, 1, 1);
  g.conv_post = x;  // [N_FRAMES, 18, 1]

  // magnitude = exp(x[:, :9]); phase = sin(x[:, 9:]).
  ggml_tensor* mag_view = ggml_view_3d(gctx, x, N_FRAMES, N_BINS, 1,
                                        x->nb[1], x->nb[2], 0);
  ggml_tensor* pha_view = ggml_view_3d(gctx, x, N_FRAMES, N_BINS, 1,
                                        x->nb[1], x->nb[2], N_BINS * x->nb[1]);
  g.magnitude = ggml_exp(gctx, mag_view);
  g.phase = ggml_sin(gctx, pha_view);

  g.gf = ggml_new_graph(gctx);
  ggml_build_forward_expand(g.gf, g.magnitude);
  ggml_build_forward_expand(g.gf, g.phase);

  ggml_backend_buffer_t gbuf = ggml_backend_alloc_ctx_tensors(gctx, impl_->backend);
  init.apply();
  ggml_backend_tensor_set(g.mel_in, mel.data(), 0, sizeof(float) * mel.size());
  ggml_backend_tensor_set(g.s_stft_in, s_stft.data(), 0, sizeof(float) * s_stft.size());
  ggml_backend_graph_compute(impl_->backend, g.gf);

  std::vector<float> magnitude, phase;
  read_tensor(g.magnitude, magnitude);
  read_tensor(g.phase, phase);

  // ---- host: ISTFT -> clamp ----
  std::vector<float> istft_raw;
  istft(magnitude, phase, istft_raw);
  audio.assign(L_S, 0.0f);
  for (int i = 0; i < L_S; i++) {
    float v = istft_raw[i];
    if (v > AUDIO_LIMIT) v = AUDIO_LIMIT;
    if (v < -AUDIO_LIMIT) v = -AUDIO_LIMIT;
    audio[i] = v;
  }

  if (debug) {
    debug->f0 = f0;
    debug->sine_wavs = sine_wavs;
    debug->sine_merge = sine_merge;
    debug->s_stft = s_stft;
    read_tensor(g.conv_pre, debug->conv_pre);
    for (int i = 0; i < NUM_UPS; i++) {
      read_tensor(g.ups_lrelu[i], debug->ups_lrelu[i]);
      read_tensor(g.ups[i], debug->ups[i]);
      read_tensor(g.source_downs[i], debug->source_downs[i]);
      read_tensor(g.source_resblocks[i], debug->source_resblocks[i]);
      read_tensor(g.fusion[i], debug->fusion[i]);
      read_tensor(g.post_resblocks[i], debug->post_resblocks[i]);
    }
    debug->resblocks.resize(NUM_UPS * NUM_KERNELS);
    for (int i = 0; i < NUM_UPS * NUM_KERNELS; i++)
      read_tensor(g.resblock_out[i], debug->resblocks[i]);
    read_tensor(g.reflection_pad, debug->reflection_pad);
    read_tensor(g.final_lrelu, debug->final_lrelu);
    read_tensor(g.conv_post, debug->conv_post);
    debug->magnitude = magnitude;
    debug->phase = phase;
    debug->istft = istft_raw;
    debug->speech = audio;
  }

  if (gbuf) ggml_backend_buffer_free(gbuf);
  ggml_free(gctx);
  return true;
}

bool HiftVocoder::vocode(const std::vector<float>& mel, std::vector<float>& audio,
                        HiFTDebug* debug) {
  if (!impl_ || !impl_->loaded || !impl_->source_loaded) return false;
  if (mel.size() % IN_CH != 0) return false;
  const int T_MEL = (int)(mel.size() / IN_CH);
  const int L_S = T_MEL * TOTAL_SCALE;
  const size_t need = (size_t)L_S * (NB_HARM + 1);
  if (impl_->rand_ini.size() != (size_t)(NB_HARM + 1) ||
      impl_->sine_waves.size() < need) {
    return false;
  }
  // The exported bank covers the full 300 s; consume only the first L_S rows.
  std::vector<float> sw(impl_->sine_waves.begin(), impl_->sine_waves.begin() + need);
  return vocode(mel, impl_->rand_ini, sw, audio, debug);
}

}  // namespace larynx::hift
