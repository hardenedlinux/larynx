#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "larynx/backend.h"

namespace larynx::hift {

// ---------------------------------------------------------------------------
// Architecture constants (cosyvoice3.yaml CausalHiFTGenerator + tests/hift_reference.py)
// ---------------------------------------------------------------------------
constexpr int IN_CH           = 80;    // mel bins
constexpr int BASE_CH         = 512;   // conv_pre output channels
constexpr int NB_HARM         = 8;     // harmonics above f0 -> 9 sine components
constexpr int SAMPLE_RATE     = 24000;
constexpr float NSF_ALPHA     = 0.1f;  // sine_amp
constexpr float NSF_SIGMA     = 0.003f; // noise_std
constexpr float NSF_VOICED    = 10.0f;  // voiced threshold
constexpr int NUM_UPS         = 3;
constexpr int UPSAMPLE_RATES[NUM_UPS]    = {8, 5, 3};
constexpr int UPSAMPLE_KERNELS[NUM_UPS]  = {16, 11, 7};
constexpr int ISTFT_N_FFT     = 16;
constexpr int ISTFT_HOP       = 4;
constexpr int N_BINS          = 9;      // n_fft/2 + 1
constexpr int OUT_CH          = 18;     // n_fft + 2
constexpr int NUM_KERNELS     = 3;
constexpr int RESBLOCK_KERNELS[3]     = {3, 7, 11};
constexpr int RESBLOCK_DILATIONS[3]   = {1, 3, 5};
constexpr int SRC_RB_KERNELS[3]       = {7, 7, 11};
constexpr int SRC_RB_DILATIONS[3]     = {1, 3, 5};
constexpr float LRELU_SLOPE   = 0.1f;
constexpr float AUDIO_LIMIT   = 0.99f;
constexpr int CONV_PRE_LOOK_RIGHT = 4;
constexpr int TOTAL_SCALE     = 8 * 5 * 3 * ISTFT_HOP;  // 480
constexpr float SNAKE_EPS     = 1e-9f;

// The mel frame count T_MEL is a *runtime* input (the validation case uses 30;
// real utterances are a few hundred). The derived sizes follow the constants:
//   L_S      = T_MEL * TOTAL_SCALE                (excitation samples, 14400 @ T_MEL=30)
//   N_FRAMES = L_S / ISTFT_HOP + 1                (STFT frames,      3601  @ T_MEL=30)
// Both are derived from the input sizes inside the host helpers (f0_predict /
// source_stft / istft), so the function signatures stay size-agnostic.

// source_downs (CausalConv1dDownSample) kernel sizes and strides, per upsample
// stage. downsample_cum_rates[::-1] = [15, 3, 1]; u==1 becomes a k=1 causal conv.
constexpr int SRC_DOWN_KERNELS[NUM_UPS]  = {30, 6, 1};
constexpr int SRC_DOWN_STRIDES[NUM_UPS]  = {15, 3, 1};

// Output channel per upsample stage: base // 2^(i+1).
inline int stage_channels(int i) { return BASE_CH >> (i + 1); }

// ---------------------------------------------------------------------------
// Loaded weights (resolved by name from hift.gguf; ne = reversed torch shape)
// ---------------------------------------------------------------------------
struct ConvW {
  ggml_tensor* w = nullptr;  // [K, IC, OC]
  ggml_tensor* b = nullptr;  // [OC]
};

struct SnakeW {
  ggml_tensor* alpha = nullptr;      // [C]  (weight ctx)
  std::vector<float> inv_alpha;      // [C] host, 1 / (alpha + 1e-9); built in the
                                     // graph ctx (the weight ctx has no headroom)
};

// A causal ResBlock: 3x (snake -> conv1(dilation) -> snake -> conv2(dilation 1)).
struct ResBlockW {
  ConvW conv1[3];
  ConvW conv2[3];
  SnakeW act1[3];
  SnakeW act2[3];
};

struct HiFTWeights {
  ggml_context* ctx = nullptr;
  ggml_backend_buffer_t buffer = nullptr;  // device buffer the tensors live in

  ConvW conv_pre;                       // [5, 80, 512]  (CausalConv1d right, k=5)
  ConvW ups[NUM_UPS];                   // [16,512,256] [11,256,128] [7,128,64]
  ConvW source_downs[NUM_UPS];          // [30,18,256] [6,18,128] [1,18,64]
  ResBlockW source_resblocks[NUM_UPS];  // ch 256 / 128 / 64
  ResBlockW resblocks[NUM_UPS * 3];     // 9 blocks, ch per stage
  ConvW conv_post;                      // [7, 64, 18]  (CausalConv1d left, k=7)

  // Host-side copies (not in the ggml graph).
  std::vector<float> m_source_w;   // [9]  Linear(9,1)
  std::vector<float> m_source_b;   // [1]
  std::vector<double> f0_w[5];     // condnet convs 0,2,4,6,8 (float64 predictor)
  std::vector<double> f0_b[5];
  std::vector<double> f0_cls_w;    // [512]
  std::vector<double> f0_cls_b;    // [1]
};

// Result of building the vocoder graph (once per load; static across calls).
struct HiFTGraph {
  ggml_cgraph* gf = nullptr;
  ggml_tensor* mel_in = nullptr;     // [T_MEL, IN_CH, 1]
  ggml_tensor* s_stft_in = nullptr;  // [N_FRAMES, OUT_CH, 1]
  ggml_tensor* magnitude = nullptr;  // [N_FRAMES, N_BINS, 1]
  ggml_tensor* phase = nullptr;      // [N_FRAMES, N_BINS, 1]

  // capture references (valid after a compute; all [T, C, 1])
  ggml_tensor* conv_pre = nullptr;
  ggml_tensor* ups_lrelu[NUM_UPS];
  ggml_tensor* ups[NUM_UPS];
  ggml_tensor* source_downs[NUM_UPS];
  ggml_tensor* source_resblocks[NUM_UPS];
  ggml_tensor* fusion[NUM_UPS];
  ggml_tensor* resblock_out[NUM_UPS * 3];
  ggml_tensor* post_resblocks[NUM_UPS];
  ggml_tensor* reflection_pad = nullptr;
  ggml_tensor* final_lrelu = nullptr;
  ggml_tensor* conv_post = nullptr;
};

// ---------------------------------------------------------------------------
// ops.cpp
// ---------------------------------------------------------------------------
// Conv1d (F32 im2col + mul_mat) with manual padding already applied. kernel =
// [K, IC, OC], input = [L, IC, N], bias = [OC] (may be null). s0 = stride,
// d0 = dilation. Output [OL, OC, N].
ggml_tensor* conv1d_f32(ggml_context* ctx, ggml_tensor* kernel, ggml_tensor* input,
                        ggml_tensor* bias, int s0, int d0);
// Snake: x + (1/(alpha+eps)) * sin(x*alpha)^2. x = [T, C, N], snake = per-channel.
ggml_tensor* snake(ggml_context* ctx, ggml_tensor* x, const SnakeW& sw, TensorInit* init);
// nearest upsample along ne0 by factor r: out[i] = in[i / r]. x = [T, C, N].
ggml_tensor* nearest_upsample(ggml_context* ctx, ggml_tensor* x, int r);
// zero-pad the time axis (ne0): left on the front, right on the back.
ggml_tensor* pad_zeros(ggml_context* ctx, ggml_tensor* x, int left, int right,
                       TensorInit* init);
// nn.ReflectionPad1d((1, 0)): prepend x[1, :]. x = [T, C, N] -> [T+1, C, N].
ggml_tensor* reflection_pad_left1(ggml_context* ctx, ggml_tensor* x);

// ---------------------------------------------------------------------------
// gguf.cpp
// ---------------------------------------------------------------------------
bool load_weights(const std::string& path, ggml_backend_t backend, HiFTWeights* out);

// ---------------------------------------------------------------------------
// source.cpp (host: float64 f0 predictor, float32 source/STFT/ISTFT)
// ---------------------------------------------------------------------------
// f0 predictor (CausalConvRNNF0Predictor) in float64. mel = [ic][t] (IN_CH*T_MEL
// doubles). f0 = [t] (T_MEL doubles).
void f0_predict(std::vector<double>& f0_out, const std::vector<double>& mel,
                const HiFTWeights& w);
// Build the source excitation + STFT. f0 (float32 [t]) -> s_stft [c][t]
// (OUT_CH*N_FRAMES floats). rand_ini (9), sine_waves (L_S*9). Optional outs:
// sine_wavs (L_S*9), sine_merge (L_S).
void source_stft(const std::vector<float>& f0, const std::vector<float>& rand_ini,
                 const std::vector<float>& sine_waves, const HiFTWeights& w,
                 std::vector<float>& s_stft_out, std::vector<float>* sine_wavs_out,
                 std::vector<float>* sine_merge_out);
// torch.istft inverse path: magnitude/phase [c][t] (each N_BINS*N_FRAMES) ->
// audio [t] (L_S floats).
void istft(const std::vector<float>& magnitude, const std::vector<float>& phase,
           std::vector<float>& audio_out);

}  // namespace larynx::hift
