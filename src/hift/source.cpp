#include "internal.h"

#include <cmath>
#include <cstring>

namespace larynx::hift {

namespace {

constexpr double PI = 3.14159265358979323846;

// ---------------------------------------------------------------------------
// f0 predictor (CausalConvRNNF0Predictor), float64
// ---------------------------------------------------------------------------

// stride-1 dilation-1 conv over an input laid out [ic][t] (length L). Left/right
// zero padding is applied by indexing (src = t + k - pad_left). W is [oc][ic][k].
void conv1d_s1(const std::vector<double>& x, int L, int IC,
               const std::vector<double>& W, const std::vector<double>& b,
               int K, int OC, int pad_left, std::vector<double>& out) {
  out.assign((size_t)OC * L, 0.0);
  for (int oc = 0; oc < OC; oc++) {
    double bias = b[oc];
    for (int t = 0; t < L; t++) {
      double acc = bias;
      for (int ic = 0; ic < IC; ic++) {
        const double* wrow = &W[(size_t)oc * IC * K + (size_t)ic * K];
        for (int k = 0; k < K; k++) {
          int src = t + k - pad_left;
          if (src >= 0 && src < L) acc += wrow[k] * x[(size_t)ic * L + src];
        }
      }
      out[(size_t)oc * L + t] = acc;
    }
  }
}

void elu_inplace(std::vector<double>& v) {
  for (double& x : v) if (x <= 0.0) x = std::exp(x) - 1.0;
}

// ---------------------------------------------------------------------------
// STFT / ISTFT (float32, 16-point DFT)
// ---------------------------------------------------------------------------

// Periodic Hann window (torch get_window('hann', 16, fftbins=True)).
void hann_window(float w[ISTFT_N_FFT]) {
  for (int n = 0; n < ISTFT_N_FFT; n++)
    w[n] = 0.5f * (1.0f - std::cos(2.0f * (float)PI * n / ISTFT_N_FFT));
}

// cos/sin tables: t[fn][n] = cos/sin(2*pi*fn*n / N), fn in [0, N), n in [0, N).
struct DftTables {
  float c[ISTFT_N_FFT][ISTFT_N_FFT];
  float s[ISTFT_N_FFT][ISTFT_N_FFT];
  DftTables() {
    for (int f = 0; f < ISTFT_N_FFT; f++)
      for (int n = 0; n < ISTFT_N_FFT; n++) {
        double a = 2.0 * PI * f * n / ISTFT_N_FFT;
        c[f][n] = (float)std::cos(a);
        s[f][n] = (float)std::sin(a);
      }
  }
};

}  // namespace

void f0_predict(std::vector<double>& f0_out, const std::vector<double>& mel,
                const HiFTWeights& w) {
  const int L = (int)(mel.size() / IN_CH);  // mel = [ic][t], ic = IN_CH
  std::vector<double> x = mel;  // [ic][t]

  // condnet[0]: CausalConv1d(80 -> 512, k=4, 'right') -> pad right 3.
  std::vector<double> y;
  conv1d_s1(x, L, IN_CH, w.f0_w[0], w.f0_b[0], 4, BASE_CH, 0, y);
  elu_inplace(y);
  // condnet[2,4,6,8]: CausalConv1d(512 -> 512, k=3, 'left') -> pad left 2.
  for (int i = 1; i < 5; i++) {
    std::vector<double> z;
    conv1d_s1(y, L, BASE_CH, w.f0_w[i], w.f0_b[i], 3, BASE_CH, 2, z);
    elu_inplace(z);
    y.swap(z);
  }

  // classifier: Linear(512 -> 1), then abs.
  f0_out.assign(L, 0.0);
  for (int t = 0; t < L; t++) {
    double acc = w.f0_cls_b[0];
    for (int ic = 0; ic < BASE_CH; ic++) acc += w.f0_cls_w[ic] * y[(size_t)ic * L + t];
    f0_out[t] = std::fabs(acc);
  }
}

void source_stft(const std::vector<float>& f0, const std::vector<float>& rand_ini,
                 const std::vector<float>& sine_waves_buf, const HiFTWeights& w,
                 std::vector<float>& s_stft_out, std::vector<float>* sine_wavs_out,
                 std::vector<float>* sine_merge_out) {
  // f0: [t] float32. Derive the derived sizes from the input length.
  const int T_MEL = (int)f0.size();
  const int L_S = T_MEL * TOTAL_SCALE;
  const int N_FRAMES = L_S / ISTFT_HOP + 1;

  // --- f0_upsamp: nearest x480 ---
  std::vector<float> f0up(L_S);
  for (int i = 0; i < L_S; i++) f0up[i] = f0[i / TOTAL_SCALE];

  // --- SineGen2 (_f02sine + noise) ---
  std::vector<float> rad(L_S * (NB_HARM + 1));
  for (int i = 0; i < L_S; i++) {
    for (int h = 0; h <= NB_HARM; h++) {
      float fn = f0up[i] * (float)(h + 1);
      rad[(size_t)i * (NB_HARM + 1) + h] = std::fmod(fn / (float)SAMPLE_RATE, 1.0f);
    }
  }
  for (int h = 0; h <= NB_HARM; h++) rad[h] += rand_ini[h];  // rad[0][h] += rand_ini[h]

  // linear downsample x(1/480): 14400 -> 30 (F.interpolate align_corners=False).
  std::vector<float> rad_down(T_MEL * (NB_HARM + 1));
  for (int i = 0; i < T_MEL; i++) {
    float src = (i + 0.5f) * (float)(L_S / T_MEL) - 0.5f;   // (i+0.5)*480 - 0.5
    int f = (int)std::floor(src);
    int c = std::min(f + 1, L_S - 1);
    float frac = src - (float)f;
    for (int h = 0; h <= NB_HARM; h++) {
      rad_down[(size_t)i * (NB_HARM + 1) + h] =
          rad[(size_t)f * (NB_HARM + 1) + h] * (1.0f - frac) +
          rad[(size_t)c * (NB_HARM + 1) + h] * frac;
    }
  }

  // cumsum along time, then phase = cumsum * 2 * pi * 480, nearest upsample.
  std::vector<float> cumsum(T_MEL * (NB_HARM + 1));
  for (int h = 0; h <= NB_HARM; h++) {
    float acc = 0.0f;
    for (int i = 0; i < T_MEL; i++) {
      acc += rad_down[(size_t)i * (NB_HARM + 1) + h];
      cumsum[(size_t)i * (NB_HARM + 1) + h] = acc;
    }
  }

  std::vector<float> sine_wavs(L_S * (NB_HARM + 1));
  for (int i = 0; i < L_S; i++) {
    int t = i / TOTAL_SCALE;  // nearest-upsample index
    for (int h = 0; h <= NB_HARM; h++) {
      float p = cumsum[(size_t)t * (NB_HARM + 1) + h];
      p *= 2.0f;
      p *= (float)PI;
      p *= (float)TOTAL_SCALE;
      float s = std::sin(p) * NSF_ALPHA;
      // uv / noise mixing (uv is per time-step, broadcast over harmonics).
      float uv = (f0up[i] > NSF_VOICED) ? 1.0f : 0.0f;
      float noise_amp = uv * NSF_SIGMA + ((1.0f - uv) * NSF_ALPHA) / 3.0f;
      float noise = noise_amp * sine_waves_buf[(size_t)i * (NB_HARM + 1) + h];
      sine_wavs[(size_t)i * (NB_HARM + 1) + h] = s * uv + noise;
    }
  }
  if (sine_wavs_out) *sine_wavs_out = sine_wavs;

  // --- m_source: tanh(Linear(9, 1)) ---
  std::vector<float> s(L_S);
  for (int i = 0; i < L_S; i++) {
    float acc = w.m_source_b[0];
    for (int h = 0; h <= NB_HARM; h++)
      acc += w.m_source_w[h] * sine_wavs[(size_t)i * (NB_HARM + 1) + h];
    s[i] = std::tanh(acc);
  }
  if (sine_merge_out) *sine_merge_out = s;

  // --- STFT (center reflect-pad 8, Hann, onesided 9 bins) ---
  const int pad = ISTFT_N_FFT / 2;
  std::vector<float> xp(L_S + 2 * pad);
  for (int i = 0; i < pad; i++) xp[i] = s[pad - i];                 // left reflect
  for (int i = 0; i < L_S; i++) xp[pad + i] = s[i];
  for (int j = 0; j < pad; j++) xp[L_S + pad + j] = s[L_S - 2 - j]; // right reflect

  static const DftTables dft;
  float win[ISTFT_N_FFT];
  hann_window(win);

  s_stft_out.assign((size_t)OUT_CH * N_FRAMES, 0.0f);
  for (int t = 0; t < N_FRAMES; t++) {
    for (int k = 0; k < N_BINS; k++) {
      float re = 0.0f, im = 0.0f;
      for (int n = 0; n < ISTFT_N_FFT; n++) {
        float seg = xp[t * ISTFT_HOP + n] * win[n];
        re += seg * dft.c[k][n];
        im -= seg * dft.s[k][n];  // torch.stft sign convention: imag = -sum sin
      }
      s_stft_out[(size_t)k * N_FRAMES + t] = re;            // real bins 0..8
      s_stft_out[(size_t)(N_BINS + k) * N_FRAMES + t] = im; // imag bins 9..17
    }
  }
}

void istft(const std::vector<float>& magnitude, const std::vector<float>& phase,
           std::vector<float>& audio_out) {
  // magnitude/phase: [c][t], each N_BINS * N_FRAMES. Derive the sizes from input.
  const int N_FRAMES = (int)(magnitude.size() / N_BINS);
  const int L_S = (N_FRAMES - 1) * ISTFT_HOP;

  static const DftTables dft;
  float win[ISTFT_N_FFT];
  hann_window(win);

  const int out_len = (N_FRAMES - 1) * ISTFT_HOP + ISTFT_N_FFT;  // L_S + n_fft
  std::vector<float> acc(out_len, 0.0f);
  std::vector<float> env(out_len, 0.0f);

  for (int t = 0; t < N_FRAMES; t++) {
    // Build full 16-bin spectrum from 9 onesided bins (conjugate symmetry).
    float Xr[ISTFT_N_FFT], Xi[ISTFT_N_FFT];
    for (int k = 0; k < N_BINS; k++) {
      float m = magnitude[(size_t)k * N_FRAMES + t];
      if (m > 100.0f) m = 100.0f;  // torch.clip(magnitude, max=1e2) in _istft
      float p = phase[(size_t)k * N_FRAMES + t];
      Xr[k] = m * std::cos(p);
      Xi[k] = m * std::sin(p);
    }
    for (int k = 1; k < ISTFT_N_FFT / 2; k++) {
      Xr[ISTFT_N_FFT - k] = Xr[k];
      Xi[ISTFT_N_FFT - k] = -Xi[k];
    }
    // inverse DFT (real part): x[n] = (1/N) sum_k (Xr cos - Xi sin).
    for (int n = 0; n < ISTFT_N_FFT; n++) {
      float v = 0.0f;
      for (int k = 0; k < ISTFT_N_FFT; k++)
        v += Xr[k] * dft.c[k][n] - Xi[k] * dft.s[k][n];
      v /= ISTFT_N_FFT;
      acc[t * ISTFT_HOP + n] += v * win[n];
      env[t * ISTFT_HOP + n] += win[n] * win[n];
    }
  }

  audio_out.assign(L_S, 0.0f);
  for (int i = 0; i < L_S; i++) {
    float e = env[ISTFT_N_FFT / 2 + i];
    if (e < 1e-8f) e = 1e-8f;
    audio_out[i] = acc[ISTFT_N_FFT / 2 + i] / e;  // center trim win/2 from each side
  }
}

}  // namespace larynx::hift
