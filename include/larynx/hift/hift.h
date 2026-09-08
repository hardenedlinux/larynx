#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace larynx::hift {

// Human-readable module banner for the CLI (Phase 3: implemented).
std::string module_name();

// Per-stage captures, each stored in the byte layout of the PyTorch reference
// (tests/hift_reference.py): ggml [T,C,N] tensors are read raw, which is the
// torch (1,C,T) row-major layout; host tensors are emitted in the same layout.
// Sizes follow the validation case: mel (1,80,30) -> 14400 samples.
struct HiFTDebug {
  // f0 / source path (host)
  std::vector<float> f0;         // (1, 30)
  std::vector<float> sine_wavs;  // (1, 14400, 9)
  std::vector<float> sine_merge; // (1, 14400)
  std::vector<float> s_stft;     // (1, 18, 3601)

  // main network
  std::vector<float> conv_pre;           // (1, 512, 30)
  std::vector<float> ups_lrelu[3];       // (1, 512/256/128, ...)
  std::vector<float> ups[3];             // (1, 256/128/64, ...)
  std::vector<float> source_downs[3];    // (1, 256/128/64, ...)
  std::vector<float> source_resblocks[3];
  std::vector<float> fusion[3];
  std::vector<std::vector<float>> resblocks; // 9 x (1, ch, ...)
  std::vector<float> post_resblocks[3];
  std::vector<float> reflection_pad;     // (1, 64, 3601)
  std::vector<float> final_lrelu;        // (1, 64, 3601)
  std::vector<float> conv_post;          // (1, 18, 3601)
  std::vector<float> magnitude;          // (1, 9, 3601)
  std::vector<float> phase;              // (1, 9, 3601)

  // ISTFT / output
  std::vector<float> istft;    // (1, 14400) raw ISTFT (pre-clamp)
  std::vector<float> speech;   // (1, 14400) clamped to [-0.99, 0.99]
};

// HiFT vocoder (CausalHiFTGenerator): speech_feat (mel) -> PCM waveform. Runs the
// f0 predictor (float64) and SineGen2/STFT/ISTFT (float32) on the host; the conv
// network (conv_pre + upsample/resblock + conv_post) runs as a single GGML graph
// (CPU this phase). Not the streaming path — whole-sequence (finalize=True).
class HiftVocoder {
 public:
  HiftVocoder();
  ~HiftVocoder();
  HiftVocoder(const HiftVocoder&) = delete;
  HiftVocoder& operator=(const HiftVocoder&) = delete;

  // Load hift.gguf (produced by tools/convert_weights.py). Returns false on any
  // error (missing file, missing tensor).
  bool load(const std::string& gguf_path);

  // Load the SineGen2 fixed source buffers (rand_ini + the full 300 s
  // sine_waves bank) from the asset exported by tests/export_hift_source.py.
  // These are not in hift.gguf — the reference model samples them from an
  // unseeded RNG at construction time, so they are frozen to a file once and
  // consumed verbatim (never regenerated). Must be called before the 3-arg
  // vocode(); the 5-arg vocode() takes explicit buffers instead.
  bool load_source(const std::string& source_path);

  // The buffers loaded by load_source(), for verification (empty until then).
  const std::vector<float>& source_rand_ini() const;
  const std::vector<float>& source_sine_waves() const;

  // mel -> PCM. `mel` is (1, 80, T_mel) numpy row-major (any T_mel); `rand_ini`
  // (1, 9) and `sine_waves` (1, T_mel*480, 9) are the fixed model-internal
  // buffers (consumed verbatim, never regenerated). `audio` is filled with the
  // (1, T_mel*480) output. Returns false if not loaded or an input has the
  // wrong size.
  bool vocode(const std::vector<float>& mel,
              const std::vector<float>& rand_ini,
              const std::vector<float>& sine_waves,
              std::vector<float>& audio,
              HiFTDebug* debug);

  // Production entry point: uses the source buffers loaded by load_source()
  // (slicing the bank to T_mel*480 rows). Equivalent to the 5-arg form called
  // with the loaded rand_ini and the first T_mel*480 rows of the loaded bank.
  bool vocode(const std::vector<float>& mel, std::vector<float>& audio,
              HiFTDebug* debug = nullptr);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace larynx::hift
