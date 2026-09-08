#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace larynx::flow {

// Human-readable module banner for the CLI (Phase 2: implemented).
std::string module_name();

// Per-stage captures, each stored in the byte layout of the PyTorch reference
// (tests/flow_reference.py). Sizes follow the validation case: 4 prompt tokens
// + 8 tokens -> 12 speech tokens -> 24 mel frames -> 16 output mel frames.
struct FlowDebug {
  // pre-DiT stages
  std::vector<float> spk;          // (1, 80)
  std::vector<float> token_embed;  // (1, 12, 80)
  std::vector<float> prelookahead; // (1, 12, 80)
  std::vector<float> mu;           // (1, 80, 24)
  std::vector<float> cond;         // (1, 80, 24)

  // DiT internals at CFM step 1 (batch = 2: conditional + unconditional)
  std::vector<float> time_embed;         // (2, 1024)
  std::vector<float> input_proj;         // (2, 24, 1024)  (proj before conv_pos_embed)
  std::vector<float> conv_pos;           // (2, 24, 1024)  (conv_pos_embed output)
  std::vector<float> input_embed;        // (2, 24, 1024)
  std::vector<std::vector<float>> blocks; // 22 x (2, 24, 1024)
  std::vector<float> norm_out;           // (2, 24, 1024)
  std::vector<float> dphi;               // (2, 80, 24)

  // final output
  std::vector<float> feat;          // (1, 80, 16)
};

// Flow decoder: PreLookaheadLayer + 22-block DiT + CFM Euler solver, running on
// GGML (CPU this phase; CUDA is a drop-in backend swap). Not the streaming
// path — this is the whole-sequence (finalize=True) inference.
class FlowDecoder {
 public:
  FlowDecoder();
  ~FlowDecoder();
  FlowDecoder(const FlowDecoder&) = delete;
  FlowDecoder& operator=(const FlowDecoder&) = delete;

  // Load flow.gguf (produced by tools/convert_weights.py). Returns false on
  // any error (missing file, missing tensor).
  bool load(const std::string& gguf_path);

  // Load the CFM deterministic seed-noise bank (1, 80, 15000) from the asset
  // exported by tests/export_flow_noise.py. Not in flow.gguf — the reference
  // builds it from torch.randn(seed 0) at construction, so it is frozen to a
  // file once and consumed verbatim. Must be called before the 6-arg infer();
  // the 7-arg infer() takes explicit noise_z instead.
  bool load_noise(const std::string& noise_path);

  // token -> mel. All inputs are laid out as the PyTorch reference emits them
  // (numpy row-major). `noise_z` is the deterministic CFM seed noise, shape
  // (1, 80, MEL_T) where MEL_T = (prompt_tokens + tokens) * 2. `mel` is filled
  // with the (1, 80, MEL_T - prompt_feat_len) output.
  bool infer(const std::vector<int32_t>& prompt_tokens,
             const std::vector<int32_t>& tokens,
             const std::vector<float>& prompt_feat,    // (1, mel_len1, 80)
             const std::vector<float>& spk_embedding,  // (1, 192)
             const std::vector<float>& noise_z,        // (1, 80, MEL_T)
             std::vector<float>& mel,
             FlowDebug* debug);

  // Production entry point: uses the noise bank loaded by load_noise() (sliced
  // to MEL_T internally). Equivalent to the 7-arg form fed the first MEL_T
  // columns of the loaded bank.
  bool infer(const std::vector<int32_t>& prompt_tokens,
             const std::vector<int32_t>& tokens,
             const std::vector<float>& prompt_feat,
             const std::vector<float>& spk_embedding,
             std::vector<float>& mel,
             FlowDebug* debug = nullptr);

 private:
  struct Impl;
  Impl* impl_;
};

}  // namespace larynx::flow
