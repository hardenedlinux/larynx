#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace larynx::pipeline {

// Pre-extracted prompt-side features. These come from the deferred Python-only
// components (speech tokenizer, campplus, matcha mel), written to disk by
// tests/extract_prompt_features.py (same binary layout the acceptance gate uses
// under wavs/flow_inputs/). The C++ pipeline reads them verbatim — it never
// runs the ONNX frontend or matcha itself.
struct PromptFeatures {
  // Flow prompt speech token (the prompt audio's speech-token sequence), (P,) int32.
  std::vector<int32_t> prompt_tokens;
  // Flow prompt_feat: matcha 80-bin mel of the prompt audio, (mel_len1*80,) float32.
  std::vector<float> prompt_feat;
  // campplus speaker embedding, (192,) float32.
  std::vector<float> spk_embedding;
};

// Outcome of a synthesize() call.
struct SynthesisResult {
  std::vector<float> audio;      // (L_s,) PCM float32 @ 24 kHz
  std::vector<int32_t> tokens;   // generated speech tokens (excludes stop token)
  std::vector<float> mel;        // Flow output (1, 80, OUT_FRAMES) row-major
  int sample_rate = 24000;
};

// Top-level CosyVoice3 orchestration: Qwen2Tokenizer -> LLM (generate speech
// tokens) -> Flow (token->mel) -> HiFT (mel->PCM). All three decoder networks run
// on GGML; the LLM's text prompt and the Flow/HiFT fixed buffers are loaded once
// at startup. This is the non-streaming, whole-sequence (finalize=True) path.
//
// Mode note: this wires the instruction-following (instruct2) flow — the LLM's
// lm_input uses an *empty* prompt-speech-token block (as frontend_instruct2 does),
// while the Flow consumes the prompt's speech token + matcha mel + speaker
// embedding. The voice-cloning branch (LLM prompt_speech_token != empty) is the
// same build_lm_input path with a non-empty vector, already verified in Phase 2.
class Pipeline {
 public:
  Pipeline();
  ~Pipeline();

  Pipeline(const Pipeline&) = delete;
  Pipeline& operator=(const Pipeline&) = delete;

  // Load the LLM/Flow/HiFT ggufs, the two fixed-asset buffers (HiFT SineGen2
  // source + Flow CFM noise) and the tokenizer data dir. Returns false on any
  // error.
  bool load(const std::string& llm_gguf,
            const std::string& flow_gguf,
            const std::string& hift_gguf,
            const std::string& hift_source_bin,
            const std::string& flow_noise_bin,
            const std::string& tokenizer_dir);

  // Synthesize `text` in the prompt's voice. `instruct` is the LLM text prompt
  // (must contain <|endofprompt|>); `seed` drives the LLM sampling RNG. Returns
  // false on any error; fills `out` on success.
  bool synthesize(const std::string& instruct,
                  const std::string& text,
                  const PromptFeatures& prompt,
                  unsigned seed,
                  SynthesisResult& out);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace larynx::pipeline
