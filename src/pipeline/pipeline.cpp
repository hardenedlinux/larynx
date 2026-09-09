#include "larynx/pipeline/pipeline.h"

#include "larynx/flow/flow.h"
#include "larynx/hift/hift.h"
#include "larynx/llm/llm.h"
#include "larynx/llm/tokenizer.h"

#include <cstdio>

namespace larynx::pipeline {

namespace {

std::vector<int> to_int(const std::vector<int32_t>& v) {
  return std::vector<int>(v.begin(), v.end());
}

std::vector<int32_t> to_i32(const std::vector<int>& v) {
  return std::vector<int32_t>(v.begin(), v.end());
}

}  // namespace

struct Pipeline::Impl {
  larynx::llm::LLM llm;
  larynx::llm::Qwen2Tokenizer tokenizer;
  larynx::flow::FlowDecoder flow;
  larynx::hift::HiftVocoder hift;
};

Pipeline::Pipeline() : impl_(new Impl()) {}
Pipeline::~Pipeline() = default;

bool Pipeline::load(const std::string& llm_gguf,
                    const std::string& flow_gguf,
                    const std::string& hift_gguf,
                    const std::string& hift_source_bin,
                    const std::string& flow_noise_bin,
                    const std::string& tokenizer_dir) {
  if (!impl_->tokenizer.load(tokenizer_dir)) {
    std::fprintf(stderr, "pipeline: tokenizer load failed (%s)\n", tokenizer_dir.c_str());
    return false;
  }
  if (!impl_->llm.load(llm_gguf)) {
    std::fprintf(stderr, "pipeline: LLM load failed (%s)\n", llm_gguf.c_str());
    return false;
  }
  if (!impl_->flow.load(flow_gguf)) {
    std::fprintf(stderr, "pipeline: Flow load failed (%s)\n", flow_gguf.c_str());
    return false;
  }
  if (!impl_->flow.load_noise(flow_noise_bin)) {
    std::fprintf(stderr, "pipeline: Flow noise load failed (%s)\n", flow_noise_bin.c_str());
    return false;
  }
  if (!impl_->hift.load(hift_gguf)) {
    std::fprintf(stderr, "pipeline: HiFT load failed (%s)\n", hift_gguf.c_str());
    return false;
  }
  if (!impl_->hift.load_source(hift_source_bin)) {
    std::fprintf(stderr, "pipeline: HiFT source load failed (%s)\n", hift_source_bin.c_str());
    return false;
  }
  return true;
}

bool Pipeline::synthesize(const std::string& instruct,
                          const std::string& text,
                          const PromptFeatures& prompt,
                          unsigned seed,
                          SynthesisResult& out) {
  // 1. Tokenize prompt text (instruct) + target text, concat.
  const std::vector<int32_t> prompt_text_tok = impl_->tokenizer.encode(instruct);
  const std::vector<int32_t> text_tok = impl_->tokenizer.encode(text);
  std::vector<int32_t> text_all = prompt_text_tok;
  text_all.insert(text_all.end(), text_tok.begin(), text_tok.end());

  // 2. Assemble lm_input = [sos; embed_tokens(text_all); task_id; (empty)].
  //    (instruct2 mode: the LLM's prompt-speech-token block is empty.)
  std::vector<float> lm_input;
  int L = 0;
  if (!impl_->llm.build_lm_input(to_int(text_all), /*prompt_speech_token=*/{},
                                 lm_input, &L)) {
    std::fprintf(stderr, "pipeline: build_lm_input failed\n");
    return false;
  }

  // 3. Generate speech tokens (min/max from the reference: 2x / 20x text len).
  const int min_len = 2 * (int)text_tok.size();
  const int max_len = 20 * (int)text_tok.size();
  larynx::llm::GenerationResult gen;
  if (!impl_->llm.generate(lm_input, L, min_len, max_len, seed, &gen)) {
    std::fprintf(stderr, "pipeline: LLM generate failed\n");
    return false;
  }
  out.tokens = gen.tokens;

  // Release the LLM's resident weights (~2.4 GiB on CUDA) before the Flow
  // decoder builds its DiT graph (~4 GiB): the two do not fit an 8 GiB card
  // together. This mirrors the Python acceptance gate, which frees model.llm
  // before running the Flow/HiFT decoders. The LLM is single-shot per synth.
  impl_->llm.release();

  // 4. Flow: speech tokens -> mel (prompt token + matcha mel + spk embedding).
  std::vector<float> mel;
  if (!impl_->flow.infer(prompt.prompt_tokens, to_i32(gen.tokens),
                         prompt.prompt_feat, prompt.spk_embedding, mel)) {
    std::fprintf(stderr, "pipeline: Flow infer failed\n");
    return false;
  }
  out.mel = mel;

  // 5. HiFT: mel -> PCM.
  if (!impl_->hift.vocode(mel, out.audio)) {
    std::fprintf(stderr, "pipeline: HiFT vocode failed\n");
    return false;
  }
  out.sample_rate = 24000;
  return true;
}

}  // namespace larynx::pipeline
