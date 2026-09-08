// larynx — native (zero-Python-at-runtime) CosyVoice3 TTS.
//
// End-to-end synthesis: text (+ prompt voice) -> PCM WAV, running the LLM,
// Flow and HiFT decoders on GGML. The prompt's speech token, matcha mel and
// speaker embedding are *pre-extracted* (Python) into a bundle directory — the
// ONNX frontend (campplus + speech tokenizer) and matcha mel are deferred, so
// this CLI reads them as files rather than computing them.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "larynx/flow/flow.h"
#include "larynx/frontend/frontend.h"
#include "larynx/hift/hift.h"
#include "larynx/llm/llm.h"
#include "larynx/pipeline/pipeline.h"

namespace {

struct Args {
  std::string instruct = "You are a helpful assistant. 请用普通话表达。<|endofprompt|>";
  std::string text;
  std::string prompt_dir;
  std::string llm_gguf = "build/llm.gguf";
  std::string flow_gguf = "build/flow.gguf";
  std::string hift_gguf = "build/hift.gguf";
  std::string hift_source = "build/hift_source.bin";
  std::string flow_noise = "build/flow_noise.bin";
  std::string tokenizer_dir = "build/tokenizer";
  std::string out_wav;
  std::string dump_tokens;
  std::string dump_mel;
  std::string dump_audio;
  unsigned seed = 0;
};

void usage(const char* argv0) {
  std::fprintf(stderr,
    "usage: %s --text <str> --prompt-dir <dir> --out <wav> [options]\n"
    "\n"
    "  --text <str>         text to synthesize (required)\n"
    "  --prompt-dir <dir>   pre-extracted prompt bundle (required):\n"
    "                         prompt_tokens.i32  (P int32 speech tokens)\n"
    "                         prompt_feat.f32    (mel_len1*80 matcha mel)\n"
    "                         spk_embedding.f32  (192 campplus embedding)\n"
    "  --out <wav>          output 16-bit PCM WAV @ 24 kHz (required)\n"
    "\n"
    "  --instruct <str>     LLM text prompt (must contain <|endofprompt|>)\n"
    "  --seed <n>           LLM sampling seed (default 0)\n"
    "  --llm/--flow/--hift <gguf>         model files (default build/*.gguf)\n"
    "  --hift-source <bin>  SineGen2 source asset (default build/hift_source.bin)\n"
    "  --flow-noise <bin>   CFM noise asset (default build/flow_noise.bin)\n"
    "  --tokenizer-dir <dir>              (default build/tokenizer)\n"
    "  --dump-tokens <i32>  also write generated speech tokens (int32)\n"
    "  --dump-mel <f32>     also write the Flow mel (80*OUT_FRAMES float32)\n"
    "  --dump-audio <f32>   also write the raw PCM (L_s float32, before 16-bit\n"
    "                       quantisation) for numerical comparison\n",
    argv0);
}

bool parse_args(int argc, char** argv, Args* a) {
  auto need = [&](int& i) -> const char* {
    if (i + 1 >= argc) return nullptr;
    return argv[++i];
  };
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    const char* v;
    if (k == "--text")           { v = need(i); if (!v) return false; a->text = v; }
    else if (k == "--prompt-dir"){ v = need(i); if (!v) return false; a->prompt_dir = v; }
    else if (k == "--out")       { v = need(i); if (!v) return false; a->out_wav = v; }
    else if (k == "--instruct")  { v = need(i); if (!v) return false; a->instruct = v; }
    else if (k == "--seed")      { v = need(i); if (!v) return false; a->seed = (unsigned)std::strtoul(v, nullptr, 10); }
    else if (k == "--llm")       { v = need(i); if (!v) return false; a->llm_gguf = v; }
    else if (k == "--flow")      { v = need(i); if (!v) return false; a->flow_gguf = v; }
    else if (k == "--hift")      { v = need(i); if (!v) return false; a->hift_gguf = v; }
    else if (k == "--hift-source"){ v = need(i); if (!v) return false; a->hift_source = v; }
    else if (k == "--flow-noise"){ v = need(i); if (!v) return false; a->flow_noise = v; }
    else if (k == "--tokenizer-dir"){ v = need(i); if (!v) return false; a->tokenizer_dir = v; }
    else if (k == "--dump-tokens"){ v = need(i); if (!v) return false; a->dump_tokens = v; }
    else if (k == "--dump-mel")  { v = need(i); if (!v) return false; a->dump_mel = v; }
    else if (k == "--dump-audio"){ v = need(i); if (!v) return false; a->dump_audio = v; }
    else return false;
  }
  return !a->text.empty() && !a->prompt_dir.empty() && !a->out_wav.empty();
}

bool read_i32(const std::string& path, std::vector<int32_t>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0 || n % 4 != 0) { std::fclose(f); return false; }
  out.resize(n / 4);
  bool ok = std::fread(out.data(), 4, out.size(), f) == out.size();
  std::fclose(f);
  return ok;
}

bool read_f32(const std::string& path, std::vector<float>& out) {
  FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return false;
  std::fseek(f, 0, SEEK_END);
  long n = std::ftell(f);
  std::fseek(f, 0, SEEK_SET);
  if (n < 0 || n % 4 != 0) { std::fclose(f); return false; }
  out.resize(n / 4);
  bool ok = std::fread(out.data(), 4, out.size(), f) == out.size();
  std::fclose(f);
  return ok;
}

bool write_i32(const std::string& path, const std::vector<int32_t>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  return ok;
}

bool write_f32(const std::string& path, const std::vector<float>& v) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;
  bool ok = std::fwrite(v.data(), 4, v.size(), f) == v.size();
  std::fclose(f);
  return ok;
}

bool write_wav(const std::string& path, const std::vector<float>& audio, int sr) {
  FILE* f = std::fopen(path.c_str(), "wb");
  if (!f) return false;

  auto put32 = [&](uint32_t x) { std::fwrite(&x, 4, 1, f); };
  auto put16 = [&](uint16_t x) { std::fwrite(&x, 2, 1, f); };

  const uint32_t n = (uint32_t)audio.size();
  const uint32_t data_bytes = n * 2;
  // 44-byte canonical WAV header (mono, 16-bit PCM).
  std::fwrite("RIFF", 1, 4, f); put32(36 + data_bytes);
  std::fwrite("WAVE", 1, 4, f);
  std::fwrite("fmt ", 1, 4, f); put32(16); put16(1); put16(1); put32(sr);
  put32(sr * 2); put16(2); put16(16);
  std::fwrite("data", 1, 4, f); put32(data_bytes);

  for (float s : audio) {
    float v = s * 32767.0f;
    if (v > 32767.0f) v = 32767.0f;
    if (v < -32768.0f) v = -32768.0f;
    put16((uint16_t)(int16_t)v);
  }
  std::fclose(f);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Args a;
  if (!parse_args(argc, argv, &a)) {
    usage(argv[0]);
    return 2;
  }

  larynx::pipeline::PromptFeatures prompt;
  if (!read_i32(a.prompt_dir + "/prompt_tokens.i32", prompt.prompt_tokens) ||
      !read_f32(a.prompt_dir + "/prompt_feat.f32", prompt.prompt_feat) ||
      !read_f32(a.prompt_dir + "/spk_embedding.f32", prompt.spk_embedding)) {
    std::fprintf(stderr, "larynx: failed to read prompt bundle in %s\n", a.prompt_dir.c_str());
    return 1;
  }
  if (prompt.spk_embedding.size() != 192) {
    std::fprintf(stderr, "larynx: spk_embedding.f32 has %zu floats, expected 192\n",
                 prompt.spk_embedding.size());
    return 1;
  }

  larynx::pipeline::Pipeline pipe;
  if (!pipe.load(a.llm_gguf, a.flow_gguf, a.hift_gguf, a.hift_source,
                 a.flow_noise, a.tokenizer_dir)) {
    std::fprintf(stderr, "larynx: pipeline load failed\n");
    return 1;
  }

  larynx::pipeline::SynthesisResult r;
  if (!pipe.synthesize(a.instruct, a.text, prompt, a.seed, r)) {
    std::fprintf(stderr, "larynx: synthesis failed\n");
    return 1;
  }

  if (!write_wav(a.out_wav, r.audio, r.sample_rate)) {
    std::fprintf(stderr, "larynx: failed to write %s\n", a.out_wav.c_str());
    return 1;
  }
  if (!a.dump_tokens.empty()) {
    std::vector<int32_t> t(r.tokens.begin(), r.tokens.end());
    write_i32(a.dump_tokens, t);
  }
  if (!a.dump_mel.empty()) write_f32(a.dump_mel, r.mel);
  if (!a.dump_audio.empty()) write_f32(a.dump_audio, r.audio);

  std::fprintf(stderr,
    "larynx: wrote %s  samples=%zu (%.2f s @ %d Hz)  tokens=%zu  mel=%zu frames\n",
    a.out_wav.c_str(), r.audio.size(), (double)r.audio.size() / r.sample_rate,
    r.sample_rate, r.tokens.size(), r.mel.size() / 80);
  return 0;
}
