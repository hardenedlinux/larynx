// Test utility: reads the corpus written by tests/tokenizer_reference.py, runs
// the C++ Qwen2Tokenizer on each text, and writes the token ids back out so
// tests/verify_tokenizer.py can compare them bit-for-bit against the Python
// CosyVoice3Tokenizer ground truth.
//
// Usage: larynx_tokenizer_dump <data_dir> <cases.bin> <out.bin>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "larynx/llm/tokenizer.h"

namespace {

uint32_t read_u32(FILE* f) {
  uint8_t b[4];
  if (std::fread(b, 1, 4, f) != 4) return 0;
  return (uint32_t)b[0] | ((uint32_t)b[1] << 8) | ((uint32_t)b[2] << 16) |
         ((uint32_t)b[3] << 24);
}

void write_u32(FILE* f, uint32_t v) {
  uint8_t b[4] = {(uint8_t)(v & 0xff), (uint8_t)((v >> 8) & 0xff),
                  (uint8_t)((v >> 16) & 0xff), (uint8_t)((v >> 24) & 0xff)};
  std::fwrite(b, 1, 4, f);
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::fprintf(stderr, "usage: %s <data_dir> <cases.bin> <out.bin>\n", argv[0]);
    return 2;
  }
  const std::string data_dir = argv[1];
  const std::string cases_path = argv[2];
  const std::string out_path = argv[3];

  larynx::llm::Qwen2Tokenizer tok;
  if (!tok.load(data_dir)) {
    std::fprintf(stderr, "failed to load tokenizer data from %s\n", data_dir.c_str());
    return 1;
  }
  std::fprintf(stderr, "loaded vocab=%zu merges=%zu special=%zu\n",
               tok.vocab_size(), tok.merge_count(), tok.special_count());

  FILE* in = std::fopen(cases_path.c_str(), "rb");
  if (!in) {
    std::fprintf(stderr, "cannot open %s\n", cases_path.c_str());
    return 1;
  }
  FILE* out = std::fopen(out_path.c_str(), "wb");
  if (!out) {
    std::fprintf(stderr, "cannot open %s\n", out_path.c_str());
    std::fclose(in);
    return 1;
  }

  uint32_t n = read_u32(in);
  write_u32(out, n);

  for (uint32_t ci = 0; ci < n; ++ci) {
    uint32_t text_len = read_u32(in);
    std::string text(text_len, '\0');
    if (text_len && std::fread(&text[0], 1, text_len, in) != text_len) {
      std::fprintf(stderr, "short read on case %u text\n", ci);
      return 1;
    }
    uint32_t id_len = read_u32(in);
    std::fseek(in, (long)id_len * 4, SEEK_CUR);  // skip ground-truth ids

    std::vector<int32_t> ids = tok.encode(text);
    write_u32(out, (uint32_t)ids.size());
    std::fwrite(ids.data(), 4, ids.size(), out);
    std::fprintf(stderr, "case %u: %zu tokens\n", ci, ids.size());
  }

  std::fclose(in);
  std::fclose(out);
  return 0;
}
