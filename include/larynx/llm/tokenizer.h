#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

namespace larynx::llm {

// Byte-level BPE text tokenizer reproducing `CosyVoice3Tokenizer.encode()`
// bit-exactly.
//
// Pipeline (identical to the serialized Rust Qwen2Tokenizer):
//   1. NFC normalize the input.
//   2. Extract the 281 special (added) tokens via longest-match on a trie
//      (split_special_tokens=false, so `[j]`, `<strong>`, `<|endofprompt|>` …
//      are matched whole, never split).
//   3. Split the remaining text with the GPT-2 regex (leftmost-first).
//   4. For each chunk: UTF-8 -> GPT-2 byte alphabet -> BPE merges -> vocab id.
//
// Data files (produced by tools/export_tokenizer.py) live in one directory:
//   - vocab.tsv         "token<TAB>id"        (151643 base BPE tokens)
//   - merges.txt        "left right"          (134935 merges, rank = line index)
//   - added_tokens.tsv  "id<TAB>content"      (281 special tokens)
class Qwen2Tokenizer {
 public:
  Qwen2Tokenizer() = default;

  // Load the three data files from `data_dir`. Returns false on any error.
  bool load(const std::string& data_dir);

  // Tokenize a UTF-8 string. Mirrors CosyVoice3Tokenizer.encode().
  std::vector<int32_t> encode(const std::string& text) const;

  size_t vocab_size() const { return vocab_.size(); }
  size_t merge_count() const { return merge_rank_.size(); }
  size_t special_count() const { return special_count_; }

 private:
  struct TrieNode {
    int32_t id = -1;  // >= 0 if this node terminates a special token
    std::unordered_map<char32_t, std::unique_ptr<TrieNode>> children;
  };

  // GPT-2 byte alphabet: byte value -> UTF-8 encoding of its byte-alphabet char.
  std::vector<std::string> byte_char_;

  // BPE model: byte-level token string (UTF-8) -> id.
  std::unordered_map<std::string, int32_t> vocab_;
  // "left right" -> merge rank (index in merges.txt; lower = higher priority).
  std::unordered_map<std::string, int32_t> merge_rank_;

  // Special-token trie (keyed by code point, longest-match).
  TrieNode special_root_;
  size_t special_count_ = 0;

  // BPE-encode one chunk (UTF-8 bytes of a single regex match).
  std::vector<int32_t> bpe(const std::string& chunk) const;
};

}  // namespace larynx::llm
