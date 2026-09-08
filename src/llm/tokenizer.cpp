#include "larynx/llm/tokenizer.h"

#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <sstream>

namespace larynx::llm {

namespace {

// --- GPT-2 byte alphabet (bytes_to_unicode) ---------------------------------
// Maps each byte value to a Unicode code point: printable ASCII (33..126) and
// Latin-1 punctuation (161..172, 174..255) map to themselves; the remaining 68
// bytes (0..32, 127..160, 173) map to 256+n in order of byte value.
std::vector<char32_t> gpt2_byte_alphabet() {
  std::vector<char32_t> table(256, 0);
  bool in_direct[256] = {false};
  for (int b = 33; b <= 126; ++b) in_direct[b] = true;
  for (int b = 161; b <= 172; ++b) in_direct[b] = true;
  for (int b = 174; b <= 255; ++b) in_direct[b] = true;
  int n = 0;
  for (int b = 0; b < 256; ++b) {
    if (in_direct[b]) {
      table[b] = static_cast<char32_t>(b);
    } else {
      table[b] = static_cast<char32_t>(256 + n);
      ++n;
    }
  }
  return table;
}

// Encode a code point as UTF-8 and append to `out`.
void append_utf8(std::string& out, char32_t cp) {
  char buf[4];
  int32_t i = 0;
  UBool is_error = false;
  U8_APPEND(buf, i, 4, cp, is_error);
  if (!is_error) out.append(buf, i);
}

// NFC-normalize a UTF-8 string (matches Python unicodedata.normalize("NFC")).
std::string nfc_normalize(const std::string& utf8) {
  UErrorCode st = U_ZERO_ERROR;
  const UNormalizer2* nfc = unorm2_getNFCInstance(&st);
  if (U_FAILURE(st)) return utf8;

  int32_t len16 = 0;
  st = U_ZERO_ERROR;
  u_strFromUTF8(nullptr, 0, &len16, utf8.data(), (int32_t)utf8.size(), &st);
  st = U_ZERO_ERROR;
  std::vector<UChar> src(len16 + 1, 0);
  u_strFromUTF8(src.data(), len16 + 1, nullptr, utf8.data(),
                (int32_t)utf8.size(), &st);
  if (U_FAILURE(st)) return utf8;

  st = U_ZERO_ERROR;
  int32_t need = unorm2_normalize(nfc, src.data(), len16, nullptr, 0, &st);
  st = U_ZERO_ERROR;
  std::vector<UChar> norm(need + 1, 0);
  unorm2_normalize(nfc, src.data(), len16, norm.data(), need, &st);
  if (U_FAILURE(st)) return utf8;

  st = U_ZERO_ERROR;
  int32_t len8 = 0;
  u_strToUTF8(nullptr, 0, &len8, norm.data(), need, &st);
  st = U_ZERO_ERROR;
  std::string out((size_t)len8 + 1, '\0');
  u_strToUTF8(out.data(), len8 + 1, nullptr, norm.data(), need, &st);
  if (U_FAILURE(st)) return utf8;
  out.resize((size_t)len8);
  return out;
}

// --- Unicode classification (mirrors Rust regex \p{L} / \p{N} / \s) ---------
bool is_L(char32_t c) {
  int8_t t = u_charType(c);
  return t >= U_UPPERCASE_LETTER && t <= U_OTHER_LETTER;  // Lu,Ll,Lt,Lm,Lo
}
bool is_N(char32_t c) {
  int8_t t = u_charType(c);
  return t >= U_DECIMAL_DIGIT_NUMBER && t <= U_OTHER_NUMBER;  // Nd,Nl,No
}
bool is_ws(char32_t c) { return u_isspace(c) != 0; }  // White_Space property

bool ascii_ci(char32_t c, char lo) {
  char up = (lo >= 'a' && lo <= 'z') ? (char)(lo - 32) : lo;
  return c == (char32_t)lo || c == (char32_t)up;
}

// --- GPT-2 regex split (leftmost-first) -------------------------------------
// The exact pattern serialized in the Rust tokenizer:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
//   ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// Returns the match length (in code points) of the leftmost-first match at
// position `pos`, or 0 if none (should not happen: the pattern covers all).
size_t gpt2_match(const std::vector<char32_t>& s, size_t pos, size_t end) {
  if (pos >= end) return 0;
  char32_t c = s[pos];

  // Branch 1: contractions (case-insensitive).
  if (c == '\'') {
    if (pos + 1 < end) {
      char32_t a = s[pos + 1];
      if (ascii_ci(a, 's')) return 2;
      if (ascii_ci(a, 't')) return 2;
      if (ascii_ci(a, 'r') && pos + 2 < end && ascii_ci(s[pos + 2], 'e')) return 3;
      if (ascii_ci(a, 'v') && pos + 2 < end && ascii_ci(s[pos + 2], 'e')) return 3;
      if (ascii_ci(a, 'm')) return 2;
      if (ascii_ci(a, 'l') && pos + 2 < end && ascii_ci(s[pos + 2], 'l')) return 3;
      if (ascii_ci(a, 'd')) return 2;
    }
  }

  // Branch 2: [^\r\n\p{L}\p{N}]?\p{L}+
  {
    if (is_L(c)) {
      size_t p = pos;
      while (p < end && is_L(s[p])) ++p;
      return p - pos;
    }
    // Optional leading char: not \r, not \n, not L, not N (space/tab/symbol ok).
    if (c != '\r' && c != '\n' && !is_N(c)) {
      if (pos + 1 < end && is_L(s[pos + 1])) {
        size_t p = pos + 1;
        while (p < end && is_L(s[p])) ++p;
        return p - pos;
      }
    }
  }

  // Branch 3: \p{N} (a single number).
  if (is_N(c)) return 1;

  // Branch 4:  ?[^\s\p{L}\p{N}]+[\r\n]*
  {
    size_t p = pos;
    if (s[p] == ' ') ++p;  // optional literal space
    if (p < end && !is_ws(s[p]) && !is_L(s[p]) && !is_N(s[p])) {
      while (p < end && !is_ws(s[p]) && !is_L(s[p]) && !is_N(s[p])) ++p;
      while (p < end && (s[p] == '\r' || s[p] == '\n')) ++p;
      return p - pos;
    }
  }

  // Branches 5/6/7: \s*[\r\n]+ | \s+(?!\S) | \s+
  if (is_ws(c)) {
    size_t w = pos;
    while (w < end && is_ws(s[w])) ++w;
    // Branch 5: whitespace* then a non-empty \r\n run. Greedy \s* backs off to
    // the LAST \r\n, so the match ends right after it.
    int last_rn = -1;
    for (size_t j = pos; j < w; ++j) {
      if (s[j] == '\r' || s[j] == '\n') last_rn = (int)j;
    }
    if (last_rn >= 0) return (size_t)(last_rn + 1) - pos;
    // Branch 6: \s+(?!\S) — a whitespace run followed by whitespace-or-end.
    // Greedy \s+ backs off one char when followed by non-whitespace, so it
    // matches [pos, w-1); if the run reaches end-of-string it matches all.
    if (w == end) return w - pos;          // run hits EOF
    if (w - pos >= 2) return w - pos - 1;  // followed by non-ws: leave last
    // Branch 7: \s+ — a single whitespace char followed by non-whitespace.
    return 1;
  }

  return 0;  // unreachable for valid input
}

}  // namespace

// --- load -------------------------------------------------------------------

bool Qwen2Tokenizer::load(const std::string& data_dir) {
  const std::string vocab_path = data_dir + "/vocab.tsv";
  const std::string merges_path = data_dir + "/merges.txt";
  const std::string added_path = data_dir + "/added_tokens.tsv";

  // Byte alphabet (fixed GPT-2 table) -> UTF-8 strings.
  const std::vector<char32_t> alphabet = gpt2_byte_alphabet();
  byte_char_.resize(256);
  for (int b = 0; b < 256; ++b) append_utf8(byte_char_[b], alphabet[b]);

  // vocab.tsv: token<TAB>id
  {
    std::ifstream f(vocab_path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      size_t tab = line.find('\t');
      if (tab == std::string::npos) return false;
      std::string token = line.substr(0, tab);
      int32_t id = std::stoi(line.substr(tab + 1));
      vocab_[token] = id;
    }
  }

  // merges.txt: "left right"
  {
    std::ifstream f(merges_path);
    if (!f.is_open()) return false;
    std::string line;
    int32_t rank = 0;
    while (std::getline(f, line)) {
      size_t sp = line.find(' ');
      if (sp == std::string::npos || sp == 0 || sp + 1 >= line.size())
        return false;
      merge_rank_[line] = rank++;  // key is exactly "left right"
    }
  }

  // added_tokens.tsv: id<TAB>content
  {
    std::ifstream f(added_path);
    if (!f.is_open()) return false;
    std::string line;
    while (std::getline(f, line)) {
      if (line.empty()) continue;
      size_t tab = line.find('\t');
      if (tab == std::string::npos) return false;
      int32_t id = std::stoi(line.substr(0, tab));
      std::string content = line.substr(tab + 1);
      // Insert into the trie keyed by code points.
      TrieNode* node = &special_root_;
      int32_t i = 0, len = (int32_t)content.size();
      while (i < len) {
        UChar32 cp;
        U8_NEXT(content.data(), i, len, cp);
        if (cp < 0) cp = 0xFFFD;
        auto& next = node->children[cp];
        if (!next) next = std::make_unique<TrieNode>();
        node = next.get();
      }
      node->id = id;
      ++special_count_;
    }
  }

  return true;
}

// --- BPE --------------------------------------------------------------------

std::vector<int32_t> Qwen2Tokenizer::bpe(const std::string& chunk) const {
  // chunk is raw UTF-8 bytes -> map each byte to its byte-alphabet char.
  std::vector<std::string> parts;
  parts.reserve(chunk.size());
  for (unsigned char b : chunk) parts.push_back(byte_char_[b]);

  // Greedy BPE: repeatedly merge the leftmost pair with the lowest rank.
  while (parts.size() >= 2) {
    int32_t best_rank = INT32_MAX;
    size_t best_i = parts.size();
    for (size_t i = 0; i + 1 < parts.size(); ++i) {
      std::string key = parts[i];
      key += ' ';
      key += parts[i + 1];
      auto it = merge_rank_.find(key);
      if (it != merge_rank_.end() && it->second < best_rank) {
        best_rank = it->second;
        best_i = i;
      }
    }
    if (best_i == parts.size()) break;  // no merge available
    parts[best_i] += parts[best_i + 1];
    parts.erase(parts.begin() + best_i + 1);
  }

  std::vector<int32_t> ids;
  ids.reserve(parts.size());
  for (const std::string& p : parts) {
    auto it = vocab_.find(p);
    // Every byte and every merge result is in vocab for GPT-2 byte-level BPE;
    // fall back to 0 defensively (never expected to trigger).
    ids.push_back(it != vocab_.end() ? it->second : 0);
  }
  return ids;
}

// --- encode -----------------------------------------------------------------

std::vector<int32_t> Qwen2Tokenizer::encode(const std::string& text) const {
  const std::string norm = nfc_normalize(text);

  // Decode to code points + byte offsets for slicing the original UTF-8.
  std::vector<char32_t> cps;
  std::vector<size_t> offs;
  offs.push_back(0);
  {
    int32_t i = 0, len = (int32_t)norm.size();
    while (i < len) {
      UChar32 cp;
      U8_NEXT(norm.data(), i, len, cp);
      if (cp < 0) cp = 0xFFFD;
      cps.push_back(cp);
      offs.push_back((size_t)i);
    }
  }
  const size_t n = cps.size();

  std::vector<int32_t> out;
  size_t pos = 0;
  bool in_normal = false;
  size_t norm_start = 0;

  auto flush_normal = [&](size_t a, size_t b) {
    // GPT-2 regex split [a, b) into chunks, then byte-level BPE each chunk.
    size_t p = a;
    while (p < b) {
      size_t m = gpt2_match(cps, p, b);
      if (m == 0) {  // defensive; the pattern covers all input
        ++p;
        continue;
      }
      std::string chunk = norm.substr(offs[p], offs[p + m] - offs[p]);
      std::vector<int32_t> ids = bpe(chunk);
      out.insert(out.end(), ids.begin(), ids.end());
      p += m;
    }
  };

  auto match_special = [&](size_t p, size_t& mlen) -> int32_t {
    const TrieNode* node = &special_root_;
    int32_t best_id = -1;
    size_t best_len = 0;
    for (size_t i = p; i < n; ++i) {
      auto it = node->children.find(cps[i]);
      if (it == node->children.end()) break;
      node = it->second.get();
      if (node->id >= 0) {
        best_id = node->id;
        best_len = i - p + 1;
      }
    }
    mlen = best_len;
    return best_id;
  };

  while (pos < n) {
    size_t mlen = 0;
    int32_t sid = match_special(pos, mlen);
    if (sid >= 0) {
      if (in_normal) {
        flush_normal(norm_start, pos);
        in_normal = false;
      }
      out.push_back(sid);
      pos += mlen;
    } else {
      if (!in_normal) {
        norm_start = pos;
        in_normal = true;
      }
      ++pos;
    }
  }
  if (in_normal) flush_normal(norm_start, n);

  return out;
}

}  // namespace larynx::llm
