#include "larynx/llm/llm.h"

#include "internal.h"
#include "larynx/backend.h"
#include "larynx/llm/sampling.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace larynx::llm {

std::string module_name() {
  return "llm (Qwen2 backbone + speech_embedding/llm_decoder; GGML CPU/CUDA)";
}

namespace {

void read_tensor(ggml_tensor* t, std::vector<float>& dst) {
  dst.resize(ggml_nelements(t));
  ggml_backend_tensor_get(t, dst.data(), 0, ggml_nbytes(t));
}

}  // namespace

struct LLM::Impl {
  ggml_backend_t backend = nullptr;
  LLMWeights w;
  bool loaded = false;

  // KV cache (host-side, populated by prefill, appended to by decode). key is
  // ROPED, value raw. Flat layout [HEAD_DIM, KV_HEADS, cache_len] (d fastest,
  // then kv head, then seq). One entry per transformer layer.
  std::vector<float> kv_k[N_LAYERS];
  std::vector<float> kv_v[N_LAYERS];
  int cache_len = 0;
};

LLM::LLM() : impl_(new Impl()) {}

LLM::~LLM() {
  if (impl_) {
    if (impl_->w.buffer) ggml_backend_buffer_free(impl_->w.buffer);
    if (impl_->w.ctx) ggml_free(impl_->w.ctx);
    if (impl_->backend) ggml_backend_free(impl_->backend);
    delete impl_;
  }
}

bool LLM::load(const std::string& path) {
  if (impl_->loaded) return true;
  impl_->backend = backend_init_best();
  if (!impl_->backend) return false;
  if (!load_weights(path, impl_->backend, &impl_->w)) {
    ggml_backend_free(impl_->backend);
    impl_->backend = nullptr;
    return false;
  }
  impl_->loaded = true;
  return true;
}

bool LLM::build_lm_input(const std::vector<int>& text_tokens,
                         const std::vector<int>& prompt_speech_token,
                         std::vector<float>& lm_input, int* L_out) {
  if (!impl_ || !impl_->loaded) return false;
  if (L_out) *L_out = 0;

  const size_t PtT = text_tokens.size();
  const size_t P = prompt_speech_token.size();
  for (int t : text_tokens) if (t < 0 || t >= TEXT_VOCAB) return false;
  for (int t : prompt_speech_token) if (t < 0 || t >= SPEECH_VOCAB) return false;

  const int L = 1 + (int)PtT + 1 + (int)P;
  lm_input.assign((size_t)L * HIDDEN, 0.0f);

  // Copy one embedding row (HIDDEN contiguous floats) from a weight table laid
  // out [HIDDEN, NROWS] (ne[0]=HIDDEN fastest) at row `row` into `dst`.
  auto copy_row = [&](ggml_tensor* table, int row, float* dst) {
    ggml_backend_tensor_get(table, dst, (size_t)row * HIDDEN * sizeof(float),
                            HIDDEN * sizeof(float));
  };

  float* p = lm_input.data();
  copy_row(impl_->w.speech_embedding, SOS_TOKEN, p);              // 1. sos
  p += HIDDEN;
  for (int t : text_tokens) {                                     // 2. embed_tokens(text)
    copy_row(impl_->w.embed_tokens, t, p);
    p += HIDDEN;
  }
  copy_row(impl_->w.speech_embedding, TASK_ID_TOKEN, p);          // 3. task_id
  p += HIDDEN;
  for (int t : prompt_speech_token) {                             // 4. speech_embedding(prompt)
    copy_row(impl_->w.speech_embedding, t, p);
    p += HIDDEN;
  }

  if (L_out) *L_out = L;
  return true;
}

bool LLM::prefill(const std::vector<float>& lm_input, int L, LLMDebug* debug) {
  if (!impl_ || !impl_->loaded) return false;
  if (L <= 0 || (int)lm_input.size() != (int64_t)L * HIDDEN) return false;

  const int B = 1;

  // no_alloc=true: the ctx holds only tensor/graph metadata; the actual data
  // lives in a backend buffer (device memory on CUDA) allocated below.
  const size_t mem =
      ggml_tensor_overhead() * (size_t)65536 +
      ggml_graph_overhead_custom(8192, false);
  ggml_init_params dp = {.mem_size = mem, .mem_buffer = nullptr, .no_alloc = true};
  ggml_context* ctx = ggml_init(dp);
  if (!ctx) return false;

  TensorInit init;

  // Input: the pre-computed embedding sequence (1, L, 896) laid out [D, L, 1].
  ggml_tensor* x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HIDDEN, L, 1);
  ggml_set_input(x_in);

  // Position ids [0..L-1] for the rotary embedding (written after buffer alloc).
  ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, L);

  // Causal mask [L, L, 1, 1]: -inf where key > query (upper triangle).
  ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, L, L, 1, 1);
  {
    std::vector<float> m((size_t)L * L);
    for (int j = 0; j < L; j++)        // j = query (ne1)
      for (int i = 0; i < L; i++)      // i = key   (ne0)
        m[i + (size_t)j * L] = (i > j) ? -INFINITY : 0.0f;
    init.fill(mask, m);
  }

  // 24 transformer layers.
  ggml_tensor* states[N_LAYERS];
  ggml_tensor* cache_k[N_LAYERS];  // roped key, pre-GQA-repeat (the KV cache)
  ggml_tensor* cache_v[N_LAYERS];  // raw value, pre-GQA-repeat
  ggml_tensor* x = x_in;
  for (int i = 0; i < N_LAYERS; i++) {
    const auto& l = impl_->w.layer[i];

    // ---- input_layernorm + self-attention ----
    ggml_tensor* h = rms_norm(ctx, x, l.in_norm);

    ggml_tensor* q = linear(ctx, l.q_w, l.q_b, h);   // [HIDDEN, L, 1]
    ggml_tensor* k = linear(ctx, l.k_w, l.k_b, h);   // [KV_DIM, L, 1]
    ggml_tensor* v = linear(ctx, l.v_w, l.v_b, h);   // [KV_DIM, L, 1]

    q = rope_full(ctx, q, pos, N_HEADS);             // [HEAD_DIM, N_HEADS, L, 1]
    k = rope_full(ctx, k, pos, KV_HEADS);            // [HEAD_DIM, KV_HEADS, L, 1]
    v = ggml_reshape_4d(ctx, v, HEAD_DIM, KV_HEADS, L, 1);

    cache_k[i] = k;   // what transformers stores in past_key_values (roped)
    cache_v[i] = v;   // raw value

    k = repeat_kv(ctx, k, N_HEADS / KV_HEADS);       // [HEAD_DIM, N_HEADS, L, 1]
    v = repeat_kv(ctx, v, N_HEADS / KV_HEADS);

    ggml_tensor* attn = linear(ctx, l.o_w, nullptr,
                               attention(ctx, q, k, v, mask));  // [HIDDEN, L, 1]
    x = ggml_add(ctx, x, attn);

    // ---- post_attention_layernorm + SwiGLU MLP ----
    h = rms_norm(ctx, x, l.post_norm);
    ggml_tensor* gate = ggml_silu(ctx, linear(ctx, l.gate_w, nullptr, h));  // [INTERMEDIATE, L, 1]
    ggml_tensor* up   = linear(ctx, l.up_w, nullptr, h);                    // [INTERMEDIATE, L, 1]
    ggml_tensor* ff   = linear(ctx, l.down_w, nullptr, ggml_mul(ctx, gate, up));
    x = ggml_add(ctx, x, ff);

    states[i] = x;
  }

  // Final RMSNorm then llm_decoder. Qwen2Encoder.forward returns outs.hidden_states[-1],
  // which transformers defines as model.norm(layer_23_output) (the final norm IS applied
  // — verified against modeling_qwen2.py 4.51.3). lm_head is unused (speech tokens come
  // from llm_decoder, not the text lm_head).
  ggml_tensor* norm = rms_norm(ctx, x, impl_->w.final_norm);            // [HIDDEN, L, 1]
  ggml_tensor* logits = linear(ctx, impl_->w.llm_decoder, nullptr, norm); // [SPEECH_VOCAB, L, 1]

  ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
  ggml_build_forward_expand(gf, logits);

  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);
  if (!buf) {
    ggml_free(ctx);
    return false;
  }
  init.apply();

  ggml_backend_tensor_set(x_in, lm_input.data(), 0, sizeof(float) * lm_input.size());
  std::vector<int32_t> pos_host((size_t)L);
  for (int i = 0; i < L; i++) pos_host[i] = i;
  ggml_backend_tensor_set(pos, pos_host.data(), 0, sizeof(int32_t) * L);

  ggml_backend_graph_compute(impl_->backend, gf);

  // Populate the KV cache (roped key + raw value per layer) for a following decode.
  for (int i = 0; i < N_LAYERS; i++) {
    read_tensor(cache_k[i], impl_->kv_k[i]);
    read_tensor(cache_v[i], impl_->kv_v[i]);
  }
  impl_->cache_len = L;

  if (debug) {
    debug->hidden_states.resize(N_LAYERS);
    for (int i = 0; i < N_LAYERS; i++) read_tensor(states[i], debug->hidden_states[i]);
    read_tensor(norm, debug->final_norm);
    read_tensor(logits, debug->logits);
  }

  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return true;
}

bool LLM::decode(int token, std::vector<float>& logits, std::vector<float>* final_norm) {
  if (!impl_ || !impl_->loaded) return false;
  if (token < 0 || token >= SPEECH_VOCAB) return false;
  if (impl_->cache_len <= 0) return false;  // prefill first

  const int T = impl_->cache_len;  // number of cached tokens
  const int B = 1;

  // Look up the token embedding (row `token` of speech_embedding, [HIDDEN, SPEECH_VOCAB]).
  std::vector<float> emb((size_t)HIDDEN);
  const size_t row_off = (size_t)token * HIDDEN;
  ggml_backend_tensor_get(impl_->w.speech_embedding, emb.data(),
                          row_off * sizeof(float), HIDDEN * sizeof(float));

  const size_t mem =
      ggml_tensor_overhead() * (size_t)65536 +
      ggml_graph_overhead_custom(8192, false);
  ggml_init_params dp = {.mem_size = mem, .mem_buffer = nullptr, .no_alloc = true};
  ggml_context* ctx = ggml_init(dp);
  if (!ctx) return false;

  TensorInit init;

  // Single-token input [HIDDEN, 1, 1] and its absolute position id = cache_len.
  ggml_tensor* x_in = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, HIDDEN, 1, 1);
  ggml_set_input(x_in);
  ggml_tensor* pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, 1);

  // Decode mask [T+1, 1, 1, 1] all-zero: the new token attends to every cached key
  // (causal is satisfied automatically — it is the newest position).
  ggml_tensor* mask = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, T + 1, 1, 1, 1);
  init.zero(mask);

  ggml_tensor* new_k[N_LAYERS];  // roped new key, for cache append
  ggml_tensor* new_v[N_LAYERS];  // raw new value
  ggml_tensor* x = x_in;
  for (int i = 0; i < N_LAYERS; i++) {
    const auto& l = impl_->w.layer[i];

    ggml_tensor* h = rms_norm(ctx, x, l.in_norm);
    ggml_tensor* q = linear(ctx, l.q_w, l.q_b, h);    // [HIDDEN, 1, 1]
    ggml_tensor* kn = linear(ctx, l.k_w, l.k_b, h);   // [KV_DIM, 1, 1]
    ggml_tensor* vn = linear(ctx, l.v_w, l.v_b, h);   // [KV_DIM, 1, 1]

    q  = rope_full(ctx, q, pos, N_HEADS);   // [HEAD_DIM, N_HEADS, 1, 1]
    kn = rope_full(ctx, kn, pos, KV_HEADS); // [HEAD_DIM, KV_HEADS, 1, 1]
    vn = ggml_reshape_4d(ctx, vn, HEAD_DIM, KV_HEADS, 1, 1);
    new_k[i] = kn;
    new_v[i] = vn;

    // Cached key/value as constant tensors, concatenated with the new token.
    ggml_tensor* kc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, KV_HEADS, T, 1);
    ggml_tensor* vc = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, HEAD_DIM, KV_HEADS, T, 1);
    init.fill(kc, impl_->kv_k[i]);
    init.fill(vc, impl_->kv_v[i]);

    ggml_tensor* k = ggml_concat(ctx, kc, kn, 2);  // [HEAD_DIM, KV_HEADS, T+1, 1]
    ggml_tensor* v = ggml_concat(ctx, vc, vn, 2);

    k = repeat_kv(ctx, k, N_HEADS / KV_HEADS);
    v = repeat_kv(ctx, v, N_HEADS / KV_HEADS);

    ggml_tensor* attn = linear(ctx, l.o_w, nullptr,
                               attention(ctx, q, k, v, mask));  // [HIDDEN, 1, 1]
    x = ggml_add(ctx, x, attn);

    h = rms_norm(ctx, x, l.post_norm);
    ggml_tensor* gate = ggml_silu(ctx, linear(ctx, l.gate_w, nullptr, h));
    ggml_tensor* up   = linear(ctx, l.up_w, nullptr, h);
    ggml_tensor* ff   = linear(ctx, l.down_w, nullptr, ggml_mul(ctx, gate, up));
    x = ggml_add(ctx, x, ff);
  }

  ggml_tensor* norm = rms_norm(ctx, x, impl_->w.final_norm);            // [HIDDEN, 1, 1]
  ggml_tensor* log  = linear(ctx, impl_->w.llm_decoder, nullptr, norm); // [SPEECH_VOCAB, 1, 1]

  ggml_cgraph* gf = ggml_new_graph_custom(ctx, 8192, false);
  ggml_build_forward_expand(gf, log);

  ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors(ctx, impl_->backend);
  if (!buf) {
    ggml_free(ctx);
    return false;
  }
  init.apply();

  ggml_backend_tensor_set(x_in, emb.data(), 0, HIDDEN * sizeof(float));
  int32_t p = T;
  ggml_backend_tensor_set(pos, &p, 0, sizeof(int32_t));

  ggml_backend_graph_compute(impl_->backend, gf);

  read_tensor(log, logits);  // [SPEECH_VOCAB] (6761 floats)
  if (final_norm) read_tensor(norm, *final_norm);

  // Append the new token's roped key / raw value to the cache.
  std::vector<float> nk((size_t)HEAD_DIM * KV_HEADS);
  std::vector<float> nv((size_t)HEAD_DIM * KV_HEADS);
  for (int i = 0; i < N_LAYERS; i++) {
    read_tensor(new_k[i], nk);
    read_tensor(new_v[i], nv);
    impl_->kv_k[i].insert(impl_->kv_k[i].end(), nk.begin(), nk.end());
    impl_->kv_v[i].insert(impl_->kv_v[i].end(), nv.begin(), nv.end());
  }
  impl_->cache_len += 1;

  ggml_backend_buffer_free(buf);
  ggml_free(ctx);
  return true;
}

bool LLM::generate(const std::vector<float>& lm_input, int L,
                   int min_len, int max_len, unsigned seed, GenerationResult* out) {
  if (!out) return false;
  *out = GenerationResult{};
  if (!impl_ || !impl_->loaded) return false;
  if (max_len <= 0) return false;

  // Prefill, then take the llm_decoder logits at the final position (the "prefill
  // logit" from which the first token is sampled).
  LLMDebug debug;
  if (!prefill(lm_input, L, &debug)) return false;
  if ((int64_t)debug.logits.size() != (int64_t)L * SPEECH_VOCAB) return false;

  std::vector<float> logits(SPEECH_VOCAB);
  std::copy(debug.logits.data() + (size_t)(L - 1) * SPEECH_VOCAB,
            debug.logits.data() + (size_t)L * SPEECH_VOCAB, logits.begin());

  std::vector<int> decoded_tokens;
  std::vector<float> ws(SPEECH_VOCAB);
  SamplingParams sp;
  int stop_token = -1;
  bool stopped = false;

  for (int i = 0; i < max_len; i++) {
    log_softmax(logits.data(), SPEECH_VOCAB, ws.data());
    if (i < min_len) ws[SPEECH_TOKEN_SIZE] = -INFINITY;  // ignore_eos gate

    // Distinct-but-deterministic RNG stream per step (the C++ multinomial is not
    // torch's RNG, so the token sequence is our own — exactly what we want to
    // exercise here: a fully self-sampled trajectory).
    sp.seed = seed + (unsigned)i;
    int token = ras_sample(ws.data(), SPEECH_VOCAB, decoded_tokens, sp);

    if (token >= SPEECH_TOKEN_SIZE) {  // stop_token_ids = [6561..6760]
      stop_token = token;
      stopped = true;
      break;
    }
    decoded_tokens.push_back(token);
    if (!decode(token, logits)) return false;
  }

  out->tokens = std::move(decoded_tokens);
  out->stop_token = stop_token;
  out->stopped = stopped;
  out->truncated = !stopped;
  out->steps = (int)out->tokens.size();
  return true;
}

void LLM::release() {
  if (!impl_) return;
  if (impl_->w.buffer) { ggml_backend_buffer_free(impl_->w.buffer); impl_->w.buffer = nullptr; }
  if (impl_->w.ctx) { ggml_free(impl_->w.ctx); impl_->w.ctx = nullptr; }
  if (impl_->backend) { ggml_backend_free(impl_->backend); impl_->backend = nullptr; }
  impl_->loaded = false;
  for (int i = 0; i < N_LAYERS; i++) {
    impl_->kv_k[i].clear();
    impl_->kv_v[i].clear();
  }
  impl_->cache_len = 0;
}

int LLM::cache_len() const {
  return impl_ ? impl_->cache_len : 0;
}

const std::vector<float>& LLM::cache_key(int layer) const {
  static const std::vector<float> empty;
  if (!impl_ || layer < 0 || layer >= N_LAYERS) return empty;
  return impl_->kv_k[layer];
}

const std::vector<float>& LLM::cache_value(int layer) const {
  static const std::vector<float> empty;
  if (!impl_ || layer < 0 || layer >= N_LAYERS) return empty;
  return impl_->kv_v[layer];
}

}  // namespace larynx::llm
