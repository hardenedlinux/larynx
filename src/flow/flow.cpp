#include "larynx/flow/flow.h"

#include "internal.h"

#include <cmath>
#include <cstring>

namespace larynx::flow {

std::string module_name() {
  return "flow (PreLookaheadLayer + DiT x22 + CFM Euler; GGML CPU, implemented)";
}

namespace {

constexpr float PI = 3.14159265358979323846f;

void read_tensor(ggml_tensor* t, std::vector<float>& dst) {
  dst.resize(ggml_nelements(t));
  memcpy(dst.data(), t->data, ggml_nbytes(t));
}

}  // namespace

struct FlowDecoder::Impl {
  ggml_backend_t backend = nullptr;
  FlowWeights w;
  TimeMlpHost time_mlp;
  bool loaded = false;
};

FlowDecoder::FlowDecoder() : impl_(new Impl()) {}

FlowDecoder::~FlowDecoder() {
  if (impl_) {
    if (impl_->w.ctx) ggml_free(impl_->w.ctx);
    if (impl_->backend) ggml_backend_free(impl_->backend);
    delete impl_;
  }
}

bool FlowDecoder::load(const std::string& path) {
  if (impl_->loaded) return true;
  impl_->backend = ggml_backend_cpu_init();
  if (!impl_->backend) return false;
  if (!load_weights(path, impl_->backend, &impl_->w, &impl_->time_mlp)) {
    ggml_backend_free(impl_->backend);
    impl_->backend = nullptr;
    return false;
  }
  impl_->loaded = true;
  return true;
}

bool FlowDecoder::infer(const std::vector<int32_t>& prompt_tokens,
                        const std::vector<int32_t>& tokens,
                        const std::vector<float>& prompt_feat,
                        const std::vector<float>& spk_embedding,
                        const std::vector<float>& noise_z,
                        std::vector<float>& mel,
                        FlowDebug* debug) {
  if (!impl_ || !impl_->loaded) return false;

  const int P  = (int)prompt_tokens.size();                 // 4
  const int Tt = (int)tokens.size();                        // 8
  const int Tseq = P + Tt;                                  // 12 speech tokens
  const int MEL_LEN1 = (int)prompt_feat.size() / MEL_DIM;   // 8
  const int MEL_T = Tseq * TOKEN_MEL_RATIO;                 // 24 mel frames
  const int OUT_FRAMES = MEL_T - MEL_LEN1;                  // 16
  const int64_t n_xy = (int64_t)MEL_T * MEL_DIM;            // 1920

  if (noise_z.size() != (size_t)n_xy) return false;

  // Clamp token ids at 0 (reference: token.clamp(min=0)).
  std::vector<int32_t> tok_idx(Tseq);
  for (int i = 0; i < P; i++)  tok_idx[i] = prompt_tokens[i] < 0 ? 0 : prompt_tokens[i];
  for (int i = 0; i < Tt; i++) tok_idx[P + i] = tokens[i] < 0 ? 0 : tokens[i];

  // ---- Frontend graph (run once): spk / token_embed / prelookahead / mu / cond ----
  ggml_init_params fp = {.mem_size = 64 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = false};
  ggml_context* fctx = ggml_init(fp);
  if (!fctx) return false;

  ggml_tensor* f_spk = ggml_new_tensor_2d(fctx, GGML_TYPE_F32, SPK_EMBED_DIM, 1);
  ggml_set_input(f_spk);
  memcpy(f_spk->data, spk_embedding.data(), sizeof(float) * spk_embedding.size());

  ggml_tensor* f_tok = ggml_new_tensor_1d(fctx, GGML_TYPE_I32, Tseq);
  ggml_set_input(f_tok);
  memcpy(f_tok->data, tok_idx.data(), sizeof(int32_t) * Tseq);

  ggml_tensor* f_pf = ggml_new_tensor_3d(fctx, GGML_TYPE_F32, MEL_DIM, MEL_LEN1, 1);
  ggml_set_input(f_pf);
  memcpy(f_pf->data, prompt_feat.data(), sizeof(float) * prompt_feat.size());

  ggml_tensor* spk = linear(fctx, impl_->w.spk_affine_w, impl_->w.spk_affine_b,
                            l2_normalize(fctx, f_spk));                    // [80, 1]
  ggml_tensor* token_embed = ggml_reshape_3d(fctx,
      ggml_get_rows(fctx, impl_->w.input_embedding, f_tok), MEL_DIM, Tseq, 1);  // [80, Tseq, 1]
  ggml_tensor* pre = build_prelookahead(fctx, impl_->w, token_embed);      // [80, Tseq, 1]
  ggml_tensor* mu = repeat_interleave_2_ne0(fctx,
      ggml_permute(fctx, pre, 1, 0, 2, 3));                                // [MEL_T, 80, 1]
  ggml_tensor* cond = ggml_pad(fctx,
      ggml_permute(fctx, f_pf, 1, 0, 2, 3), MEL_T - MEL_LEN1, 0, 0, 0);    // [MEL_T, 80, 1]

  ggml_cgraph* fg = ggml_new_graph(fctx);
  ggml_build_forward_expand(fg, spk);
  ggml_build_forward_expand(fg, mu);
  ggml_build_forward_expand(fg, cond);
  ggml_backend_graph_compute(impl_->backend, fg);

  std::vector<float> spk_h, mu_h, cond_h;
  read_tensor(spk, spk_h);    // [80]
  read_tensor(mu, mu_h);      // [MEL_T*80] == [t][c]
  read_tensor(cond, cond_h);  // [MEL_T*80]

  if (debug) {
    read_tensor(spk, debug->spk);                // (1, 80)
    read_tensor(token_embed, debug->token_embed);   // (1, 12, 80)
    read_tensor(pre, debug->prelookahead);          // (1, 12, 80)
    read_tensor(mu, debug->mu);                     // (1, 80, 24)
    read_tensor(cond, debug->cond);                 // (1, 80, 24)
  }
  ggml_free(fctx);

  // ---- DiT graph (built once, run 10x across the CFM steps) ----
  const int B = 2;
  ggml_init_params dp = {.mem_size = 512 * 1024 * 1024, .mem_buffer = nullptr, .no_alloc = false};
  ggml_context* dctx = ggml_init(dp);
  if (!dctx) return false;

  DiTGraph g = build_dit(dctx, impl_->w, MEL_T, B);

  // t_span = 1 - cos(linspace(0, 1, 11) * pi/2)
  std::vector<float> t_span(N_TIMESTEPS + 1);
  for (int i = 0; i <= N_TIMESTEPS; i++) {
    float tt = (float)i / (float)N_TIMESTEPS;
    t_span[i] = 1.0f - cosf(tt * 0.5f * PI);
  }

  // x holds the evolving sample in Space A layout [t][c] (== ggml [T,80,B] ne0=T).
  std::vector<float> x((size_t)n_xy);
  for (int c = 0; c < MEL_DIM; c++)
    for (int t = 0; t < MEL_T; t++)
      x[t + c * MEL_T] = noise_z[c * MEL_T + t];

  // Batch-2 fixed inputs: conditional (index 0) | unconditional (index 1 = 0).
  std::vector<float> mu_in((size_t)2 * n_xy, 0.0f);
  std::vector<float> cond_in((size_t)2 * n_xy, 0.0f);
  std::vector<float> spks_in((size_t)2 * MEL_DIM, 0.0f);
  memcpy(mu_in.data(), mu_h.data(), sizeof(float) * n_xy);
  memcpy(cond_in.data(), cond_h.data(), sizeof(float) * n_xy);
  memcpy(spks_in.data(), spk_h.data(), sizeof(float) * MEL_DIM);
  memcpy(g.mu_in->data, mu_in.data(), mu_in.size() * sizeof(float));
  memcpy(g.cond_in->data, cond_in.data(), cond_in.size() * sizeof(float));
  memcpy(g.spks_in->data, spks_in.data(), spks_in.size() * sizeof(float));

  std::vector<float> x_in((size_t)2 * n_xy);
  std::vector<float> t_emb_in((size_t)2 * DIM);
  std::vector<float> dphi((size_t)2 * n_xy);

  float t = t_span[0];
  float dt = t_span[1] - t_span[0];
  const float a = 1.0f + CFG_RATE;  // 1.7
  const float b = CFG_RATE;         // 0.7

  for (int step = 1; step <= N_TIMESTEPS; step++) {
    // x_in = [x; x]
    memcpy(x_in.data(), x.data(), sizeof(float) * n_xy);
    memcpy(x_in.data() + n_xy, x.data(), sizeof(float) * n_xy);
    memcpy(g.x_in->data, x_in.data(), x_in.size() * sizeof(float));

    // t_emb_in = [time_embed(t); time_embed(t)]
    std::vector<float> te = time_embed_host(t, impl_->time_mlp);
    memcpy(t_emb_in.data(), te.data(), sizeof(float) * DIM);
    memcpy(t_emb_in.data() + DIM, te.data(), sizeof(float) * DIM);
    memcpy(g.t_emb_in->data, t_emb_in.data(), t_emb_in.size() * sizeof(float));

    ggml_backend_graph_compute(impl_->backend, g.gf);

    read_tensor(g.dphi, dphi);  // [MEL_T, 80, 2] == [t][c][b]

    // combined = (1+cfg)*dphi_cond - cfg*dphi_uncond ; x += dt * combined
    for (int64_t i = 0; i < n_xy; i++) {
      x[i] += dt * (a * dphi[i] - b * dphi[i + n_xy]);
    }

    if (step == 1 && debug) {
      debug->time_embed = t_emb_in;                    // (2, 1024)
      read_tensor(g.input_proj, debug->input_proj);    // (2, 24, 1024)
      read_tensor(g.conv_pos, debug->conv_pos);        // (2, 24, 1024)
      read_tensor(g.input_embed, debug->input_embed);  // (2, 24, 1024)
      debug->blocks.resize(DEPTH);
      for (int i = 0; i < DEPTH; i++) read_tensor(g.blocks[i], debug->blocks[i]);
      read_tensor(g.norm_out, debug->norm_out);        // (2, 24, 1024)
      debug->dphi = dphi;                              // (2, 80, 24)
    }

    t += dt;
    if (step < N_TIMESTEPS) dt = t_span[step + 1] - t;
  }

  // feat = x[:, :, MEL_LEN1:] -> mel (1, 80, OUT_FRAMES)
  mel.assign((size_t)MEL_DIM * OUT_FRAMES, 0.0f);
  for (int c = 0; c < MEL_DIM; c++)
    for (int t = 0; t < OUT_FRAMES; t++)
      mel[c * OUT_FRAMES + t] = x[(MEL_LEN1 + t) + c * MEL_T];

  if (debug) debug->feat = mel;

  ggml_free(dctx);
  return true;
}

}  // namespace larynx::flow
