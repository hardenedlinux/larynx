# LLM backbone — implementation notes and numerical validation

The `src/llm/` module implements the **LLM backbone** — the stage that turns a
text + instruct prompt into a speech-token sequence — as a single GGML compute
path (per `docs/ARCHITECTURE.md` §3.3 and `docs/adr/0001`). It reimplements
`cosyvoice/llm/llm.py` `CosyVoice3LM` (a Qwen2-0.5B backbone with the
CosyVoice3 `speech_embedding` / `llm_decoder` heads), in the non-vLLM
`inference_wrapper` path.

| Component | Reference | Notes |
|---|---|---|
| `larynx::llm::LLM` | `CosyVoice3LM.inference_wrapper` (non-vLLM) | prefill → sample → KV-cache decode |
| Qwen2 backbone | `cosyvoice/llm/llm.py` `Qwen2Encoder` (`forward_one_step`) | 24 layers, GQA 7:1 |
| `Qwen2Tokenizer` | `CosyVoice3Tokenizer.encode()` | byte-level BPE, bit-exact |
| sampling | `cosyvoice/utils/common.py` `ras_sampling` | top_p 0.8 / top_k 25, own RNG |

All compute runs on the GGML backend chosen by `ggml_backend_init_best()` —
CUDA when `LARYNX_ENABLE_CUDA=ON` and a device is present, else CPU; the
verification scripts force `LARYNX_BACKEND=cpu` so both sides are float32 for
the numerical comparison. On CUDA the cuBLAS math mode is pinned to
`CUBLAS_DEFAULT_MATH` (TF32 disabled) by `patches/0001`. Weights stay F32, no
quantization.

---

## 1. Architecture constants

### Qwen2 backbone (`CosyVoice-BlankEN/config.json`)

| param | value |
|---|---|
| hidden_size | 896 |
| intermediate_size (SwiGLU) | 4864 |
| num_attention_heads | 14 |
| num_key_value_heads | 2 (GQA 7:1) |
| head_dim | 64 (`hidden_size / heads`) |
| num_hidden_layers | 24 |
| rms_norm_eps | 1e-6 |
| rope_theta | 1e6 (NEOX, `rope_full`) |
| hidden_act | silu (SwiGLU: `silu(gate)·up`, no separate gate bias) |
| vocab_size | 151936 |
| tie_word_embeddings | true (`embed_tokens` ≡ `lm_head`) |
| max_position_embeddings / sliding_window | 32768 / 32768 (`use_sliding_window=false`) |

### CosyVoice3LM speech-token heads (`cosyvoice/llm/llm.py` `CosyVoice3LM`)

`speech_token_size = 6561` (from `cosyvoice3.yaml`). The heads use the
`speech_token_size + 200 = 6761` vocabulary:

| member | value | role |
|---|---|---|
| `sos` | `speech_token_size + 0` = **6561** | prefill start marker (also a stop id) |
| `eos_token` | `speech_token_size + 1` = **6562** | end-of-speech marker |
| `task_id` | `speech_token_size + 2` = **6563** | task control token |
| `fill_token` | `speech_token_size + 3` = **6564** | LM target filler |
| `stop_token_ids` | `[6561 .. 6760]` (200 ids) | any of these ends decoding |
| `speech_embedding` | `Embedding(6761, 896)` | row lookup feeds decode |
| `llm_decoder` | `Linear(896, 6761, bias=False)` | the output head |

The Qwen2 text `lm_head` is **not** used — the speech token comes from
`llm_decoder`, not the text head. The text `embed_tokens` (151936×896) is used
only to build the prefill `lm_input` (see §2); the C++ side does **not** load
`embed_tokens` (a frontend/text-input gap, see §5).

---

## 2. Reference semantics

### Prefill input (`lm_input`)

`CosyVoice3LM.inference` builds a single `(1, L, 896)` sequence:

```
lm_input = cat([ speech_embedding[sos],          # 1 token  (sos = 6561)
                 embed_tokens(cat([prompt, text])),  # prompt_text + tts_text tokens
                 speech_embedding[task_id] ],     # 1 token  (task_id = 6563)
               dim=1)
```

`prompt` is the INSTRUCT string, `text` the TTS text. `min_len`/`max_len`:

```
min_len = int(text_len * 2)     # text_len = tts text token count
max_len = int(text_len * 20)
```

### Autoregressive loop (`inference_wrapper`, non-vLLM)

```python
for i in range(max_len):
    y_pred, cache = llm.forward_one_step(lm_input, masks=tril(L,L), cache=cache)
    logp = llm_decoder(y_pred[:, -1]).log_softmax(dim=-1)          # (6761,)
    top_ids = sampling_ids(logp, out_tokens, sampling, ignore_eos=(i < min_len))
    if top_ids in stop_token_ids:            # stop_token_ids = [6561..6760]
        break
    out_tokens.append(top_ids)
    lm_input = speech_embedding.weight[top_ids].reshape(1, 1, -1)
```

`forward_one_step` is the full Qwen2 forward with a causal `tril` mask; on
decode the cache (KV) is carried so the single new token attends to all prior
positions. `min_len`/`max_len` bound the loop; `ignore_eos` gates the `sos`
slot (see pitfall #1).

### `sampling_ids` and `ras_sampling`

```python
def sampling_ids(weighted_scores, decoded_tokens, sampling, ignore_eos=True):
    if ignore_eos is True:
        weighted_scores[self.speech_token_size] = -float('inf')   # masks 6561
    return self.sampling(weighted_scores, decoded_tokens, sampling)

def ras_sampling(weighted_scores, decoded_tokens, sampling,
                 top_p=0.8, top_k=25, win_size=10, tau_r=0.1):
    top_ids = nucleus_sampling(weighted_scores, top_p, top_k)
    rep_num = (torch.tensor(decoded_tokens[-win_size:]) == top_ids).sum().item()
    if rep_num >= win_size * tau_r:            # 10 * 0.1 = 1 repeat
        weighted_scores[top_ids] = -float('inf')
        top_ids = random_sampling(weighted_scores, decoded_tokens, sampling)
    return top_ids

def nucleus_sampling(weighted_scores, top_p=0.8, top_k=25):
    sorted_value, sorted_idx = weighted_scores.softmax(dim=0).sort(descending=True, stable=True)
    cum_prob, prob, indices = 0.0, [], []
    for i in range(len(sorted_idx)):
        if cum_prob < top_p and len(prob) < top_k:
            cum_prob += sorted_value[i]; prob.append(sorted_value[i]); indices.append(sorted_idx[i])
        else: break
    return indices[torch.tensor(prob).multinomial(1, replacement=True)].item()

def random_sampling(weighted_scores, decoded_tokens, sampling):
    return weighted_scores.softmax(dim=0).multinomial(1, replacement=True).item()
```

`sampling` config = `ras_sampling` (confirmed from `cosyvoice3.yaml`, not
assumed). The C++ `ras_sample` (`src/llm/sampling.cpp`) reproduces the exact
nucleus candidate set and the repetition fallback; the multinomial draw uses a
`std::mt19937` seeded per step, so the sampled **sequence** differs from a torch
run but is drawn from the same distribution.

---

## 3. Must-replicate pitfalls

These are the subtle points that a reimplementation gets wrong. Each was either
caught by the numerical verification or read directly from the CosyVoice source.

1. **`ignore_eos` masks the `sos` slot (6561), not `eos_token` (6562).**
   `sampling_ids` masks `weighted_scores[self.speech_token_size]` = index 6561.
   In `CosyVoice3LM`, index 6561 is `sos = speech_token_size + 0`, while
   `eos_token = speech_token_size + 1` = 6562. So the flag named "ignore eos"
   actually suppresses the *first* control slot, one *below* the token it's
   named after. Worse, `stop_token_ids` has 200 entries, so masking only 6561
   means the model can still stop early during `i < min_len` via 6562..6760 —
   `min_len` is a **soft** floor, not a hard one. (Empirically the model stops
   at 6562, matching the PyTorch reference's stop token exactly.)

2. **`q/k/v` projections have bias; `o_proj` does not.** Qwen2's
   `q_proj/k_proj/v_proj` are `bias=True`, `o_proj` is `bias=False`. The weight
   struct (`src/llm/internal.h`) carries `q_b/k_b/v_b` and no `o_b`.

3. **The `sampling` int parameter (25) is dead code.** `ras_sampling`/`random_sampling`
   accept `sampling` but never read it; `nucleus_sampling` doesn't take it at
   all. The only randomness is `torch.multinomial`. It only matters in the vLLM
   parallel path, which is not implemented.

4. **The output head is `llm_decoder`, not `lm_head`.** Speech tokens come from
   `llm_decoder` (`Linear(896, 6761, bias=False)`); the Qwen2 `lm_head` is tied
   to `embed_tokens` and unused. Applying the text head instead produces a
   text-vocab (151936) distribution that is silently wrong in shape.

5. **`sos`/`task_id` are embedding rows 6561/6563, not 0/1.** The base
   `TransformerLM`/`Qwen2LM` classes use `sos=0, task_id=1` (rows of a tiny
   `Embedding(2)`); `CosyVoice3LM` *overrides* them to
   `speech_token_size + {0,2}` = 6561/6563 (rows of `speech_embedding`). Using
   the base-class values builds a prefill from the wrong embedding.

6. **GPT-2 regex `\s+(?!\S)` negative lookahead.** The Rust tokenizer's split
   pattern is

   ```
   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|
    ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
   ```

   `\s+(?!\S)` (branch 6) is the subtle one: `\s+` is greedy and the
   `(?!\S)` lookahead forces it to **back off one character** when the run is
   followed by non-whitespace, and to match the **whole run** only at
   end-of-string. A naive `\s+`-equivalent handling mis-tokenizes trailing and
   multiple whitespace (`"a  "` must split as `["a", "  "]`, and `"a  b"` as
   `["a", " ", " ", "b"]`). The fix (`src/llm/tokenizer.cpp` `gpt2_match`,
   branch 6/7) is explicit:

   ```cpp
   if (w == end) return w - pos;          // run hits EOF: lookahead sees EOF
   if (w - pos >= 2) return w - pos - 1;  // followed by non-ws: greedy back-off
   return 1;                              // single ws before non-ws: \s+ branch
   ```

7. **`text_normalize` runs before the tokenizer, and does not touch the special
   markers.** Official order is `text_normalize` (WeText/ttsfrd + cleaning)
   **then** `tokenizer.encode`. The Pronunciation-Inpainting ASCII-bracket
   tokens (`[j][ǐ]`), `<strong>` stress markers, and `<|endofprompt|>` all
   survive `text_normalize` verbatim: `remove_bracket` strips only *fullwidth*
   `（）【】` + backtick + `——`, not ASCII `[]` or `<...>`, and `<|…|>` triggers an
   explicit skip (`if '<|' in text and '|>' in text: text_frontend = False`).
   So **no special-token bypass is needed before text_normalize** at CLI time;
   the real gap is that the C++ text_normalize itself is not yet implemented
   (the frontend is a shell). The C++ tokenizer already handles these markers
   bit-exactly (Checkpoint 2).

8. **GGML layout is feature-major.** A PyTorch `(1, L, 896)` tensor is stored
   flat as `[t*D + d]`, byte-identical to GGML's `[ne0=D, ne1=L, ne2=1]` order
   `[d + t*D]`. The KV cache is stored flat as `[HEAD_DIM, KV_HEADS, T]` (d
   fastest, then kv head, then seq), and the rope is the full NEOX rotary
   applied to the `[HEAD_DIM, KV_HEADS, T]` layout *before* GQA repeat.

---

## 4. GGML op mapping

**Native (used directly):** `ggml_mul_mat` (Linear), `ggml_rms_norm`, `ggml_silu`
(SwiGLU gate), `ggml_add`, `ggml_reshape_4d`, `ggml_concat`, `ggml_repeat`
(GQA repeat via `repeat_kv`), `ggml_soft_max_ext` (attention), `ggml_rope_ext`
(NEOX rotary).

**Assembled from primitives** (`src/llm/ops.cpp`):

- `rms_norm` — `x / sqrt(mean(x²)+eps) · w` on `[D, T, B]`.
- `linear` — `x @ Wᵀ + b`; `W = [in, out]`, `b = [out]` (nullable).
- `rope_full` — full NEOX rotary on `[heads·head_dim, T, B]` reshaped to
  `[head_dim, heads, T, B]`, positions from a `[T]` i32 tensor; `rotate_pairs`
  matches `apply_rotary_pos_emb`.
- `repeat_kv` — GQA `[d, kv, T, B] → [d, kv·n_rep, T, B]` (consecutive,
  matching HF `repeat_kv`).
- `attention` — scaled-dot-product with a causal mask `[T, T, 1, 1]` (0/−inf).

The prefill builds one graph over the full `L`; each decode step rebuilds a
graph with the cached K/V as constants concatenated with the new token's K/V.
`no_alloc=true` + `ggml_backend_alloc_ctx_tensors` + `TensorInit.apply()` puts
the tensors on the backend buffer (device memory on CUDA).

---

## 5. Numerical validation (Checkpoints 1–6)

Both sides float32 (C++ forced to CPU via `LARYNX_BACKEND=cpu`). `max`/`mean`
are absolute error; `rel` is `max_abs / max|ref|` (scale-normalized).

| checkpoint | what it verifies | result |
|---|---|---|
| 1 — architecture | Qwen2 config + CosyVoice3LM constants read from source | confirmed (this doc §1) |
| 2 — tokenizer | `Qwen2Tokenizer.encode` vs `CosyVoice3Tokenizer.encode` over a 14-case corpus (special tokens, contractions, NFC, mixed CJK/ASCII, overlapping `[i]/[in]/[ing]`) | **bit-exact** (0 id mismatches) |
| 3 — prefill | `prefill(lm_input, L)` vs `forward_one_step` on L=36: 24 hidden states + final_norm + logits | worst `h23` rel **3.4e-6**; final_norm rel 7.7e-7; logits rel 2.6e-6 |
| 4 — decode single-step | prefill + one `decode(token)`: decode logits, final_norm, and 48 KV-cache stages (24 k + 24 v) | decode_logits rel **2.7e-6**, final_norm rel 4.2e-6, cache worst 6.1e-6 |
| 5 — decode sequence | full autoregressive loop with a *captured* token trajectory (80 steps) injected — proves KV-cache append, absolute rope positions, causal mask over the whole trajectory | worst step rel **1.795e-6**, argmax 0/80 mismatches |
| 6a — sampling policy | `log_softmax` + `ignore_eos` mask + nucleus candidate set vs torch (RNG excluded) | weighted_scores rel **2.2e-7**, candidate sets identical (0 mismatches) |
| 6b — autonomous generation | C++ runs the *whole* loop on its own (own RNG, own KV cache, own stop) for 5 texts × 2 seeds | 10/10 normal stops, stop id = 6562 (∈ [6561,6760]), 0 truncations, no stop-at-min_len anomaly; lengths ≈ 5× text_len |

**Checkpoint 5 strategy (why the RNG is excluded):** the reference seeds torch
(`manual_seed(0)`) and captures its token trajectory; the C++ injects that
trajectory verbatim and compares only the deterministic per-step logits. This
proves the KV-cache mechanics are bit-correct over the whole trajectory without
conflating the (expected) `mt19937` ≠ torch RNG difference, which would
otherwise be impossible to compare exactly. Checkpoint 6b then exercises the
C++'s own RNG end-to-end to prove the autonomous loop terminates correctly.

The reference `stop_token` in the Checkpoint-5 capture is **6562**, and the
autonomous C++ runs also stop at **6562** — the model's natural done-signal
(`ignore_eos` masks 6561, so the stop concentrates on 6562). Token *counts*
differ from the reference (e.g. seed 0: reference 80 vs C++ 98) solely because
of the RNG stream; the stop behaviour is identical.

To reproduce: build, then run the reference scripts under the CosyVoice
python3.10 env (`tests/{tokenizer_reference,llm_reference,llm_decode_reference,
llm_decode_seq_reference,llm_sample_reference,generate_reference}.py`) and the
`.venv/bin/python tests/verify_*.py` comparators (or `ctest -R llm`).

---

## 6. Deferred (per the task scope)

- **`embed_tokens` (text embedding) is not loaded** (`src/llm/gguf.cpp`), and
  the frontend `text_normalize` (WeText) is a shell (`src/frontend/frontend.cpp`).
  Consequently the C++ cannot yet build `lm_input` from raw text — the reference
  builds it. This is the concrete gap to close when wiring Flow/CLI, *not* any
  special-token bypass (pitfall #7).
- **vLLM path** (`CosyVoice3LM.inference_wrapper` vLLM branch, `sampling` int,
  parallel sampling) — not implemented.
- **CLI main flow / Flow / HiFT wiring** — separate later phase.

## 7. Files

- `include/larynx/llm/llm.h` — `LLM` (`load`/`prefill`/`decode`/`generate`),
  `LLMDebug`, `GenerationResult`.
- `src/llm/llm.cpp` — prefill graph, decode step, KV-cache management, `generate`.
- `src/llm/ops.cpp` — `rms_norm`, `linear`, `rope_full`, `repeat_kv`, `attention`.
- `src/llm/gguf.cpp` — `load_weights` (per-layer q/k/v/o/gate/up/down/in_norm/post_norm,
  `llm_decoder`, `final_norm`, `speech_embedding`).
- `src/llm/tokenizer.cpp` — byte-level BPE + NFC + special-token trie + GPT-2 regex.
- `src/llm/sampling.cpp` — `log_softmax`, `softmax`, `nucleus_candidates`, `ras_sample`.
- `src/llm/internal.h` — constants + weight structs.
- `tests/llm_*_reference.py` / `tests/llm_*_dump.cpp` / `tests/verify_llm*.py` —
  the reference / dump / comparison harness per checkpoint.
