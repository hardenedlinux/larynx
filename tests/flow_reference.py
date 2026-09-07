#!/usr/bin/env python3
"""Self-contained PyTorch numerical reference for the Flow decoder.

This is the *ground truth* for tests/verify_flow.py. It does not import the
cosyvoice package (whose deps target a missing Python 3.10 on this box);
instead it inlines the verbatim reference modules that make up
``CausalMaskedDiffWithDiT`` + its DiT estimator:

  - x_transformers: ``apply_rotary_pos_emb`` / ``rotate_half`` / ``RotaryEmbedding``
  - cosyvoice/utils/mask.py: ``make_pad_mask`` (+ the non-streaming
    ``add_optional_chunk_mask`` result, which is just ``mask.bool()`` here)
  - cosyvoice/flow/DiT/modules.py: SinusPositionEmbedding,
    CausalConvPositionEmbedding, AdaLayerNormZero, AdaLayerNormZero_Final,
    FeedForward, Attention, AttnProcessor, DiTBlock, TimestepEmbedding
  - cosyvoice/flow/DiT/dit.py: InputEmbedding, DiT
  - cosyvoice/transformer/upsample_encoder.py: PreLookaheadLayer
  - cosyvoice/flow/flow_matching.py: ConditionalCFM / CausalConditionalCFM
  - cosyvoice/flow/flow.py: CausalMaskedDiffWithDiT

The math is identical to the source; only the ONNX-trace-specific mask
workaround (which is a provable no-op for the all-True full-context mask) is
omitted. It loads ``flow.pt``'s state_dict, runs ``inference()`` (non-streaming,
``finalize=True``) on small deterministic inputs, and dumps per-stage tensors to
``flow_ref.npz`` so the C++ GGML implementation can be compared stage-by-stage.

Only torch + numpy are required (both in .venv).
"""

import os
import math
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Constants for the validation case (must match tests/verify_flow.py / flow_dump.cpp)
# ---------------------------------------------------------------------------
P = 4          # prompt tokens
T = 8          # main (decoding) tokens
MEL_LEN1 = 8   # prompt_feat frames (= P * token_mel_ratio)
VOCAB = 6561
INPUT_SEED = 1234   # seed for the *inputs* (tokens / prompt_feat / spk embedding)
                    # NB: the CFM noise is a fixed buffer seeded with 0, not this.

# ---------------------------------------------------------------------------
# x_transformers rotary (verbatim)
# ---------------------------------------------------------------------------
def rotate_half(x):
    x = x.reshape(*x.shape[:-1], x.shape[-1] // 2, 2)
    x1, x2 = x[..., 0], x[..., 1]
    x = torch.stack((-x2, x1), dim=-1)
    return x.reshape(*x.shape[:-2], x.shape[-2] * x.shape[-1])


def apply_rotary_pos_emb(t, freqs, scale=1):
    rot_dim, seq_len, orig_dtype = freqs.shape[-1], t.shape[-2], t.dtype
    freqs = freqs[:, -seq_len:, :]
    if torch.is_tensor(scale):
        scale = scale[:, -seq_len:, :]
    if t.ndim == 4 and freqs.ndim == 3:
        freqs = freqs.unsqueeze(1)
        if torch.is_tensor(scale):
            scale = scale.unsqueeze(1)
    # partial rotary (GPT-J): rotate only the first `rot_dim` channels.
    t_rot, t_unrot = t[..., :rot_dim], t[..., rot_dim:]
    t_rot = (t_rot * freqs.cos() * scale) + (rotate_half(t_rot) * freqs.sin() * scale)
    return torch.cat((t_rot, t_unrot), dim=-1).type(orig_dtype)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, use_xpos=False, scale_base=512, interpolation_factor=1.0,
                 base=10000, base_rescale_factor=1.0):
        super().__init__()
        base *= base_rescale_factor ** (dim / (dim - 2))
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.interpolation_factor = interpolation_factor
        if not use_xpos:
            self.register_buffer('scale', None)
            return
        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)
        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward_from_seq_len(self, seq_len):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        return self.forward(t)

    def forward(self, t, offset=0):
        if t.ndim == 1:
            t = t.unsqueeze(0)
        freqs = torch.einsum('b i, j -> b i j', t.type_as(self.inv_freq), self.inv_freq) / self.interpolation_factor
        freqs = torch.stack((freqs, freqs), dim=-1)
        freqs = freqs.reshape(*freqs.shape[:-2], freqs.shape[-2] * freqs.shape[-1])
        if self.scale is None:
            return freqs, 1.0
        max_pos = t.max() + 1
        power = (t - (max_pos // 2)) / self.scale_base
        scale = self.scale ** power.unsqueeze(-1)
        scale = torch.stack((scale, scale), dim=-1)
        scale = scale.reshape(*scale.shape[:-2], scale.shape[-2] * scale.shape[-1])
        return freqs, scale


# ---------------------------------------------------------------------------
# cosyvoice/utils/mask.py
# ---------------------------------------------------------------------------
def make_pad_mask(lengths, max_len=0):
    batch_size = lengths.size(0)
    max_len = max_len if max_len > 0 else lengths.max().item()
    seq_range = torch.arange(0, max_len, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_length_expand = lengths.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand
    return mask


# ---------------------------------------------------------------------------
# cosyvoice/flow/DiT/modules.py (verbatim, minus unused joints)
# ---------------------------------------------------------------------------
class SinusPositionEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class CausalConvPositionEmbedding(nn.Module):
    def __init__(self, dim, kernel_size=31, groups=16):
        super().__init__()
        assert kernel_size % 2 != 0
        self.kernel_size = kernel_size
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, dim, kernel_size, groups=groups, padding=0),
            nn.Mish(),
        )

    def forward(self, x, mask=None):
        if mask is not None:
            mask = mask[..., None]
            x = x.masked_fill(~mask, 0.0)
        x = x.permute(0, 2, 1)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv1(x)
        x = F.pad(x, (self.kernel_size - 1, 0, 0, 0))
        x = self.conv2(x)
        out = x.permute(0, 2, 1)
        if mask is not None:
            out = out.masked_fill(~mask, 0.0)
        return out


class AdaLayerNormZero(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 6)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb=None):
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = torch.chunk(emb, 6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class AdaLayerNormZero_Final(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = nn.Linear(dim, dim * 2)
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)

    def forward(self, x, emb):
        emb = self.linear(self.silu(emb))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, dropout=0.0, approximate="none"):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim
        activation = nn.GELU(approximate=approximate)
        project_in = nn.Sequential(nn.Linear(dim, inner_dim), activation)
        self.ff = nn.Sequential(project_in, nn.Dropout(dropout), nn.Linear(inner_dim, dim_out))

    def forward(self, x):
        return self.ff(x)


class AttnProcessor:
    def __call__(self, attn, x, mask=None, rope=None):
        batch_size = x.shape[0]
        query = attn.to_q(x)
        key = attn.to_k(x)
        value = attn.to_v(x)

        if rope is not None:
            freqs, xpos_scale = rope
            q_xpos_scale, k_xpos_scale = (xpos_scale, xpos_scale ** -1.0) if xpos_scale is not None else (1.0, 1.0)
            query = apply_rotary_pos_emb(query, freqs, q_xpos_scale)
            key = apply_rotary_pos_emb(key, freqs, k_xpos_scale)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if mask is not None:
            attn_mask = mask
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(1)
                attn_mask = attn_mask.expand(batch_size, attn.heads, query.shape[-2], key.shape[-2])
        else:
            attn_mask = None

        x = F.scaled_dot_product_attention(query, key, value, attn_mask=attn_mask, dropout_p=0.0, is_causal=False)
        x = x.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        x = x.to(query.dtype)
        x = attn.to_out[0](x)
        x = attn.to_out[1](x)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(-1)
            else:
                mask = mask[:, 0, -1].unsqueeze(-1)
            x = x.masked_fill(~mask, 0.0)
        return x


class Attention(nn.Module):
    def __init__(self, processor, dim, heads=8, dim_head=64, dropout=0.0,
                 context_dim=None, context_pre_only=None):
        super().__init__()
        self.processor = processor
        self.dim = dim
        self.heads = heads
        self.inner_dim = dim_head * heads
        self.dropout = dropout
        self.context_dim = context_dim
        self.context_pre_only = context_pre_only
        self.to_q = nn.Linear(dim, self.inner_dim)
        self.to_k = nn.Linear(dim, self.inner_dim)
        self.to_v = nn.Linear(dim, self.inner_dim)
        self.to_out = nn.ModuleList([])
        self.to_out.append(nn.Linear(self.inner_dim, dim))
        self.to_out.append(nn.Dropout(dropout))

    def forward(self, x, c=None, mask=None, rope=None, c_rope=None):
        if c is not None:
            return self.processor(self, x, c=c, mask=mask, rope=rope, c_rope=c_rope)
        return self.processor(self, x, mask=mask, rope=rope)


class DiTBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, ff_mult=4, dropout=0.1):
        super().__init__()
        self.attn_norm = AdaLayerNormZero(dim)
        self.attn = Attention(processor=AttnProcessor(), dim=dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout, approximate="tanh")

    def forward(self, x, t, mask=None, rope=None):
        norm, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.attn_norm(x, emb=t)
        attn_output = self.attn(x=norm, mask=mask, rope=rope)
        x = x + gate_msa.unsqueeze(1) * attn_output
        ff_norm = self.ff_norm(x) * (1 + scale_mlp[:, None]) + shift_mlp[:, None]
        ff_output = self.ff(ff_norm)
        x = x + gate_mlp.unsqueeze(1) * ff_output
        return x


class TimestepEmbedding(nn.Module):
    def __init__(self, dim, freq_embed_dim=256):
        super().__init__()
        self.time_embed = SinusPositionEmbedding(freq_embed_dim)
        self.time_mlp = nn.Sequential(nn.Linear(freq_embed_dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, timestep):
        time_hidden = self.time_embed(timestep)
        time_hidden = time_hidden.to(timestep.dtype)
        time = self.time_mlp(time_hidden)
        return time


# ---------------------------------------------------------------------------
# cosyvoice/flow/DiT/dit.py (verbatim)
# ---------------------------------------------------------------------------
class InputEmbedding(nn.Module):
    def __init__(self, mel_dim, text_dim, out_dim, spk_dim=None):
        super().__init__()
        spk_dim = 0 if spk_dim is None else spk_dim
        self.spk_dim = spk_dim
        self.proj = nn.Linear(mel_dim * 2 + text_dim + spk_dim, out_dim)
        self.conv_pos_embed = CausalConvPositionEmbedding(dim=out_dim)

    def forward(self, x, cond, text_embed, spks, capture=None):
        to_cat = [x, cond, text_embed]
        if self.spk_dim > 0:
            spks = spks.unsqueeze(1).expand(-1, x.shape[1], -1)  # 'b c -> b t c'
            to_cat.append(spks)
        x = self.proj(torch.cat(to_cat, dim=-1))
        if capture is not None:
            capture['input_proj'] = x.detach()
        pos = self.conv_pos_embed(x)
        if capture is not None:
            capture['conv_pos'] = pos.detach()
        x = pos + x
        return x


class DiT(nn.Module):
    def __init__(self, *, dim, depth=8, heads=8, dim_head=64, dropout=0.1, ff_mult=4,
                 mel_dim=80, mu_dim=None, long_skip_connection=False, spk_dim=None,
                 out_channels=None, static_chunk_size=50, num_decoding_left_chunks=2):
        super().__init__()
        self.time_embed = TimestepEmbedding(dim)
        if mu_dim is None:
            mu_dim = mel_dim
        self.input_embed = InputEmbedding(mel_dim, mu_dim, dim, spk_dim)
        self.rotary_embed = RotaryEmbedding(dim_head)
        self.dim = dim
        self.depth = depth
        self.transformer_blocks = nn.ModuleList(
            [DiTBlock(dim=dim, heads=heads, dim_head=dim_head, ff_mult=ff_mult, dropout=dropout) for _ in range(depth)]
        )
        self.long_skip_connection = nn.Linear(dim * 2, dim, bias=False) if long_skip_connection else None
        self.norm_out = AdaLayerNormZero_Final(dim)
        self.proj_out = nn.Linear(dim, mel_dim)
        self.out_channels = out_channels
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False, capture=None):
        x = x.transpose(1, 2)
        mu = mu.transpose(1, 2)
        cond = cond.transpose(1, 2)
        spks = spks.unsqueeze(dim=1)
        batch, seq_len = x.shape[0], x.shape[1]
        if t.ndim == 0:
            t = t.repeat(batch)

        t = self.time_embed(t)
        if capture is not None:
            capture['time_embed'] = t.detach()

        x = self.input_embed(x, cond, mu, spks.squeeze(1), capture=capture)

        rope = self.rotary_embed.forward_from_seq_len(seq_len)

        if self.long_skip_connection is not None:
            residual = x

        if streaming is True:
            attn_mask = mask.bool().unsqueeze(dim=1)
        else:
            # add_optional_chunk_mask(..., use_dynamic_chunk=False, static_chunk_size=0)
            # returns `mask.bool()` (the ONNX row-repair is a no-op for all-True).
            attn_mask = mask.bool().repeat(1, x.size(1), 1).unsqueeze(dim=1)

        if capture is not None:
            capture['input_embed'] = x.detach()
            capture['blocks'] = []

        for i, block in enumerate(self.transformer_blocks):
            x = block(x, t, mask=attn_mask.bool(), rope=rope)
            if capture is not None:
                capture['blocks'].append(x.detach())

        if self.long_skip_connection is not None:
            x = self.long_skip_connection(torch.cat((x, residual), dim=-1))

        x = self.norm_out(x, t)
        if capture is not None:
            capture['norm_out'] = x.detach()
        output = self.proj_out(x).transpose(1, 2)
        if capture is not None:
            capture['proj_out'] = output.detach()
        return output


# ---------------------------------------------------------------------------
# cosyvoice/transformer/upsample_encoder.py: PreLookaheadLayer (verbatim)
# ---------------------------------------------------------------------------
class PreLookaheadLayer(nn.Module):
    def __init__(self, in_channels, channels, pre_lookahead_len=1):
        super().__init__()
        self.in_channels = in_channels
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(in_channels, channels, kernel_size=pre_lookahead_len + 1, stride=1, padding=0)
        self.conv2 = nn.Conv1d(channels, in_channels, kernel_size=3, stride=1, padding=0)

    def forward(self, inputs, context=torch.zeros(0, 0, 0)):
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        else:
            assert self.training is False
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(torch.concat([outputs, context], dim=2), (0, self.pre_lookahead_len - context.size(2)), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()
        outputs = outputs + inputs
        return outputs


# ---------------------------------------------------------------------------
# cosyvoice/flow/flow_matching.py: CFM (verbatim, non-TRT path)
# ---------------------------------------------------------------------------
def set_all_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class BASECFM(nn.Module):
    def __init__(self, n_feats, cfm_params, n_spks=1, spk_emb_dim=64):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.sigma_min = cfm_params.sigma_min
        self.estimator = None


class ConditionalCFM(BASECFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator=None):
        super().__init__(n_feats=in_channels, cfm_params=cfm_params, n_spks=n_spks, spk_emb_dim=spk_emb_dim)
        self.t_scheduler = cfm_params.t_scheduler
        self.training_cfg_rate = cfm_params.training_cfg_rate
        self.inference_cfg_rate = cfm_params.inference_cfg_rate
        self.estimator = estimator


class CausalConditionalCFM(ConditionalCFM):
    def __init__(self, in_channels, cfm_params, n_spks=1, spk_emb_dim=64, estimator=None):
        super().__init__(in_channels, cfm_params, n_spks, spk_emb_dim, estimator)
        set_all_random_seed(0)
        self.rand_noise = torch.randn([1, 80, 50 * 300])

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps, temperature=1.0, spks=None, cond=None, streaming=False):
        z = self.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device, dtype=mu.dtype)
        if self.t_scheduler == 'cosine':
            t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming), None

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming=False):
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        t = t.unsqueeze(dim=0)
        sol = []
        x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=spks.dtype)
        mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        t_in = torch.zeros([2], device=x.device, dtype=spks.dtype)
        spks_in = torch.zeros([2, 80], device=x.device, dtype=spks.dtype)
        cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=spks.dtype)
        for step in range(1, len(t_span)):
            x_in[:] = x
            mask_in[:] = mask
            mu_in[0] = mu
            t_in[:] = t.unsqueeze(0)
            spks_in[0] = spks
            cond_in[0] = cond
            dphi_dt = self.forward_estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming)
            dphi_dt, cfg_dphi_dt = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
            dphi_dt = ((1.0 + self.inference_cfg_rate) * dphi_dt - self.inference_cfg_rate * cfg_dphi_dt)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1].float()

    def forward_estimator(self, x, mask, mu, t, spks, cond, streaming=False):
        if isinstance(self.estimator, torch.nn.Module):
            return self.estimator(x, mask, mu, t, spks, cond, streaming=streaming)
        raise NotImplementedError


# ---------------------------------------------------------------------------
# cosyvoice/flow/flow.py: CausalMaskedDiffWithDiT (inference path only)
# ---------------------------------------------------------------------------
class CausalMaskedDiffWithDiT(nn.Module):
    def __init__(self, input_size=512, output_size=80, spk_embed_dim=192, output_type="mel",
                 vocab_size=4096, input_frame_rate=50, only_mask_loss=True, token_mel_ratio=2,
                 pre_lookahead_len=3, pre_lookahead_layer=None, decoder=None, decoder_conf=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.decoder_conf = decoder_conf
        self.vocab_size = vocab_size
        self.output_type = output_type
        self.input_frame_rate = input_frame_rate
        self.input_embedding = nn.Embedding(vocab_size, input_size)
        self.spk_embed_affine_layer = torch.nn.Linear(spk_embed_dim, output_size)
        self.pre_lookahead_len = pre_lookahead_len
        self.pre_lookahead_layer = pre_lookahead_layer
        self.decoder = decoder
        self.only_mask_loss = only_mask_loss
        self.token_mel_ratio = token_mel_ratio

    def inference(self, token, token_len, prompt_token, prompt_token_len, prompt_feat,
                  prompt_feat_len, embedding, streaming, finalize):
        assert token.shape[0] == 1
        embedding = F.normalize(embedding, dim=1)
        embedding = self.spk_embed_affine_layer(embedding)

        token, token_len = torch.concat([prompt_token, token], dim=1), prompt_token_len + token_len
        mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
        token = self.input_embedding(torch.clamp(token, min=0)) * mask

        if finalize is True:
            h = self.pre_lookahead_layer(token)
        else:
            h = self.pre_lookahead_layer(token[:, :-self.pre_lookahead_len], context=token[:, -self.pre_lookahead_len:])
        h = h.repeat_interleave(self.token_mel_ratio, dim=1)
        mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

        conds = torch.zeros([1, mel_len1 + mel_len2, self.output_size], device=token.device).to(h.dtype)
        conds[:, :mel_len1] = prompt_feat
        conds = conds.transpose(1, 2)

        mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
        feat, _ = self.decoder(
            mu=h.transpose(1, 2).contiguous(),
            mask=mask.unsqueeze(1),
            spks=embedding,
            cond=conds,
            n_timesteps=10,
            streaming=streaming
        )
        feat = feat[:, :, mel_len1:]
        assert feat.shape[2] == mel_len2
        return feat.float(), None


# ---------------------------------------------------------------------------
# Building the model from cosyvoice3.yaml
# ---------------------------------------------------------------------------
class _CFMParams:
    sigma_min = 1e-6
    solver = 'euler'
    t_scheduler = 'cosine'
    training_cfg_rate = 0.2
    inference_cfg_rate = 0.7


def build_model():
    cfm_params = _CFMParams()
    estimator = DiT(dim=1024, depth=22, heads=16, dim_head=64, ff_mult=2, mel_dim=80,
                    mu_dim=80, spk_dim=80, out_channels=80, static_chunk_size=50,
                    num_decoding_left_chunks=-1)
    decoder = CausalConditionalCFM(in_channels=240, n_spks=1, spk_emb_dim=80,
                                   cfm_params=cfm_params, estimator=estimator)
    pre_lookahead = PreLookaheadLayer(in_channels=80, channels=1024, pre_lookahead_len=3)
    model = CausalMaskedDiffWithDiT(
        input_size=80, output_size=80, spk_embed_dim=192, output_type="mel",
        vocab_size=VOCAB, input_frame_rate=25, token_mel_ratio=2, pre_lookahead_len=3,
        pre_lookahead_layer=pre_lookahead, decoder=decoder,
    )
    return model


# ---------------------------------------------------------------------------
# Manual step-by-step inference (identical math to inference(), but with
# capture hooks). Used to dump per-stage tensors; cross-checked against the
# real inference() at the end.
# ---------------------------------------------------------------------------
def manual_inference(model, prompt_tokens, tokens, prompt_feat, spk_embedding, capture):
    """Replicates CausalMaskedDiffWithDiT.inference(finalize=True)."""
    token_len1, token_len2 = prompt_tokens.shape[1], tokens.shape[1]
    embedding = F.normalize(spk_embedding, dim=1)
    embedding = model.spk_embed_affine_layer(embedding)
    capture['spk'] = embedding.detach()

    token = torch.concat([prompt_tokens, tokens], dim=1)
    token_len = torch.tensor([token_len1 + token_len2])
    mask = (~make_pad_mask(token_len)).unsqueeze(-1).to(embedding)
    token = model.input_embedding(torch.clamp(token, min=0)) * mask
    capture['token_embed'] = token.detach()

    h = model.pre_lookahead_layer(token)
    capture['prelookahead'] = h.detach()

    h = h.repeat_interleave(model.token_mel_ratio, dim=1)
    mel_len1, mel_len2 = prompt_feat.shape[1], h.shape[1] - prompt_feat.shape[1]

    conds = torch.zeros([1, mel_len1 + mel_len2, model.output_size], device=token.device).to(h.dtype)
    conds[:, :mel_len1] = prompt_feat
    conds = conds.transpose(1, 2)
    capture['cond'] = conds.detach()

    mask = (~make_pad_mask(torch.tensor([mel_len1 + mel_len2]))).to(h)
    mu = h.transpose(1, 2).contiguous()
    capture['mu'] = mu.detach()
    capture['mask'] = mask.detach()

    decoder = model.decoder
    z = decoder.rand_noise[:, :, :mu.size(2)].to(mu.device).to(mu.dtype)
    capture['noise_z'] = z.detach()

    t_span = torch.linspace(0, 1, 10 + 1, device=mu.device, dtype=mu.dtype)
    t_span = 1 - torch.cos(t_span * 0.5 * torch.pi)
    capture['t_span'] = t_span.detach()

    # solve_euler, with a DiT capture on step 1.
    x = z
    t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
    t = t.unsqueeze(0)
    x_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=embedding.dtype)
    mask_in = torch.zeros([2, 1, x.size(2)], device=x.device, dtype=embedding.dtype)
    mu_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=embedding.dtype)
    t_in = torch.zeros([2], device=x.device, dtype=embedding.dtype)
    spks_in = torch.zeros([2, 80], device=x.device, dtype=embedding.dtype)
    cond_in = torch.zeros([2, 80, x.size(2)], device=x.device, dtype=embedding.dtype)

    for step in range(1, len(t_span)):
        x_in[:] = x
        mask_in[:] = mask.unsqueeze(1)
        mu_in[0] = mu
        t_in[:] = t.unsqueeze(0)
        spks_in[0] = embedding
        cond_in[0] = conds
        step_capture = capture if step == 1 else None
        dphi_dt = decoder.estimator(x_in, mask_in, mu_in, t_in, spks_in, cond_in, streaming=False, capture=step_capture)
        if step == 1:
            capture['dphi_step1'] = dphi_dt.detach()
        dphi_dt_cond, dphi_dt_uncond = torch.split(dphi_dt, [x.size(0), x.size(0)], dim=0)
        dphi_dt = ((1.0 + decoder.inference_cfg_rate) * dphi_dt_cond - decoder.inference_cfg_rate * dphi_dt_uncond)
        x = x + dt * dphi_dt
        t = t + dt
        if step < len(t_span) - 1:
            dt = t_span[step + 1] - t

    feat = x[:, :, mel_len1:]
    return feat.float().detach()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=os.path.expanduser(
        '~/Project/CosyVoice/pretrained_models/Fun-CosyVoice3-0.5B/flow.pt'))
    ap.add_argument('--out', default=os.path.join(os.path.dirname(__file__), 'flow_ref.npz'))
    args = ap.parse_args()

    torch.set_num_threads(1)
    torch.manual_seed(INPUT_SEED)

    model = build_model()
    model.eval()

    sd = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(sd, strict=True)
    print(f"loaded {len(sd)} tensors from {args.checkpoint}")

    # deterministic inputs (seed INPUT_SEED; the CFM noise is separately seed 0).
    prompt_tokens = torch.randint(0, VOCAB, (1, P))
    tokens = torch.randint(0, VOCAB, (1, T))
    prompt_feat = torch.randn(1, MEL_LEN1, 80)
    spk_embedding = torch.randn(1, 192)

    capture = {}
    feat = manual_inference(model, prompt_tokens, tokens, prompt_feat, spk_embedding, capture)

    # Cross-check: the real inference() must agree with the manual replication.
    prompt_token_len = torch.tensor([P])
    token_len = torch.tensor([T])
    prompt_feat_len = torch.tensor([MEL_LEN1])
    feat_model, _ = model.inference(
        tokens, token_len, prompt_tokens, prompt_token_len,
        prompt_feat, prompt_feat_len, spk_embedding, streaming=False, finalize=True,
    )
    delta = (feat - feat_model).abs().max().item()
    print(f"manual vs model.inference() max abs diff = {delta:.3e}")
    assert delta < 1e-5, "manual replication diverged from model.inference()!"

    # Assemble npz.
    out = {
        # inputs (C++ reads these)
        'prompt_tokens': prompt_tokens.numpy(),
        'tokens': tokens.numpy(),
        'prompt_feat': prompt_feat.numpy(),
        'spk_embedding': spk_embedding.numpy(),
        'noise_z': capture['noise_z'].numpy(),
        # pre-DiT stages
        'spk': capture['spk'].numpy(),
        'token_embed': capture['token_embed'].numpy(),
        'prelookahead': capture['prelookahead'].numpy(),
        'mu': capture['mu'].numpy(),
        'cond': capture['cond'].numpy(),
        't_span': capture['t_span'].numpy(),
        # DiT internals at step 1 (batch = 2N = 2 rows)
        'time_embed_step1': capture['time_embed'].numpy(),
        'dit_input_embed_step1': capture['input_embed'].numpy(),
        'dit_input_proj_step1': capture['input_proj'].numpy(),
        'dit_conv_pos_step1': capture['conv_pos'].numpy(),
        'dit_norm_out_step1': capture['norm_out'].numpy(),
        'dphi_step1': capture['dphi_step1'].numpy(),
        # final output
        'feat': feat.numpy(),
    }
    for i, b in enumerate(capture['blocks']):
        out[f'dit_block{i}'] = b.numpy()

    np.savez(args.out, **out)
    print(f"wrote {args.out}:")
    for k, v in out.items():
        print(f"  {k:24s} {str(v.dtype):8s} {str(v.shape)}")


if __name__ == '__main__':
    main()
