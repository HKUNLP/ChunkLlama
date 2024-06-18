# -*- coding:utf-8 -*-
from typing import List, Optional, Tuple, Union, Dict

from torch import nn
import math

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv, LLAMA_INPUTS_DOCSTRING
import torch
import transformers
from transformers.utils import add_start_docstrings_to_model_forward
from flash_attn.flash_attn_interface import flash_attn_with_kvcache, flash_attn_func
from transformers.cache_utils import Cache, DynamicCache
import gc
import math

from transformers.modeling_outputs import CausalLMOutputWithPast
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from typing import Optional, Union

import torch

import flash_attn
import flash_attn_2_cuda as flash_attn_cuda


def new_flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    block_table: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    rotary_interleaved=True,
    alibi_slopes=None,
    num_splits=0,
):
    """
    If k and v are not None, k_cache and v_cache will be updated *inplace* with the new values from
    k and v. This is useful for incremental decoding: you can pass in the cached keys/values from
    the previous step, and update them with the new keys/values from the current step, and do
    attention with the updated cache, all in 1 kernel.

    If you pass in k / v, you must make sure that the cache is large enough to hold the new values.
    For example, the KV cache could be pre-allocated with the max sequence length, and you can use
    cache_seqlens to keep track of the current sequence lengths of each sequence in the batch.

    Also apply rotary embedding if rotary_cos and rotary_sin are passed in. The key @k will be
    rotated by rotary_cos and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If causal or local (i.e., window_size != (-1, -1)), the query @q will be rotated by rotary_cos
    and rotary_sin at indices cache_seqlens, cache_seqlens + 1, etc.
    If not causal and not local, the query @q will be rotated by rotary_cos and rotary_sin at
    indices cache_seqlens only (i.e. we consider all tokens in @q to be at position cache_seqlens).

    See tests/test_flash_attn.py::test_flash_attn_kvcache for examples of how to use this function.

    Supports multi-query and grouped-query attention (MQA/GQA) by passing in KV with fewer heads
    than Q. Note that the number of heads in Q must be divisible by the number of heads in KV.
    For example, if Q has 6 heads and K, V have 2 heads, head 0, 1, 2 of Q will attention to head
    0 of K, V, and head 3, 4, 5 of Q will attention to head 1 of K, V.

    If causal=True, the causal mask is aligned to the bottom right corner of the attention matrix.
    For example, if seqlen_q = 2 and seqlen_k = 5, the causal mask (1 = keep, 0 = masked out) is:
        1 1 1 1 0
        1 1 1 1 1
    If seqlen_q = 5 and seqlen_k = 2, the causal mask is:
        0 0
        0 0
        0 0
        1 0
        1 1
    If the row of the mask is all zero, the output will be zero.

    If window_size != (-1, -1), implements sliding window local attention. Query at position i
    will only attend to keys between
    [i + seqlen_k - seqlen_q - window_size[0], i + seqlen_k - seqlen_q + window_size[1]] inclusive.

    Note: Does not support backward pass.

    Arguments:
        q: (batch_size, seqlen, nheads, headdim)
        k_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
            page_block_size must be a multiple of 256.
        v_cache: (batch_size_cache, seqlen_cache, nheads_k, headdim) if there's no block_table,
            or (num_blocks, page_block_size, nheads_k, headdim) if there's a block_table (i.e. paged KV cache)
        k [optional]: (batch_size, seqlen_new, nheads_k, headdim). If not None, we concatenate
            k with k_cache, starting at the indices specified by cache_seqlens.
        v [optional]: (batch_size, seqlen_new, nheads_k, headdim). Similar to k.
        rotary_cos [optional]: (seqlen_ro, rotary_dim / 2). If not None, we apply rotary embedding
            to k and q. Only applicable if k and v are passed in. rotary_dim must be divisible by 16.
        rotary_sin [optional]: (seqlen_ro, rotary_dim / 2). Similar to rotary_cos.
        cache_seqlens: int, or (batch_size,), dtype torch.int32. The sequence lengths of the
            KV cache.
        block_table [optional]: (batch_size, max_num_blocks_per_seq), dtype torch.int32.
        cache_batch_idx: (batch_size,), dtype torch.int32. The indices used to index into the KV cache.
            If None, we assume that the batch indices are [0, 1, 2, ..., batch_size - 1].
            If the indices are not distinct, and k and v are provided, the values updated in the cache
                 might come from any of the duplicate indices.
        softmax_scale: float. The scaling of QK^T before applying softmax.
            Default to 1 / sqrt(headdim).
        causal: bool. Whether to apply causal attention mask (e.g., for auto-regressive modeling).
        window_size: (left, right). If not (-1, -1), implements sliding window local attention.
        rotary_interleaved: bool. Only applicable if rotary_cos and rotary_sin are passed in.
            If True, rotary embedding will combine dimensions 0 & 1, 2 & 3, etc. If False,
            rotary embedding will combine dimensions 0 & rotary_dim / 2, 1 & rotary_dim / 2 + 1
            (i.e. GPT-NeoX style).
        alibi_slopes: (nheads,) or (batch_size, nheads), fp32. A bias of
            (-alibi_slope * |i + seqlen_k - seqlen_q - j|)
            is added to the attention score of query i and key j.
        num_splits: int. If > 1, split the key/value into this many chunks along the sequence.
           If num_splits == 1, we don't split the key/value. If num_splits == 0, we use a heuristic
           to automatically determine the number of splits.
           Don't change this unless you know what you are doing.

    Return:
        out: (batch_size, seqlen, nheads, headdim).
    """
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    maybe_contiguous = lambda x: x.contiguous() if x is not None and x.stride(-1) != 1 else x
    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)
    cache_batch_idx = maybe_contiguous(cache_batch_idx)
    block_table = maybe_contiguous(block_table)
    out, softmax_lse = flash_attn_cuda.fwd_kvcache(
        q,
        k_cache,
        v_cache,
        k,
        v,
        cache_seqlens,
        rotary_cos,
        rotary_sin,
        cache_batch_idx,
        block_table,
        alibi_slopes,
        None,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        rotary_interleaved,
        num_splits,
    )
    return out, softmax_lse

# flash_attn.flash_attn_interface.flash_attn_with_kvcache = new_flash_attn_with_kvcache


def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class ChunkLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, scaling_factor=1.0, device=None):
        super().__init__()

        self.max_seq_len = max_position_embeddings
        self.dim = dim
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len,
            device=device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        # employing yarn will lead to better performance but results reported in our paper did not use yarn.
        scale = seq_len / self.max_position_embeddings
        mscale = get_mscale(scale)

        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        chunk_len = chunk_size - local_window
        q_t = torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
        qc_t = (torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) + chunk_len).clamp(
            max=chunk_size) / self.scaling_factor
        k_t = (torch.arange(seq_len + MAX_NEW_TOKENS, device=device,
                            dtype=self.inv_freq.dtype) % chunk_len) / self.scaling_factor

        q_freqs = torch.outer(q_t, self.inv_freq)  # seq_len x dim/2
        qc_freqs = torch.outer(qc_t, self.inv_freq)
        k_freqs = torch.outer(k_t, self.inv_freq)  # seq_len x dim/2

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)  # seq_len x dim
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1)
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)  # seq_len x dim
        self.register_buffer("q_cos_cached", q_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("q_sin_cached", q_emb.sin().to(dtype) * mscale, persistent=False)
        self.register_buffer("qc_cos_cached", qc_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("qc_sin_cached", qc_emb.sin().to(dtype) * mscale, persistent=False)
        self.register_buffer("k_cos_cached", k_emb.cos().to(dtype) * mscale, persistent=False)
        self.register_buffer("k_sin_cached", k_emb.sin().to(dtype) * mscale, persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # no token will exceed chunk_size
        # chunk1_q,
        if seq_len > self.max_seq_len:
            self._set_cos_sin_cache(seq_len=seq_len, device=self.inv_freq.device, dtype=torch.float32)
            self.max_seq_len = seq_len
        return (
            self.q_cos_cached[:seq_len].to(dtype=x.dtype),
            self.q_sin_cached[:seq_len].to(dtype=x.dtype),
            self.qc_cos_cached[:seq_len].to(dtype=x.dtype),
            self.qc_sin_cached[:seq_len].to(dtype=x.dtype),
            self.k_cos_cached[:seq_len].to(dtype=x.dtype),
            self.k_sin_cached[:seq_len].to(dtype=x.dtype),
        )


def apply_rotary_pos_emb(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb


def _merge_single_chunk(softmax_lse, attn_outputs):
    softmax_lse = softmax_lse.to(torch.float32)
    max_softmax_sum = torch.max(softmax_lse, dim=0).values
    stable_softmax_sum = softmax_lse - max_softmax_sum.unsqueeze(0)
    lse_s = torch.exp(stable_softmax_sum).detach()
    lse_sum = torch.sum(lse_s, dim=0)
    lse_s /= lse_sum
    lse_s = lse_s.to(torch.bfloat16)
    attn_outputs *= lse_s.unsqueeze(-1)
    return attn_outputs.sum(dim=0)


def merge_attn_outputs(flash_results, decoding=False):
    if decoding:
        attn_outputs = torch.stack([flash_attn_output[0] for flash_attn_output in flash_results])
        softmax_lse = torch.stack([flash_attn_output[1] for flash_attn_output in flash_results])
        return _merge_single_chunk(softmax_lse, attn_outputs)
    attn_outputs_all = [flash_results[0][0]]
    flash_results = flash_results[1:]
    for flash_per_chunk in flash_results:
        attn_outputs = torch.stack([flash_attn_output[0] for flash_attn_output in flash_per_chunk])
        softmax_lse = torch.stack([flash_attn_output[1] for flash_attn_output in flash_per_chunk])
        attn_outputs_all.append(_merge_single_chunk(softmax_lse, attn_outputs))
    return torch.cat(attn_outputs_all, dim=2)


def do_flash_attn(query_states, key_states, value_states, causal=True, layer_idx=0):

    output, softmax_lse, _ = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2),
                                             value_states.transpose(1, 2), causal=causal, return_attn_probs=True)

    return output.transpose(1, 2), softmax_lse


def do_flash_decoding(query_states, key_states, value_states, k_cache, v_cache, cache_seqlens, intra=False):
    
    if not intra:
        temp = torch.zeros_like(k_cache[:, 0:1, :, :])
        k_cache = torch.cat([k_cache, temp], dim=1)
        v_cache = torch.cat([v_cache, temp], dim=1)
        output, softmax_lse = new_flash_attn_with_kvcache(query_states.transpose(1, 2), k_cache, v_cache, cache_seqlens=cache_seqlens)
    else:
        output, softmax_lse = new_flash_attn_with_kvcache(query_states.transpose(1, 2), k_cache, v_cache,
                                                    key_states.transpose(1, 2),
                                                    value_states.transpose(1, 2), cache_seqlens=cache_seqlens)
    return output.transpose(1, 2), softmax_lse





def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value=None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    bsz, q_len, _ = hidden_states.size()
    chunk_len = chunk_size - local_window


    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    kv_seq_len += past_key_value["cache_seqlens"].item()
    past_key_value["cache_seqlens"] += key_states.shape[-2]
   
    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len
    # covert to b x head x len x h
    # need to chunk query states
    q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    key_states = apply_rotary_pos_emb(key_states, k_cos, k_sin, position_ids)
    position_ids = position_ids % chunk_len

    # update kv cache
    key_cache = past_key_value[0][:, :, 0, :, :]
    value_cache = past_key_value[0][:, :, 1, :, :]


    if not has_kv_cache:
        key_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = key_states.transpose(1, 2)
        value_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = value_states.transpose(1, 2)

    flash_results = []

    if not has_kv_cache:
        q_states_intra = apply_rotary_pos_emb(query_states[:, :, :chunk_len, :], q_cos, q_sin,
                                              position_ids[:, :chunk_len])
        k_states_prev = key_states[:, :, :chunk_len, :]
        v_states_prev = value_states[:, :, :chunk_len, :]
        flash_results.append(do_flash_attn(q_states_intra, k_states_prev, v_states_prev))
        remain_len = kv_seq_len - chunk_len

        while remain_len > 0:
            flash_per_chunk = []
            begin = kv_seq_len - remain_len
            curr_chunk_len = min(chunk_len, remain_len)
            end = begin + curr_chunk_len
            q_states_intra = apply_rotary_pos_emb(query_states[:, :, begin:end, :], q_cos, q_sin,
                                                  position_ids[:, begin:end])

            k_states_intra = key_states[:, :, begin:end, :]
            v_states_intra = value_states[:, :, begin:end, :]
            flash_per_chunk.append(do_flash_attn(q_states_intra, k_states_intra, v_states_intra))

            q_states_succ = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                 position_ids[:, begin:end])
            flash_per_chunk.append(do_flash_attn(q_states_succ, k_states_prev, v_states_prev, False, self.layer_idx))

            if begin - (k_states_prev.size(-2)) > 0:
                prev_len = k_states_prev.size(-2)
                q_states_inter = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                      position_ids[:, chunk_len - 1][:, None].repeat(1, curr_chunk_len))
                k_states_inter = key_states[:, :, :begin - prev_len, :]
                v_states_inter = value_states[:, :, :begin - prev_len, :]
                flash_per_chunk.append(
                    do_flash_attn(q_states_inter, k_states_inter, v_states_inter, False, self.layer_idx + 1))

            flash_results.append(flash_per_chunk)
            k_states_prev = k_states_intra
            v_states_prev = v_states_intra
            remain_len = remain_len - chunk_len

        attn_output = merge_attn_outputs(flash_results)
    else:
        flash_results = []
        chunk_num_curr = (kv_seq_len - 1) // chunk_len
        q_states_intra = apply_rotary_pos_emb(query_states, q_cos, q_sin, position_ids)
        k_cache_intra = key_cache[:, chunk_len * chunk_num_curr:, :, :]
        v_cache_intra = value_cache[:, chunk_len * chunk_num_curr:, :, :]
        cache_seqlens = kv_seq_len - 1 - chunk_len * chunk_num_curr
        flash_results.append(do_flash_decoding(q_states_intra, key_states, value_states, k_cache_intra, v_cache_intra,
                                               cache_seqlens=cache_seqlens, intra=True))

        if chunk_num_curr >= 1:
            q_states_succ = apply_rotary_pos_emb(query_states, qc_cos, qc_sin, position_ids)

            k_cache_succ = key_cache[:, chunk_len * (chunk_num_curr - 1):chunk_len * chunk_num_curr, :, :]
            v_cache_succ = value_cache[:, chunk_len * (chunk_num_curr - 1):chunk_len * chunk_num_curr, :, :]
            cache_seqlens = v_cache_succ.size(1)

            flash_results.append(
                do_flash_decoding(q_states_succ, None, None, k_cache_succ, v_cache_succ,
                                  cache_seqlens=cache_seqlens, intra=False))

        if chunk_num_curr >= 2:
            q_states_inter = apply_rotary_pos_emb(query_states, qc_cos, qc_sin,
                                                  torch.tensor([[chunk_len - 1]], device=query_states.device))
            k_cache_inter = key_cache[:, :chunk_len * (chunk_num_curr - 1), :, :]
            v_cache_inter = value_cache[:, :chunk_len * (chunk_num_curr - 1), :, :]
            cache_seqlens = v_cache_inter.size(1)
            flash_results.append(
                do_flash_decoding(q_states_inter, None, None, k_cache_inter, v_cache_inter,
                                  cache_seqlens=cache_seqlens, intra=False))

        attn_output = merge_attn_outputs(flash_results, True)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value


def allocate_inference_cache(
        max_batch_size,
        max_seqlen,
        nheads,
        headdim,
        layers,
        dtype=torch.float16,
):
    assert dtype in [torch.float16, torch.bfloat16, torch.float32]
    kv_cache_shape = (max_batch_size, max_seqlen, 2, nheads, headdim)
    # print(max_batch_size)
    # input()
    allc_kv_cache = {i: {0:torch.empty(kv_cache_shape, device=layer.self_attn.k_proj.weight.device, dtype=dtype), "cache_seqlens":torch.tensor([0],  device=layer.self_attn.k_proj.weight.device).long()} for
                     i, layer in enumerate(layers)}

    return allc_kv_cache



@add_start_docstrings_to_model_forward(LLAMA_INPUTS_DOCSTRING)
def LlamaModel_forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache

    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # retrieve input_ids and inputs_embeds
    if input_ids is not None and inputs_embeds is not None:
        raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
    elif input_ids is not None:
        batch_size, seq_length = input_ids.shape[:2]
    elif inputs_embeds is not None:
        batch_size, seq_length = inputs_embeds.shape[:2]
    else:
        raise ValueError("You have to specify either input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training:
        if use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

    past_key_values_length = 0

    if past_key_values:
        input_ids = input_ids[:, -1].unsqueeze(-1)
        position_ids = position_ids[:, -1].unsqueeze(-1) if position_ids is not None else None

    if use_cache and past_key_values is None:
        num_kv_heads = self.config.num_key_value_heads
        num_attention_heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // num_attention_heads
        past_key_values = allocate_inference_cache(
            batch_size,
            MAX_CACHE_LEN,
            num_kv_heads,
            head_dim,
            self.layers,
            dtype=self.dtype,
        )

    if position_ids is None:
        device = input_ids.device if input_ids is not None else inputs_embeds.device
        position_ids = torch.arange(
            past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=device
        )
        position_ids = position_ids.unsqueeze(0)

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    if self._use_flash_attention_2:
        # 2d mask is passed through the layers
        attention_mask = attention_mask if (attention_mask is not None and 0 in attention_mask) else None
    elif self._use_sdpa and not output_attentions:
        # output_attentions=True can not be supported when using SDPA, and we fall back on
        # the manual implementation that requires a 4D causal mask in all cases.
        attention_mask = _prepare_4d_causal_attention_mask_for_sdpa(
            attention_mask,
            (batch_size, seq_length),
            inputs_embeds,
            past_key_values_length,
        )
    else:
        # 4d mask is passed through the layers
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), inputs_embeds, past_key_values_length
        )

    # embed positions
    hidden_states = inputs_embeds
    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None
    for i, decoder_layer in enumerate(self.layers):
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                attention_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
            )
        else:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values[i],
                output_attentions=output_attentions,
                use_cache=use_cache,
            )

        hidden_states = layer_outputs[0]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    if not return_dict:
        return tuple(v for v in [hidden_states, past_key_values, all_hidden_states, all_self_attns] if v is not None)

    return BaseModelOutputWithPast(
        last_hidden_state=hidden_states,
        past_key_values=past_key_values,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
    )


def causal_forward(self,
                   input_ids: torch.LongTensor = None,
                   attention_mask: Optional[torch.Tensor] = None,
                   position_ids: Optional[torch.LongTensor] = None,
                   past_key_values: Optional[List[torch.FloatTensor]] = None,
                   inputs_embeds: Optional[torch.FloatTensor] = None,
                   labels: Optional[torch.LongTensor] = None,
                   use_cache: Optional[bool] = None,
                   output_attentions: Optional[bool] = None,
                   output_hidden_states: Optional[bool] = None,
                   return_dict: Optional[bool] = None,
                   ) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
    )

    hidden_states = outputs[0]
    full_logits_length = 32000

    if hidden_states.shape[-2] < full_logits_length:
        logits = self.lm_head(hidden_states)
        logits = logits.float()
        loss = None

        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)

            loss = loss_fct(shift_logits, shift_labels)
    else:
        res = 0
        div_len = full_logits_length // 2
        if labels is None:
            # only produce the last logits
            logits = self.lm_head(hidden_states[..., -1:, :])
            logits = logits.float()
            # logits = logits.expand(-1, hidden_states.shape[-2], -1)
            loss = None
        else:
            # calculate loss by chunk
            shift_hidden_states = hidden_states[..., :-1, :]
            shift_labels = labels[..., 1:].contiguous()

            for i in range(0, shift_hidden_states.shape[-2], div_len):
                st = i
                ed = min(i + div_len, shift_hidden_states.shape[-2])
                logits = self.lm_head(shift_hidden_states[..., st:ed, :])
                logits = logits.float()

                shift_logits = logits.contiguous()
                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)

                res = res + loss_fct(shift_logits, shift_labels[st:ed]) * (ed - st)
            loss = res / (hidden_states.shape[-2] - 1)
            logits = None

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
                       
chunk_size = None
local_window = None
linear_factor = None
MAX_NEW_TOKENS = 512
MAX_CACHE_LEN = 32 * 1024 + MAX_NEW_TOKENS # default max_len


def replace_with_chunkllama(pretraining_length=4096, local_window_size=None, max_prompt_length=None):
    global chunk_size
    global local_window
    global MAX_CACHE_LEN
    chunk_size = pretraining_length * 3 // 4
    if max_prompt_length:
        MAX_CACHE_LEN = max_prompt_length + MAX_NEW_TOKENS
    local_window = local_window_size if local_window_size else pretraining_length // 16
    transformers.models.llama.modeling_llama.LlamaForCausalLM.forward = causal_forward
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward
    transformers.models.llama.modeling_llama.LlamaRotaryEmbedding = ChunkLlamaRotaryEmbedding
    transformers.models.llama.modeling_llama.LlamaLinearScalingRotaryEmbedding = ChunkLlamaRotaryEmbedding
