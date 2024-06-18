# -*- coding:utf-8 -*-
from typing import List, Optional, Tuple, Union, Dict

from torch import nn
import math

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv, LLAMA_INPUTS_DOCSTRING, apply_rotary_pos_emb
import torch
import transformers
from transformers.utils import add_start_docstrings_to_model_forward

from flash_attn import flash_attn_func, flash_attn_with_kvcache


def do_flash_attn(query_states, key_states, value_states, causal=True):
    output, softmax_lse, _ = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2),
                                             value_states.transpose(1, 2), causal=causal, return_attn_probs=True)

    return output.transpose(1, 2)


def do_flash_decoding(query_states, key_states, value_states, k_cache, v_cache, cache_seqlens):
    output = flash_attn_with_kvcache(query_states.transpose(1, 2), k_cache, v_cache,
                                                  key_states.transpose(1, 2),
                                                  value_states.transpose(1, 2), cache_seqlens=cache_seqlens)
    return output.transpose(1, 2)




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

    cos, sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # update kv cache
    key_cache = past_key_value[0][:, :, 0, :, :]
    value_cache = past_key_value[0][:, :, 1, :, :]

    if not has_kv_cache:
        key_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = key_states.transpose(1, 2)
        value_cache[:, kv_seq_len - key_states.shape[-2]:kv_seq_len, :, :] = value_states.transpose(1, 2)


    if not has_kv_cache:
        attn_output =  do_flash_attn(query_states, key_states, value_states)

    else:
        cache_seqlens = kv_seq_len-1
        attn_output = do_flash_decoding(query_states, key_states, value_states, key_cache, value_cache,
                          cache_seqlens=cache_seqlens)



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
    allc_kv_cache = {i: {0: torch.empty(kv_cache_shape, device=layer.self_attn.k_proj.weight.device, dtype=dtype),
                         "cache_seqlens": torch.tensor([0], device=layer.self_attn.k_proj.weight.device).long()} for
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



MAX_CACHE_LEN = 32 * 1024
MAX_NEW_TOKENS = 512

def replace_with_flashdecoding(max_prompt_length=None):
    global MAX_CACHE_LEN
    if max_prompt_length is not None:
        MAX_CACHE_LEN = max_prompt_length + MAX_NEW_TOKENS
    transformers.models.llama.modeling_llama.LlamaModel.forward = LlamaModel_forward
    transformers.models.llama.modeling_llama.LlamaAttention.forward = forward
    transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = forward
