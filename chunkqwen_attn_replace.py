# -*- coding:utf-8 -*-

from typing import List, Optional, Tuple, Union

from torch import nn
import math
from transformers.models.llama.modeling_llama import rotate_half, repeat_kv
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss
import torch
import transformers

from transformers.cache_utils import Cache
from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func


class ChunkLlamaRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=4096, base=10000, scaling_factor=1.0, device=None):
        super().__init__()

        self.max_seq_len = 16384
        self.dim = dim
        self.max_length = None
        self.scaling_factor = scaling_factor
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        
        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=self.max_seq_len,
            device=device, dtype=torch.float32
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        chunk_len = chunk_size - local_window
        q_t = torch.arange(chunk_len, device=device, dtype=self.inv_freq.dtype) / self.scaling_factor
        qc_t = (q_t + chunk_len).clamp(max=chunk_size) / self.scaling_factor
        k_t = (torch.arange(seq_len + MAX_NEW_TOKENS, device=device,
                            dtype=self.inv_freq.dtype) % chunk_len) / self.scaling_factor

        q_freqs = torch.outer(q_t, self.inv_freq)  # seq_len x dim/2
        qc_freqs = torch.outer(qc_t, self.inv_freq)
        k_freqs = torch.outer(k_t, self.inv_freq)  # seq_len x dim/2

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        q_emb = torch.cat((q_freqs, q_freqs), dim=-1)  # seq_len x dim
        qc_emb = torch.cat((qc_freqs, qc_freqs), dim=-1)
        k_emb = torch.cat((k_freqs, k_freqs), dim=-1)  # seq_len x dim
        self.register_buffer("q_cos_cached", q_emb.cos().to(dtype), persistent=False)
        self.register_buffer("q_sin_cached", q_emb.sin().to(dtype), persistent=False)
        self.register_buffer("qc_cos_cached", qc_emb.cos().to(dtype), persistent=False)
        self.register_buffer("qc_sin_cached", qc_emb.sin().to(dtype), persistent=False)
        self.register_buffer("k_cos_cached", k_emb.cos().to(dtype), persistent=False)
        self.register_buffer("k_sin_cached", k_emb.sin().to(dtype), persistent=False)

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


def merge_attn_outputs(flash_results):
    attn_outputs_all = [flash_results[0][0]]
    flash_results = flash_results[1:]
    for flash_per_chunk in flash_results:
        attn_outputs = torch.stack([flash_attn_output[0] for flash_attn_output in flash_per_chunk])
        logits = torch.stack([flash_attn_output[1] for flash_attn_output in flash_per_chunk])
        max_logits = torch.max(logits, dim=0).values  
        stable_logits = logits - max_logits.unsqueeze(0)
        lse_s = torch.exp(stable_logits).detach()
        lse_sum = torch.sum(lse_s, dim=0)
        lse_s /= lse_sum
        attn_outputs *= lse_s.unsqueeze(-1)
        attn_outputs_all.append(attn_outputs.sum(dim=0))
    return torch.cat(attn_outputs_all, dim=2)


def do_flash_attn(query_states, key_states, value_states, causal=True):
    # flash_attention
    output, softmax_lse, _ = flash_attn_func(query_states.transpose(1, 2), key_states.transpose(1, 2),
                                             value_states.transpose(1, 2), causal=causal, return_attn_probs=True)
    return output.transpose(1, 2), softmax_lse


def apply_rotary_pos_emb(x, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    x_emb = (x * cos) + (rotate_half(x) * sin)
    return x_emb


def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Cache] = None,
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

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    kv_seq_len = key_states.shape[-2]
    # during inference
    if past_key_value is not None:
        kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    q_seq_len = query_states.shape[-2]
    has_kv_cache = q_seq_len != kv_seq_len
    # covert to b x head x len x h
    # need to chunk query states
    q_cos, q_sin, qc_cos, qc_sin, k_cos, k_sin = self.rotary_emb(value_states, seq_len=kv_seq_len)
    key_states = apply_rotary_pos_emb(key_states, k_cos, k_sin, position_ids)
    position_ids = position_ids % chunk_len

    if past_key_value is not None:
        key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs=None)

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    flash_results = []
    if not has_kv_cache:
        q_states_intra = apply_rotary_pos_emb(query_states[:, :, :chunk_len, :], q_cos, q_sin,
                                              position_ids[:, :chunk_len])
        k_states_prev = key_states[:, :, :chunk_len, :]
        v_states_prev = value_states[:, :, :chunk_len, :]
        flash_result = do_flash_attn(q_states_intra, k_states_prev, v_states_prev)
        flash_results.append(flash_result)
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
            flash_result = do_flash_attn(q_states_intra, k_states_intra, v_states_intra)
            flash_per_chunk.append(flash_result)

            q_states_succ = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                 position_ids[:, begin:end])
            flash_result = do_flash_attn(q_states_succ, k_states_prev, v_states_prev, False)
            flash_per_chunk.append(flash_result)

            if begin - (k_states_prev.size(-2)) > 0:
                prev_len = k_states_prev.size(-2)
                q_states_inter = apply_rotary_pos_emb(query_states[:, :, begin:end, :], qc_cos, qc_sin,
                                                      position_ids[:, chunk_len - 1][:, None].repeat(1, curr_chunk_len))
                k_states_inter = key_states[:, :, :begin - prev_len, :]
                v_states_inter = value_states[:, :, :begin - prev_len, :]
                flash_result = do_flash_attn(q_states_inter, k_states_inter, v_states_inter, False)
                flash_per_chunk.append(flash_result)

            flash_results.append(flash_per_chunk)
            k_states_prev = k_states_intra
            v_states_prev = v_states_intra
            remain_len = remain_len - chunk_len

        attn_output = merge_attn_outputs(flash_results)
    else:
        chunk_num_curr = (kv_seq_len - 1) // chunk_len
        q_states_intra = apply_rotary_pos_emb(query_states, q_cos, q_sin, position_ids)
        k_states_intra = key_states[:, :, chunk_len * chunk_num_curr:kv_seq_len, :]
        attn_weights = torch.matmul(q_states_intra, k_states_intra.transpose(2, 3)) / math.sqrt(
            self.head_dim)
        attn_scores = [attn_weights]

        if chunk_num_curr >= 1:
            q_states_succ = apply_rotary_pos_emb(query_states, qc_cos, qc_sin, position_ids)

            k_states_succ = key_states[:, :, chunk_len * (chunk_num_curr - 1):chunk_len * chunk_num_curr, :]
            attn_weights = torch.matmul(q_states_succ, k_states_succ.transpose(2, 3)) / math.sqrt(
                self.head_dim)
            attn_scores = [attn_weights] + attn_scores

        if chunk_num_curr >= 2:
            q_states_inter = apply_rotary_pos_emb(query_states, qc_cos, qc_sin,
                                                  torch.tensor([[chunk_len - 1]], device=query_states.device))
            k_states_inter = key_states[:, :, :chunk_len * (chunk_num_curr - 1), :]
            attn_weights = torch.matmul(q_states_inter, k_states_inter.transpose(2, 3)) / math.sqrt(
                self.head_dim)
            attn_scores = [attn_weights] + attn_scores

        attn_weights = torch.cat(attn_scores, dim=-1)
        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)
    return attn_output, None, past_key_value

def qwen_forward(
    self,
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
    *args, 
    **kwargs
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
    >>> from transformers import AutoTokenizer, Qwen2ForCausalLM

    >>> model = Qwen2ForCausalLM.from_pretrained(PATH_TO_CONVERTED_WEIGHTS)
    >>> tokenizer = AutoTokenizer.from_pretrained(PATH_TO_CONVERTED_TOKENIZER)

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
    global full_logits_length

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
        chunk_size = full_logits_length // 2
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

            for i in range(0, shift_hidden_states.shape[-2], chunk_size):
                st = i
                ed = min(i + chunk_size, shift_hidden_states.shape[-2])
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


def replace_with_chunkqwen(pretraining_length=4096, local_window_size=None, full_logits_size=32000):
    global chunk_size
    global local_window
    global full_logits_length
    chunk_size = pretraining_length * 3 // 4
    local_window = local_window_size if local_window_size else pretraining_length // 16
    full_logits_length = full_logits_size
    transformers.models.qwen2.modeling_qwen2.Qwen2FlashAttention2.forward = forward
    transformers.models.qwen2.modeling_qwen2.Qwen2RotaryEmbedding = ChunkLlamaRotaryEmbedding
    transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.forward = qwen_forward

