import inspect
import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn

from transformers.activations import ACT2FN

from transformers.utils.import_utils import is_torch_fx_available

from torch.utils.checkpoint import checkpoint
from functools import partial

from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from einops import rearrange, repeat, reduce, pack, unpack
from einops.layers.torch import Rearrange




def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if q is not None:
        q_embed = (q * cos) + (rotate_half(q) * sin)

    else:
        q_embed = None
    
    if k is not None:
        k_embed = (k * cos) + (rotate_half(k) * sin)
    else:
        k_embed = None
    return q_embed, k_embed    


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, base=10000):
        super().__init__()
        self.dim = dim
        self.base = base
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float() / self.dim))
            
        self.register_buffer("inv_freq", inv_freq, persistent=False)
                
    @torch.no_grad()
    def forward(self, x, position_ids=None):
        if position_ids is None:
            # position_ids: [bsz, seq_len]
            position_ids = torch.arange(x.shape[2], device=x.device, dtype=torch.int64).unsqueeze(0).expand(x.shape[0], -1)
            

        # x: [bs, num_attention_heads, seq_len, head_size]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 since bfloat16 loses precision on long contexts
        # See https://github.com/huggingface/transformers/pull/29285
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)




class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)






class Attention(nn.Module):

    def __init__(self, num_attention_heads, num_key_value_heads, attention_head_size, attention_window_size=None,  seq_length=None, use_positional_embedding=False, rope_base=None):
        super().__init__()

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = attention_head_size
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads 
        self.attention_window_size = attention_window_size
        self.seq_length = seq_length

        self.use_positional_embedding = use_positional_embedding
        self.rope_base = rope_base


        if self.use_positional_embedding:
            self.rotary_emb = RotaryEmbedding(
                dim=self.attention_head_size,
                base=self.rope_base)


    

    def forward(self, query_states, key_states, value_states):
        bsz, q_len, _ = query_states.size()

        query_states = query_states.view(bsz, q_len, self.num_attention_heads, self.attention_head_size).transpose(1, 2).contiguous()

        key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()

        value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.attention_head_size).transpose(1, 2).contiguous()

        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        if self.use_positional_embedding:
            cos, sin = self.rotary_emb(query_states)
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        query_states = query_states.transpose(1, 2)  
        key_states = key_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
        value_states = value_states.transpose(1, 2)  # (batch, slen, num_heads, head_dim)
        
        if self.attention_window_size is not None:
            attn_outputs = flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
                window_size=(self.attention_window_size, self.attention_window_size),
            )
        else:
            attn_outputs = flash_attn_func(
                query_states,
                key_states,
                value_states,
                causal=True,
            )

        attn_outputs = attn_outputs.reshape(bsz, q_len, int(self.num_attention_heads * self.attention_head_size)).contiguous()


        return attn_outputs





class Block(nn.Module):
    def __init__(
        self,
        hidden_size=768,
        num_attention_heads=12,
        num_key_value_heads=4,
        attention_window_size=None,
        seq_length=None,
        use_positional_embedding=False,
        rope_base=None,
        ):
        super().__init__()


        self.hidden_size = hidden_size


        self.intermediate_size = self.hidden_size

        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.attention_head_size = int(self.intermediate_size / self.num_attention_heads)
         

        self.latent_dim = self.intermediate_size 
        self.latent_dim += self.attention_head_size * self.num_key_value_heads * 2 

        self.pre_avg_layernorm = RMSNorm(self.intermediate_size)

        self.in_proj = nn.Linear(self.hidden_size, self.latent_dim, bias=True)
        self.out_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=True)


        self.self_attn = Attention(self.num_attention_heads, self.num_key_value_heads, self.attention_head_size, attention_window_size, seq_length, use_positional_embedding, rope_base)


    def forward(self, hidden_states):

        batch_size, seq_len, hidden_size = hidden_states.shape

        

        hidden_states = self.in_proj(hidden_states).transpose(1, 2)


        query_states, key_states, value_states,hidden_states = hidden_states.tensor_split((self.intermediate_size, self.intermediate_size + self.attention_head_size * self.num_key_value_heads, self.intermediate_size + self.attention_head_size * self.num_key_value_heads*2,), dim=1)


        query_states = query_states.transpose(1,2)
        key_states = key_states.transpose(1,2)
        value_states = value_states.transpose(1,2)


        attn_outputs = self.self_attn(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states
        )


        hidden_states = self.pre_avg_layernorm(attn_outputs)
        contextualized_states = self.out_proj(hidden_states)

        return contextualized_states


