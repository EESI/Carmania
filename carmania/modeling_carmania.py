import math
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from .model import Block, RMSNorm
from transformers.activations import ACT2FN
from typing import Optional, List, Dict, Tuple, Union
from transformers.modeling_outputs import CausalLMOutput
from .configuration_carmania import CarmaniaConfig


class CarmaniaPreTrainedModel(PreTrainedModel):
    base_model_prefix = "carmania"
    supports_gradient_checkpointing = True

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.initializer_range)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=self.config.initializer_range)


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.act_fn = ACT2FN[config.mlp_hidden_act]
        self.ffn_dim = config.intermediate_size
        self.hidden_dim = config.hidden_size

        if config.mlp_hidden_act == "silu":
            self.gate_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)
        self.down_proj = nn.Linear(self.ffn_dim, self.hidden_dim, bias=False)
        self.up_proj = nn.Linear(self.hidden_dim, self.ffn_dim, bias=False)

    def forward(self, x):
        if hasattr(self, "gate_proj"):
            return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        return self.down_proj(self.act_fn(self.up_proj(x)))

class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.block = Block(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            attention_window_size=config.attention_window_size,
            seq_length=config.seq_length,
            use_positional_embedding=config.use_positional_embedding,
            rope_base=config.rope_base,
        )
        self.mlp = MLP(config)
        self.norm1 = RMSNorm(config.hidden_size, eps=1e-6)
        self.norm2 = RMSNorm(config.hidden_size, eps=1e-6)

    def forward(self, x):
        residual = x
        x = self.norm1(x)
        x = self.block(x)
        x = x + residual

        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual
        return x





class CarmaniaModel(CarmaniaPreTrainedModel):
    config_class = CarmaniaConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.token_embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.decoders = nn.ModuleList([Decoder(config) for _ in range(config.num_layers)])
        self.output_layer = nn.Linear(config.hidden_size, config.vocab_size)

        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = True,
        output_hidden_states: Optional[bool] = True,  # <-- add this
    ) -> CausalLMOutput:
        hidden_states = []
        x = self.token_embedding(input_ids)

        for decoder in self.decoders:
            x = decoder(x)
            if output_hidden_states:
                hidden_states.append(x)

        logits = self.output_layer(x)

        return CausalLMOutput(
            logits=logits,
            hidden_states=tuple(hidden_states) if output_hidden_states else None
        )


