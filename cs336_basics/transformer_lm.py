import torch
from torch import nn

from cs336_basics.embedding import Embedding
from cs336_basics.transformer_block import TransformerBlock
from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.linear import Linear

class TransformerLM(nn.Module):
    def __init__(
            self,
            vocab_size,
            context_length,
            d_model,
            num_layers,
            num_heads,
            d_ff,
            rope_theta,
            device=None,
            dtype=None
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model,device=device,dtype=dtype)
        blocks = []
        for i in range(num_layers):
            layer = TransformerBlock(
                d_model,
                num_heads,
                d_ff,
                rope_theta,
                context_length,
                device=device,
                dtype=dtype
            )
            blocks.append(layer)
        self.layers = nn.ModuleList(blocks)
        self.ln_final = RMSNorm(d_model,device=device,dtype=dtype)
        self.lm_head = Linear(d_model,vocab_size,device=device,dtype=dtype)

    def forward(self,in_indices, token_positions=None, past_key_values=None, use_cache=False):
        if use_cache and past_key_values is None:
            past_key_values = [None for _ in self.layers]
        elif past_key_values is not None:
            if len(past_key_values) != len(self.layers):
                raise ValueError("Invalid past_key_values")
        
        present_key_values = []

        hidden_states = self.token_embeddings(in_indices)
        for layer_index, layer in enumerate(self.layers):
            if use_cache is False:
                hidden_states = layer(hidden_states,
                                      token_positions=token_positions)
            else:
                layer_past_key_value = past_key_values[layer_index]
                hidden_states, present_key_value = layer(hidden_states,
                                                          token_positions=token_positions,
                                                          past_key_value=layer_past_key_value,
                                                          use_cache=True)
                present_key_values.append(present_key_value)

        normalized = self.ln_final(hidden_states)
        logits = self.lm_head(normalized)
        if use_cache is False:
            return logits
        else:
            return logits, present_key_values
            