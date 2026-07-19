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

    def forward(self,in_indices):
        hidden_states = self.token_embeddings(in_indices)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        normalized = self.ln_final(hidden_states)
        logits = self.lm_head(normalized)
        return logits
        