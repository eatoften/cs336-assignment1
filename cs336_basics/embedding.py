import torch
from torch import nn


class Embedding(nn.Module):
    def __init__(self,num_embeddings,embedding_dim,device=None,dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        raw_weight = torch.empty(num_embeddings,embedding_dim,device = device)
        self.weight = nn.Parameter(raw_weight)
        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0,
            std = 1,
            a = -3,
            b = 3
        )

    def forward(self,token_ids:torch.Tensor) -> torch.Tensor:
        output = self.weight[token_ids]
        return output
