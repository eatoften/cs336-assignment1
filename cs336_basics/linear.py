import torch 
import math
from torch import nn

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        raw_weight = torch.empty(out_features,in_features,device=device,dtype=dtype)
        self.weight = nn.Parameter(raw_weight)
        sigma = math.sqrt(2/(in_features+out_features))
        torch.nn.init.trunc_normal_(
            self.weight,
            0,
            sigma,
            -3*sigma,
            3*sigma
            )
        

    def forward(self,x:torch.Tensor) -> torch.Tensor:
        # x: (..., in_features)
        # weight: (out_features, in_features)
        # output:(..., out_features)
        y = torch.einsum("...i,oi -> ...o", x, self.weight)
        return y
