import torch
from torch import nn

class RMSNorm(nn.Module):
    def __init__(self,d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        raw_weight = torch.empty(d_model, device = device, dtype = dtype)
        self.weight = nn.Parameter(raw_weight)
        torch.nn.init.constant_(self.weight,1)
        
    
    def forward(self,x: torch.Tensor):
        in_dtype = x.dtype
        x = x.to(torch.float32)
        x_square = x*x
        mean_square = torch.mean(x_square,dim = -1,keepdim = True)
        rms = torch.sqrt(mean_square + self.eps)
        normalized = x/rms
        result = normalized * self.weight
        result = result.to(in_dtype)
        return result
        

