import torch
from torch import nn
from cs336_basics.linear import Linear

class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff, device = None, dtype = None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model,d_ff,device=device,dtype=dtype)
        self.w2 = Linear(d_ff,d_model,device=device,dtype=dtype)
        self.w3 = Linear(d_model,d_ff,device=device,dtype=dtype)

    def forward(self, x:torch.Tensor):
        a = self.w1(x)
        gate = a * torch.sigmoid(a)
        content = self.w3(x)
        hidden = gate * content
        output = self.w2(hidden)
        return output