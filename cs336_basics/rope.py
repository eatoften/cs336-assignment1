import torch
from torch import nn

class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta:float,d_k:int,max_seq_len:int,device=None):
        super().__init__()
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        if d_k % 2 != 0:
            raise ValueError("d_k should be even")
        index = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        exponent = - index / d_k
        freq = theta ** exponent
        position = torch.arange(0,max_seq_len,1,device=device,dtype=torch.float32)
        angles = torch.outer(position,freq)
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        self.register_buffer("cos_cache", cos_cache, persistent=False)
        self.register_buffer("sin_cache", sin_cache, persistent=False)

    def forward(self, x:torch.Tensor,token_positions:torch.Tensor):
        selected_cos = self.cos_cache[token_positions]
        selected_sin = self.sin_cache[token_positions]
        x_even = x[...,0::2]  #x_even shape = (batch,heads,sequence,d_k/2)
        x_odd = x[...,1::2]
        y_even = x_even*selected_cos - x_odd*selected_sin
        y_odd = x_even*selected_sin + x_odd*selected_cos
        stack = torch.stack((y_even,y_odd),dim=-1)
        result = torch.flatten(stack,start_dim=-2)
        return result