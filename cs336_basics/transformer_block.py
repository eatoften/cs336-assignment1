import torch
from torch import nn

from cs336_basics.rmsnorm import RMSNorm
from cs336_basics.multihead_attention import CausalMultiHeadSelfAttention
from cs336_basics.swiglu import SwiGLU


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, theta, max_seq_len, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.ln1 = RMSNorm(d_model,eps=1e-5, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(d_model, num_heads,theta=theta, max_seq_len=max_seq_len,device=device,dtype=dtype)
        self.ln2 = RMSNorm(d_model, eps=1e-5,device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
    
    def forward(self, x:torch.Tensor):
        normalized_x = self.ln1(x)
        attn_output = self.attn(normalized_x)
        z = x + attn_output

        normalized_z = self.ln2(z)
        ffn_output = self.ffn(normalized_z)
        result = z + ffn_output
        return result

        