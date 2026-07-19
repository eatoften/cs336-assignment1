import torch
from torch import nn
from cs336_basics.linear import Linear
from einops import rearrange
from cs336_basics.attention import scaled_dot_product_attention
from cs336_basics.rope import RotaryPositionalEmbedding

class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model,num_heads,theta=None,max_seq_len=None,device=None,dtype=None):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        if self.d_model % self.num_heads != 0:
            raise ValueError("d_model mod num_heads should be 0")
        self.d_head = self.d_model // self.num_heads
        if theta is None and max_seq_len is None:
            self.rope = None
        elif theta is not None and max_seq_len is not None:
            self.rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len,device=device)
        else:
            raise ValueError("theta and max_seq_len value error")
        self.q_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.k_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.v_proj = Linear(d_model,d_model,device=device,dtype=dtype)
        self.output_proj = Linear(d_model,d_model,device=device,dtype=dtype)

    def forward(self, x:torch.Tensor, token_positions=None):
        #x,Q,K,V: (...,sequence,d_model)
        Q = self.q_proj(x)
        K = self.k_proj(x)
        V = self.v_proj(x)
        Q = rearrange(Q,"... seq (head d_head) -> ... head seq d_head",head = self.num_heads,d_head = self.d_head)
        K = rearrange(K,"... seq (head d_head) -> ... head seq d_head",head = self.num_heads,d_head = self.d_head)
        V = rearrange(V,"... seq (head d_head) -> ... head seq d_head",head = self.num_heads,d_head = self.d_head)
        seq_len = x.shape[-2]

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(0, seq_len,device=x.device)
            token_positions = rearrange(token_positions,"... seq -> ... 1 seq")
            Q = self.rope(Q,token_positions)
            K = self.rope(K,token_positions)

        mask = torch.ones(seq_len,seq_len,dtype=torch.bool,device=x.device)
        mask = torch.tril(mask,diagonal=0)
        result = scaled_dot_product_attention(Q,K,V,mask)
        #result: (..., head, sequence, d_head)
        result = rearrange(result,"... head seq d -> ... seq (head d)")
        result = self.output_proj(result)
        return result





