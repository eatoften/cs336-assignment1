import torch
import math
from torch import nn
from cs336_basics.softmax import softmax

def scaled_dot_product_attention(Q,K,V,mask=None):
    #Q:(...,queries,d_k)
    #k:(...,keys,d_k)
    #V:(...,keys,d_v)
    #mask:(...,queries,keys)
    #scores:(...,queries,keys)
    scores = torch.einsum("...qd,...kd->...qk",Q,K)
    d_k = Q.shape[-1]
    scaled_scores = scores/math.sqrt(d_k)
    if mask is not None:
        scaled_scores = scaled_scores.masked_fill(~mask,-math.inf)
    attention_weights= softmax(scaled_scores,dim=-1)
    #attention_weights: (...,queries,keys)
    output = torch.einsum("...qk,...kv->...qv",attention_weights,V)
    return output
    
    
