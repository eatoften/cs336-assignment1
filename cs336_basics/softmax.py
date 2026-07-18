import torch
from torch import nn

def softmax(x:torch.Tensor,dim):
    m = torch.amax(x,dim,keepdim = True)
    shifted = x - m
    numerator = torch.exp(shifted)
    denominator = torch.sum(numerator,dim=dim,keepdim=True)
    output = numerator/denominator
    return output