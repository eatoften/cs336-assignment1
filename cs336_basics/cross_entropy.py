import torch

def cross_entropy(inputs,targets):
    #inputs:(..., vocab_size)
    #targets:(...)
    shifted_logits = inputs - torch.amax(inputs,dim=-1,keepdim=True)
    exp_shifted_logits = torch.exp(shifted_logits)
    log_denominator = torch.log(torch.sum(exp_shifted_logits,dim=-1))
    target_indices = torch.unsqueeze(targets, dim=-1)
    correct_logits = torch.squeeze(torch.gather(shifted_logits,dim=-1, index=target_indices),dim=-1)
    loss = torch.mean(log_denominator - correct_logits)
    return loss
    




