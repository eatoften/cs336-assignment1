import torch
import numpy as np

def data_loader(dataset, batch_size, context_length, device):
    starting_indices= np.random.randint(
        low=0,
        high=len(dataset) - context_length,
        size=batch_size
    )
    input_seq = []
    target_seq = []
    for start in starting_indices:
        input_slice = dataset[start: start + context_length]
        target_slice = dataset[start+1 : start + context_length + 1]
        input_seq.append(input_slice)
        target_seq.append(target_slice)
    
    input_batch = np.stack(input_seq, axis=0)
    target_batch = np.stack(target_seq, axis=0)

    input_tensor = torch.tensor(
        input_batch,
        dtype=torch.long,
        device=device
    )

    target_tensor = torch.tensor(
        target_batch,
        dtype=torch.long,
        device=device
    )
    return input_tensor, target_tensor