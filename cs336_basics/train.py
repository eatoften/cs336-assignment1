import torch
import numpy as np
import time
from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.data_loader import data_loader
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import lr_schedule
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

if __name__ == "__main__":
    vocab_size = 10000
    context_length = 16
    d_model = 64
    num_layers = 2
    num_heads = 4
    d_ff = 128
    rope_theta = 10000
    batch_size = 4
    max_iters = 10
    max_learning_rate = 0.001
    min_learning_rate = 0.0001
    warmup_iters = 2
    cosine_cycle_iters = 10
    max_l2_norm = 1.0
    device = "cpu"
    dtype = torch.float32
    eval_iters = 5

    train_data = np.load(
        "./data/tinystories_train_tokens.npy",
        mmap_mode="r",
    )
    valid_data = np.load(
        "./data/tinystories_valid_tokens.npy",
        mmap_mode="r",
    )

    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
        rope_theta=rope_theta,
        device=device,
        dtype=dtype
    )

    optimizer = AdamW(
        model.parameters(),
        lr = max_learning_rate,
        betas= (0.9,0.95),
        eps= 1e-8,
        weight_decay=0.01
    )

    start_time = time.perf_counter()

    model.train()

    for iteration in range(max_iters):
        current_lr = lr_schedule(
                                    iteration,
                                    max_learning_rate=max_learning_rate,
                                    min_learning_rate=min_learning_rate,
                                    warmup_iters=warmup_iters,
                                    cosine_cycle_iters=cosine_cycle_iters
                    )
        for group in optimizer.param_groups:
            group["lr"] = current_lr

        input_ids, target_ids = data_loader(train_data,
                                batch_size=batch_size,
                                context_length=context_length,
                                device=device
        )

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)

        loss.backward()

        gradient_clipping(
            model.parameters(),
            max_l2_norm,
        )

        optimizer.step()
        print("iteration:", iteration+1)
        print("current_lr:", current_lr)
        print("training loss:", loss.item())

    model.eval()
    validation_losses = []
    with torch.no_grad():
        for i in range(eval_iters):
            valid_input_ids, valid_target_ids = data_loader(valid_data,
                            batch_size=batch_size,
                            context_length=context_length,
                            device=device

            )
            valid_logits = model(valid_input_ids)
            valid_loss = cross_entropy(valid_logits,valid_target_ids)
            validation_losses.append(valid_loss.item())

    mean_validation_loss = sum(validation_losses) / len(validation_losses)
    model.train()
    elapsed_time = time.perf_counter() - start_time
    print("elapsed_time:", elapsed_time)
    print("validation loss:", mean_validation_loss)

    save_checkpoint(
    model,
    optimizer,
    max_iters,
    "./smoke_checkpoint.pt",
    )

    print("checkpoint saved")