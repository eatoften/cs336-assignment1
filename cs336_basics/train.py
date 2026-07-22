import torch
import numpy as np
import time
from datetime import datetime

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.adamw import AdamW
from cs336_basics.data_loader import data_loader
from cs336_basics.cross_entropy import cross_entropy
from cs336_basics.gradient_clipping import gradient_clipping
from cs336_basics.lr_schedule import lr_schedule
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.experiment_logger import ExperimentLogger


if __name__ == "__main__":
    vocab_size = 10000
    context_length = 256
    d_model = 512
    num_layers = 4
    num_heads = 16
    d_ff = 1344
    rope_theta = 10000
    batch_size = 8
    max_iters = 500
    max_learning_rate = 0.0003
    min_learning_rate = 0.00003
    warmup_iters = 50
    cosine_cycle_iters = max_iters - 1
    max_l2_norm = 1.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    eval_iters = 20
    log_interval = 25
    eval_interval = 100

    betas = (0.9, 0.95)
    adam_eps = 1e-8
    weight_decay = 0.01

    seed = 42
    np.random.seed(seed=seed)
    torch.manual_seed(seed=seed)

    config = {
        "experiment": "tinystories_lr_3e-4",
        "seed": seed,
        "vocab_size": vocab_size,
        "context_length": context_length,
        "d_model": d_model,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "d_ff": d_ff,
        "rope_theta": rope_theta,
        "batch_size": batch_size,
        "max_iters": max_iters,
        "max_learning_rate": max_learning_rate,
        "min_learning_rate": min_learning_rate,
        "warmup_iters": warmup_iters,
        "cosine_cycle_iters": cosine_cycle_iters,
        "max_l2_norm": max_l2_norm,
        "eval_iters": eval_iters,
        "betas": betas,
        "adam_eps": adam_eps,
        "weight_decay": weight_decay,
        "device": str(device),
        "dtype": str(dtype),
        "log_interval": log_interval,
        "eval_interval": eval_interval,
    }

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
        lr=max_learning_rate,
        betas=betas,
        eps=adam_eps,
        weight_decay=weight_decay,
    )


    run_name = datetime.now().strftime(
        "tinystories_lr_3e-4_%Y%m%d_%H%M%S"
    )

    logger = ExperimentLogger(
        run_name=run_name,
        config=config,
    )

    start_time = time.perf_counter()

    model.train()

    # overfit small batch test
    # input_ids, target_ids = data_loader(train_data,
    #                             batch_size=batch_size,
    #                             context_length=context_length,
    #                             device=device
    # )

    print("selected device:", device)
    print("model device:", next(model.parameters()).device)


    for iteration in range(max_iters):
        current_lr = lr_schedule(
                                    iteration,
                                    max_learning_rate=max_learning_rate,
                                    min_learning_rate=min_learning_rate,
                                    warmup_iters=warmup_iters,
                                    cosine_cycle_iters=cosine_cycle_iters
                    )
        # current_lr = max_learning_rate

        for group in optimizer.param_groups:
            group["lr"] = current_lr

        input_ids, target_ids = data_loader(train_data,
                                batch_size=batch_size,
                                context_length=context_length,
                                device=device
        )

        if iteration == 0:
            print("batch device:", input_ids.device)

        optimizer.zero_grad()
        logits = model(input_ids)
        loss = cross_entropy(logits, target_ids)

        loss.backward()

        gradient_clipping(
            model.parameters(),
            max_l2_norm,
        )

        optimizer.step()

        step = iteration + 1
        tokens_processed = step * batch_size * context_length

        mean_validation_loss = None

        should_evaluate = (
            step % eval_interval == 0
            or step == max_iters
        )

        if should_evaluate:
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

        should_log = (
            step == 1
            or step % log_interval == 0
            or step == max_iters
        )

        if should_log:
            logger.log(
                step=step,
                tokens_processed=tokens_processed,
                train_loss=loss.item(),
                validation_loss=mean_validation_loss,
                learning_rate=current_lr,
            )

            print(
                "step:", step,
                "train loss:", loss.item(),
                "validation loss:", mean_validation_loss,
                "lr:", current_lr,
            )


    elapsed_time = time.perf_counter() - start_time
    print("elapsed_time:", elapsed_time)
    print("validation loss:", mean_validation_loss)

    # checkpoint_path = logger.run_dir / "checkpoint.pt"

    # save_checkpoint(
    #     model,
    #     optimizer,
    #     max_iters,
    #     checkpoint_path,
    # )

    # print("checkpoint saved")

    logger.close()