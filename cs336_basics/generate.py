import argparse
import json
from pathlib import Path

import torch

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.decoding import generate_token_ids


def load_model_from_run(run_dir: Path, device):
    config_path = run_dir / "config.json"
    checkpoint_path = run_dir / "checkpoint.pt"

    with open(config_path, encoding="utf-8") as config_file:
        config = json.load(config_file)

    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        device=device,
        dtype=torch.float32,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model state"]
    )

    completed_steps = checkpoint["iteration"]

    model.eval()

    return model, config, completed_steps


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "run_dir",
        type=Path,
    )

    parser.add_argument(
        "--vocab-path",
        type=Path,
        default=Path("./data/tinystories_train_vocab.pkl"),
    )

    parser.add_argument(
        "--merges-path",
        type=Path,
        default=Path("./data/tinystories_train_merges.pkl"),
    )

    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model, config, completed_steps = load_model_from_run(
        args.run_dir,
        device,
    )

    tokenizer = Tokenizer.from_files(
        vocab_filepath=args.vocab_path,
        merges_filepath=args.merges_path,
        special_tokens=["<|endoftext|>"],
    )

    eos_id = tokenizer.special_to_id[
        "<|endoftext|>"
    ]

    prompt_ids = tokenizer.encode(args.prompt)

    generated_ids = generate_token_ids(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=eos_id,
        context_length=config["context_length"],
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    generated_text = tokenizer.decode(
        generated_ids
    )

    print("\n--- Generation settings ---")
    print("prompt tokens:", len(prompt_ids))
    print("generated tokens:", len(generated_ids) - len(prompt_ids))
    print("eos id:", eos_id)

    print("\n--- Generated text ---")
    print(generated_text)

    print("device:", next(model.parameters()).device)
    print("completed steps:", completed_steps)
    print("context length:", config["context_length"])
    print("vocab size:", config["vocab_size"])