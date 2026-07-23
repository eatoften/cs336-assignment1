import argparse
import csv
import gc
import hashlib
import json
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from cs336_basics.decoding import generate_token_ids, generate_token_ids_cached
from cs336_basics.generate import load_model_from_run


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as input_file:
        while chunk := input_file.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def git_metadata() -> tuple[str | None, bool | None]:
    repository_root = Path(__file__).resolve().parents[1]
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repository_root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None, None

    return commit, bool(status.strip())


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_generation(
    generation_function,
    model: torch.nn.Module,
    prompt_ids: list[int],
    context_length: int,
    max_new_tokens: int,
) -> list[int]:
    return generation_function(
        model=model,
        prompt_ids=prompt_ids,
        eos_id=-1,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        temperature=0,
        top_p=1.0,
    )


def time_generation(
    generation_function,
    model: torch.nn.Module,
    prompt_ids: list[int],
    context_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> tuple[float, list[int]]:
    synchronize(device)
    start_time = time.perf_counter()
    generated_ids = run_generation(
        generation_function=generation_function,
        model=model,
        prompt_ids=prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
    )
    synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time

    actual_new_tokens = len(generated_ids) - len(prompt_ids)
    if actual_new_tokens != max_new_tokens:
        raise RuntimeError(
            f"Expected {max_new_tokens} generated tokens, got {actual_new_tokens}"
        )

    return elapsed_seconds, generated_ids


def measure_peak_extra_memory_mib(
    generation_function,
    model: torch.nn.Module,
    prompt_ids: list[int],
    context_length: int,
    max_new_tokens: int,
    device: torch.device,
) -> float | None:
    if device.type != "cuda":
        return None

    gc.collect()
    torch.cuda.empty_cache()
    synchronize(device)
    baseline_bytes = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)

    run_generation(
        generation_function=generation_function,
        model=model,
        prompt_ids=prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
    )
    synchronize(device)

    peak_bytes = torch.cuda.max_memory_allocated(device)
    return max(0, peak_bytes - baseline_bytes) / (1024**2)


def first_mismatch(
    first_ids: list[int],
    second_ids: list[int],
) -> tuple[int, int | None, int | None] | None:
    for index, (first_id, second_id) in enumerate(zip(first_ids, second_ids)):
        if first_id != second_id:
            return index, first_id, second_id

    if len(first_ids) != len(second_ids):
        index = min(len(first_ids), len(second_ids))
        first_id = first_ids[index] if index < len(first_ids) else None
        second_id = second_ids[index] if index < len(second_ids) else None
        return index, first_id, second_id

    return None


def benchmark_prompt_length(
    model: torch.nn.Module,
    prompt_ids: list[int],
    context_length: int,
    max_new_tokens: int,
    warmup_runs: int,
    repeats: int,
    device: torch.device,
) -> tuple[dict, list[dict]]:
    uncached_reference = run_generation(
        generation_function=generate_token_ids,
        model=model,
        prompt_ids=prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
    )
    cached_reference = run_generation(
        generation_function=generate_token_ids_cached,
        model=model,
        prompt_ids=prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
    )

    mismatch = first_mismatch(uncached_reference, cached_reference)
    if mismatch is not None:
        index, uncached_id, cached_id = mismatch
        raise AssertionError(
            "Cached and uncached generation differ at "
            f"position {index}: uncached={uncached_id}, cached={cached_id}"
        )

    for _ in range(warmup_runs):
        run_generation(
            generation_function=generate_token_ids,
            model=model,
            prompt_ids=prompt_ids,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
        run_generation(
            generation_function=generate_token_ids_cached,
            model=model,
            prompt_ids=prompt_ids,
            context_length=context_length,
            max_new_tokens=max_new_tokens,
        )
    synchronize(device)

    timings = {
        "uncached": [],
        "cached": [],
    }
    raw_rows = []

    for repeat_index in range(repeats):
        method_order = (
            ("uncached", generate_token_ids),
            ("cached", generate_token_ids_cached),
        )
        if repeat_index % 2 == 1:
            method_order = tuple(reversed(method_order))

        for method_name, generation_function in method_order:
            elapsed_seconds, generated_ids = time_generation(
                generation_function=generation_function,
                model=model,
                prompt_ids=prompt_ids,
                context_length=context_length,
                max_new_tokens=max_new_tokens,
                device=device,
            )
            if generated_ids != uncached_reference:
                raise AssertionError(
                    f"{method_name} output changed during repeat {repeat_index + 1}"
                )

            timings[method_name].append(elapsed_seconds)
            raw_rows.append(
                {
                    "prompt_tokens": len(prompt_ids),
                    "new_tokens": max_new_tokens,
                    "method": method_name,
                    "repeat": repeat_index + 1,
                    "elapsed_seconds": elapsed_seconds,
                    "tokens_per_second": max_new_tokens / elapsed_seconds,
                    "token_ids_equal": True,
                }
            )

    uncached_median_seconds = statistics.median(timings["uncached"])
    cached_median_seconds = statistics.median(timings["cached"])
    uncached_tokens_per_second = max_new_tokens / uncached_median_seconds
    cached_tokens_per_second = max_new_tokens / cached_median_seconds

    uncached_peak_extra_memory_mib = measure_peak_extra_memory_mib(
        generation_function=generate_token_ids,
        model=model,
        prompt_ids=prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        device=device,
    )
    cached_peak_extra_memory_mib = measure_peak_extra_memory_mib(
        generation_function=generate_token_ids_cached,
        model=model,
        prompt_ids=prompt_ids,
        context_length=context_length,
        max_new_tokens=max_new_tokens,
        device=device,
    )

    summary = {
        "prompt_tokens": len(prompt_ids),
        "new_tokens": max_new_tokens,
        "token_ids_equal": True,
        "uncached_median_seconds": uncached_median_seconds,
        "cached_median_seconds": cached_median_seconds,
        "uncached_tokens_per_second": uncached_tokens_per_second,
        "cached_tokens_per_second": cached_tokens_per_second,
        "speedup": uncached_median_seconds / cached_median_seconds,
        "uncached_peak_extra_memory_mib": uncached_peak_extra_memory_mib,
        "cached_peak_extra_memory_mib": cached_peak_extra_memory_mib,
        "uncached_timings_seconds": timings["uncached"],
        "cached_timings_seconds": timings["cached"],
    }
    return summary, raw_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark uncached and KV-cached autoregressive decoding."
    )
    parser.add_argument(
        "run_dir",
        type=Path,
        help="Training run directory containing config.json and checkpoint.pt.",
    )
    parser.add_argument(
        "--tokens-path",
        type=Path,
        default=Path("./data/tinystories_valid_tokens.npy"),
    )
    parser.add_argument(
        "--prompt-lengths",
        type=int,
        nargs="+",
        default=[16, 64, 128],
    )
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--start-offset", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    context_length: int,
    dataset_length: int,
) -> None:
    if not args.prompt_lengths:
        raise ValueError("At least one prompt length is required")
    if any(prompt_length <= 0 for prompt_length in args.prompt_lengths):
        raise ValueError("All prompt lengths must be positive")
    if args.new_tokens <= 0:
        raise ValueError("new_tokens must be positive")
    if args.warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative")
    if args.repeats <= 0:
        raise ValueError("repeats must be positive")
    if args.start_offset < 0:
        raise ValueError("start_offset must be non-negative")

    largest_prompt_length = max(args.prompt_lengths)
    if args.start_offset + largest_prompt_length > dataset_length:
        raise ValueError("Requested prompt exceeds the token dataset")

    for prompt_length in args.prompt_lengths:
        required_positions = prompt_length + args.new_tokens - 1
        if required_positions > context_length:
            raise ValueError(
                f"prompt_length={prompt_length} and new_tokens={args.new_tokens} "
                f"require {required_positions} positions, but context_length="
                f"{context_length}"
            )


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config, completed_steps = load_model_from_run(args.run_dir, device)

    token_dataset = np.load(args.tokens_path, mmap_mode="r")
    validate_args(
        args=args,
        context_length=config["context_length"],
        dataset_length=len(token_dataset),
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = args.run_dir / f"kv_cache_benchmark_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU"
    )
    all_summaries = []
    all_raw_rows = []

    print("device:", device)
    print("device name:", device_name)
    print("completed training steps:", completed_steps)
    print("context length:", config["context_length"])
    print("warm-up runs:", args.warmup_runs)
    print("measured repeats:", args.repeats)

    for prompt_length in args.prompt_lengths:
        prompt_slice = token_dataset[
            args.start_offset : args.start_offset + prompt_length
        ]
        prompt_ids = [int(token_id) for token_id in prompt_slice]

        print(
            f"\nBenchmarking prompt={prompt_length}, "
            f"new_tokens={args.new_tokens}..."
        )
        summary, raw_rows = benchmark_prompt_length(
            model=model,
            prompt_ids=prompt_ids,
            context_length=config["context_length"],
            max_new_tokens=args.new_tokens,
            warmup_runs=args.warmup_runs,
            repeats=args.repeats,
            device=device,
        )
        all_summaries.append(summary)
        all_raw_rows.extend(raw_rows)

        print("token IDs equal: yes")
        print(
            "uncached median: "
            f"{summary['uncached_tokens_per_second']:.2f} tokens/second"
        )
        print(
            "cached median:   "
            f"{summary['cached_tokens_per_second']:.2f} tokens/second"
        )
        print(f"speedup:         {summary['speedup']:.2f}x")
        if summary["uncached_peak_extra_memory_mib"] is not None:
            print(
                "uncached peak extra allocated memory: "
                f"{summary['uncached_peak_extra_memory_mib']:.2f} MiB"
            )
            print(
                "cached peak extra allocated memory:   "
                f"{summary['cached_peak_extra_memory_mib']:.2f} MiB"
            )

    csv_path = output_dir / "timings.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "prompt_tokens",
                "new_tokens",
                "method",
                "repeat",
                "elapsed_seconds",
                "tokens_per_second",
                "token_ids_equal",
            ],
        )
        writer.writeheader()
        writer.writerows(all_raw_rows)

    git_commit, git_dirty = git_metadata()
    checkpoint_path = args.run_dir / "checkpoint.pt"
    results = {
        "timestamp": timestamp,
        "run_dir": str(args.run_dir.resolve()),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_size_bytes": checkpoint_path.stat().st_size,
        "tokens_path": str(args.tokens_path.resolve()),
        "tokens_sha256": file_sha256(args.tokens_path),
        "checkpoint_step": completed_steps,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "device": str(device),
        "device_name": device_name,
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "model_config": config,
        "benchmark_config": {
            "batch_size": 1,
            "model_dtype": str(next(model.parameters()).dtype),
            "prompt_lengths": args.prompt_lengths,
            "new_tokens": args.new_tokens,
            "correctness_runs_per_method": 1,
            "warmup_runs": args.warmup_runs,
            "repeats": args.repeats,
            "start_offset": args.start_offset,
            "temperature": 0,
            "top_p": 1.0,
            "eos_id": -1,
            "timing_scope": (
                "End-to-end generation after model and token loading; includes "
                "prefill, Python decoding loop, input tensor construction, "
                "sampling, and CUDA synchronization."
            ),
            "memory_metric": (
                "Peak extra PyTorch CUDA memory_allocated above the resident "
                "model baseline, measured once per method."
            ),
        },
        "results": all_summaries,
    }
    json_path = output_dir / "results.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(results, json_file, indent=2)

    print("\nSaved raw timings:", csv_path)
    print("Saved benchmark summary:", json_path)


if __name__ == "__main__":
    main()
