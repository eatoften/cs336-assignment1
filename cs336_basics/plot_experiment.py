import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def load_metrics(metrics_path):
    train_steps = []
    train_elapsed_seconds = []
    train_losses = []

    validation_steps = []
    validation_elapsed_seconds = []
    validation_losses = []

    with open(metrics_path, encoding="utf-8") as metrics_file:
        reader = csv.DictReader(metrics_file)

        for row in reader:
            step = int(row["step"])
            elapsed_seconds = float(row["elapsed_seconds"])
            train_loss = float(row["train_loss"])

            train_steps.append(step)
            train_elapsed_seconds.append(elapsed_seconds)
            train_losses.append(train_loss)

            if row["validation_loss"]:
                validation_steps.append(step)
                validation_elapsed_seconds.append(elapsed_seconds)
                validation_losses.append(
                    float(row["validation_loss"])
                )

    return {
        "train_steps": train_steps,
        "train_elapsed_seconds": train_elapsed_seconds,
        "train_losses": train_losses,
        "validation_steps": validation_steps,
        "validation_elapsed_seconds": validation_elapsed_seconds,
        "validation_losses": validation_losses,
    }



def plot_loss_vs_steps(metrics, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        metrics["train_steps"],
        metrics["train_losses"],
        color="#2563EB",
        marker="o",
        linestyle="-",
        label="Training loss",
    )

    if metrics["validation_steps"]:
        ax.plot(
            metrics["validation_steps"],
            metrics["validation_losses"],
            color="#EA580C",
            marker="s",
            linestyle="--",
            label="Validation loss",
        )

    ax.set_title("Loss vs Gradient Steps")
    ax.set_xlabel("Gradient step")
    ax.set_ylabel("Cross-entropy loss per token")

    ax.grid(True, color="#D1D5DB", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)



def plot_loss_vs_time(metrics, output_path):
    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(
        metrics["train_elapsed_seconds"],
        metrics["train_losses"],
        color="#2563EB",
        marker="o",
        linestyle="-",
        label="Training loss",
    )

    if metrics["validation_elapsed_seconds"]:
        ax.plot(
            metrics["validation_elapsed_seconds"],
            metrics["validation_losses"],
            color="#EA580C",
            marker="s",
            linestyle="--",
            label="Validation loss",
        )

    ax.set_title("Loss vs Wall-Clock Time")
    ax.set_xlabel("Wall-clock time (seconds)")
    ax.set_ylabel("Cross-entropy loss per token")

    ax.grid(True, color="#D1D5DB", alpha=0.6)
    ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()

    metrics_path = args.run_dir / "metrics.csv"
    metrics = load_metrics(metrics_path)
    steps_plot_path = args.run_dir / "loss_vs_steps.png"

    plot_loss_vs_steps(
        metrics,
        steps_plot_path,
    )

    print(f"Saved plot to: {steps_plot_path}")


    time_plot_path = args.run_dir / "loss_vs_time.png"

    plot_loss_vs_time(
        metrics,
        time_plot_path,
    )

    print(f"Saved plot to: {time_plot_path}")

    for metric_name, values in metrics.items():
        print(f"{metric_name}:")
        print(f"  {values}")