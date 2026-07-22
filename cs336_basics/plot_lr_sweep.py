from pathlib import Path
import matplotlib.pyplot as plt
from cs336_basics.plot_experiment import load_metrics


runs = {
    "3e-4": Path("./runs") / "tinystories_lr_3e-4_20260722_181443",
    "1e-3": Path("./runs") / "tinystories_lr_1e-3_20260722_175210",
    "3e-3": Path("./runs") / "tinystories_lr_3e-3_20260722_180234"
}

fig, ax = plt.subplots(figsize = (8,5))


for (label, run_dir) in runs.items():
    metrics_path = run_dir / "metrics.csv"

    metrics = load_metrics(metrics_path)
    ax.plot(metrics["validation_steps"],
            metrics["validation_losses"],
            label = label,
            marker = "o")
    print(label)
    print(metrics["validation_steps"])
    print(metrics["validation_losses"])

    

ax.set_title("Validation Loss by Learning Rate")
ax.set_xlabel("Gradient step")
ax.set_ylabel("Validation cross-entropy loss")
ax.grid(True,color = "#D1D5DB", alpha = 0.6)
ax.legend(title = "Maximum learning rate")
fig.tight_layout()
output_path = Path("./runs") / "lr_sweep_validation_loss.png"
fig.savefig(output_path, dpi=160)
plt.close(fig)

