import csv
import json
import time
from pathlib import Path


class ExperimentLogger:
    def __init__(self, run_name, config, root_dir="runs"):
        self.run_dir = Path(root_dir) / run_name
        self.run_dir.mkdir(parents=True, exist_ok=False)

        with open(
            self.run_dir / "config.json",
            "w",
            encoding="utf-8",
        ) as config_file:
            json.dump(config, config_file, indent=2)

        self.metrics_file = open(
            self.run_dir / "metrics.csv",
            "w",
            newline="",
            encoding="utf-8",
        )

        fieldnames = [
            "step",
            "tokens_processed",
            "train_loss",
            "validation_loss",
            "learning_rate",
            "elapsed_seconds",
        ]

        self.writer = csv.DictWriter(
            self.metrics_file,
            fieldnames=fieldnames,
        )
        self.writer.writeheader()

        self.start_time = time.perf_counter()

    def log(
        self,
        step,
        tokens_processed,
        train_loss,
        validation_loss,
        learning_rate,
    ):
        elapsed_seconds = time.perf_counter() - self.start_time

        row = {
            "step": step,
            "tokens_processed": tokens_processed,
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "learning_rate": learning_rate,
            "elapsed_seconds": elapsed_seconds,
        }

        self.writer.writerow(row)
        self.metrics_file.flush()

    def close(self):
        self.metrics_file.close()