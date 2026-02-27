"""
Unified Evaluation Module for CV and LLM Fine-tuning.

Provides metrics, visualizations, and comparison tools.

Interview Topics:
  - mAP calculation and IoU thresholds
  - Perplexity interpretation
  - Overfitting detection and mitigation
  - Train/val/test split best practices
"""

import json
import math
from pathlib import Path
from typing import Optional

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use("Agg")  # non-interactive backend
except ImportError:
    plt = None

try:
    import numpy as np
except ImportError:
    np = None


class FinetuningEvaluator:
    """Evaluates and visualizes fine-tuning results."""

    def __init__(self, output_dir: str = "output"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def summarize_cv_results(self, results_dir: str) -> dict:
        """
        Summarize YOLOv8 training results from results.csv.

        Interview: "How to detect overfitting?"
        1. Train loss decreases but val loss increases → overfitting
        2. mAP reaches plateau and starts dropping → early stopping
        3. Huge train/val gap → model is too large or data is too small
        4. Solutions: augmentation, dropout, smaller model, more data
        """
        results_path = Path(results_dir) / "results.csv"
        if not results_path.exists():
            print(f"Results file not found: {results_path}")
            return {}

        import csv
        with open(results_path) as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {}

        # Clean header whitespace
        rows = [{k.strip(): v.strip() for k, v in row.items()} for row in rows]

        # Extract key metrics from last epoch
        last = rows[-1]
        best_map50 = max(float(r.get("metrics/mAP50(B)", 0)) for r in rows)
        best_map = max(float(r.get("metrics/mAP50-95(B)", 0)) for r in rows)

        summary = {
            "total_epochs": len(rows),
            "final_metrics": {
                "train_box_loss": float(last.get("train/box_loss", 0)),
                "train_cls_loss": float(last.get("train/cls_loss", 0)),
                "val_box_loss": float(last.get("val/box_loss", 0)),
                "val_cls_loss": float(last.get("val/cls_loss", 0)),
                "mAP50": float(last.get("metrics/mAP50(B)", 0)),
                "mAP50-95": float(last.get("metrics/mAP50-95(B)", 0)),
                "precision": float(last.get("metrics/precision(B)", 0)),
                "recall": float(last.get("metrics/recall(B)", 0)),
            },
            "best_metrics": {
                "best_mAP50": best_map50,
                "best_mAP50-95": best_map,
            },
        }

        return summary

    def plot_cv_training(self, results_dir: str, save_path: str = None):
        """Plot YOLOv8 training curves: loss and mAP over epochs."""
        if plt is None:
            print("matplotlib required for plotting")
            return

        results_path = Path(results_dir) / "results.csv"
        if not results_path.exists():
            print(f"Results not found: {results_path}")
            return

        import csv
        with open(results_path) as f:
            reader = csv.DictReader(f)
            rows = [{k.strip(): v.strip() for k, v in row.items()} for row in reader]

        epochs = list(range(1, len(rows) + 1))

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("YOLOv8 Fine-tuning Results", fontsize=14, fontweight="bold")

        # 1. Box Loss
        train_box = [float(r.get("train/box_loss", 0)) for r in rows]
        val_box = [float(r.get("val/box_loss", 0)) for r in rows]
        axes[0, 0].plot(epochs, train_box, "b-", label="Train")
        axes[0, 0].plot(epochs, val_box, "r-", label="Val")
        axes[0, 0].set_title("Box Loss")
        axes[0, 0].set_xlabel("Epoch")
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. Classification Loss
        train_cls = [float(r.get("train/cls_loss", 0)) for r in rows]
        val_cls = [float(r.get("val/cls_loss", 0)) for r in rows]
        axes[0, 1].plot(epochs, train_cls, "b-", label="Train")
        axes[0, 1].plot(epochs, val_cls, "r-", label="Val")
        axes[0, 1].set_title("Classification Loss")
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. mAP
        map50 = [float(r.get("metrics/mAP50(B)", 0)) for r in rows]
        map50_95 = [float(r.get("metrics/mAP50-95(B)", 0)) for r in rows]
        axes[1, 0].plot(epochs, map50, "g-", label="mAP50")
        axes[1, 0].plot(epochs, map50_95, "m-", label="mAP50-95")
        axes[1, 0].set_title("mAP Metrics")
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. Precision / Recall
        precision = [float(r.get("metrics/precision(B)", 0)) for r in rows]
        recall = [float(r.get("metrics/recall(B)", 0)) for r in rows]
        axes[1, 1].plot(epochs, precision, "c-", label="Precision")
        axes[1, 1].plot(epochs, recall, "y-", label="Recall")
        axes[1, 1].set_title("Precision & Recall")
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "cv_training_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curves saved: {save_path}")

    def plot_llm_training(self, trainer_log: list = None, save_path: str = None):
        """Plot LLM training curves: loss and perplexity."""
        if plt is None:
            print("matplotlib required for plotting")
            return

        if trainer_log is None:
            print("No training log provided")
            return

        # Extract metrics from trainer log
        train_steps = []
        train_losses = []
        eval_steps = []
        eval_losses = []

        for entry in trainer_log:
            if "loss" in entry:
                train_steps.append(entry.get("step", 0))
                train_losses.append(entry["loss"])
            if "eval_loss" in entry:
                eval_steps.append(entry.get("step", 0))
                eval_losses.append(entry["eval_loss"])

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle("LLM Fine-tuning (LoRA) Results", fontsize=14, fontweight="bold")

        # Loss
        if train_losses:
            axes[0].plot(train_steps, train_losses, "b-", alpha=0.7, label="Train Loss")
        if eval_losses:
            axes[0].plot(eval_steps, eval_losses, "r-o", label="Eval Loss")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Step")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Perplexity
        if eval_losses:
            perplexities = [math.exp(l) for l in eval_losses]
            axes[1].plot(eval_steps, perplexities, "g-o", label="Perplexity")
            axes[1].set_title("Perplexity")
            axes[1].set_xlabel("Step")
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_path = save_path or str(self.output_dir / "llm_training_curves.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"Training curves saved: {save_path}")

    def generate_report(
        self,
        cv_metrics: dict = None,
        llm_metrics: dict = None,
        save_path: str = None,
    ) -> dict:
        """Generate a comprehensive fine-tuning report."""
        report = {
            "title": "Fine-tuning Lab Report",
            "cv_finetuning": cv_metrics or {},
            "llm_finetuning": llm_metrics or {},
        }

        # Print summary
        print(f"\n{'='*60}")
        print("FINE-TUNING LAB REPORT")
        print(f"{'='*60}")

        if cv_metrics:
            print(f"\n--- CV Fine-tuning (YOLOv8) ---")
            for key, val in cv_metrics.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for k, v in val.items():
                        print(f"    {k}: {v}")
                else:
                    print(f"  {key}: {val}")

        if llm_metrics:
            print(f"\n--- LLM Fine-tuning (LoRA) ---")
            for key, val in llm_metrics.items():
                print(f"  {key}: {val}")

        # Save report
        save_path = save_path or str(self.output_dir / "finetuning_report.json")
        with open(save_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved: {save_path}")

        return report
