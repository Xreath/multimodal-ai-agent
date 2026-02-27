"""
YOLOv8 Fine-tuning Module.

Demonstrates transfer learning strategies for object detection:
1. Full fine-tune — train all layers
2. Feature extraction — freeze backbone, train head only
3. Progressive unfreezing — gradually unfreeze layers

Interview Topics:
  - Transfer learning: when full fine-tune vs feature extraction?
  - Backbone freezing strategies and learning rate scheduling
  - Early stopping and overfitting detection
  - mAP calculation: IoU thresholds, precision-recall curves
  - Data augmentation in YOLO: mosaic, mixup, color jitter
"""

import os
import json
import time
from pathlib import Path
from typing import Optional

import yaml

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLOFineTuner:
    """
    Fine-tunes YOLOv8 on custom datasets with various transfer learning strategies.

    Transfer Learning Strategies:
    ─────────────────────────────
    1. FULL FINE-TUNE (freeze=0):
       - All weights updated
       - Best when: large dataset, domain very different from COCO
       - Risk: overfitting on small datasets

    2. FEATURE EXTRACTION (freeze=backbone):
       - Freeze backbone (feature extractor), train detection head only
       - Best when: small dataset, domain similar to COCO
       - Faster training, less GPU memory

    3. PROGRESSIVE UNFREEZING:
       - Start frozen → gradually unfreeze layers
       - Best when: medium dataset, moderate domain shift
       - Balances speed and accuracy
    """

    def __init__(self, config_path: str = "configs/cv_config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.results = None

    def load_model(self, pretrained: str = None) -> "YOLO":
        """Load YOLOv8 with pretrained weights."""
        if YOLO is None:
            raise ImportError("ultralytics required: pip install ultralytics")

        model_path = pretrained or self.config["model"]["base"]
        self.model = YOLO(model_path)
        print(f"Loaded model: {model_path}")
        print(f"  Parameters: {sum(p.numel() for p in self.model.model.parameters()):,}")
        print(f"  Trainable: {sum(p.numel() for p in self.model.model.parameters() if p.requires_grad):,}")
        return self.model

    def freeze_layers(self, num_layers: int = 10):
        """
        Freeze first N layers of the model for transfer learning.

        YOLOv8n architecture (simplified):
          Layer 0-9: Backbone (feature extraction) — CSPDarknet
          Layer 10-21: Neck + Head (detection) — PANet + Detect

        Interview: "When do you freeze the backbone?"
        - Small dataset (<500 images) → freeze backbone
        - Large dataset (>5000 images) → full fine-tune
        - Similar domain (COCO-like) → freeze more
        - Different domain (medical, satellite) → freeze less
        """
        if self.model is None:
            raise ValueError("Load model first with load_model()")

        frozen_count = 0
        for i, (name, param) in enumerate(self.model.model.named_parameters()):
            if i < num_layers:
                param.requires_grad = False
                frozen_count += 1

        trainable = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.model.parameters())
        print(f"Froze {frozen_count} parameter groups")
        print(f"  Trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    def train(
        self,
        data_yaml: str,
        epochs: int = None,
        batch_size: int = None,
        imgsz: int = None,
        freeze: int = None,
        project: str = "output",
        name: str = "safety_detection",
        resume: bool = False,
    ) -> dict:
        """
        Train YOLOv8 on custom dataset.

        Returns training results with metrics.

        Interview: "How do you choose training hyperparameters?"
        - Epochs: start with 50-100, use early stopping (patience=10)
        - Batch size: largest that fits in GPU memory (power of 2)
        - Learning rate: 0.01 for SGD, 0.001 for Adam/AdamW
        - Image size: match dataset resolution (640 standard, 1280 for small objects)
        - Augmentation: heavier for small datasets (mosaic, mixup)
        """
        if self.model is None:
            self.load_model()

        cfg = self.config["training"]
        epochs = epochs or cfg["epochs"]
        batch_size = batch_size or cfg["batch_size"]
        imgsz = imgsz or cfg["imgsz"]
        freeze_layers = freeze if freeze is not None else self.config["freeze"]["layers"]

        print(f"\n{'='*60}")
        print(f"YOLOv8 Fine-tuning")
        print(f"{'='*60}")
        print(f"  Dataset: {data_yaml}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Image size: {imgsz}")
        print(f"  Freeze layers: {freeze_layers}")
        print(f"  Device: {cfg['device']}")
        print(f"{'='*60}\n")

        start_time = time.time()

        self.results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            freeze=freeze_layers,
            patience=cfg["patience"],
            optimizer=cfg["optimizer"],
            lr0=cfg["lr0"],
            lrf=cfg["lrf"],
            warmup_epochs=cfg["warmup_epochs"],
            weight_decay=cfg["weight_decay"],
            device=cfg["device"],
            project=project,
            name=name,
            exist_ok=True,
            resume=resume,
            # Augmentation
            hsv_h=self.config["augmentation"]["hsv_h"],
            hsv_s=self.config["augmentation"]["hsv_s"],
            hsv_v=self.config["augmentation"]["hsv_v"],
            degrees=self.config["augmentation"]["degrees"],
            translate=self.config["augmentation"]["translate"],
            scale=self.config["augmentation"]["scale"],
            fliplr=self.config["augmentation"]["fliplr"],
            mosaic=self.config["augmentation"]["mosaic"],
            mixup=self.config["augmentation"]["mixup"],
            verbose=True,
        )

        training_time = time.time() - start_time

        # Extract metrics
        metrics = self._extract_metrics()
        metrics["training_time_seconds"] = round(training_time, 2)
        metrics["config"] = {
            "epochs": epochs,
            "batch_size": batch_size,
            "imgsz": imgsz,
            "freeze_layers": freeze_layers,
        }

        return metrics

    def _extract_metrics(self) -> dict:
        """Extract key metrics from training results."""
        if self.results is None:
            return {}

        results_dict = self.results.results_dict if hasattr(self.results, "results_dict") else {}

        return {
            "mAP50": round(results_dict.get("metrics/mAP50(B)", 0), 4),
            "mAP50-95": round(results_dict.get("metrics/mAP50-95(B)", 0), 4),
            "precision": round(results_dict.get("metrics/precision(B)", 0), 4),
            "recall": round(results_dict.get("metrics/recall(B)", 0), 4),
            "box_loss": round(results_dict.get("train/box_loss", 0), 4),
            "cls_loss": round(results_dict.get("train/cls_loss", 0), 4),
        }

    def evaluate(self, data_yaml: str = None, split: str = "test") -> dict:
        """
        Evaluate fine-tuned model on test set.

        Interview: "How is mAP calculated?"
        1. Draw precision-recall curve for each class
        2. Calculate the area under the PR curve (AP = Average Precision)
        3. mAP50: Average of APs with IoU=0.5 threshold
        4. mAP50-95: Average of APs with IoU=0.5, 0.55, ..., 0.95 (COCO metric)
        """
        if self.model is None:
            raise ValueError("No model loaded")

        print(f"\nEvaluating on {split} split...")
        results = self.model.val(
            data=data_yaml,
            split=split,
            conf=self.config["evaluation"]["conf_threshold"],
            iou=self.config["evaluation"]["iou_threshold"],
        )

        eval_metrics = {
            "mAP50": round(results.box.map50, 4),
            "mAP50-95": round(results.box.map, 4),
            "precision": round(results.box.mp, 4),
            "recall": round(results.box.mr, 4),
            "per_class": {},
        }

        # Per-class metrics
        if hasattr(results.box, "maps") and results.names:
            for i, name in results.names.items():
                if i < len(results.box.maps):
                    eval_metrics["per_class"][name] = round(results.box.maps[i], 4)

        return eval_metrics

    def predict(self, image_path: str, conf: float = 0.25) -> list:
        """Run inference with fine-tuned model."""
        if self.model is None:
            raise ValueError("No model loaded")

        results = self.model.predict(image_path, conf=conf)
        detections = []
        for r in results:
            for box in r.boxes:
                detections.append({
                    "label": r.names[int(box.cls)],
                    "confidence": round(float(box.conf), 4),
                    "bbox": [round(x, 1) for x in box.xyxy[0].tolist()],
                })
        return detections

    def compare_strategies(self, data_yaml: str, epochs: int = 20) -> dict:
        """
        Compare different transfer learning strategies.

        Runs 3 experiments:
        1. Full fine-tune (freeze=0)
        2. Freeze backbone (freeze=10)
        3. Head only (freeze larger number)

        Interview: "Comparison of transfer learning strategies?"
        """
        strategies = {
            "full_finetune": {"freeze": 0, "description": "Train all layers"},
            "freeze_backbone": {"freeze": 10, "description": "Freeze first 10 layers (backbone)"},
            "head_only": {"freeze": 20, "description": "Train detection head only"},
        }

        comparison = {}
        for name, config in strategies.items():
            print(f"\n{'='*60}")
            print(f"Strategy: {name} — {config['description']}")
            print(f"{'='*60}")

            self.load_model()  # fresh model each time
            metrics = self.train(
                data_yaml=data_yaml,
                epochs=epochs,
                freeze=config["freeze"],
                name=f"compare_{name}",
            )
            comparison[name] = {
                "description": config["description"],
                "freeze_layers": config["freeze"],
                "metrics": metrics,
            }

        # Print comparison table
        print(f"\n{'='*60}")
        print("Transfer Learning Strategy Comparison")
        print(f"{'='*60}")
        print(f"{'Strategy':<20} {'mAP50':>8} {'mAP50-95':>10} {'Precision':>10} {'Recall':>8} {'Time(s)':>8}")
        print("-" * 70)
        for name, data in comparison.items():
            m = data["metrics"]
            print(f"{name:<20} {m['mAP50']:>8.4f} {m['mAP50-95']:>10.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['training_time_seconds']:>8.1f}")

        return comparison

    def export_model(self, format: str = "onnx") -> str:
        """
        Export fine-tuned model for deployment.

        Formats: onnx, torchscript, coreml, tflite, openvino

        Interview: "Which format for model deployment?"
        - ONNX: cross-platform, TensorRT support
        - TorchScript: PyTorch native, mobile deployment
        - CoreML: Apple devices (iOS/macOS)
        - TFLite: Android/edge devices
        - OpenVINO: Intel hardware optimization
        """
        if self.model is None:
            raise ValueError("No model loaded")

        export_path = self.model.export(format=format)
        print(f"Model exported: {export_path}")
        return str(export_path)
