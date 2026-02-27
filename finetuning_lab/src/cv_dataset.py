"""
CV Dataset Preparation for YOLOv8 Fine-tuning.

Creates a synthetic safety equipment detection dataset in YOLO format.
In production, you'd use real labeled data (e.g., from Roboflow, CVAT, or Label Studio).

YOLO format per image:
  - images/train/img_001.jpg
  - labels/train/img_001.txt  (class_id cx cy w h — normalized)

Interview Topics:
  - Dataset formats: YOLO, COCO, VOC (Pascal XML)
  - Data augmentation strategies and when to use them
  - Class imbalance handling: oversampling, weighted loss, focal loss
  - Train/val/test split best practices
"""

import os
import json
import random
import shutil
from pathlib import Path
from typing import Optional

import yaml
import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class CVDatasetBuilder:
    """Builds and manages YOLO-format datasets for fine-tuning."""

    # Safety equipment classes
    CLASSES = ["hardhat", "no-hardhat", "vest", "no-vest", "person"]

    def __init__(self, data_dir: str = "data/cv/safety"):
        self.data_dir = Path(data_dir)
        self.classes = self.CLASSES

    def create_dataset_structure(self) -> Path:
        """Create YOLO dataset directory structure."""
        for split in ["train", "val", "test"]:
            (self.data_dir / "images" / split).mkdir(parents=True, exist_ok=True)
            (self.data_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

        # Create data.yaml (YOLO dataset config)
        data_yaml = {
            "path": str(self.data_dir.resolve()),
            "train": "images/train",
            "val": "images/val",
            "test": "images/test",
            "names": {i: name for i, name in enumerate(self.classes)},
            "nc": len(self.classes),
        }

        yaml_path = self.data_dir / "data.yaml"
        with open(yaml_path, "w") as f:
            yaml.dump(data_yaml, f, default_flow_style=False)

        print(f"Dataset structure created at: {self.data_dir}")
        print(f"  Classes: {self.classes}")
        print(f"  data.yaml: {yaml_path}")
        return self.data_dir

    def generate_synthetic_dataset(
        self,
        num_train: int = 100,
        num_val: int = 20,
        num_test: int = 20,
        img_size: int = 640,
    ) -> dict:
        """
        Generate a synthetic dataset with colored shapes representing safety items.

        In production, you'd use real images with proper annotations.
        This synthetic version demonstrates the pipeline and YOLO format.
        """
        if cv2 is None:
            raise ImportError("OpenCV required: pip install opencv-python")

        self.create_dataset_structure()

        stats = {}
        for split, count in [("train", num_train), ("val", num_val), ("test", num_test)]:
            class_counts = {c: 0 for c in self.classes}
            for i in range(count):
                img, labels = self._generate_synthetic_image(img_size)
                # Save image
                img_path = self.data_dir / "images" / split / f"img_{i:04d}.jpg"
                cv2.imwrite(str(img_path), img)
                # Save labels
                label_path = self.data_dir / "labels" / split / f"img_{i:04d}.txt"
                with open(label_path, "w") as f:
                    for label in labels:
                        f.write(f"{label[0]} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")
                        class_counts[self.classes[label[0]]] += 1

            stats[split] = {"images": count, "class_counts": class_counts}

        print(f"\nSynthetic dataset generated:")
        for split, info in stats.items():
            print(f"  {split}: {info['images']} images")
            for cls, cnt in info["class_counts"].items():
                print(f"    {cls}: {cnt} instances")

        return stats

    def _generate_synthetic_image(self, size: int = 640) -> tuple:
        """Generate a single synthetic image with random safety objects."""
        # Create a scene background (random color)
        bg_color = random.choice([
            (200, 200, 200),  # gray
            (180, 200, 180),  # greenish
            (200, 190, 170),  # brownish
        ])
        img = np.full((size, size, 3), bg_color, dtype=np.uint8)
        # Add noise for realism
        noise = np.random.randint(0, 20, img.shape, dtype=np.uint8)
        img = cv2.add(img, noise)

        labels = []
        num_objects = random.randint(2, 6)

        for _ in range(num_objects):
            cls_id = random.randint(0, len(self.classes) - 1)
            cls_name = self.classes[cls_id]

            # Random position and size
            w = random.randint(30, 120)
            h = random.randint(30, 150)
            x1 = random.randint(0, size - w)
            y1 = random.randint(0, size - h)
            x2, y2 = x1 + w, y1 + h

            # Draw shape based on class
            color = self._class_color(cls_name)
            shape = self._class_shape(cls_name)

            if shape == "circle":
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                r = min(w, h) // 2
                cv2.circle(img, (cx, cy), r, color, -1)
                cv2.circle(img, (cx, cy), r, (0, 0, 0), 2)
            elif shape == "rectangle":
                cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), 2)
            elif shape == "triangle":
                pts = np.array([
                    [(x1 + x2) // 2, y1],
                    [x1, y2],
                    [x2, y2],
                ], np.int32)
                cv2.fillPoly(img, [pts], color)
                cv2.polylines(img, [pts], True, (0, 0, 0), 2)
            else:
                # Ellipse for person
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, color, -1)
                cv2.ellipse(img, (cx, cy), (w // 2, h // 2), 0, 0, 360, (0, 0, 0), 2)

            # Add class label text
            cv2.putText(img, cls_name[:3], (x1 + 2, y1 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

            # Convert to YOLO format (normalized cx, cy, w, h)
            cx_norm = ((x1 + x2) / 2) / size
            cy_norm = ((y1 + y2) / 2) / size
            w_norm = w / size
            h_norm = h / size
            labels.append((cls_id, cx_norm, cy_norm, w_norm, h_norm))

        return img, labels

    def _class_color(self, cls_name: str) -> tuple:
        colors = {
            "hardhat": (0, 200, 255),       # yellow
            "no-hardhat": (0, 0, 200),       # red
            "vest": (0, 255, 100),           # green
            "no-vest": (100, 0, 200),        # purple
            "person": (200, 150, 100),       # blue-ish
        }
        return colors.get(cls_name, (128, 128, 128))

    def _class_shape(self, cls_name: str) -> str:
        shapes = {
            "hardhat": "circle",
            "no-hardhat": "triangle",
            "vest": "rectangle",
            "no-vest": "rectangle",
            "person": "ellipse",
        }
        return shapes.get(cls_name, "rectangle")

    def from_coco_format(self, coco_json: str, images_dir: str, split: str = "train"):
        """
        Convert COCO format annotations to YOLO format.

        COCO format: {images: [...], annotations: [{image_id, category_id, bbox: [x,y,w,h]}]}
        YOLO format: class_id cx cy w h (normalized)

        Interview: "What is the difference between COCO and YOLO formats?"
        - COCO: absolute [x, y, width, height] from top-left
        - YOLO: normalized [cx, cy, width, height] center-based
        - VOC: absolute [xmin, ymin, xmax, ymax]
        """
        with open(coco_json) as f:
            coco = json.load(f)

        # Build image lookup
        img_lookup = {img["id"]: img for img in coco["images"]}
        # Build category mapping
        cat_map = {}
        for cat in coco.get("categories", []):
            if cat["name"] in self.classes:
                cat_map[cat["id"]] = self.classes.index(cat["name"])

        # Group annotations by image
        img_anns = {}
        for ann in coco["annotations"]:
            img_id = ann["image_id"]
            if img_id not in img_anns:
                img_anns[img_id] = []
            img_anns[img_id].append(ann)

        converted = 0
        for img_id, anns in img_anns.items():
            img_info = img_lookup[img_id]
            img_w, img_h = img_info["width"], img_info["height"]

            # Copy image
            src = Path(images_dir) / img_info["file_name"]
            if not src.exists():
                continue
            dst = self.data_dir / "images" / split / img_info["file_name"]
            shutil.copy2(src, dst)

            # Write YOLO labels
            label_name = Path(img_info["file_name"]).stem + ".txt"
            label_path = self.data_dir / "labels" / split / label_name
            with open(label_path, "w") as f:
                for ann in anns:
                    if ann["category_id"] not in cat_map:
                        continue
                    cls_id = cat_map[ann["category_id"]]
                    x, y, w, h = ann["bbox"]  # COCO: [x, y, w, h] absolute
                    # Convert to YOLO: [cx, cy, w, h] normalized
                    cx = (x + w / 2) / img_w
                    cy = (y + h / 2) / img_h
                    nw = w / img_w
                    nh = h / img_h
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            converted += 1

        print(f"Converted {converted} images from COCO to YOLO format ({split} split)")

    def get_dataset_stats(self) -> dict:
        """Analyze dataset statistics — class distribution, image counts, etc."""
        stats = {}
        for split in ["train", "val", "test"]:
            label_dir = self.data_dir / "labels" / split
            if not label_dir.exists():
                continue

            class_counts = {c: 0 for c in self.classes}
            total_objects = 0
            num_images = 0

            for label_file in label_dir.glob("*.txt"):
                num_images += 1
                with open(label_file) as f:
                    for line in f:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            if cls_id < len(self.classes):
                                class_counts[self.classes[cls_id]] += 1
                                total_objects += 1

            stats[split] = {
                "images": num_images,
                "total_objects": total_objects,
                "class_counts": class_counts,
                "avg_objects_per_image": total_objects / max(num_images, 1),
            }

        return stats
