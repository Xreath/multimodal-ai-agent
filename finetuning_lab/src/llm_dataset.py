"""
LLM Instruction Dataset Preparation for Fine-tuning.

Creates instruction-following datasets for visual reasoning tasks.
Format: instruction/input/output triplets in JSON.

Interview Topics:
  - Dataset quality > quantity (1000 good samples > 10000 noisy)
  - Instruction format: Alpaca, ShareGPT, ChatML
  - Data filtering: deduplication, quality scoring, length filtering
  - Synthetic data generation with stronger models
"""

import json
import random
from pathlib import Path
from typing import Optional


class LLMDatasetBuilder:
    """Builds instruction-following datasets for visual reasoning fine-tuning."""

    # Instruction templates for visual reasoning tasks
    TASK_TEMPLATES = {
        "scene_description": {
            "instructions": [
                "Describe the scene based on the following CV pipeline output.",
                "What objects are present in this image? Analyze the detection results.",
                "Provide a detailed scene analysis from the following visual data.",
                "Summarize what you see based on these computer vision results.",
                "Analyze the following object detection and OCR results.",
            ],
            "generator": "_generate_scene_description",
        },
        "safety_inspection": {
            "instructions": [
                "Check for safety violations in this workplace image analysis.",
                "Are there any safety concerns based on the detected objects?",
                "Evaluate workplace safety compliance from the CV results.",
                "Identify potential hazards in this scene analysis.",
                "Review the following detection results for safety violations.",
            ],
            "generator": "_generate_safety_inspection",
        },
        "object_counting": {
            "instructions": [
                "How many vehicles are detected in this image?",
                "Count the people visible in the detection results.",
                "What is the total number of detected objects by category?",
                "Summarize the object counts from this detection output.",
                "How many instances of each class were detected?",
            ],
            "generator": "_generate_object_counting",
        },
        "spatial_reasoning": {
            "instructions": [
                "Describe the spatial relationships between detected objects.",
                "Where are the objects positioned relative to each other?",
                "Analyze the layout and positioning of objects in this scene.",
                "What can you infer about the scene layout from the bounding boxes?",
                "Describe the relative positions of objects using bbox coordinates.",
            ],
            "generator": "_generate_spatial_reasoning",
        },
        "ocr_analysis": {
            "instructions": [
                "What text was detected in this image? Interpret its meaning.",
                "Analyze the OCR results and explain what the text indicates.",
                "What can you determine from the text found in this image?",
                "Interpret the detected text regions in context of the scene.",
                "What information do the text detections reveal about this image?",
            ],
            "generator": "_generate_ocr_analysis",
        },
    }

    # Object categories for synthetic data
    OBJECT_CATEGORIES = {
        "vehicles": ["car", "bus", "truck", "motorcycle", "bicycle"],
        "people": ["person"],
        "safety": ["hardhat", "vest", "fire extinguisher", "safety sign"],
        "outdoor": ["traffic light", "stop sign", "bench", "tree"],
        "indoor": ["chair", "table", "monitor", "keyboard", "phone"],
    }

    def __init__(self, output_dir: str = "data/llm"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_dataset(
        self,
        num_samples: int = 200,
        output_file: str = "visual_reasoning_instructions.json",
    ) -> dict:
        """
        Generate a synthetic instruction dataset for visual reasoning.

        Each sample contains:
        - instruction: what the user asks
        - input: CV pipeline output (simulated)
        - output: expected response

        Interview: "How to prepare an instruction dataset?"
        - Instruction diversity: different expressions for the same task
        - Input variety: different scenarios, edge cases
        - Output quality: consistent, accurate, detailed answers
        - Balance: equal amount from each task type
        """
        dataset = []
        samples_per_task = num_samples // len(self.TASK_TEMPLATES)
        remainder = num_samples % len(self.TASK_TEMPLATES)

        for i, (task_name, template) in enumerate(self.TASK_TEMPLATES.items()):
            count = samples_per_task + (1 if i < remainder else 0)
            generator = getattr(self, template["generator"])

            for _ in range(count):
                instruction = random.choice(template["instructions"])
                cv_output, response = generator()

                dataset.append({
                    "instruction": instruction,
                    "input": json.dumps(cv_output, indent=2),
                    "output": response,
                    "task_type": task_name,
                })

        # Shuffle
        random.shuffle(dataset)

        # Save
        output_path = self.output_dir / output_file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        # Stats
        task_counts = {}
        for sample in dataset:
            task = sample["task_type"]
            task_counts[task] = task_counts.get(task, 0) + 1

        stats = {
            "total_samples": len(dataset),
            "output_path": str(output_path),
            "task_distribution": task_counts,
            "avg_instruction_length": round(
                sum(len(s["instruction"]) for s in dataset) / len(dataset), 1
            ),
            "avg_output_length": round(
                sum(len(s["output"]) for s in dataset) / len(dataset), 1
            ),
        }

        print(f"Dataset generated: {output_path}")
        print(f"  Total samples: {stats['total_samples']}")
        for task, count in stats["task_distribution"].items():
            print(f"  {task}: {count}")

        return stats

    def _random_cv_output(self, num_objects: int = None, include_text: bool = False) -> dict:
        """Generate a random CV pipeline output."""
        if num_objects is None:
            num_objects = random.randint(2, 8)

        # Pick a scene type
        scene_type = random.choice(["traffic", "workplace", "indoor", "outdoor"])
        category_pool = {
            "traffic": self.OBJECT_CATEGORIES["vehicles"] + self.OBJECT_CATEGORIES["people"],
            "workplace": self.OBJECT_CATEGORIES["people"] + self.OBJECT_CATEGORIES["safety"],
            "indoor": self.OBJECT_CATEGORIES["indoor"] + self.OBJECT_CATEGORIES["people"],
            "outdoor": self.OBJECT_CATEGORIES["outdoor"] + self.OBJECT_CATEGORIES["people"],
        }

        objects = []
        for _ in range(num_objects):
            label = random.choice(category_pool[scene_type])
            x1 = random.randint(10, 800)
            y1 = random.randint(10, 600)
            w = random.randint(50, 300)
            h = random.randint(50, 300)
            objects.append({
                "label": label,
                "confidence": round(random.uniform(0.5, 0.99), 2),
                "bbox": [x1, y1, x1 + w, y1 + h],
            })

        output = {
            "image_info": {"width": 1280, "height": 720},
            "objects": objects,
            "scene_type": scene_type,
        }

        if include_text:
            text_regions = []
            text_options = {
                "traffic": ["STOP", "SPEED LIMIT 50", "NO PARKING", "ONE WAY", "ABC 1234"],
                "workplace": ["DANGER", "WEAR HELMET", "EXIT", "FIRE EXTINGUISHER", "AUTHORIZED ONLY"],
                "indoor": ["MEETING ROOM", "DO NOT DISTURB", "WIFI: guest123", "EMERGENCY EXIT"],
                "outdoor": ["PARK ENTRANCE", "NO SMOKING", "OPEN 9-5", "CAUTION WET FLOOR"],
            }
            num_text = random.randint(1, 3)
            for _ in range(num_text):
                text_regions.append({
                    "text": random.choice(text_options[scene_type]),
                    "confidence": round(random.uniform(0.7, 0.99), 2),
                })
            output["text_regions"] = text_regions

        return output

    def _generate_scene_description(self) -> tuple:
        """Generate a scene description sample."""
        cv_output = self._random_cv_output(include_text=random.choice([True, False]))
        objects = cv_output["objects"]

        # Count objects by label
        counts = {}
        for obj in objects:
            counts[obj["label"]] = counts.get(obj["label"], 0) + 1

        # Build response
        obj_list = ", ".join(f"{count} {label}(s)" for label, count in counts.items())
        confidence_avg = sum(o["confidence"] for o in objects) / len(objects)

        response = f"The scene contains {len(objects)} detected objects: {obj_list}. "
        response += f"The average detection confidence is {confidence_avg:.2f}. "

        if "text_regions" in cv_output:
            texts = [t["text"] for t in cv_output["text_regions"]]
            response += f"Text detected in the image: {', '.join(texts)}. "

        response += f"This appears to be a {cv_output['scene_type']} scene "
        response += f"with image dimensions {cv_output['image_info']['width']}x{cv_output['image_info']['height']}."

        return cv_output, response

    def _generate_safety_inspection(self) -> tuple:
        """Generate a safety inspection sample."""
        cv_output = self._random_cv_output(include_text=True)

        # Force safety-relevant objects
        safety_objects = []
        num_people = random.randint(2, 5)
        for i in range(num_people):
            safety_objects.append({
                "label": "person",
                "confidence": round(random.uniform(0.7, 0.99), 2),
                "bbox": [random.randint(10, 800), random.randint(10, 600),
                         random.randint(100, 400), random.randint(100, 400)],
            })

        num_helmets = random.randint(0, num_people)
        num_vests = random.randint(0, num_people)
        for _ in range(num_helmets):
            safety_objects.append({
                "label": "hardhat",
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "bbox": [random.randint(10, 800), random.randint(10, 200),
                         random.randint(50, 100), random.randint(50, 100)],
            })
        for _ in range(num_vests):
            safety_objects.append({
                "label": "vest",
                "confidence": round(random.uniform(0.6, 0.95), 2),
                "bbox": [random.randint(10, 800), random.randint(100, 400),
                         random.randint(80, 200), random.randint(80, 200)],
            })

        cv_output["objects"] = safety_objects
        cv_output["scene_type"] = "workplace"

        # Build response
        violations = []
        if num_helmets < num_people:
            violations.append(f"{num_people - num_helmets} worker(s) not wearing hardhats")
        if num_vests < num_people:
            violations.append(f"{num_people - num_vests} worker(s) not wearing safety vests")

        if violations:
            response = f"SAFETY VIOLATIONS DETECTED: {len(violations)} issue(s) found.\n"
            for i, v in enumerate(violations, 1):
                response += f"{i}. {v}\n"
            response += f"\nSummary: {num_people} workers detected, {num_helmets} hardhats, {num_vests} vests. "
            response += "Immediate corrective action recommended."
        else:
            response = f"No safety violations detected. All {num_people} workers are wearing "
            response += "proper safety equipment (hardhats and vests). Compliance: 100%."

        return cv_output, response

    def _generate_object_counting(self) -> tuple:
        """Generate an object counting sample."""
        cv_output = self._random_cv_output()
        objects = cv_output["objects"]

        counts = {}
        for obj in objects:
            counts[obj["label"]] = counts.get(obj["label"], 0) + 1

        response = f"Object count summary ({len(objects)} total detections):\n"
        for label, count in sorted(counts.items(), key=lambda x: -x[1]):
            response += f"- {label}: {count}\n"

        most_common = max(counts, key=counts.get)
        response += f"\nMost frequent: {most_common} ({counts[most_common]} instances)."

        return cv_output, response

    def _generate_spatial_reasoning(self) -> tuple:
        """Generate a spatial reasoning sample."""
        cv_output = self._random_cv_output(num_objects=random.randint(3, 5))
        objects = cv_output["objects"]

        response = "Spatial analysis of detected objects:\n\n"
        img_w = cv_output["image_info"]["width"]
        img_h = cv_output["image_info"]["height"]

        for obj in objects:
            x1, y1, x2, y2 = obj["bbox"]
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2

            # Determine position
            h_pos = "left" if cx < img_w / 3 else ("center" if cx < 2 * img_w / 3 else "right")
            v_pos = "top" if cy < img_h / 3 else ("middle" if cy < 2 * img_h / 3 else "bottom")

            response += f"- {obj['label']} (conf: {obj['confidence']}): {v_pos}-{h_pos} of frame\n"

        # Add relative relationships
        if len(objects) >= 2:
            o1, o2 = objects[0], objects[1]
            dx = ((o1["bbox"][0] + o1["bbox"][2]) / 2) - ((o2["bbox"][0] + o2["bbox"][2]) / 2)
            rel = "to the left of" if dx < 0 else "to the right of"
            response += f"\nThe {o1['label']} is {rel} the {o2['label']}."

        return cv_output, response

    def _generate_ocr_analysis(self) -> tuple:
        """Generate an OCR analysis sample."""
        cv_output = self._random_cv_output(include_text=True)

        if "text_regions" not in cv_output or not cv_output["text_regions"]:
            cv_output["text_regions"] = [
                {"text": "CAUTION", "confidence": 0.95},
                {"text": "AREA 51", "confidence": 0.88},
            ]

        texts = cv_output["text_regions"]
        response = f"OCR Analysis: {len(texts)} text region(s) detected.\n\n"
        for t in texts:
            response += f'- "{t["text"]}" (confidence: {t["confidence"]})\n'

        high_conf = [t for t in texts if t["confidence"] > 0.8]
        if high_conf:
            response += f"\nHigh-confidence readings: {', '.join(t['text'] for t in high_conf)}. "

        response += "These text elements provide contextual information about the scene."

        return cv_output, response

    def format_for_training(
        self,
        dataset_path: str = None,
        format: str = "chatml",
    ) -> list:
        """
        Convert instruction dataset to training format.

        Formats:
        - chatml: <|im_start|>system\n...<|im_end|> (Qwen, TinyLlama)
        - alpaca: ### Instruction:\n...\n### Response:\n...
        - sharegpt: [{from: human, value: ...}, {from: gpt, value: ...}]

        Interview: "Differences between instruction formats?"
        - ChatML: modern, multi-turn support, special tokens
        - Alpaca: simple, single-turn, common
        - ShareGPT: multi-turn conversation format
        """
        dataset_path = dataset_path or str(self.output_dir / "visual_reasoning_instructions.json")
        with open(dataset_path) as f:
            raw_data = json.load(f)

        formatted = []
        for sample in raw_data:
            if format == "chatml":
                text = (
                    f"<|im_start|>system\n"
                    f"You are a visual reasoning assistant that analyzes CV pipeline outputs.<|im_end|>\n"
                    f"<|im_start|>user\n"
                    f"{sample['instruction']}\n\n{sample['input']}<|im_end|>\n"
                    f"<|im_start|>assistant\n"
                    f"{sample['output']}<|im_end|>"
                )
            elif format == "alpaca":
                text = (
                    f"Below is an instruction that describes a task, paired with an input. "
                    f"Write a response that appropriately completes the request.\n\n"
                    f"### Instruction:\n{sample['instruction']}\n\n"
                    f"### Input:\n{sample['input']}\n\n"
                    f"### Response:\n{sample['output']}"
                )
            elif format == "sharegpt":
                text = json.dumps({
                    "conversations": [
                        {"from": "system", "value": "You are a visual reasoning assistant."},
                        {"from": "human", "value": f"{sample['instruction']}\n\n{sample['input']}"},
                        {"from": "gpt", "value": sample["output"]},
                    ]
                })
            else:
                raise ValueError(f"Unknown format: {format}")

            formatted.append({"text": text, "task_type": sample["task_type"]})

        return formatted

    def split_dataset(
        self,
        dataset_path: str = None,
        train_ratio: float = 0.85,
        val_ratio: float = 0.15,
    ) -> dict:
        """Split dataset into train/validation sets."""
        dataset_path = dataset_path or str(self.output_dir / "visual_reasoning_instructions.json")
        with open(dataset_path) as f:
            data = json.load(f)

        random.shuffle(data)
        split_idx = int(len(data) * train_ratio)

        train_data = data[:split_idx]
        val_data = data[split_idx:]

        train_path = self.output_dir / "train.json"
        val_path = self.output_dir / "val.json"

        with open(train_path, "w") as f:
            json.dump(train_data, f, indent=2)
        with open(val_path, "w") as f:
            json.dump(val_data, f, indent=2)

        stats = {
            "train": {"samples": len(train_data), "path": str(train_path)},
            "val": {"samples": len(val_data), "path": str(val_path)},
        }
        print(f"Dataset split: train={len(train_data)}, val={len(val_data)}")
        return stats
