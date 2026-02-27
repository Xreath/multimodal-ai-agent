"""
Visual Perception Engine — Main Pipeline
Combines all CV modules to produce a single structured JSON output.
"""

import cv2
import json
import time
import numpy as np
import random
from typing import Dict, Any, Optional
import os

from .preprocessor import ImagePreprocessor
from .detector import ObjectDetector
from .segmentor import InstanceSegmentor
from .ocr_engine import OCREngine


def random_color() -> tuple:
    """Generate a random bright color."""
    return (random.randint(50, 255), random.randint(50, 255), random.randint(50, 255))


class VisualPerceptionPipeline:
    """
    Main CV pipeline: Detection + Segmentation + OCR → Structured JSON

    Usage:
        pipeline = VisualPerceptionPipeline()
        result = pipeline.analyze("image.jpg")
    """

    def __init__(
        self,
        detection_model: str = "yolov8n.pt",
        segmentation_model: str = "yolov8n-seg.pt",
        ocr_languages: Optional[list[str]] = None,
        confidence: float = 0.25
    ):
        self.preprocessor = ImagePreprocessor()
        self.detector = ObjectDetector(model_name=detection_model, confidence=confidence)
        self.segmentor = InstanceSegmentor(model_name=segmentation_model, confidence=confidence)
        self.ocr_engine = OCREngine(languages=ocr_languages or ["en", "tr"])
        self.confidence = confidence

    def analyze(
        self,
        image_path: str,
        run_detection: bool = True,
        run_segmentation: bool = True,
        run_ocr: bool = True,
        max_size: int = 1280
    ) -> Dict[str, Any]:
        """
        Runs the full analysis pipeline.

        Args:
            image_path: Image file path
            run_detection: Run object detection
            run_segmentation: Run instance segmentation
            run_ocr: Run OCR
            max_size: Maximum image dimension

        Returns:
            Structured JSON dict
        """
        start_time = time.time()

        # 1. Load and preprocess image
        image = self.preprocessor.load_image(image_path)
        image = self.preprocessor.resize(image, max_size=max_size)
        image_info = self.preprocessor.get_image_info(image)

        result = {
            "image_path": image_path,
            "image_info": image_info,
            "objects": [],
            "segments": [],
            "text_regions": [],
            "scene_description": "",
            "processing_time": {}
        }

        # 2. Object Detection
        if run_detection:
            t = time.time()
            result["objects"] = self.detector.detect(image)
            result["processing_time"]["detection"] = round(time.time() - t, 3)

        # 3. Instance Segmentation
        if run_segmentation:
            t = time.time()
            segments = self.segmentor.segment(image)
            # mask_base64 can be very large — optionally remove it
            result["segments"] = segments
            result["processing_time"]["segmentation"] = round(time.time() - t, 3)

        # 4. OCR
        if run_ocr:
            t = time.time()
            result["text_regions"] = self.ocr_engine.extract_text(image)
            result["processing_time"]["ocr"] = round(time.time() - t, 3)

        # 5. Generate simple scene description
        result["scene_description"] = self._generate_scene_description(result)

        result["processing_time"]["total"] = round(time.time() - start_time, 3)

        return result

    def _generate_scene_description(self, result: Dict) -> str:
        """Generate a simple scene description from Detection + OCR results."""
        parts = []

        # Object count
        if result["objects"]:
            label_counts = {}
            for obj in result["objects"]:
                label = obj["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            obj_parts = [f"{count} {label}{'s' if count > 1 else ''}"
                         for label, count in label_counts.items()]
            parts.append(f"Detected: {', '.join(obj_parts)}")

        # Text info
        if result["text_regions"]:
            texts = [r["text"] for r in result["text_regions"][:5]]  # First 5 texts
            parts.append(f"Text found: {', '.join(texts)}")

        # Segment count
        if result["segments"]:
            parts.append(f"{len(result['segments'])} segmented regions")

        return ". ".join(parts) if parts else "No significant content detected"

    def visualize(
        self,
        image_path: str,
        result: Dict[str, Any],
        output_path: str,
        show_detections: bool = True,
        show_segments: bool = True,
        show_ocr: bool = True
    ) -> str:
        """
        Visualize and save analysis results.

        Args:
            image_path: Original image path
            result: Output of analyze()
            output_path: Output image path
            show_detections: Show bounding boxes
            show_segments: Show segmentation masks
            show_ocr: Show OCR texts

        Returns:
            Saved file path
        """
        # Load original image
        image = self.preprocessor.load_image(image_path)
        image = self.preprocessor.resize(image, max_size=1280)

        # Create copy
        vis_image = image.copy()

        # 1. Draw segmentation masks (in background, semi-transparent)
        if show_segments and result["segments"]:
            overlay = vis_image.copy()
            for seg in result["segments"]:
                if "mask_base64" not in seg:
                    continue
                import base64
                mask_bytes = base64.b64decode(seg["mask_base64"])
                mask = cv2.imdecode(np.frombuffer(mask_bytes, np.uint8), cv2.IMREAD_GRAYSCALE)

                if mask is not None and mask.shape == image.shape[:2]:
                    color = random_color()
                    # Draw mask on overlay
                    overlay[mask > 0] = color

            # Semi-transparent blend
            cv2.addWeighted(vis_image, 0.7, overlay, 0.3, 0, vis_image)

        # 2. Draw detection bounding boxes
        if show_detections and result["objects"]:
            for obj in result["objects"]:
                bbox = obj["bbox"]
                label = obj["label"]
                conf = obj["confidence"]
                x1, y1, x2, y2 = map(int, bbox)

                color = random_color()
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), color, 2)

                # Label text
                label_text = f"{label} {conf:.2f}"
                (text_w, text_h), _ = cv2.getTextSize(
                    label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                )
                cv2.rectangle(vis_image, (x1, y1 - text_h - 8),
                             (x1 + text_w + 8, y1), color, -1)
                cv2.putText(vis_image, label_text, (x1 + 4, y1 - 4),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # 3. Draw OCR texts
        if show_ocr and result["text_regions"]:
            for text_region in result["text_regions"]:
                bbox = text_region["bbox"]
                text = text_region["text"]
                conf = text_region.get("confidence", 0)
                x1, y1, x2, y2 = map(int, bbox)

                # Yellow bounding box (for OCR)
                cv2.rectangle(vis_image, (x1, y1), (x2, y2), (0, 255, 255), 2)

                # Write text on top
                (text_w, text_h), _ = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
                )
                cv2.rectangle(vis_image, (x1, y1 - text_h - 6),
                             (x1 + text_w + 4, y1), (0, 255, 255), -1)
                cv2.putText(vis_image, text, (x1 + 2, y1 - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        # Save
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        cv2.imwrite(output_path, vis_image)
        return output_path

    def analyze_and_save(
        self,
        image_path: str,
        output_path: str,
        include_masks: bool = False,
        save_image: bool = True,
        image_output_path: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Analyze and save.

        Args:
            image_path: Image file path
            output_path: JSON output file path
            include_masks: Include mask data in JSON
            save_image: Save visual output
            image_output_path: Visual output file path (if None, automatic)
        """
        result = self.analyze(image_path, **kwargs)

        # mask_base64 takes up a lot of space, optionally remove it
        if not include_masks:
            for seg in result["segments"]:
                seg.pop("mask_base64", None)

        # Save JSON
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"JSON saved to {output_path}")

        # Visual output
        if save_image:
            if image_output_path is None:
                # Derive from JSON path
                base = os.path.splitext(output_path)[0]
                image_output_path = f"{base}_annotated.jpg"

            # Load masks temporarily for segment visualization
            if include_masks:
                # Already exists in result from analyze
                pass
            else:
                # Recalculate masks (for visualization)
                image = self.preprocessor.load_image(image_path)
                image = self.preprocessor.resize(image, max_size=kwargs.get("max_size", 1280))
                if kwargs.get("run_segmentation", True):
                    segments = self.segmentor.segment(image)
                    result["segments"] = segments

            self.visualize(image_path, result, image_output_path)
            print(f"Annotated image saved to {image_output_path}")

        return result
