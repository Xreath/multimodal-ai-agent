"""
Object Detection Module — YOLOv8
Detects objects in an image, returns bbox and confidence.
"""

from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Any


class ObjectDetector:
    """YOLOv8-based object detection module."""

    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.25):
        """
        Args:
            model_name: YOLOv8 model file (n/s/m/l/x)
            confidence: Minimum confidence threshold
        """
        self.model = YOLO(model_name)
        self.confidence = confidence

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detects objects in the image.

        Args:
            image: OpenCV image in BGR format

        Returns:
            List of detections: [{"label": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]
        """
        results = self.model(image, conf=self.confidence, verbose=False)

        detections = []
        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                detections.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)]
                })

        return detections

    def detect_and_draw(self, image: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Perform detection and draw on the image."""
        detections = self.detect(image)
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            label = f"{det['label']} {det['confidence']:.2f}"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return annotated, detections
