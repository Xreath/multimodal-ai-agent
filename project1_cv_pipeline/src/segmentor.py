"""
Instance Segmentation Module — YOLOv8-seg
Nesnelerin piksel düzeyinde mask'larını çıkarır.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import base64
from typing import List, Dict, Any


class InstanceSegmentor:
    """YOLOv8-seg tabanlı instance segmentation modülü."""

    def __init__(self, model_name: str = "yolov8n-seg.pt", confidence: float = 0.25):
        """
        Args:
            model_name: YOLOv8-seg model dosyası
            confidence: Minimum confidence threshold
        """
        self.model = YOLO(model_name)
        self.confidence = confidence

    def segment(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Instance segmentation yapar.

        Args:
            image: BGR formatında OpenCV görüntüsü

        Returns:
            List of segments: [{"label", "confidence", "bbox", "mask_base64", "area_pixels"}]
        """
        results = self.model(image, conf=self.confidence, verbose=False)

        segments = []
        for result in results:
            if result.masks is None:
                continue

            boxes = result.boxes
            masks = result.masks

            for i, (box, mask) in enumerate(zip(boxes, masks)):
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]

                # Mask'ı binary numpy array olarak al
                mask_array = mask.data[0].cpu().numpy()
                # Orijinal görüntü boyutuna resize et
                mask_resized = cv2.resize(
                    mask_array.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )

                # Mask'ı base64 encode et (PNG olarak)
                _, mask_png = cv2.imencode(".png", mask_resized * 255)
                mask_b64 = base64.b64encode(mask_png).decode("utf-8")

                area = int(np.sum(mask_resized > 0))

                segments.append({
                    "label": label,
                    "confidence": round(conf, 4),
                    "bbox": [round(x1), round(y1), round(x2), round(y2)],
                    "mask_base64": mask_b64,
                    "area_pixels": area
                })

        return segments

    def segment_and_draw(self, image: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Segmentation yap ve görüntü üzerine overlay çiz."""
        results = self.model(image, conf=self.confidence, verbose=False)

        segments = self.segment(image)
        annotated = image.copy()

        for result in results:
            if result.masks is None:
                continue
            for mask in result.masks:
                mask_array = mask.data[0].cpu().numpy()
                mask_resized = cv2.resize(
                    mask_array.astype(np.uint8),
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_NEAREST
                )
                # Rastgele renk ile overlay
                color = np.random.randint(0, 255, 3).tolist()
                overlay = annotated.copy()
                overlay[mask_resized > 0] = color
                annotated = cv2.addWeighted(annotated, 0.7, overlay, 0.3, 0)

        return annotated, segments
