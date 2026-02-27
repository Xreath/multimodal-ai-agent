"""
OCR Module — EasyOCR
Detects and reads text regions in an image.
"""

import easyocr
import cv2
import numpy as np
from typing import List, Dict, Any


class OCREngine:
    """EasyOCR-based text extraction module."""

    def __init__(self, languages: List[str] = None):
        """
        Args:
            languages: List of languages to recognize. Default: ['en']
        """
        if languages is None:
            languages = ["en"]
        self.reader = easyocr.Reader(languages, gpu=True)

    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts text regions from the image.

        Args:
            image: OpenCV image in BGR format

        Returns:
            List of text regions: [{"text": str, "confidence": float, "bbox": [x1, y1, x2, y2]}]
        """
        # EasyOCR expects RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.reader.readtext(rgb_image)

        text_regions = []
        for (bbox_points, text, confidence) in results:
            # EasyOCR returns in [[x1,y1],[x2,y2],[x3,y3],[x4,y4]] format
            # Convert this to [x_min, y_min, x_max, y_max] format
            pts = np.array(bbox_points)
            x_min = int(pts[:, 0].min())
            y_min = int(pts[:, 1].min())
            x_max = int(pts[:, 0].max())
            y_max = int(pts[:, 1].max())

            text_regions.append({
                "text": text,
                "confidence": round(float(confidence), 4),
                "bbox": [x_min, y_min, x_max, y_max]
            })

        return text_regions

    def extract_and_draw(self, image: np.ndarray) -> tuple[np.ndarray, List[Dict]]:
        """Detect text and draw on the image."""
        text_regions = self.extract_text(image)
        annotated = image.copy()

        for region in text_regions:
            x1, y1, x2, y2 = region["bbox"]
            label = f"{region['text']} ({region['confidence']:.2f})"

            cv2.rectangle(annotated, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cv2.putText(annotated, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

        return annotated, text_regions
