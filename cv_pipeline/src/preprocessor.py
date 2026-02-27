"""
Image Preprocessing Module — OpenCV
Preprocessing operations to prepare image for the pipeline.
"""

import cv2
import numpy as np
from typing import Optional, Tuple


class ImagePreprocessor:
    """OpenCV-based image preprocessing module."""

    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load image from disk."""
        image = cv2.imread(path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {path}")
        return image

    @staticmethod
    def resize(image: np.ndarray, max_size: int = 1280) -> np.ndarray:
        """
        Resize to maximum dimension while maintaining aspect ratio.
        Usually 640 or 1280 is used for YOLO.
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_size:
            return image

        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def enhance_for_ocr(image: np.ndarray) -> np.ndarray:
        """
        Enhance image for OCR:
        - Grayscale conversion
        - Adaptive thresholding
        - Noise reduction
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Contrast enhancement with CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Noise reduction
        denoised = cv2.fastNlMeansDenoising(enhanced, h=10)
        return denoised

    @staticmethod
    def crop_region(image: np.ndarray, bbox: list) -> np.ndarray:
        """Crop region with bbox [x1, y1, x2, y2]."""
        x1, y1, x2, y2 = [int(c) for c in bbox]
        h, w = image.shape[:2]
        # Boundary check
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        return image[y1:y2, x1:x2]

    @staticmethod
    def get_image_info(image: np.ndarray) -> dict:
        """Return basic information about the image."""
        h, w = image.shape[:2]
        channels = image.shape[2] if len(image.shape) == 3 else 1
        return {
            "width": w,
            "height": h,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_mb": round(image.nbytes / (1024 * 1024), 2)
        }
