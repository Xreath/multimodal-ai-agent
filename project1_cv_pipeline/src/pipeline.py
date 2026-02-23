"""
Visual Perception Engine — Ana Pipeline
Tüm CV modüllerini birleştirerek tek bir structured JSON çıktı üretir.
"""

import cv2
import json
import time
import numpy as np
from typing import Dict, Any, Optional

from .preprocessor import ImagePreprocessor
from .detector import ObjectDetector
from .segmentor import InstanceSegmentor
from .ocr_engine import OCREngine


class VisualPerceptionPipeline:
    """
    Ana CV pipeline: Detection + Segmentation + OCR → Structured JSON

    Kullanım:
        pipeline = VisualPerceptionPipeline()
        result = pipeline.analyze("image.jpg")
    """

    def __init__(
        self,
        detection_model: str = "yolov8n.pt",
        segmentation_model: str = "yolov8n-seg.pt",
        ocr_languages: list = None,
        confidence: float = 0.25
    ):
        self.preprocessor = ImagePreprocessor()
        self.detector = ObjectDetector(model_name=detection_model, confidence=confidence)
        self.segmentor = InstanceSegmentor(model_name=segmentation_model, confidence=confidence)
        self.ocr_engine = OCREngine(languages=ocr_languages)
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
        Tam analiz pipeline'ı çalıştırır.

        Args:
            image_path: Görüntü dosya yolu
            run_detection: Object detection çalıştır
            run_segmentation: Instance segmentation çalıştır
            run_ocr: OCR çalıştır
            max_size: Görüntü max boyutu

        Returns:
            Structured JSON dict
        """
        start_time = time.time()

        # 1. Görüntüyü yükle ve ön-işle
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
            # mask_base64 çok büyük olabilir — opsiyonel olarak kaldır
            result["segments"] = segments
            result["processing_time"]["segmentation"] = round(time.time() - t, 3)

        # 4. OCR
        if run_ocr:
            t = time.time()
            result["text_regions"] = self.ocr_engine.extract_text(image)
            result["processing_time"]["ocr"] = round(time.time() - t, 3)

        # 5. Basit sahne açıklaması oluştur
        result["scene_description"] = self._generate_scene_description(result)

        result["processing_time"]["total"] = round(time.time() - start_time, 3)

        return result

    def _generate_scene_description(self, result: Dict) -> str:
        """Detection + OCR sonuçlarından basit bir sahne açıklaması üret."""
        parts = []

        # Object sayımı
        if result["objects"]:
            label_counts = {}
            for obj in result["objects"]:
                label = obj["label"]
                label_counts[label] = label_counts.get(label, 0) + 1

            obj_parts = [f"{count} {label}{'s' if count > 1 else ''}"
                         for label, count in label_counts.items()]
            parts.append(f"Detected: {', '.join(obj_parts)}")

        # Text bilgisi
        if result["text_regions"]:
            texts = [r["text"] for r in result["text_regions"][:5]]  # İlk 5 text
            parts.append(f"Text found: {', '.join(texts)}")

        # Segment sayısı
        if result["segments"]:
            parts.append(f"{len(result['segments'])} segmented regions")

        return ". ".join(parts) if parts else "No significant content detected"

    def analyze_and_save(
        self,
        image_path: str,
        output_path: str,
        include_masks: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """Analiz yap ve JSON olarak kaydet."""
        result = self.analyze(image_path, **kwargs)

        # mask_base64 çok yer kaplar, opsiyonel olarak kaldır
        if not include_masks:
            for seg in result["segments"]:
                seg.pop("mask_base64", None)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

        print(f"Results saved to {output_path}")
        return result
