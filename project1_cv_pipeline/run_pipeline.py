"""
CV Pipeline — Çalıştırma Scripti
Kullanım: python run_pipeline.py <image_path> [--output output.json]
"""

import argparse
import json
import sys
import os

# src modülünü import edebilmek için path ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import VisualPerceptionPipeline


def main():
    parser = argparse.ArgumentParser(description="Visual Perception Engine — CV Pipeline")
    parser.add_argument("image", help="Analiz edilecek görüntü dosya yolu")
    parser.add_argument("--output", "-o", default=None, help="JSON çıktı dosya yolu")
    parser.add_argument("--no-detection", action="store_true", help="Object detection'ı atla")
    parser.add_argument("--no-segmentation", action="store_true", help="Segmentation'ı atla")
    parser.add_argument("--no-ocr", action="store_true", help="OCR'ı atla")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--include-masks", action="store_true", help="Mask base64 verilerini JSON'a dahil et")
    parser.add_argument("--max-size", type=int, default=1280, help="Maks görüntü boyutu (default: 1280)")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"HATA: Görüntü bulunamadı: {args.image}")
        sys.exit(1)

    print(f"Analiz ediliyor: {args.image}")
    print(f"Confidence: {args.confidence}")
    print("-" * 50)

    pipeline = VisualPerceptionPipeline(confidence=args.confidence)

    if args.output:
        result = pipeline.analyze_and_save(
            args.image,
            args.output,
            include_masks=args.include_masks,
            run_detection=not args.no_detection,
            run_segmentation=not args.no_segmentation,
            run_ocr=not args.no_ocr,
            max_size=args.max_size
        )
    else:
        result = pipeline.analyze(
            args.image,
            run_detection=not args.no_detection,
            run_segmentation=not args.no_segmentation,
            run_ocr=not args.no_ocr,
            max_size=args.max_size
        )

    # Sonuçları yazdır (mask'lar hariç — çok büyük)
    display_result = result.copy()
    for seg in display_result.get("segments", []):
        seg.pop("mask_base64", None)

    print("\n=== SONUÇLAR ===")
    print(json.dumps(display_result, indent=2, ensure_ascii=False))
    print(f"\nToplam süre: {result['processing_time']['total']:.3f}s")
    print(f"Tespit edilen nesne: {len(result['objects'])}")
    print(f"Segment sayısı: {len(result['segments'])}")
    print(f"Metin bölgesi: {len(result['text_regions'])}")


if __name__ == "__main__":
    main()
