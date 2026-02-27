"""
CV Pipeline — Execution Script
Usage: python run_pipeline.py <image_path> [--output output.json]
"""

import argparse
import json
import sys
import os

# Add path to import src module
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import VisualPerceptionPipeline


def main():
    parser = argparse.ArgumentParser(description="Visual Perception Engine — CV Pipeline")
    parser.add_argument("image", help="Path to the image to be analyzed")
    parser.add_argument("--output", "-o", default=None, help="JSON output file path")
    parser.add_argument("--image-output", "-i", default=None, help="Visual output file path (annotated image)")
    parser.add_argument("--no-detection", action="store_true", help="Skip Object detection")
    parser.add_argument("--no-segmentation", action="store_true", help="Skip Segmentation")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR")
    parser.add_argument("--confidence", type=float, default=0.25, help="Confidence threshold (default: 0.25)")
    parser.add_argument("--include-masks", action="store_true", help="Include mask base64 data in JSON")
    parser.add_argument("--max-size", type=int, default=1280, help="Max image size (default: 1280)")
    parser.add_argument("--no-image", action="store_true", help="Do not save visual output")

    args = parser.parse_args()

    if not os.path.exists(args.image):
        print(f"ERROR: Image not found: {args.image}")
        sys.exit(1)

    print(f"Analyzing: {args.image}")
    print(f"Confidence: {args.confidence}")
    print("-" * 50)

    pipeline = VisualPerceptionPipeline(confidence=args.confidence)

    if args.output:
        result = pipeline.analyze_and_save(
            args.image,
            args.output,
            include_masks=args.include_masks,
            save_image=not args.no_image,
            image_output_path=args.image_output,
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

        # If --output is not given but visual output is requested
        if not args.no_image:
            image_out = args.image_output or "output/annotated_image.jpg"
            os.makedirs(os.path.dirname(image_out) or ".", exist_ok=True)
            pipeline.visualize(args.image, result, image_out)
            print(f"Annotated image saved to {image_out}")

    # Print results (excluding masks — they are too large)
    display_result = result.copy()
    for seg in display_result.get("segments", []):
        seg.pop("mask_base64", None)

    print("\n=== RESULTS ===")
    print(json.dumps(display_result, indent=2, ensure_ascii=False))
    print(f"\nTotal time: {result['processing_time']['total']:.3f}s")
    print(f"Objects detected: {len(result['objects'])}")
    print(f"Number of segments: {len(result['segments'])}")
    print(f"Text regions: {len(result['text_regions'])}")


if __name__ == "__main__":
    main()
