#!/usr/bin/env python3
"""
Fine-tuning Lab CLI — YOLOv8 CV Fine-tuning + LLM LoRA/QLoRA

Usage:
    # CV Fine-tuning
    python run_finetuning.py cv-dataset                    # Generate synthetic dataset
    python run_finetuning.py cv-train                      # Train YOLOv8 on custom data
    python run_finetuning.py cv-train --freeze 10          # Freeze backbone (transfer learning)
    python run_finetuning.py cv-eval                       # Evaluate on test set
    python run_finetuning.py cv-compare                    # Compare transfer learning strategies
    python run_finetuning.py cv-predict -i image.jpg       # Predict with fine-tuned model

    # LLM Fine-tuning
    python run_finetuning.py llm-dataset                   # Generate instruction dataset
    python run_finetuning.py llm-train                     # LoRA fine-tune
    python run_finetuning.py llm-generate -q "query"       # Generate with fine-tuned model

    # Combined
    python run_finetuning.py report                        # Full evaluation report
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


def cmd_cv_dataset(args):
    """Generate synthetic CV dataset."""
    from src.cv_dataset import CVDatasetBuilder

    builder = CVDatasetBuilder(data_dir=args.data_dir)
    stats = builder.generate_synthetic_dataset(
        num_train=args.num_train,
        num_val=args.num_val,
        num_test=args.num_test,
        img_size=args.img_size,
    )

    # Show stats
    full_stats = builder.get_dataset_stats()
    print(f"\nDataset statistics:")
    for split, info in full_stats.items():
        print(f"  {split}: {info['images']} images, {info['total_objects']} objects "
              f"(avg {info['avg_objects_per_image']:.1f}/img)")

    # Save stats
    stats_path = Path(args.data_dir) / "dataset_stats.json"
    with open(stats_path, "w") as f:
        json.dump(full_stats, f, indent=2)
    print(f"\nStats saved: {stats_path}")


def cmd_cv_train(args):
    """Fine-tune YOLOv8 on custom dataset."""
    from src.cv_finetuning import YOLOFineTuner

    tuner = YOLOFineTuner(config_path=args.config)
    tuner.load_model(pretrained=args.model)

    data_yaml = str(Path(args.data_dir) / "data.yaml")
    metrics = tuner.train(
        data_yaml=data_yaml,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze=args.freeze,
        project=args.output,
        name=args.name,
    )

    print(f"\nTraining results:")
    print(json.dumps(metrics, indent=2))

    # Save metrics
    metrics_path = Path(args.output) / args.name / "training_metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def cmd_cv_eval(args):
    """Evaluate fine-tuned YOLOv8 model."""
    from src.cv_finetuning import YOLOFineTuner

    tuner = YOLOFineTuner(config_path=args.config)
    model_path = args.model or str(Path(args.output) / args.name / "weights" / "best.pt")
    tuner.load_model(pretrained=model_path)

    data_yaml = str(Path(args.data_dir) / "data.yaml")
    metrics = tuner.evaluate(data_yaml=data_yaml, split="test")

    print(f"\nEvaluation results:")
    print(json.dumps(metrics, indent=2))


def cmd_cv_compare(args):
    """Compare transfer learning strategies."""
    from src.cv_finetuning import YOLOFineTuner

    tuner = YOLOFineTuner(config_path=args.config)
    data_yaml = str(Path(args.data_dir) / "data.yaml")

    comparison = tuner.compare_strategies(
        data_yaml=data_yaml,
        epochs=args.epochs or 10,
    )

    # Save comparison
    compare_path = Path(args.output) / "strategy_comparison.json"
    with open(compare_path, "w") as f:
        json.dump(comparison, f, indent=2, default=str)
    print(f"\nComparison saved: {compare_path}")


def cmd_cv_predict(args):
    """Run inference with fine-tuned model."""
    from src.cv_finetuning import YOLOFineTuner

    tuner = YOLOFineTuner(config_path=args.config)
    model_path = args.model or str(Path(args.output) / args.name / "weights" / "best.pt")
    tuner.load_model(pretrained=model_path)

    detections = tuner.predict(args.image, conf=args.confidence)
    print(f"\nDetections ({len(detections)}):")
    for d in detections:
        print(f"  {d['label']}: {d['confidence']:.2f} bbox={d['bbox']}")


def cmd_llm_dataset(args):
    """Generate LLM instruction dataset."""
    from src.llm_dataset import LLMDatasetBuilder

    builder = LLMDatasetBuilder(output_dir=args.data_dir)
    stats = builder.generate_dataset(
        num_samples=args.num_samples,
        output_file=args.output_file,
    )
    print(f"\nDataset stats:")
    print(json.dumps(stats, indent=2))

    # Also split into train/val
    builder.split_dataset()


def cmd_llm_train(args):
    """Fine-tune LLM with LoRA."""
    from src.llm_finetuning import LLMFineTuner

    tuner = LLMFineTuner(config_path=args.config)
    tuner.load_model(model_name=args.model)
    tuner.apply_lora()

    data_path = args.data_path or str(Path(args.data_dir) / "visual_reasoning_instructions.json")
    train_dataset, eval_dataset = tuner.prepare_dataset(data_path)

    metrics = tuner.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        output_dir=str(Path(args.output) / "llm_finetuned"),
    )

    tuner.save_model(str(Path(args.output) / "llm_finetuned"))

    print(f"\nTraining metrics:")
    print(json.dumps(metrics, indent=2))

    # Save metrics
    metrics_path = Path(args.output) / "llm_finetuned" / "training_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)


def cmd_llm_generate(args):
    """Generate with fine-tuned LLM."""
    from src.llm_finetuning import LLMFineTuner

    tuner = LLMFineTuner(config_path=args.config)
    adapter_path = args.adapter or str(Path(args.output) / "llm_finetuned")
    tuner.load_finetuned(adapter_path, base_model=args.model)

    response = tuner.generate(args.query, max_new_tokens=args.max_tokens)
    print(f"\nQuery: {args.query}")
    print(f"\nResponse: {response}")


def cmd_report(args):
    """Generate comprehensive fine-tuning report."""
    from src.evaluation import FinetuningEvaluator

    evaluator = FinetuningEvaluator(output_dir=args.output)

    # Check for CV results
    cv_results_dir = Path(args.output) / args.name
    cv_metrics = None
    if cv_results_dir.exists():
        cv_metrics = evaluator.summarize_cv_results(str(cv_results_dir))
        evaluator.plot_cv_training(str(cv_results_dir))

    # Check for LLM results
    llm_metrics_path = Path(args.output) / "llm_finetuned" / "training_metrics.json"
    llm_metrics = None
    if llm_metrics_path.exists():
        with open(llm_metrics_path) as f:
            llm_metrics = json.load(f)

    evaluator.generate_report(cv_metrics=cv_metrics, llm_metrics=llm_metrics)


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tuning Lab — CV & LLM Fine-tuning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Global args
    parser.add_argument("--config", default=None, help="Config file path")
    parser.add_argument("--output", "-o", default="output", help="Output directory")
    parser.add_argument("--data-dir", default=None, help="Data directory")
    parser.add_argument("--name", default="safety_detection", help="Experiment name")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # CV Dataset
    p_cv_data = subparsers.add_parser("cv-dataset", help="Generate CV dataset")
    p_cv_data.add_argument("--num-train", type=int, default=100, help="Training images")
    p_cv_data.add_argument("--num-val", type=int, default=20, help="Validation images")
    p_cv_data.add_argument("--num-test", type=int, default=20, help="Test images")
    p_cv_data.add_argument("--img-size", type=int, default=640, help="Image size")

    # CV Train
    p_cv_train = subparsers.add_parser("cv-train", help="Fine-tune YOLOv8")
    p_cv_train.add_argument("--model", default=None, help="Pretrained model path")
    p_cv_train.add_argument("--epochs", type=int, default=None, help="Training epochs")
    p_cv_train.add_argument("--batch-size", type=int, default=None, help="Batch size")
    p_cv_train.add_argument("--freeze", type=int, default=None, help="Freeze N layers")

    # CV Eval
    p_cv_eval = subparsers.add_parser("cv-eval", help="Evaluate CV model")
    p_cv_eval.add_argument("--model", default=None, help="Model path")

    # CV Compare
    p_cv_compare = subparsers.add_parser("cv-compare", help="Compare strategies")
    p_cv_compare.add_argument("--epochs", type=int, default=10, help="Epochs per strategy")

    # CV Predict
    p_cv_pred = subparsers.add_parser("cv-predict", help="Predict with model")
    p_cv_pred.add_argument("-i", "--image", required=True, help="Image path")
    p_cv_pred.add_argument("--model", default=None, help="Model path")
    p_cv_pred.add_argument("--confidence", type=float, default=0.25, help="Conf threshold")

    # LLM Dataset
    p_llm_data = subparsers.add_parser("llm-dataset", help="Generate LLM dataset")
    p_llm_data.add_argument("--num-samples", type=int, default=200, help="Number of samples")
    p_llm_data.add_argument("--output-file", default="visual_reasoning_instructions.json")

    # LLM Train
    p_llm_train = subparsers.add_parser("llm-train", help="LoRA fine-tune LLM")
    p_llm_train.add_argument("--model", default=None, help="Base model name")
    p_llm_train.add_argument("--data-path", default=None, help="Dataset path")

    # LLM Generate
    p_llm_gen = subparsers.add_parser("llm-generate", help="Generate with fine-tuned model")
    p_llm_gen.add_argument("-q", "--query", required=True, help="Query text")
    p_llm_gen.add_argument("--adapter", default=None, help="Adapter path")
    p_llm_gen.add_argument("--model", default=None, help="Base model name")
    p_llm_gen.add_argument("--max-tokens", type=int, default=256, help="Max new tokens")

    # Report
    subparsers.add_parser("report", help="Generate evaluation report")

    args = parser.parse_args()

    # Set defaults based on command type
    if args.command and args.command.startswith("cv"):
        args.config = args.config or "configs/cv_config.yaml"
        args.data_dir = args.data_dir or "data/cv/safety"
    elif args.command and args.command.startswith("llm"):
        args.config = args.config or "configs/llm_config.yaml"
        args.data_dir = args.data_dir or "data/llm"

    if not args.command:
        parser.print_help()
        return

    # Dispatch
    commands = {
        "cv-dataset": cmd_cv_dataset,
        "cv-train": cmd_cv_train,
        "cv-eval": cmd_cv_eval,
        "cv-compare": cmd_cv_compare,
        "cv-predict": cmd_cv_predict,
        "llm-dataset": cmd_llm_dataset,
        "llm-train": cmd_llm_train,
        "llm-generate": cmd_llm_generate,
        "report": cmd_report,
    }

    if args.command in commands:
        commands[args.command](args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
