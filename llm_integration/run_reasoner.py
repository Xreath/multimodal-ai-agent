"""
Visual Reasoner — CLI Entry Point

Usage:
    # Single turn analysis
    python run_reasoner.py --image data/bus.jpg --question "What is in this scene?"

    # Tool calling analysis
    python run_reasoner.py --image data/bus.jpg --question "How many people are there?" --use-tools

    # Multi-modal comparison
    python run_reasoner.py --image data/bus.jpg --question "What do you see?" --compare

    # Safety inspection
    python run_reasoner.py --image data/bus.jpg --safety

    # Interactive mode (multi-turn)
    python run_reasoner.py --image data/bus.jpg --interactive

    # Select provider
    python run_reasoner.py --image data/bus.jpg --question "..." --provider anthropic --model claude-sonnet-4-20250514

    # Select prompt strategy
    python run_reasoner.py --image data/bus.jpg --question "..." --strategy few_shot

    # Use pre-computed CV result
    python run_reasoner.py --cv-result output/result.json --question "What is in this scene?"
"""

import argparse
import json
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

from src.visual_reasoner import VisualReasoner


def main():
    parser = argparse.ArgumentParser(
        description="Visual Reasoner — CV + LLM Integration",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    # Input
    parser.add_argument("--image", "-i", type=str, help="Image file path")
    parser.add_argument("--cv-result", type=str, help="Pre-computed CV result (JSON)")
    parser.add_argument("--question", "-q", type=str, help="User question")

    # LLM Settings
    parser.add_argument("--provider", "-p", type=str, default="deepseek",
                        choices=["deepseek", "openai", "anthropic", "gemini"],
                        help="LLM provider (default: deepseek)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model name (default: dependent on provider)")
    parser.add_argument("--strategy", "-s", type=str, default="cot",
                        choices=["direct", "few_shot", "cot"],
                        help="Prompt strategy (default: cot)")

    # Modes
    parser.add_argument("--use-tools", action="store_true", help="Enable tool calling mode")
    parser.add_argument("--compare", action="store_true", help="Compare with multi-modal LLM")
    parser.add_argument("--safety", action="store_true", help="Safety inspection mode")
    parser.add_argument("--interactive", action="store_true", help="Interactive (multi-turn) mode")

    # Output
    parser.add_argument("--output", "-o", type=str, default=None, help="JSON output file")

    args = parser.parse_args()

    # Validation
    if not args.image and not args.cv_result:
        parser.error("--image or --cv-result required")
    if not args.question and not args.safety and not args.interactive:
        parser.error("--question, --safety or --interactive required")

    # Load CV result (if provided)
    cv_result = None
    if args.cv_result:
        with open(args.cv_result, "r") as f:
            cv_result = json.load(f)
        print(f"[Info] CV result loaded from {args.cv_result}")

    # Create reasoner
    reasoner = VisualReasoner(
        provider=args.provider,
        model=args.model,
        prompt_strategy=args.strategy
    )

    print(f"\n{'='*60}")
    print(f"  Visual Reasoner")
    print(f"  Provider: {args.provider} / {reasoner.llm.model_name}")
    print(f"  Strategy: {args.strategy}")
    print(f"{'='*60}\n")

    result = None

    if args.safety:
        # Safety inspection
        result = reasoner.safety_inspection(args.image, cv_result=cv_result)
        _print_safety_result(result)

    elif args.compare:
        # Multi-modal comparison
        result = reasoner.compare_with_multimodal(args.image, args.question)
        _print_comparison_result(result)

    elif args.interactive:
        # Interactive mode
        _run_interactive(reasoner, args.image, cv_result)
        return  # interactive mode manages its own output

    elif args.use_tools:
        # Tool calling
        result = reasoner.analyze_with_tools(args.image, args.question)
        _print_analysis_result(result)

    else:
        # Standard analysis
        result = reasoner.analyze(args.image, args.question, cv_result=cv_result)
        _print_analysis_result(result)

    # Save JSON output
    if result and args.output:
        # cv_result can be large, save separately
        save_result = {k: v for k, v in result.items() if k != "cv_result"}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] {args.output}")


def _print_analysis_result(result: dict):
    """Format and print analysis result."""
    print("\n" + "─" * 60)
    print("ANSWER:")
    print("─" * 60)
    print(result.get("answer", "No answer"))

    steps = result.get("reasoning_steps", [])
    if steps:
        print("\nREASONING:")
        for step in steps:
            print(f"  {step}")

    evidence = result.get("evidence", [])
    if evidence:
        print("\nEVIDENCE:")
        for ev in evidence:
            print(f"  - {ev}")

    conf = result.get("confidence", "?")
    print(f"\nCONFIDENCE: {conf}")

    follow_up = result.get("follow_up_questions", [])
    if follow_up:
        print("\nSUGGESTED FOLLOW-UPS:")
        for q in follow_up:
            print(f"  ? {q}")

    print("─" * 60)


def _print_safety_result(result: dict):
    """Print safety inspection result."""
    print("\n" + "─" * 60)
    print("SAFETY INSPECTION REPORT")
    print("─" * 60)

    violations = result.get("violations", [])
    if violations:
        for i, v in enumerate(violations, 1):
            severity = v.get("severity", "?").upper()
            print(f"\n  [{severity}] Violation #{i}: {v.get('description', '?')}")
            if v.get("affected_objects"):
                print(f"    Objects: {', '.join(v['affected_objects'])}")
            print(f"    Recommendation: {v.get('recommendation', '?')}")
    else:
        print("\n  No safety violations detected.")

    print(f"\n  Overall Risk: {result.get('overall_risk', '?').upper()}")
    print(f"  Summary: {result.get('summary', '?')}")
    print("─" * 60)


def _print_comparison_result(result: dict):
    """Print comparison result."""
    print("\n" + "─" * 60)
    print("CV PIPELINE vs MULTI-MODAL LLM COMPARISON")
    print("─" * 60)

    print("\n[Pipeline Analysis]")
    _print_analysis_result(result["pipeline_analysis"])

    print("\n[Multi-modal LLM Analysis]")
    print(result["multimodal_analysis"])

    print("\n[Comparison]")
    print(result["comparison"])
    print("─" * 60)


def _run_interactive(reasoner: VisualReasoner, image_path: str, cv_result=None):
    """Interactive multi-turn conversation mode."""
    print("Interactive mode started. Type 'quit' to exit.\n")

    # First question
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        return

    result = reasoner.analyze(image_path, question, cv_result=cv_result)
    _print_analysis_result(result)

    # Follow-up questions
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Conversation ended.")
            break

        result = reasoner.follow_up(question)
        _print_analysis_result(result)


if __name__ == "__main__":
    main()
