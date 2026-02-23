"""
Visual Reasoner — CLI Entry Point

Kullanım:
    # Tek seferlik analiz
    python run_reasoner.py --image data/bus.jpg --question "Bu sahnede ne var?"

    # Tool calling ile analiz
    python run_reasoner.py --image data/bus.jpg --question "Kaç kişi var?" --use-tools

    # Multi-modal karşılaştırma
    python run_reasoner.py --image data/bus.jpg --question "Ne görüyorsun?" --compare

    # Güvenlik denetimi
    python run_reasoner.py --image data/bus.jpg --safety

    # İnteraktif mod (multi-turn)
    python run_reasoner.py --image data/bus.jpg --interactive

    # Provider seçimi
    python run_reasoner.py --image data/bus.jpg --question "..." --provider anthropic --model claude-sonnet-4-20250514

    # Prompt stratejisi seçimi
    python run_reasoner.py --image data/bus.jpg --question "..." --strategy few_shot

    # Önceden hesaplanmış CV sonucu kullan
    python run_reasoner.py --cv-result output/result.json --question "Bu sahnede ne var?"
"""

import argparse
import json
import os
import sys

# Proje root'unu path'e ekle
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

    # Girdi
    parser.add_argument("--image", "-i", type=str, help="Görüntü dosya yolu")
    parser.add_argument("--cv-result", type=str, help="Önceden hesaplanmış CV sonucu (JSON)")
    parser.add_argument("--question", "-q", type=str, help="Kullanıcı sorusu")

    # LLM ayarları
    parser.add_argument("--provider", "-p", type=str, default="deepseek",
                        choices=["deepseek", "openai", "anthropic", "gemini"],
                        help="LLM provider (default: deepseek)")
    parser.add_argument("--model", "-m", type=str, default=None,
                        help="Model adı (default: provider'a göre)")
    parser.add_argument("--strategy", "-s", type=str, default="cot",
                        choices=["direct", "few_shot", "cot"],
                        help="Prompt stratejisi (default: cot)")

    # Modlar
    parser.add_argument("--use-tools", action="store_true", help="Tool calling modunu aktifleştir")
    parser.add_argument("--compare", action="store_true", help="Multi-modal LLM ile karşılaştır")
    parser.add_argument("--safety", action="store_true", help="Güvenlik denetimi modu")
    parser.add_argument("--interactive", action="store_true", help="İnteraktif (multi-turn) mod")

    # Çıktı
    parser.add_argument("--output", "-o", type=str, default=None, help="JSON çıktı dosyası")

    args = parser.parse_args()

    # Validasyon
    if not args.image and not args.cv_result:
        parser.error("--image veya --cv-result gerekli")
    if not args.question and not args.safety and not args.interactive:
        parser.error("--question, --safety veya --interactive gerekli")

    # CV sonucu yükle (verilmişse)
    cv_result = None
    if args.cv_result:
        with open(args.cv_result, "r") as f:
            cv_result = json.load(f)
        print(f"[Info] CV result loaded from {args.cv_result}")

    # Reasoner oluştur
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
        # Güvenlik denetimi
        result = reasoner.safety_inspection(args.image, cv_result=cv_result)
        _print_safety_result(result)

    elif args.compare:
        # Multi-modal karşılaştırma
        result = reasoner.compare_with_multimodal(args.image, args.question)
        _print_comparison_result(result)

    elif args.interactive:
        # İnteraktif mod
        _run_interactive(reasoner, args.image, cv_result)
        return  # interactive mod kendi çıktısını yönetir

    elif args.use_tools:
        # Tool calling
        result = reasoner.analyze_with_tools(args.image, args.question)
        _print_analysis_result(result)

    else:
        # Standart analiz
        result = reasoner.analyze(args.image, args.question, cv_result=cv_result)
        _print_analysis_result(result)

    # JSON çıktı kaydet
    if result and args.output:
        # cv_result büyük olabilir, ayrı kaydet
        save_result = {k: v for k, v in result.items() if k != "cv_result"}
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(save_result, f, indent=2, ensure_ascii=False)
        print(f"\n[Saved] {args.output}")


def _print_analysis_result(result: dict):
    """Analiz sonucunu formatla ve yazdır."""
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
    """Güvenlik denetimi sonucunu yazdır."""
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
    """Karşılaştırma sonucunu yazdır."""
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
    """İnteraktif multi-turn conversation modu."""
    print("Interactive mode started. Type 'quit' to exit.\n")

    # İlk soru
    question = input("You: ").strip()
    if question.lower() in ("quit", "exit", "q"):
        return

    result = reasoner.analyze(image_path, question, cv_result=cv_result)
    _print_analysis_result(result)

    # Takip soruları
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            print("Conversation ended.")
            break

        result = reasoner.follow_up(question)
        _print_analysis_result(result)


if __name__ == "__main__":
    main()
