"""
Multi-Modal Agent CLI — LangGraph tabanlı agent'ı çalıştırır.

Kullanım:
    # Görüntü analizi (CV pipeline + LLM reasoning)
    python run_agent.py analyze --image ../project1_cv_pipeline/data/bus.jpg --query "Bu sahnede ne var?"

    # Sadece reasoning (görüntüsüz)
    python run_agent.py ask --query "YOLO'nun çalışma prensibi nedir?"

    # Interactive mod (multi-turn konuşma)
    python run_agent.py interactive --image ../project1_cv_pipeline/data/bus.jpg

    # Graph yapısını görselleştir
    python run_agent.py graph

    # Memory demo (vector store ile hatırlama)
    python run_agent.py memory-demo
"""

import argparse
import json
import os
import sys

# Proje path'lerini ekle
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MONOREPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.state import create_initial_state
from src.graph import build_agent_graph, visualize_graph


def run_analyze(args):
    """
    Tek seferlik analiz modu.

    Akış:
    1. State oluştur (query + image)
    2. Graph'ı derle
    3. Çalıştır (planner → router → vision → reasoner → evaluator → respond)
    4. Sonucu göster
    """
    print("\n" + "=" * 70)
    print("🤖 MULTI-MODAL AGENT — Analyze Mode")
    print("=" * 70)

    # Initial state
    state = create_initial_state(
        user_query=args.query,
        image_path=args.image,
        max_iterations=args.max_iter
    )

    # Graph compile & run
    graph = build_agent_graph(with_memory=False)

    print(f"\nSorgu: {args.query}")
    if args.image:
        print(f"Görüntü: {args.image}")
    print(f"Max iteration: {args.max_iter}")
    print("-" * 70)

    # Invoke — graph'ı baştan sona çalıştır
    result = graph.invoke(state)

    # Sonuç
    print("\n" + "=" * 70)
    print("📋 SONUÇ")
    print("=" * 70)
    print(result.get("final_answer", "Cevap üretilemedi."))

    # Detayları kaydet
    if args.output:
        output_data = {
            "query": args.query,
            "image_path": args.image,
            "plan": result.get("plan", []),
            "final_answer": result.get("final_answer"),
            "evaluation_score": result.get("evaluation_score"),
            "evaluation_feedback": result.get("evaluation_feedback"),
            "tool_results": result.get("tool_results", []),
            "iteration_count": result.get("iteration_count"),
            "messages": result.get("messages", []),
        }
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Sonuç kaydedildi: {args.output}")


def run_ask(args):
    """Görüntüsüz soru-cevap modu."""
    print("\n" + "=" * 70)
    print("🤖 MULTI-MODAL AGENT — Ask Mode (no image)")
    print("=" * 70)

    state = create_initial_state(
        user_query=args.query,
        image_path=None,
        max_iterations=args.max_iter
    )

    graph = build_agent_graph(with_memory=False)
    result = graph.invoke(state)

    print("\n" + "=" * 70)
    print("📋 SONUÇ")
    print("=" * 70)
    print(result.get("final_answer", "Cevap üretilemedi."))


def run_interactive(args):
    """
    Interactive mod — multi-turn konuşma.

    Her turda:
    1. Kullanıcıdan soru al
    2. Agent graph'ını çalıştır
    3. Cevabı göster
    4. Memory'e kaydet (long-term)
    5. Tekrarla

    Mülakat notu:
    - Multi-turn state: her turda graph yeniden çalışır
    - Memory persistence: geçmiş analizler hatırlanır
    - 'q' ile çıkış
    """
    from src.memory import MemoryManager

    print("\n" + "=" * 70)
    print("🤖 MULTI-MODAL AGENT — Interactive Mode")
    print("=" * 70)
    print("Komutlar: 'q' → çıkış, 'memory' → hafızadaki kayıtlar")

    if args.image:
        print(f"Görüntü: {args.image}")

    memory = MemoryManager(
        persist_dir=os.path.join(PROJECT_ROOT, "data", "memory")
    )

    turn = 0
    while True:
        turn += 1
        try:
            query = input(f"\n[Turn {turn}] Soru: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("👋 Görüşmek üzere!")
            break
        if query.lower() == "memory":
            print(f"📚 Hafızada {memory.vector_store.count} kayıt var")
            if memory.vector_store.count > 0:
                recent = memory.vector_store.search("son analiz", n_results=3)
                for i, mem in enumerate(recent, 1):
                    print(f"  {i}. {mem['text'][:200]}...")
            continue

        # Memory'den ilgili context al
        relevant_context = memory.get_full_context(query)
        full_query = query
        if relevant_context:
            full_query = f"{query}\n\nGeçmiş bilgi:\n{relevant_context}"

        # Agent çalıştır
        state = create_initial_state(
            user_query=full_query,
            image_path=args.image if turn == 1 or args.image else None,
            max_iterations=args.max_iter
        )

        graph = build_agent_graph(with_memory=False)
        result = graph.invoke(state)

        answer = result.get("final_answer", "Cevap üretilemedi.")
        print(f"\n💬 Cevap:\n{answer}")

        # Memory'e kaydet
        memory.add_conversation_message("user", query)
        memory.add_conversation_message("assistant", answer)
        memory.store_analysis(
            f"Soru: {query}\nCevap: {answer[:500]}",
            metadata={"turn": turn, "has_image": bool(args.image)}
        )


def run_graph_viz(args):
    """Graph yapısını görselleştir."""
    print("\n" + "=" * 70)
    print("📊 AGENT GRAPH VISUALIZATION")
    print("=" * 70)

    graph = build_agent_graph(with_memory=False)
    mermaid = visualize_graph(graph)
    print(mermaid)

    # Mermaid dosyası kaydet
    output_path = os.path.join(PROJECT_ROOT, "output", "graph.mmd")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(mermaid)
    print(f"\n💾 Mermaid diagram kaydedildi: {output_path}")
    print("Görselleştirmek için: https://mermaid.live/")


def run_memory_demo(args):
    """
    Memory demo — Vector store'un nasıl çalıştığını gösterir.

    Bu demo:
    1. Birkaç örnek analiz sonucunu vector store'a kaydet
    2. Semantic search ile ilgili kayıtları bul
    3. RAG-style context oluştur
    """
    from src.memory import MemoryManager

    print("\n" + "=" * 70)
    print("🧠 MEMORY DEMO — Vector Store")
    print("=" * 70)

    memory = MemoryManager(
        persist_dir=os.path.join(PROJECT_ROOT, "data", "memory_demo")
    )

    # Örnek analizleri kaydet
    analyses = [
        ("Otobüs fotoğrafında 4 kişi, 1 otobüs, 2 araba tespit edildi. "
         "Plaka numarası: 34 ABC 123. Sahne bir şehir sokağı.",
         {"type": "vehicle_detection", "image": "bus.jpg"}),

        ("Depo görüntüsünde 3 işçi baret takmıyor. Forklift geçiş alanında "
         "engel var. 2 güvenlik ihlali tespit edildi.",
         {"type": "safety_inspection", "image": "warehouse.jpg"}),

        ("Otopark görüntüsünde 12 araç var. 3'ü kırmızı, 5'i beyaz, 4'ü siyah. "
         "Boş park yeri sayısı: 8. Doluluk oranı: %60.",
         {"type": "parking_analysis", "image": "parking.jpg"}),

        ("Trafik kamerası: saat 08:30'da yoğun trafik. 45 araç/dakika geçiş. "
         "Kırmızı ışık ihlali: 2 araç. Ortalama hız: 25 km/h.",
         {"type": "traffic_analysis", "image": "traffic_cam.jpg"}),
    ]

    print("\n📝 Analizler kaydediliyor...")
    for text, metadata in analyses:
        memory.store_analysis(text, metadata)
        print(f"  ✅ {metadata['type']} ({metadata['image']})")

    print(f"\n📚 Toplam kayıt: {memory.vector_store.count}")

    # Semantic search demo
    queries = [
        "araçlar ve park yeri",
        "güvenlik ihlali var mı?",
        "trafik yoğunluğu nasıl?",
    ]

    for query in queries:
        print(f"\n🔍 Arama: \"{query}\"")
        results = memory.vector_store.search(query, n_results=2)
        for i, r in enumerate(results, 1):
            distance = f"{r['distance']:.4f}" if r['distance'] is not None else "N/A"
            print(f"  {i}. [mesafe: {distance}] {r['text'][:120]}...")

    # RAG context demo
    print(f"\n{'='*60}")
    print("📄 RAG Context örneği:")
    print("=" * 60)
    context = memory.get_full_context("depo güvenliği")
    print(context)


def run_tts_demo(args):
    """
    TTS Demo — Metni sese çevirir (Edge-TTS).

    Edge-TTS Microsoft'un neural TTS motorunu kullanır:
    - Ücretsiz, API key gerektirmez
    - 300+ ses, 80+ dil
    - Türkçe: EmelNeural (kadın), AhmetNeural (erkek)
    """
    from src.voice import EdgeTTS

    print("\n" + "=" * 70)
    print("🔊 TTS DEMO — Edge-TTS")
    print("=" * 70)

    tts = EdgeTTS(voice=args.voice)
    output_path = args.output or os.path.join(PROJECT_ROOT, "output", "tts_output.mp3")

    print(f"Metin: \"{args.text}\"")
    print(f"Ses: {tts.voice}")
    print(f"Hız: {args.rate}")

    result = tts.synthesize_sync(args.text, output_path, rate=args.rate)

    print(f"\n✅ Ses dosyası oluşturuldu: {result['output_path']}")
    print(f"   Kelime sayısı: {result['word_count']}")
    print(f"   Tahmini süre: {result['duration_estimate_seconds']}s")
    print(f"\n▶️  Dinlemek için: open {result['output_path']}")


def run_asr_demo(args):
    """
    ASR Demo — Ses dosyasını metne çevirir (Whisper).

    Whisper OpenAI'ın ses tanıma modeli:
    - Offline çalışır (API gerekmez)
    - 100+ dil desteği
    - Zaman damgalı çıktı (altyazı için)
    """
    from src.voice import WhisperASR

    print("\n" + "=" * 70)
    print("🎤 ASR DEMO — Whisper")
    print("=" * 70)

    asr = WhisperASR(model_size=args.model)

    if args.detect_language:
        print(f"Dil tespiti: {args.audio}")
        result = asr.detect_language(args.audio)
        print(f"Tespit edilen dil: {result['detected_language']} "
              f"(confidence: {result['confidence']:.2%})")
        print(f"Top 5: {result['top_5']}")
        return

    print(f"Ses dosyası: {args.audio}")
    print(f"Model: {args.model}")
    lang_info = f", dil: {args.language}" if args.language else ", dil: auto-detect"
    print(f"Ayarlar{lang_info}")

    result = asr.transcribe(
        args.audio,
        language=args.language,
        task=args.task
    )

    if "error" in result:
        print(f"❌ Hata: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"📝 TRANSKRIPT")
    print(f"{'='*60}")
    print(result["text"])
    print(f"\nDil: {result['language']}")
    print(f"Süre: {result['duration']}s")
    print(f"Segment sayısı: {len(result['segments'])}")

    if args.segments:
        print(f"\n📋 Segmentler (zaman damgalı):")
        for seg in result["segments"]:
            print(f"  [{seg['start']:>6.2f}s → {seg['end']:>6.2f}s] {seg['text']}")

    # JSON kaydet
    if args.output:
        import json
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"\n💾 Sonuç kaydedildi: {args.output}")


def run_voice_pipeline(args):
    """
    Voice Pipeline — Tam akış: Ses → ASR → Agent → TTS → Ses.

    End-to-end voice assistant:
    1. Kullanıcı ses dosyası verir
    2. Whisper metin çıkarır
    3. Agent graph cevap üretir
    4. Edge-TTS cevabı seslendirir
    """
    from src.voice import VoiceAssistant

    print("\n" + "=" * 70)
    print("🎙️  VOICE PIPELINE — Ses → Agent → Ses")
    print("=" * 70)

    assistant = VoiceAssistant(
        whisper_model=args.whisper_model,
        tts_voice=args.tts_voice
    )

    result = assistant.process_voice_query(
        audio_path=args.audio,
        image_path=args.image,
        output_audio_path=args.output
    )

    if "error" in result:
        print(f"❌ Hata: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"📋 SONUÇLAR")
    print(f"{'='*60}")
    print(f"🎤 Transkript: \"{result['user_query']}\"")
    print(f"💬 Agent cevabı:\n{result['agent_response']}")
    print(f"🔊 Ses çıktısı: {result['tts_output']['output_path']}")
    print(f"\n▶️  Dinlemek için: open {result['tts_output']['output_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Agent — LangGraph tabanlı agentic sistem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Örnekler:
  python run_agent.py analyze --image ../project1_cv_pipeline/data/bus.jpg -q "Bu sahnede ne var?"
  python run_agent.py ask -q "Object detection nasıl çalışır?"
  python run_agent.py interactive --image ../project1_cv_pipeline/data/bus.jpg
  python run_agent.py graph
  python run_agent.py memory-demo
  python run_agent.py tts --text "Merhaba, ben bir AI asistanıyım."
  python run_agent.py asr --audio ses.wav
  python run_agent.py voice --audio soru.wav --image bus.jpg
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Çalışma modu")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Görüntü + soru analizi")
    p_analyze.add_argument("--image", "-i", help="Görüntü dosya yolu")
    p_analyze.add_argument("--query", "-q", required=True, help="Soru")
    p_analyze.add_argument("--output", "-o", default="output/agent_result.json", help="Çıktı JSON yolu")
    p_analyze.add_argument("--max-iter", type=int, default=5, help="Max iteration sayısı")

    # ask
    p_ask = subparsers.add_parser("ask", help="Görüntüsüz soru-cevap")
    p_ask.add_argument("--query", "-q", required=True, help="Soru")
    p_ask.add_argument("--max-iter", type=int, default=3, help="Max iteration sayısı")

    # interactive
    p_inter = subparsers.add_parser("interactive", help="Interactive multi-turn mod")
    p_inter.add_argument("--image", "-i", help="Görüntü dosya yolu")
    p_inter.add_argument("--max-iter", type=int, default=5, help="Max iteration sayısı")

    # graph
    subparsers.add_parser("graph", help="Graph yapısını görselleştir")

    # memory-demo
    subparsers.add_parser("memory-demo", help="Memory (vector store) demo")

    # tts — Text-to-Speech
    p_tts = subparsers.add_parser("tts", help="Metin → Ses (Edge-TTS)")
    p_tts.add_argument("--text", "-t", required=True, help="Seslendirilecek metin")
    p_tts.add_argument("--voice", "-v", default="tr_female",
                       help="Ses: tr_female, tr_male, en_female, en_male (varsayılan: tr_female)")
    p_tts.add_argument("--rate", "-r", default="+0%", help="Hız: '+20%%' daha hızlı, '-20%%' daha yavaş")
    p_tts.add_argument("--output", "-o", help="Çıktı dosya yolu (.mp3)")

    # asr — Speech-to-Text
    p_asr = subparsers.add_parser("asr", help="Ses → Metin (Whisper)")
    p_asr.add_argument("--audio", "-a", required=True, help="Ses dosyası yolu")
    p_asr.add_argument("--model", "-m", default="base",
                       help="Whisper model: tiny, base, small, medium, large (varsayılan: base)")
    p_asr.add_argument("--language", "-l", help="Dil kodu: tr, en, vb. (varsayılan: auto-detect)")
    p_asr.add_argument("--task", default="transcribe",
                       help="transcribe (aynı dil) veya translate (İngilizce'ye çevir)")
    p_asr.add_argument("--segments", "-s", action="store_true", help="Zaman damgalı segmentleri göster")
    p_asr.add_argument("--detect-language", action="store_true", help="Sadece dil tespiti yap")
    p_asr.add_argument("--output", "-o", help="JSON çıktı yolu")

    # voice — Full voice pipeline
    p_voice = subparsers.add_parser("voice", help="Ses → Agent → Ses (tam pipeline)")
    p_voice.add_argument("--audio", "-a", required=True, help="Giriş ses dosyası")
    p_voice.add_argument("--image", "-i", help="Opsiyonel görüntü (multi-modal analiz)")
    p_voice.add_argument("--whisper-model", default="base", help="Whisper model boyutu")
    p_voice.add_argument("--tts-voice", default="tr_female", help="TTS ses seçimi")
    p_voice.add_argument("--output", "-o", help="TTS çıktı dosya yolu (.mp3)")

    args = parser.parse_args()

    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "ask":
        run_ask(args)
    elif args.command == "interactive":
        run_interactive(args)
    elif args.command == "graph":
        run_graph_viz(args)
    elif args.command == "memory-demo":
        run_memory_demo(args)
    elif args.command == "tts":
        run_tts_demo(args)
    elif args.command == "asr":
        run_asr_demo(args)
    elif args.command == "voice":
        run_voice_pipeline(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
