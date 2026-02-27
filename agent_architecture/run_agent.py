"""
Multi-Modal Agent CLI — Runs the LangGraph-based agent.

Usage:
    # Image analysis (CV pipeline + LLM reasoning)
    python run_agent.py analyze --image ../project1_cv_pipeline/data/bus.jpg --query "What is in this scene?"

    # Reasoning only (no image)
    python run_agent.py ask --query "What is the working principle of YOLO?"

    # Reasoning with web search (Search Node)
    python run_agent.py search --query "What are the latest LangGraph updates?"

    # Bring context from past memory (Memory Node)
    python run_agent.py memory-query --query "Which vehicles did we detect before?"

    # Voice → Agent (Speech Node integration — audio passes through graph)
    python run_agent.py voice-agent --audio speech.wav --image bus.jpg

    # Interactive mode (multi-turn conversation)
    python run_agent.py interactive --image ../project1_cv_pipeline/data/bus.jpg

    # Visualize graph structure
    python run_agent.py graph

    # Memory demo (remembering with vector store)
    python run_agent.py memory-demo
"""

import argparse
import json
import os
import sys

# Add project paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MONOREPO_ROOT = os.path.dirname(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

from src.state import create_initial_state
from src.graph import build_agent_graph, visualize_graph


def run_analyze(args):
    """
    Single-shot analysis mode.

    Flow:
    1. Create state (query + image)
    2. Compile graph
    3. Run (planner → router → vision → reasoner → evaluator → respond)
    4. Display result
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

    print(f"\nQuery: {args.query}")
    if args.image:
        print(f"Image: {args.image}")
    print(f"Max iteration: {args.max_iter}")
    print("-" * 70)

    # Invoke — run graph end-to-end
    result = graph.invoke(state)

    # Result
    print("\n" + "=" * 70)
    print("📋 RESULT")
    print("=" * 70)
    print(result.get("final_answer", "Could not generate an answer."))

    # Save details
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
        print(f"\n💾 Result saved: {args.output}")


def run_ask(args):
    """Image-less question-answer mode."""
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
    print("📋 RESULT")
    print("=" * 70)
    print(result.get("final_answer", "Could not generate an answer."))


def run_interactive(args):
    """
    Interactive mode — multi-turn conversation.

    Each turn:
    1. Get question from user
    2. Run agent graph
    3. Display answer
    4. Save to memory (long-term)
    5. Repeat

    Interview note:
    - Multi-turn state: graph runs from scratch each turn
    - Memory persistence: past analyses are remembered
    - 'q' to exit
    """
    from src.memory import MemoryManager

    print("\n" + "=" * 70)
    print("🤖 MULTI-MODAL AGENT — Interactive Mode")
    print("=" * 70)
    print("Commands: 'q' → exit, 'memory' → view stored records")

    if args.image:
        print(f"Görüntü: {args.image}")

    memory = MemoryManager(
        persist_dir=os.path.join(PROJECT_ROOT, "data", "memory")
    )

    turn = 0
    while True:
        turn += 1
        try:
            query = input(f"\n[Turn {turn}] Question: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not query:
            continue
        if query.lower() in ("q", "quit", "exit"):
            print("👋 See you later!")
            break
        if query.lower() == "memory":
            print(f"📚 {memory.vector_store.count} records in memory")
            if memory.vector_store.count > 0:
                recent = memory.vector_store.search("last analysis", n_results=3)
                for i, mem in enumerate(recent, 1):
                    print(f"  {i}. {mem['text'][:200]}...")
            continue

        # Get relevant context from Memory
        relevant_context = memory.get_full_context(query)
        full_query = query
        if relevant_context:
            full_query = f"{query}\n\nPast knowledge:\n{relevant_context}"

        # Run Agent
        state = create_initial_state(
            user_query=full_query,
            image_path=args.image if turn == 1 or args.image else None,
            max_iterations=args.max_iter
        )

        graph = build_agent_graph(with_memory=False)
        result = graph.invoke(state)

        answer = result.get("final_answer", "Could not generate an answer.")
        print(f"\n💬 Answer:\n{answer}")

        # Save to memory
        memory.add_conversation_message("user", query)
        memory.add_conversation_message("assistant", answer)
        memory.store_analysis(
            f"Question: {query}\nAnswer: {answer[:500]}",
            metadata={"turn": turn, "has_image": bool(args.image)}
        )


def run_graph_viz(args):
    """Visualize graph structure."""
    print("\n" + "=" * 70)
    print("📊 AGENT GRAPH VISUALIZATION")
    print("=" * 70)

    graph = build_agent_graph(with_memory=False)
    mermaid = visualize_graph(graph)
    print(mermaid)

    # Save Mermaid file
    output_path = os.path.join(PROJECT_ROOT, "output", "graph.mmd")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(mermaid)
    print(f"\n💾 Mermaid diagram saved: {output_path}")
    print("To visualize: https://mermaid.live/")


def run_memory_demo(args):
    """
    Memory demo — Shows how vector store works.

    This demo:
    1. Saves some sample analyses to the vector store
    2. Finds relevant records using semantic search
    3. Creates RAG-style context
    """
    from src.memory import MemoryManager

    print("\n" + "=" * 70)
    print("🧠 MEMORY DEMO — Vector Store")
    print("=" * 70)

    memory = MemoryManager(
        persist_dir=os.path.join(PROJECT_ROOT, "data", "memory_demo")
    )

    # Save sample analyses
    analyses = [
        ("Detected 4 people, 1 bus, 2 cars in the bus photo. "
         "License plate: 34 ABC 123. The scene is a city street.",
         {"type": "vehicle_detection", "image": "bus.jpg"}),

        ("3 workers are not wearing hardhats in the warehouse image. "
         "There is an obstacle in the forklift crossing area. 2 safety violations detected.",
         {"type": "safety_inspection", "image": "warehouse.jpg"}),

        ("There are 12 vehicles in the parking lot image. 3 red, 5 white, 4 black. "
         "Number of empty parking spots: 8. Occupancy rate: 60%.",
         {"type": "parking_analysis", "image": "parking.jpg"}),

        ("Traffic camera: heavy traffic at 08:30. 45 vehicles/minute passing. "
         "Red light violation: 2 vehicles. Average speed: 25 km/h.",
         {"type": "traffic_analysis", "image": "traffic_cam.jpg"}),
    ]

    print("\n📝 Saving analyses...")
    for text, metadata in analyses:
        memory.store_analysis(text, metadata)
        print(f"  ✅ {metadata['type']} ({metadata['image']})")

    print(f"\n📚 Total records: {memory.vector_store.count}")

    # Semantic search demo
    queries = [
        "vehicles and parking spot",
        "is there any safety violation?",
        "how is the traffic density?",
    ]

    for query in queries:
        print(f"\n🔍 Search: \"{query}\"")
        results = memory.vector_store.search(query, n_results=2)
        for i, r in enumerate(results, 1):
            distance = f"{r['distance']:.4f}" if r['distance'] is not None else "N/A"
            print(f"  {i}. [distance: {distance}] {r['text'][:120]}...")

    # RAG context demo
    print(f"\n{'='*60}")
    print("📄 RAG Context example:")
    print("=" * 60)
    context = memory.get_full_context("warehouse safety")
    print(context)


def run_tts_demo(args):
    """
    TTS Demo — Converts text to speech (Edge-TTS).

    Edge-TTS uses Microsoft's neural TTS engine:
    - Free, no API key required
    - 300+ voices, 80+ languages
    - Turkish: EmelNeural (female), AhmetNeural (male)
    """
    from src.voice import EdgeTTS

    print("\n" + "=" * 70)
    print("🔊 TTS DEMO — Edge-TTS")
    print("=" * 70)

    tts = EdgeTTS(voice=args.voice)
    output_path = args.output or os.path.join(PROJECT_ROOT, "output", "tts_output.mp3")

    print(f"Text: \"{args.text}\"")
    print(f"Voice: {tts.voice}")
    print(f"Rate: {args.rate}")

    result = tts.synthesize_sync(args.text, output_path, rate=args.rate)

    print(f"\n✅ Audio file created: {result['output_path']}")
    print(f"   Word count: {result['word_count']}")
    print(f"   Estimated duration: {result['duration_estimate_seconds']}s")
    print(f"\n▶️  To listen: open {result['output_path']}")


def run_asr_demo(args):
    """
    ASR Demo — Converts audio file to text (Whisper).

    Whisper is OpenAI's speech recognition model:
    - Works offline (no API needed)
    - 100+ languages support
    - Timestamped output (for subtitles)
    """
    from src.voice import WhisperASR

    print("\n" + "=" * 70)
    print("🎤 ASR DEMO — Whisper")
    print("=" * 70)

    asr = WhisperASR(model_size=args.model)

    if args.detect_language:
        print(f"Language detection: {args.audio}")
        result = asr.detect_language(args.audio)
        print(f"Detected language: {result['detected_language']} "
              f"(confidence: {result['confidence']:.2%})")
        print(f"Top 5: {result['top_5']}")
        return

    print(f"Audio file: {args.audio}")
    print(f"Model: {args.model}")
    lang_info = f", language: {args.language}" if args.language else ", language: auto-detect"
    print(f"Settings{lang_info}")

    result = asr.transcribe(
        args.audio,
        language=args.language,
        task=args.task
    )

    if "error" in result:
        print(f"❌ Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"📝 TRANSCRIPT")
    print(f"{'='*60}")
    print(result["text"])
    print(f"\nLanguage: {result['language']}")
    print(f"Duration: {result['duration']}s")
    print(f"Segment count: {len(result['segments'])}")

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
        print(f"\n💾 Result saved: {args.output}")


def run_search(args):
    """
    Search mode — Answers question using web search (Search Node).

    Planner chooses 'search' action → search_node gets updated info
    from DuckDuckGo → reasoner synthesizes → evaluator evaluates.
    """
    print("\n" + "=" * 70)
    print("🔍 MULTI-MODAL AGENT — Search Mode (Web Search + Reasoning)")
    print("=" * 70)

    state = create_initial_state(
        user_query=args.query,
        image_path=None,
        max_iterations=args.max_iter
    )
    # Add hint to force search in planner
    state["user_query"] = f"[Perform web search] {args.query}"

    graph = build_agent_graph(with_memory=False)
    result = graph.invoke(state)

    print("\n" + "=" * 70)
    print("📋 RESULT")
    print("=" * 70)
    print(result.get("final_answer", "Could not generate an answer."))
    search_results = result.get("search_results", [])
    if search_results:
        print(f"\n📎 {len(search_results)} web results used")


def run_memory_query(args):
    """
    Memory query mode — Answers questions by bringing context from past analyses.

    memory_node brings relevant past analyses from ChromaDB →
    reasoner produces a more informed answer with this context.
    """
    print("\n" + "=" * 70)
    print("🧠 MULTI-MODAL AGENT — Memory Query Mode (RAG)")
    print("=" * 70)

    state = create_initial_state(
        user_query=args.query,
        image_path=None,
        max_iterations=args.max_iter
    )
    # Hint to force memory usage in planner
    state["user_query"] = f"[Bring context from memory] {args.query}"

    graph = build_agent_graph(with_memory=False)
    result = graph.invoke(state)

    print("\n" + "=" * 70)
    print("📋 RESULT")
    print("=" * 70)
    print(result.get("final_answer", "Could not generate an answer."))
    memory_ctx = result.get("memory_context")
    if memory_ctx:
        print(f"\n📚 {len(memory_ctx.split(chr(10)))} lines of context from memory used")


def run_voice_agent(args):
    """
    Voice Agent mode — Audio passes through LangGraph (Voice Node integration).

    Difference: Unlike the old voice-pipeline, audio now passes INSIDE the GRAPH.
    voice_node → planner → router → [vision/search/memory/reason]
    → evaluator → respond flow is followed.
    Finally, the answer is optionally synthesized with TTS.
    """
    print("\n" + "=" * 70)
    print("🎙️  MULTI-MODAL AGENT — Voice Agent Mode (Graph Entegrasyonu)")
    print("=" * 70)
    print(f"Audio file: {args.audio}")
    if args.image:
        print(f"Image: {args.image}")

    # audio_path passed to graph — voice_node will automatically transcribe
    state = create_initial_state(
        user_query="[Waiting for voice input — voice_node will transcribe]",
        image_path=args.image,
        audio_path=args.audio,
        max_iterations=args.max_iter
    )

    graph = build_agent_graph(with_memory=False)
    result = graph.invoke(state)

    transcription = result.get("transcription", "")
    final_answer = result.get("final_answer", "Could not generate an answer.")

    print("\n" + "=" * 70)
    print("📋 RESULT")
    print("=" * 70)
    if transcription:
        print(f"🎤 Transcript: \"{transcription}\"")
    print(f"💬 Agent answer:\n{final_answer}")

    # Synthesize with optional TTS
    if args.tts:
        from src.voice import EdgeTTS
        tts = EdgeTTS(voice=args.tts_voice)
        output_path = args.output or os.path.join(PROJECT_ROOT, "output", "voice_agent_response.mp3")
        clean_answer = final_answer.split("\n📎")[0].split("\n📊")[0].strip()
        tts_result = tts.synthesize_sync(clean_answer, output_path)
        print(f"\n🔊 Answer synthesized: {tts_result['output_path']}")
        print(f"▶️  To listen: open {tts_result['output_path']}")


def run_voice_pipeline(args):
    """
    Voice Pipeline — Old flow: Audio → ASR (external) → Agent → TTS → Audio.

    This is the old version; audio is transcribed outside the graph.
    Use the voice-agent command for the new version.
    """
    from src.voice import VoiceAssistant

    print("\n" + "=" * 70)
    print("🎙️  VOICE PIPELINE — Ses → Agent → Ses (Legacy)")
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
        print(f"❌ Error: {result['error']}")
        return

    print(f"\n{'='*60}")
    print(f"📋 RESULTS")
    print(f"{'='*60}")
    print(f"🎤 Transcript: \"{result['user_query']}\"")
    print(f"💬 Agent answer:\n{result['agent_response']}")
    print(f"🔊 Audio output: {result['tts_output']['output_path']}")
    print(f"\n▶️  To listen: open {result['tts_output']['output_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-Modal Agent — LangGraph based agentic system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_agent.py analyze --image ../project1_cv_pipeline/data/bus.jpg -q "What's in this scene?"
  python run_agent.py ask -q "How does object detection work?"
  python run_agent.py interactive --image ../project1_cv_pipeline/data/bus.jpg
  python run_agent.py graph
  python run_agent.py memory-demo
  python run_agent.py tts --text "Hello, I am an AI assistant."
  python run_agent.py asr --audio speech.wav
  python run_agent.py voice --audio question.wav --image bus.jpg
        """
    )

    subparsers = parser.add_subparsers(dest="command", help="Run mode")

    # analyze
    p_analyze = subparsers.add_parser("analyze", help="Image + question analysis")
    p_analyze.add_argument("--image", "-i", help="Image file path")
    p_analyze.add_argument("--query", "-q", required=True, help="Question")
    p_analyze.add_argument("--output", "-o", default="output/agent_result.json", help="Output JSON path")
    p_analyze.add_argument("--max-iter", type=int, default=5, help="Max iteration count")

    # ask
    p_ask = subparsers.add_parser("ask", help="Image-less Q&A")
    p_ask.add_argument("--query", "-q", required=True, help="Question")
    p_ask.add_argument("--max-iter", type=int, default=3, help="Max iteration count")

    # interactive
    p_inter = subparsers.add_parser("interactive", help="Interactive multi-turn mode")
    p_inter.add_argument("--image", "-i", help="Image file path")
    p_inter.add_argument("--max-iter", type=int, default=5, help="Max iteration count")

    # graph
    subparsers.add_parser("graph", help="Visualize graph structure")

    # memory-demo
    subparsers.add_parser("memory-demo", help="Memory (vector store) demo")

    # tts — Text-to-Speech
    p_tts = subparsers.add_parser("tts", help="Text → Speech (Edge-TTS)")
    p_tts.add_argument("--text", "-t", required=True, help="Text to synthesize")
    p_tts.add_argument("--voice", "-v", default="tr_female",
                       help="Voice: tr_female, tr_male, en_female, en_male (default: tr_female)")
    p_tts.add_argument("--rate", "-r", default="+0%", help="Rate: '+20%%' faster, '-20%%' slower")
    p_tts.add_argument("--output", "-o", help="Output file path (.mp3)")

    # asr — Speech-to-Text
    p_asr = subparsers.add_parser("asr", help="Speech → Text (Whisper)")
    p_asr.add_argument("--audio", "-a", required=True, help="Audio file path")
    p_asr.add_argument("--model", "-m", default="base",
                       help="Whisper model: tiny, base, small, medium, large (default: base)")
    p_asr.add_argument("--language", "-l", help="Language code: tr, en, etc. (default: auto-detect)")
    p_asr.add_argument("--task", default="transcribe",
                       help="transcribe (same language) or translate (translate to English)")
    p_asr.add_argument("--segments", "-s", action="store_true", help="Show timestamped segments")
    p_asr.add_argument("--detect-language", action="store_true", help="Only perform language detection")
    p_asr.add_argument("--output", "-o", help="JSON output path")

    # voice — Legacy voice pipeline (outside graph)
    p_voice = subparsers.add_parser("voice", help="Speech → Agent → Speech (old pipeline)")
    p_voice.add_argument("--audio", "-a", required=True, help="Input audio file")
    p_voice.add_argument("--image", "-i", help="Optional image (multi-modal analysis)")
    p_voice.add_argument("--whisper-model", default="base", help="Whisper model size")
    p_voice.add_argument("--tts-voice", default="tr_female", help="TTS voice selection")
    p_voice.add_argument("--output", "-o", help="TTS output file path (.mp3)")

    # voice-agent — New: audio passes through graph (Voice Node integration)
    p_va = subparsers.add_parser("voice-agent", help="Speech → Graph (Voice Node) → Answer")
    p_va.add_argument("--audio", "-a", required=True, help="Input audio file")
    p_va.add_argument("--image", "-i", help="Optional image")
    p_va.add_argument("--tts", action="store_true", help="Synthesize answer with TTS")
    p_va.add_argument("--tts-voice", default="tr_female", help="TTS voice selection")
    p_va.add_argument("--output", "-o", help="TTS output file path (.mp3)")
    p_va.add_argument("--max-iter", type=int, default=3, help="Max iteration count")

    # search — Reasoning with web search (Search Node)
    p_search = subparsers.add_parser("search", help="Web search + reasoning (Search Node)")
    p_search.add_argument("--query", "-q", required=True, help="Search query")
    p_search.add_argument("--max-iter", type=int, default=3, help="Max iteration count")

    # memory-query — Getting context from past memory (Memory Node)
    p_mq = subparsers.add_parser("memory-query", help="Get context from memory + reasoning (Memory Node)")
    p_mq.add_argument("--query", "-q", required=True, help="Question")
    p_mq.add_argument("--max-iter", type=int, default=3, help="Max iteration count")

    args = parser.parse_args()

    if args.command == "analyze":
        run_analyze(args)
    elif args.command == "ask":
        run_ask(args)
    elif args.command == "search":
        run_search(args)
    elif args.command == "memory-query":
        run_memory_query(args)
    elif args.command == "voice-agent":
        run_voice_agent(args)
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
