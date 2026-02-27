"""
Agent Nodes — Nodes of the LangGraph graph (execution units).

╔══════════════════════════════════════════════════════════════════╗
║  LangGraph Node Concept                                         ║
║                                                                  ║
║  Node = Python function                                         ║
║  - Input: AgentState (or a part of it)                          ║
║  - Output: State update (dict)                                 ║
║                                                                  ║
║  Graph flow:                                                    ║
║  START → planner → router ─┬─→ vision → reasoner → evaluator   ║
║                             ├─→ reasoner → evaluator             ║
║                             └─→ respond → END                    ║
║                                                                  ║
║  If Evaluator gives a bad score → returns to planner (loop)      ║
╚══════════════════════════════════════════════════════════════════╝

Agent Patterns (Interview note):

1. ReAct (Reason + Act):
   - Think → Act → Observe → Repeat
   - Simple, tool selection with a single LLM call
   - In this project: router + tool nodes

2. Plan-and-Execute:
   - Plan first (determine all steps)
   - Then execute them sequentially
   - In this project: planner node → executor nodes

3. Reflection:
   - Generate answer → Evaluate → Correct if necessary
   - In this project: evaluator node → loop back

We combine all three: Plan → Execute (ReAct) → Reflect
"""

import json
import sys
import os
from typing import Optional
from dotenv import load_dotenv

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONOREPO_ROOT = os.path.dirname(PROJECT_ROOT)


load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# ─── LLM Helper ──────────────────────────────────────────────────

def _get_openai_client():
    """
    Create DeepSeek LLM client — with OpenAI SDK.

    Since DeepSeek's API is OpenAI-compatible, we use the OpenAI SDK directly.
    This eliminates dependency on project2 and provides a cleaner architecture.

    Interview note:
    - Many LLM providers (DeepSeek, Together, Groq) offer OpenAI-compatible APIs
    - This allows using multiple providers with a single SDK (openai)
    - Just changing the base_url is enough
    """
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable not set.\n"
            "Create project3_agent_architecture/.env with:\n"
            "  DEEPSEEK_API_KEY=sk-your-key-here"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# DeepSeek model name
_LLM_MODEL = "deepseek-chat"

# Singleton client — reuse instead of creating a new instance on every call
_openai_client = None


def _call_llm(prompt: str, system_prompt: str = "", json_mode: bool = False) -> str:
    """
    Simple LLM call helper.

    All nodes use this function to access the LLM.
    Centralized LLM access → provider replacement from a single point.
    Client is cached as a singleton — not recreated over and over.
    """
    global _openai_client
    if _openai_client is None:
        _openai_client = _get_openai_client()

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": _LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    try:
        response = _openai_client.chat.completions.create(**kwargs)
        return response.choices[0].message.content
    except Exception as e:
        error_msg = f"LLM API error: {type(e).__name__}: {e}"
        print(f"⚠️  {error_msg}")
        # Even in JSON mode, return a valid fallback
        if json_mode:
            return json.dumps({"error": error_msg, "answer": error_msg})
        return error_msg


# ═══════════════════════════════════════════════════════════════════
# NODE 1: PLANNER — Breaks down the task into steps
# ═══════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """You are a task planner. Analyze the user's request and
break it down into actionable steps.

Available capabilities:
- vision: Image analysis (object detection, segmentation, OCR) — use if image_path is present
- search: Search the web for up-to-date info (DuckDuckGo) — use if current/external info is needed
- memory: Remember past analysis (vector store) — use if similar past context is needed
- reason: Synthesize all information and draw a conclusion
- respond: Give a direct, simple response

Priority: vision (if image present) > search (if current info needed) > memory (if past context needed) > reason

Write each step short and clear. Return in JSON format:
{
  "steps": ["step1", "step2", ...],
  "requires_vision": true/false,
  "requires_search": true/false,
  "requires_memory": true/false,
  "complexity": "simple" | "moderate" | "complex"
}"""


def planner_node(state: dict) -> dict:
    """
    PLANNER NODE — Analyzes the user's request and creates a plan.

    This node is the "Plan" part of the Plan-and-Execute pattern.
    It breaks down a complex task into steps using the LLM.

    Interview note:
    - Planning is the agent's most critical capability
    - Bad plan = bad result (garbage in, garbage out)
    - Plan must adapt to complexity: simple question → 1 step, complex → multiple steps
    - Writing the plan to state provides transparency (explainability)

    Input (from state): user_query, image_path
    Output (state update): plan, next_action, messages
    """
    user_query = state["user_query"]
    image_path = state.get("image_path")

    print(f"\n{'='*60}")
    print(f"🧠 PLANNER NODE")
    print(f"{'='*60}")
    print(f"Query: {user_query}")
    print(f"Image: {image_path or 'None'}")

    context = f"User request: {user_query}"
    if image_path:
        context += f"\nImage available: {image_path}"

    raw_response = _call_llm(context, PLANNER_SYSTEM_PROMPT, json_mode=True)

    # Parse plan
    try:
        parsed = json.loads(raw_response)
        steps = parsed.get("steps", [])
        requires_vision = parsed.get("requires_vision", bool(image_path))
        requires_search = parsed.get("requires_search", False)
        requires_memory = parsed.get("requires_memory", False)
        complexity = parsed.get("complexity", "moderate")
    except json.JSONDecodeError:
        steps = [f"Answer directly: {user_query}"]
        requires_vision = bool(image_path)
        requires_search = False
        requires_memory = False
        complexity = "simple"

    print(f"Plan ({complexity}):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    # Initial action priority order: vision > search > memory > reason
    if requires_vision and image_path:
        next_action = "vision"
    elif requires_search:
        next_action = "search"
    elif requires_memory:
        next_action = "memory"
    else:
        next_action = "reason"

    return {
        "plan": steps,
        "current_step": 0,
        "next_action": next_action,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [{
            "role": "assistant",
            "content": f"[Planner] Plan generated ({len(steps)} steps, {complexity}): {', '.join(steps)}"
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 2: ROUTER — Conditional edge (decision point)
# ═══════════════════════════════════════════════════════════════════

def router_node(state: dict) -> str:
    """
    ROUTER — Conditional edge function.

    In LangGraph, conditional_edges decide which node to go to
    by looking at the state. This is not a "node", it's a "decision function".

    Interview note:
    - There are two types of edges in LangGraph:
      1. Normal edge: A → B (always)
      2. Conditional edge: A → router → B or C (based on state)
    - The router function returns a string → goes to the node mapped in the edge mapping
    - Infinite loop protection: max_iterations check is required

    Input: state
    Output: string → node name ("vision", "reason", "respond", "human_approval")
    """
    next_action = state.get("next_action", "reason")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    needs_approval = state.get("needs_human_approval", False)

    print(f"\n{'='*60}")
    print(f"🔀 ROUTER NODE")
    print(f"{'='*60}")

    # Infinite loop protection
    if iteration_count >= max_iterations:
        print(f"⚠️  Max iterations ({max_iterations}) exceeded → respond")
        return "respond"

    # Human-in-the-loop check
    if needs_approval:
        print(f"👤 Human approval required → human_approval")
        return "human_approval"

    print(f"Decision: {next_action} (iteration {iteration_count}/{max_iterations})")
    return next_action


# ═══════════════════════════════════════════════════════════════════
# NODE 3: VISION — Runs the CV Pipeline
# ═══════════════════════════════════════════════════════════════════

def vision_node(state: dict) -> dict:
    """
    VISION NODE — Passes the image through the CV pipeline.

    Calls VisualPerceptionPipeline from Project 1:
    - Object Detection (YOLOv8)
    - Instance Segmentation (YOLOv8-seg)
    - OCR (EasyOCR)

    Interview note:
    - This node is the "perception" layer in the agentic system
    - The "eyes" of the agent — perceives the world
    - CV pipeline is lazy-loaded: only loaded when needed
    - Result is written to state → other nodes can use it

    Input (from state): image_path
    Output (state update): cv_result, tool_results, next_action, messages
    """
    image_path = state.get("image_path")

    print(f"\n{'='*60}")
    print(f"👁️  VISION NODE")
    print(f"{'='*60}")

    if not image_path:
        print("⚠️  No image path — skipping")
        return {
            "cv_result": None,
            "next_action": "reason",
            "tool_results": [{"tool": "vision", "error": "No image path provided"}],
            "messages": [{"role": "assistant", "content": "[Vision] Image path not specified."}]
        }

    # Load and run CV Pipeline
    # sys.path management to import project1's src/ directory:
    # 1. Add project1 to path (to resolve relative imports)
    # 2. Temporarily remove project3's 'src' module from sys.modules
    # 3. Import
    # 4. Restore
    project1_path = os.path.join(MONOREPO_ROOT, "project1_cv_pipeline")

    # Temporary sys.path and modules management
    old_src_module = sys.modules.pop("src", None)
    if project1_path not in sys.path:
        sys.path.insert(0, project1_path)

    from src.pipeline import VisualPerceptionPipeline
    pipeline = VisualPerceptionPipeline()

    # Restore
    if old_src_module is not None:
        sys.modules["src"] = old_src_module

    print(f"Analyzing image: {image_path}")
    cv_result = pipeline.analyze(image_path)

    # Summary info
    n_objects = len(cv_result.get("objects", []))
    n_segments = len(cv_result.get("segments", []))
    n_text = len(cv_result.get("text_regions", []))
    proc_time = cv_result.get("processing_time", {}).get("total", 0)

    summary = (
        f"Detected: {n_objects} objects, {n_segments} segments, {n_text} text regions "
        f"({proc_time:.2f}s)"
    )
    print(f"✅ {summary}")

    return {
        "cv_result": cv_result,
        "next_action": "reason",
        "tool_results": [{
            "tool": "vision",
            "summary": summary,
            "objects": n_objects,
            "segments": n_segments,
            "text_regions": n_text,
            "processing_time": proc_time
        }],
        "messages": [{"role": "assistant", "content": f"[Vision] {summary}"}]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 4: REASONER — Synthesizes info, generates an answer
# ═══════════════════════════════════════════════════════════════════

REASONER_SYSTEM_PROMPT = """You are a multi-modal AI analyst. Synthesize the given information
and provide a comprehensive and accurate answer to the user's question.

Provide your answer in this JSON format:
{
  "answer": "Main answer (detailed, explanatory)",
  "reasoning_steps": ["step1", "step2", ...],
  "confidence": 0.0-1.0,
  "evidence": ["evidence1", "evidence2", ...],
  "follow_up_suggestions": ["suggestion1", "suggestion2"]
}

Rules:
- Use CV pipeline results as evidence
- Lower your confidence where you are uncertain
- Provide concrete numbers and data
- Answer in English"""


def reasoner_node(state: dict) -> dict:
    """
    REASONER NODE — Synthesizes all info and generates an answer.

    This node is the "reasoning" layer in the agentic system.
    It combines CV results, tool results, and conversation history
    to produce a meaningful answer.

    Interview note:
    - Reasoning = "sense-making" — converting raw data into understanding
    - Context window management is critical: must fit all information
    - Chain-of-Thought (CoT) prompting improves reasoning quality
    - Evidence-based reasoning: state the evidence for the answer

    Input (from state): user_query, cv_result, tool_results, plan
    Output (state update): reasoning, final_answer, next_action, messages
    """
    user_query = state["user_query"]
    cv_result = state.get("cv_result")
    tool_results = state.get("tool_results", [])
    plan = state.get("plan", [])
    memory_context = state.get("memory_context")
    search_results = state.get("search_results", [])

    print(f"\n{'='*60}")
    print(f"🤔 REASONER NODE")
    print(f"{'='*60}")

    # Create context — the info package to send to the LLM
    context_parts = [f"User question: {user_query}"]

    if plan:
        context_parts.append(f"Plan: {', '.join(plan)}")

    if memory_context:
        context_parts.append(f"Relevant Past Info From Memory:\n{memory_context}")

    if cv_result:
        cv_summary = _summarize_cv_result(cv_result)
        context_parts.append(f"CV Analysis Result:\n{cv_summary}")

    if search_results:
        search_summary = "\n".join(
            f"- {r.get('title', 'No title')}: {r.get('body', r.get('snippet', ''))[:200]}"
            for r in search_results[:5]
        )
        context_parts.append(f"Web Search Results:\n{search_summary}")

    if tool_results:
        context_parts.append(f"Tool Results:\n{json.dumps(tool_results, indent=2, ensure_ascii=False)}")

    full_context = "\n\n".join(context_parts)
    print(f"Context length: {len(full_context)} characters")

    # Send to LLM
    raw_response = _call_llm(full_context, REASONER_SYSTEM_PROMPT, json_mode=True)

    # Parse
    try:
        parsed = json.loads(raw_response)
        answer = parsed.get("answer", raw_response)
        reasoning_steps = parsed.get("reasoning_steps", [])
        confidence = parsed.get("confidence", 0.5)
    except json.JSONDecodeError:
        answer = raw_response
        reasoning_steps = []
        confidence = 0.5

    print(f"Answer length: {len(answer)} characters")
    print(f"Confidence: {confidence}")
    print(f"Reasoning steps: {len(reasoning_steps)}")

    return {
        "reasoning": raw_response,
        "final_answer": answer,
        "messages": [{
            "role": "assistant",
            "content": f"[Reasoner] Confidence: {confidence} | {answer[:200]}..."
        }]
    }


def _summarize_cv_result(cv_result: dict) -> str:
    """Convert CV result to a summary string (to save tokens)."""
    parts = []

    # Objects
    objects = cv_result.get("objects", [])
    if objects:
        # Group object counts
        from collections import Counter
        label_counts = Counter(o["label"] for o in objects)
        obj_summary = ", ".join(f"{count}x {label}" for label, count in label_counts.items())
        parts.append(f"Detected objects: {obj_summary}")

        # Highlight max confidence
        max_conf = max(o["confidence"] for o in objects)
        parts.append(f"Highest confidence: {max_conf:.2f}")

    # Text regions
    text_regions = cv_result.get("text_regions", [])
    if text_regions:
        texts = [t["text"] for t in text_regions[:5]]  # First 5
        parts.append(f"Detected texts: {', '.join(texts)}")

    # Segments
    segments = cv_result.get("segments", [])
    if segments:
        parts.append(f"Segmentation: {len(segments)} segments")

    # Image info
    img_info = cv_result.get("image_info", {})
    if img_info:
        parts.append(f"Image info: {img_info.get('width', '?')}x{img_info.get('height', '?')}")

    return "\n".join(parts) if parts else "CV result empty"


# ═══════════════════════════════════════════════════════════════════
# NODE 5: EVALUATOR — Evaluates answer quality
# ═══════════════════════════════════════════════════════════════════

EVALUATOR_SYSTEM_PROMPT = """You are a quality assurance expert. Evaluate the generated answer.

Return in JSON format:
{
  "score": 0.0-1.0,
  "feedback": "Short evaluation",
  "pass": true/false,
  "improvement_suggestion": "Suggestion for improvement if any"
}

Evaluation criteria:
- Accuracy: Does the answer align with the question?
- Evidence: Are CV results used appropriately?
- Completeness: Is the question completely answered?
- Clarity: Is the answer clear and understandable?

Score > 0.7 → PASS, < 0.7 → FAIL (try again)"""


def evaluator_node(state: dict) -> dict:
    """
    EVALUATOR NODE — Reflection pattern: evaluates the quality of the answer.

    This node is a "self-critique" mechanism. It evaluates the agent's
    own answer and makes it try again if it's inadequate.

    Interview note:
    - Reflection/Self-critique: LLM evaluating its own output
    - This pattern significantly increases answer quality
    - Trade-off: Extra LLM call = higher cost + latency
    - Infinite loop risk: Limit with max_iterations
    - In production: Hybrid of simple heuristic (length, format) + LLM evaluation

    Input (from state): user_query, final_answer, reasoning
    Output (state update): evaluation_score, evaluation_feedback, next_action
    """
    user_query = state["user_query"]
    final_answer = state.get("final_answer", "")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    print(f"\n{'='*60}")
    print(f"📊 EVALUATOR NODE")
    print(f"{'='*60}")

    eval_context = (
        f"Original question: {user_query}\n\n"
        f"Generated answer: {final_answer}"
    )

    raw_response = _call_llm(eval_context, EVALUATOR_SYSTEM_PROMPT, json_mode=True)

    try:
        parsed = json.loads(raw_response)
        score = parsed.get("score", 0.5)
        feedback = parsed.get("feedback", "")
        passed = parsed.get("pass", score >= 0.7)
    except json.JSONDecodeError:
        score = 0.7
        feedback = "Evaluation parsing failed — passing by default"
        passed = True

    print(f"Score: {score:.2f}")
    print(f"Feedback: {feedback}")
    print(f"Passed: {'✅ YES' if passed else '❌ NO'}")

    if passed or iteration >= max_iter - 1:
        next_action = "respond"
        if not passed:
            print(f"⚠️  Score is low but max iterations reached → respond")
    else:
        next_action = "reason"
        print(f"🔄 Retrying (iteration {iteration}/{max_iter})")

    return {
        "evaluation_score": score,
        "evaluation_feedback": feedback,
        "next_action": next_action,
        "iteration_count": iteration + 1,  # Increase iteration in the loop — infinite loop protection
        "messages": [{
            "role": "assistant",
            "content": f"[Evaluator] Score: {score:.2f} — {feedback}"
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 6: RESPOND — Formats the final answer
# ═══════════════════════════════════════════════════════════════════

def respond_node(state: dict) -> dict:
    """
    RESPOND NODE — Prepares the final answer to be given to the user.

    A simple formatting node. Formats the final_answer in the state
    to be suitable for the user.

    Input (from state): final_answer, evaluation_score, plan, tool_results
    Output (state update): final_answer (formatted), messages
    """
    final_answer = state.get("final_answer", "Could not generate an answer.")
    score = state.get("evaluation_score")
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])

    print(f"\n{'='*60}")
    print(f"💬 RESPOND NODE")
    print(f"{'='*60}")

    # Rich answer format
    response_parts = [final_answer]

    if tool_results:
        tools_used = set(tr.get("tool", "unknown") for tr in tool_results)
        response_parts.append(f"\n📎 Tools used: {', '.join(tools_used)}")

    if score is not None:
        response_parts.append(f"📊 Confidence score: {score:.0%}")

    formatted_answer = "\n".join(response_parts)
    print(f"Final answer ({len(formatted_answer)} characters)")

    return {
        "final_answer": formatted_answer,
        "messages": [{
            "role": "assistant",
            "content": formatted_answer
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 7: HUMAN APPROVAL — Waits for human approval
# ═══════════════════════════════════════════════════════════════════

def human_approval_node(state: dict) -> dict:
    """
    HUMAN-IN-THE-LOOP NODE — Waits for human approval on critical decisions.

    In LangGraph, human-in-the-loop can be done in two ways:
    1. interrupt_before/interrupt_after — stop the graph, let human decide
    2. Approval node — ask for approval based on state

    We use the 2nd method here (simpler, more flexible).

    Interview note:
    - Why is human-in-the-loop necessary?
      → Security: when costs of wrong decisions are high (deleting, sending)
      → Ethics: when working with sensitive data
      → Regulatory: compliance requirements
    - When is it NOT used?
      → When latency is critical (real-time systems)
      → If the decision is low-risk
    - LangGraph interrupt: graph checkpointed → can be paused and resumed
    """
    print(f"\n{'='*60}")
    print(f"👤 HUMAN APPROVAL NODE")
    print(f"{'='*60}")

    plan = state.get("plan", [])
    print(f"Plan: {plan}")
    print(f"Waiting for approval...")

    # Get approval via input in CLI
    try:
        approval = input("\n✋ Do you approve this plan? (y/n): ").strip().lower()
    except EOFError:
        approval = "y"  # Auto-approve in non-interactive mode

    if approval in ("e", "evet", "y", "yes"):
        print("✅ Approved — continuing")
        return {
            "needs_human_approval": False,
            "next_action": "vision" if state.get("image_path") else "reason",
            "messages": [{"role": "user", "content": "[Human] Plan approved."}]
        }
    else:
        print("❌ Rejected — replanning")
        return {
            "needs_human_approval": False,
            "next_action": "respond",
            "final_answer": "Operation cancelled by user.",
            "messages": [{"role": "user", "content": "[Human] Plan rejected."}]
        }


# ═══════════════════════════════════════════════════════════════════
# NODE 8: VOICE — Converts audio file to text with Whisper
# ═══════════════════════════════════════════════════════════════════

def voice_node(state: dict) -> dict:
    """
    VOICE NODE — audio → text with ASR (Automatic Speech Recognition).

    Runs immediately after START in the graph (pre-planner).
    If there is no audio_path, it's a pass-through (state unchanged).
    If there is an audio_path → transcribes with Whisper →
    updates user_query with transcript → planner kicks in.

    Interview note:
    - Voice node is the audio component of the "perception" layer in the graph
    - Vision node can be thought of as eyes, Voice node as ears
    - By updating user_query, the planner processes the voice command
      as if it were a text command — clean abstraction
    - Whisper is lazy-loaded: model is downloaded on first call (~74MB base)
    - Pass-through pattern: if no audio_path returns {} → state unchanged

    Input (from state): audio_path
    Output (state update): transcription, user_query, tool_results, messages
    """
    audio_path = state.get("audio_path")

    print(f"\n{'='*60}")
    print(f"🎤 VOICE NODE (Speech Node)")
    print(f"{'='*60}")

    if not audio_path:
        print("ℹ️  No audio file — pass-through")
        return {}  # State unchanged, moving to planner

    if not os.path.exists(audio_path):
        print(f"⚠️  Audio file not found: {audio_path}")
        return {
            "transcription": None,
            "tool_results": [{"tool": "voice/asr", "error": f"File not found: {audio_path}"}],
            "messages": [{"role": "assistant", "content": f"[Voice] Audio file not found: {audio_path}"}]
        }

    print(f"Audio file: {audio_path}")

    # Whisper ASR — lazy import (large model, load only when needed)
    from .voice import WhisperASR
    asr = WhisperASR(model_size="base")
    result = asr.transcribe(audio_path)

    if "error" in result:
        return {
            "transcription": None,
            "tool_results": [{"tool": "voice/asr", "error": result["error"]}],
            "messages": [{"role": "assistant", "content": f"[Voice] Transcription error: {result['error']}"}]
        }

    transcription = result["text"]
    language = result.get("language", "unknown")
    duration = result.get("duration", 0)

    print(f"✅ Transcript ({language}, {duration:.1f}s): \"{transcription}\"")

    return {
        "transcription": transcription,
        "user_query": transcription,       # Convert voice command to text query
        "tool_results": [{
            "tool": "voice/asr",
            "transcription": transcription,
            "language": language,
            "duration_seconds": duration,
            "segments": len(result.get("segments", []))
        }],
        "messages": [{
            "role": "user",
            "content": f"[Voice/ASR] Transcript: \"{transcription}\" ({language}, {duration:.1f}s)"
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 9: SEARCH — Web search with DuckDuckGo
# ═══════════════════════════════════════════════════════════════════

def search_node(state: dict) -> dict:
    """
    SEARCH NODE — Performs web search with DuckDuckGo.

    Extracts search query from Plan or user_query,
    searches the web and saves results to state.
    Later, the reasoner synthesizes this information.

    Interview note:
    - Search grounding: Access to current info beyond LLM's knowledge cutoff
    - DuckDuckGo: No API key required, privacy-focused, free
    - Alternatives: SerpAPI ($), Tavily API (integrated with LangChain), Bing Search API
    - Search quality: query engineering is critical — vague query → irrelevant results
    - Results sent to reasoner → LLM synthesizes (not raw URLs)
    - Saved as tool result → evaluator can also assess quality

    Input (from state): user_query, plan
    Output (state update): search_results, tool_results, next_action, messages
    """
    user_query = state["user_query"]
    plan = state.get("plan", [])

    print(f"\n{'='*60}")
    print(f"🔍 SEARCH NODE (Web Search)")
    print(f"{'='*60}")

    # Determine search query: first "search" step from plan or user_query
    search_query = user_query
    for step in plan:
        if "search" in step.lower() or "find" in step.lower():
            # Extract search terms from plan step
            search_query = step.replace("search:", "").replace("find:", "").strip()
            break

    print(f"Search query: \"{search_query}\"")

    try:
        from ddgs import DDGS

        # region='wt-wt' → language-independent global results
        raw_results = list(DDGS().text(search_query, max_results=5, region="wt-wt"))

        results = [
            {
                "title": r.get("title", ""),
                "url": r.get("href", ""),
                "body": r.get("body", "")[:500]  # Truncate to save tokens
            }
            for r in raw_results
        ]

        print(f"✅ {len(results)} results found")
        for i, r in enumerate(results[:3], 1):
            print(f"  {i}. {r['title'][:60]}...")

    except Exception as e:
        print(f"⚠️  DuckDuckGo error: {e} — continuing with LLM knowledge")
        results = []

    return {
        "search_results": results,
        "next_action": "reason",
        "tool_results": [{
            "tool": "search/duckduckgo",
            "query": search_query,
            "result_count": len(results),
            "top_titles": [r["title"][:60] for r in results[:3]]
        }],
        "messages": [{
            "role": "assistant",
            "content": f"[Search] Found {len(results)} results for \"{search_query}\"."
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 10: MEMORY — Retrieves relevant past info from VectorMemory
# ═══════════════════════════════════════════════════════════════════

def memory_node(state: dict) -> dict:
    """
    MEMORY NODE — Retrieves relevant context from long-term memory (ChromaDB).

    Finds semantically similar past analysis to the user's query
    and adds it to the state as memory_context. Reasoner uses this context.

    This is the RAG (Retrieval-Augmented Generation) pattern:
    1. Retrieve: Get relevant documents from vector store
    2. Augment: Add these documents to the LLM prompt
    3. Generate: LLM generates a more informed answer

    Interview note:
    - RAG vs Fine-tuning: RAG is better for up-to-date/changing information
    - Embedding similarity: top k documents chosen via cosine similarity
    - ChromaDB: embedded, built on SQLite, for production use Pinecone/Weaviate
    - Without memory node: agent starts from scratch every time (stateless)
    - With memory node: past analyses accumulate, agent gets "smarter"

    Input (from state): user_query
    Output (state update): memory_context, tool_results, next_action, messages
    """
    user_query = state["user_query"]

    print(f"\n{'='*60}")
    print(f"🧠 MEMORY NODE (Vector Store Retrieval)")
    print(f"{'='*60}")
    print(f"Query: \"{user_query}\"")

    # Load VectorMemory — from default persist_dir
    from .memory import VectorMemory

    default_persist_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "memory"
    )
    vector_memory = VectorMemory(persist_dir=default_persist_dir)

    # Pass if memory is empty
    if vector_memory.count == 0:
        print("ℹ️  Memory is empty — no past analysis")
        return {
            "memory_context": None,
            "next_action": "reason",
            "tool_results": [{
                "tool": "memory/chromadb",
                "query": user_query,
                "result_count": 0,
                "message": "Vector store empty"
            }],
            "messages": [{
                "role": "assistant",
                "content": "[Memory] No saved analysis yet."
            }]
        }

    # Semantic search
    relevant_context = vector_memory.get_relevant_context(user_query, n_results=3)
    memories = vector_memory.search(user_query, n_results=3)

    if relevant_context:
        print(f"✅ {len(memories)} relevant records found:")
        for i, mem in enumerate(memories[:3], 1):
            print(f"  {i}. {mem['text'][:80]}...")
    else:
        print("ℹ️  No relevant records found")

    return {
        "memory_context": relevant_context if relevant_context else None,
        "next_action": "reason",
        "tool_results": [{
            "tool": "memory/chromadb",
            "query": user_query,
            "result_count": len(memories),
            "retrieved": [m["text"][:100] for m in memories[:3]]
        }],
        "messages": [{
            "role": "assistant",
            "content": (
                f"[Memory] Found {len(memories)} past analyses and added to context."
                if memories else "[Memory] No relevant past analysis found."
            )
        }]
    }
