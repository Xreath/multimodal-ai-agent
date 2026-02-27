"""
Visual Reasoner — Main orchestrator.
Feeds CV pipeline output to the LLM to reason about images.

Flow:
1. Pass image through CV pipeline → structured JSON
2. Convert JSON to LLM format via prompt engine
3. Send to LLM → reasoning response
4. (Optional) Tool calling loop — call tool if LLM requests more data
5. (Optional) Multi-turn — user asks a follow-up question

Interview note:
- This file is the heart of the "Visual Reasoner" project
- Single-point management of CV + LLM integration → orchestration pattern
- Multi-turn conversation state management → introduction to memory management
- Tool calling loop → foundation of the agent loop in Project 3
"""

import json
import sys
import os
from typing import Optional

from .prompt_engine import PromptEngine, SYSTEM_PROMPT_VISUAL_ANALYST, SYSTEM_PROMPT_SAFETY_INSPECTOR
from .llm_client import LLMClient
from .tool_registry import ToolRegistry, create_default_registry
from .output_parser import ReasoningResponse, SceneAnalysis, parse_json_from_text


class VisualReasoner:
    """
    Image analysis + LLM reasoning orchestrator.

    Usage:
        reasoner = VisualReasoner(provider="deepseek")
        result = reasoner.analyze("bus.jpg", "How many people are in this scene?")
    """

    def __init__(
        self,
        provider: str = "deepseek",
        model: Optional[str] = None,
        prompt_strategy: str = "cot",
        tool_registry: Optional[ToolRegistry] = None
    ):
        """
        Args:
            provider: LLM provider ("deepseek", "openai", "anthropic", "gemini")
            model: Model name (None → provider default)
            prompt_strategy: "direct", "few_shot", or "cot"
            tool_registry: Tools (None → default registry)
        """
        self.llm = LLMClient(provider=provider, model=model)
        self.prompt_engine = PromptEngine()
        self.strategy = prompt_strategy
        self.tool_registry = tool_registry or create_default_registry()

        # Multi-turn conversation state
        self._conversation_history: list[dict] = []
        self._current_cv_result: Optional[dict] = None

        # Lazy-load CV pipeline
        self._cv_pipeline = None

    @property
    def cv_pipeline(self):
        """Lazy load the CV pipeline — import only when needed."""
        if self._cv_pipeline is None:
            project1_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "project1_cv_pipeline"
            )

            # project2's "src" module is already loaded in sys.modules.
            # project1 also uses a "src" package → name collision.
            # Temporarily remove project2's src and load project1's.
            saved_src = sys.modules.pop("src", None)
            saved_src_subs = {k: sys.modules.pop(k) for k in list(sys.modules) if k.startswith("src.")}

            try:
                sys.path.insert(0, project1_path)
                from src.pipeline import VisualPerceptionPipeline
                self._cv_pipeline = VisualPerceptionPipeline()
            finally:
                # Restore project2's src module
                if project1_path in sys.path:
                    sys.path.remove(project1_path)
                if saved_src is not None:
                    sys.modules["src"] = saved_src
                sys.modules.update(saved_src_subs)

        return self._cv_pipeline

    # ─── Main Analysis ────────────────────────────────────────────────

    def analyze(
        self,
        image_path: str,
        question: str,
        cv_result: Optional[dict] = None
    ) -> dict:
        """
        One-shot analysis: image + question → structured answer.

        Args:
            image_path: Image file path
            question: User question
            cv_result: Pre-computed CV result (None → run pipeline)

        Returns:
            {"answer": "...", "reasoning_steps": [...], "cv_result": {...}, ...}
        """
        # 1. Run CV pipeline (or use provided)
        if cv_result is None:
            print(f"[CV Pipeline] Analyzing {image_path}...")
            cv_result = self.cv_pipeline.analyze(image_path)
            print(f"[CV Pipeline] Done in {cv_result['processing_time']['total']}s")

        self._current_cv_result = cv_result

        # 2. Build prompt (based on chosen strategy)
        messages = self._build_prompt(cv_result, question)

        # 3. Send to LLM
        print(f"[LLM] Sending to {self.llm.provider_name}/{self.llm.model_name}...")
        raw_response = self.llm.chat(messages, json_mode=True)
        print(f"[LLM] Response received")

        # 4. Parse response
        result = self._parse_response(raw_response)
        result["cv_result"] = cv_result
        result["provider"] = self.llm.provider_name
        result["model"] = self.llm.model_name
        result["strategy"] = self.strategy

        # 5. Add to conversation history (for multi-turn)
        self._conversation_history = messages + [
            {"role": "assistant", "content": raw_response}
        ]

        return result

    def analyze_with_tools(
        self,
        image_path: str,
        question: str,
        max_tool_rounds: int = 3
    ) -> dict:
        """
        Analysis with tool calling — LLM can call tools when needed.

        Flow:
        1. Send initial prompt (along with tool definitions)
        2. If LLM calls a tool → execute → send result back
        3. Repeat until LLM returns text response or max_tool_rounds reached

        This is a simplified version of the agent loop in Project 3.
        """
        # Run CV pipeline
        print(f"[CV Pipeline] Analyzing {image_path}...")
        cv_result = self.cv_pipeline.analyze(image_path)
        self._current_cv_result = cv_result

        # Get tool definitions (format depends on provider)
        if self.llm.provider_name == "openai":
            tools = self.tool_registry.get_openai_tools()
        elif self.llm.provider_name == "anthropic":
            tools = self.tool_registry.get_anthropic_tools()
        else:
            tools = self.tool_registry.get_openai_tools()

        # Initial prompt
        cv_context = self.prompt_engine.format_cv_context(cv_result)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_VISUAL_ANALYST},
            {
                "role": "user",
                "content": f"CV Analysis:\n{cv_context}\n\nQuestion: {question}"
            }
        ]

        # Tool calling loop
        for round_num in range(max_tool_rounds):
            print(f"[Tool Loop] Round {round_num + 1}/{max_tool_rounds}")

            response = self.llm.chat_with_tools(messages, tools)

            if response["type"] == "text":
                # LLM gave final answer
                result = self._parse_response(response["content"])
                result["cv_result"] = cv_result
                result["tool_rounds"] = round_num + 1
                return result

            # LLM called a tool → execute
            for tc in response["tool_calls"]:
                print(f"  [Tool Call] {tc['name']}({tc['arguments']})")
                tool_result = self.tool_registry.execute_tool(tc["name"], tc["arguments"])
                print(f"  [Tool Result] {tool_result[:200]}...")

                # OpenAI format: send tool result back
                if self.llm.provider_name == "openai":
                    messages.append(response["raw_message"])
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result
                    })
                elif self.llm.provider_name == "anthropic":
                    # Anthropic format
                    messages.append({"role": "assistant", "content": response["raw_response"].content})
                    messages.append({
                        "role": "user",
                        "content": [{
                            "type": "tool_result",
                            "tool_use_id": tc["id"],
                            "content": tool_result
                        }]
                    })

        # Max rounds reached — get final response
        final_response = self.llm.chat(messages)
        result = self._parse_response(final_response)
        result["cv_result"] = cv_result
        result["tool_rounds"] = max_tool_rounds
        result["note"] = "Max tool rounds reached"
        return result

    # ─── Multi-turn Conversation ──────────────────────────────────

    def follow_up(self, question: str) -> dict:
        """
        Ask a follow-up question about the same image.
        Maintains previous conversation history.

        Interview note:
        - Multi-turn conversation state management
        - We send the entire history each turn → token cost increases
        - In Project 3, we will optimize this with memory management
        """
        if not self._conversation_history:
            raise ValueError("No active conversation. Call analyze() first.")

        # Add new question to history
        self._conversation_history.append({
            "role": "user",
            "content": f"Follow-up question: {question}"
        })

        # Send to LLM
        print(f"[LLM] Follow-up to {self.llm.provider_name}/{self.llm.model_name}...")
        raw_response = self.llm.chat(self._conversation_history, json_mode=True)

        # Add to history
        self._conversation_history.append({
            "role": "assistant",
            "content": raw_response
        })

        result = self._parse_response(raw_response)
        result["turn"] = len([m for m in self._conversation_history if m["role"] == "user"])
        return result

    def reset_conversation(self):
        """Reset conversation state."""
        self._conversation_history = []
        self._current_cv_result = None

    # ─── Multi-modal LLM Comparison ──────────────────────────────

    def compare_with_multimodal(
        self,
        image_path: str,
        question: str,
        multimodal_provider: Optional[str] = None,
        multimodal_model: Optional[str] = None
    ) -> dict:
        """
        CV pipeline + LLM reasoning vs direct multi-modal LLM comparison.

        Interview question: When to use dedicated CV model, when multi-modal LLM?
        - CV pipeline: more accurate bbox/mask, quantitative data, cheaper
        - Multi-modal LLM: better scene understanding, context, nuance
        - Best approach: use both together (as this project does)
        """
        # 1. CV pipeline + LLM reasoning
        pipeline_result = self.analyze(image_path, question)

        # 2. Multi-modal LLM direct analysis
        mm_provider = multimodal_provider or self.llm.provider_name
        mm_model = multimodal_model or self.llm.model_name

        mm_client = LLMClient(provider=mm_provider, model=mm_model)
        print(f"[Multi-modal] Sending image to {mm_provider}/{mm_model}...")
        multimodal_response = mm_client.chat_with_image(image_path, question)

        # 3. Comparison prompt
        comparison_messages = self.prompt_engine.build_comparison_prompt(
            pipeline_result["cv_result"],
            multimodal_response,
            question
        )
        comparison = self.llm.chat(comparison_messages, json_mode=True)

        return {
            "question": question,
            "pipeline_analysis": pipeline_result,
            "multimodal_analysis": multimodal_response,
            "comparison": comparison
        }

    # ─── Special Analysis Modes ───────────────────────────────────

    def safety_inspection(self, image_path: str, cv_result: Optional[dict] = None) -> dict:
        """Safety inspection mode — detects safety violations."""
        if cv_result is None:
            cv_result = self.cv_pipeline.analyze(image_path)

        cv_context = self.prompt_engine.format_cv_context(cv_result)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT_SAFETY_INSPECTOR},
            {
                "role": "user",
                "content": f"Analyze this scene for safety violations:\n\n{cv_context}"
            }
        ]

        raw_response = self.llm.chat(messages, json_mode=True)
        result = self._parse_response(raw_response)
        result["cv_result"] = cv_result
        result["mode"] = "safety_inspection"
        return result

    # ─── Internal Helpers ─────────────────────────────────────────

    def _build_prompt(self, cv_result: dict, question: str) -> list[dict]:
        """Build prompt based on chosen strategy."""
        if self.strategy == "direct":
            return self.prompt_engine.build_direct_prompt(cv_result, question)
        elif self.strategy == "few_shot":
            return self.prompt_engine.build_few_shot_prompt(cv_result, question)
        elif self.strategy == "cot":
            return self.prompt_engine.build_cot_prompt(cv_result, question)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _parse_response(self, raw_response: str) -> dict:
        """Parse LLM response — JSON or text."""
        try:
            parsed = parse_json_from_text(raw_response)
            return parsed
        except (json.JSONDecodeError, ValueError):
            # JSON parsing failed → return as raw text
            return {
                "answer": raw_response,
                "reasoning_steps": [],
                "evidence": [],
                "confidence": 0.5,
                "follow_up_questions": [],
                "parse_warning": "Response was not valid JSON, returned as raw text"
            }
