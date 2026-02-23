"""
Visual Reasoner — Ana orkestratör.
CV pipeline çıktısını LLM'e besleyerek görüntüler hakkında reasoning yapar.

Akış:
1. Görüntüyü CV pipeline'dan geçir → structured JSON
2. JSON'ı prompt engine ile LLM formatına dönüştür
3. LLM'e gönder → reasoning response
4. (Opsiyonel) Tool calling loop — LLM ek veri isterse tool çağır
5. (Opsiyonel) Multi-turn — kullanıcı takip sorusu sorar

Mülakat notu:
- Bu dosya "Visual Reasoner" projesinin kalbi
- CV + LLM entegrasyonunun tek noktadan yönetilmesi → orchestration pattern
- Multi-turn conversation state yönetimi → memory management'a giriş
- Tool calling loop → Proje 3'teki agent loop'un temeli
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
    Görüntü analizi + LLM reasoning orkestratörü.

    Kullanım:
        reasoner = VisualReasoner(provider="deepseek")
        result = reasoner.analyze("bus.jpg", "Bu sahnede kaç kişi var?")
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
            model: Model adı (None → provider varsayılanı)
            prompt_strategy: "direct", "few_shot", veya "cot"
            tool_registry: Tool'lar (None → varsayılan registry)
        """
        self.llm = LLMClient(provider=provider, model=model)
        self.prompt_engine = PromptEngine()
        self.strategy = prompt_strategy
        self.tool_registry = tool_registry or create_default_registry()

        # Multi-turn conversation state
        self._conversation_history: list[dict] = []
        self._current_cv_result: Optional[dict] = None

        # CV pipeline lazy-load
        self._cv_pipeline = None

    @property
    def cv_pipeline(self):
        """CV pipeline'ı lazy load et — sadece gerektiğinde import."""
        if self._cv_pipeline is None:
            project1_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                "project1_cv_pipeline"
            )
            if project1_path not in sys.path:
                sys.path.insert(0, project1_path)

            from src.pipeline import VisualPerceptionPipeline
            self._cv_pipeline = VisualPerceptionPipeline()
        return self._cv_pipeline

    # ─── Ana Analiz ───────────────────────────────────────────────

    def analyze(
        self,
        image_path: str,
        question: str,
        cv_result: Optional[dict] = None
    ) -> dict:
        """
        Tek seferlik analiz: görüntü + soru → yapısal cevap.

        Args:
            image_path: Görüntü dosya yolu
            question: Kullanıcı sorusu
            cv_result: Önceden hesaplanmış CV sonucu (None → pipeline çalıştır)

        Returns:
            {"answer": "...", "reasoning_steps": [...], "cv_result": {...}, ...}
        """
        # 1. CV pipeline çalıştır (veya önceden verilmişi kullan)
        if cv_result is None:
            print(f"[CV Pipeline] Analyzing {image_path}...")
            cv_result = self.cv_pipeline.analyze(image_path)
            print(f"[CV Pipeline] Done in {cv_result['processing_time']['total']}s")

        self._current_cv_result = cv_result

        # 2. Prompt oluştur (seçilen stratejiye göre)
        messages = self._build_prompt(cv_result, question)

        # 3. LLM'e gönder
        print(f"[LLM] Sending to {self.llm.provider_name}/{self.llm.model_name}...")
        raw_response = self.llm.chat(messages, json_mode=True)
        print(f"[LLM] Response received")

        # 4. Parse et
        result = self._parse_response(raw_response)
        result["cv_result"] = cv_result
        result["provider"] = self.llm.provider_name
        result["model"] = self.llm.model_name
        result["strategy"] = self.strategy

        # 5. Conversation history'e ekle (multi-turn için)
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
        Tool calling ile analiz — LLM gerektiğinde tool'ları çağırabilir.

        Akış:
        1. İlk prompt'u gönder (tool tanımlarıyla birlikte)
        2. LLM tool çağırırsa → çalıştır → sonucu geri gönder
        3. LLM text cevap dönene kadar veya max_tool_rounds'a kadar tekrarla

        Bu Proje 3'teki agent loop'un basitleştirilmiş hali.
        """
        # CV pipeline çalıştır
        print(f"[CV Pipeline] Analyzing {image_path}...")
        cv_result = self.cv_pipeline.analyze(image_path)
        self._current_cv_result = cv_result

        # Tool tanımlarını al (provider'a göre format)
        if self.llm.provider_name == "openai":
            tools = self.tool_registry.get_openai_tools()
        elif self.llm.provider_name == "anthropic":
            tools = self.tool_registry.get_anthropic_tools()
        else:
            tools = self.tool_registry.get_openai_tools()

        # İlk prompt
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
                # LLM final cevap verdi
                result = self._parse_response(response["content"])
                result["cv_result"] = cv_result
                result["tool_rounds"] = round_num + 1
                return result

            # LLM tool çağırdı → çalıştır
            for tc in response["tool_calls"]:
                print(f"  [Tool Call] {tc['name']}({tc['arguments']})")
                tool_result = self.tool_registry.execute_tool(tc["name"], tc["arguments"])
                print(f"  [Tool Result] {tool_result[:200]}...")

                # OpenAI format: tool sonucunu geri gönder
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

        # Max rounds reached — son response'u al
        final_response = self.llm.chat(messages)
        result = self._parse_response(final_response)
        result["cv_result"] = cv_result
        result["tool_rounds"] = max_tool_rounds
        result["note"] = "Max tool rounds reached"
        return result

    # ─── Multi-turn Conversation ──────────────────────────────────

    def follow_up(self, question: str) -> dict:
        """
        Aynı görüntü hakkında takip sorusu sor.
        Önceki conversation history'yi korur.

        Mülakat notu:
        - Multi-turn conversation state yönetimi
        - Her turda tüm history'yi gönderiyoruz → token maliyeti artıyor
        - Proje 3'te bunu memory management ile optimize edeceğiz
        """
        if not self._conversation_history:
            raise ValueError("No active conversation. Call analyze() first.")

        # Yeni soruyu history'e ekle
        self._conversation_history.append({
            "role": "user",
            "content": f"Follow-up question: {question}"
        })

        # LLM'e gönder
        print(f"[LLM] Follow-up to {self.llm.provider_name}/{self.llm.model_name}...")
        raw_response = self.llm.chat(self._conversation_history, json_mode=True)

        # History'e ekle
        self._conversation_history.append({
            "role": "assistant",
            "content": raw_response
        })

        result = self._parse_response(raw_response)
        result["turn"] = len([m for m in self._conversation_history if m["role"] == "user"])
        return result

    def reset_conversation(self):
        """Conversation state'i sıfırla."""
        self._conversation_history = []
        self._current_cv_result = None

    # ─── Multi-modal LLM Karşılaştırma ───────────────────────────

    def compare_with_multimodal(
        self,
        image_path: str,
        question: str,
        multimodal_provider: Optional[str] = None,
        multimodal_model: Optional[str] = None
    ) -> dict:
        """
        CV pipeline + LLM reasoning vs direkt multi-modal LLM karşılaştırması.

        Mülakat sorusu: Ne zaman dedicated CV model, ne zaman multi-modal LLM?
        - CV pipeline: daha doğru bbox/mask, quantitative data, daha ucuz
        - Multi-modal LLM: daha iyi scene understanding, context, nuance
        - En iyisi: ikisini birlikte kullan (bu projenin yaptığı gibi)
        """
        # 1. CV pipeline + LLM reasoning
        pipeline_result = self.analyze(image_path, question)

        # 2. Multi-modal LLM direkt analiz
        mm_provider = multimodal_provider or self.llm.provider_name
        mm_model = multimodal_model or self.llm.model_name

        mm_client = LLMClient(provider=mm_provider, model=mm_model)
        print(f"[Multi-modal] Sending image to {mm_provider}/{mm_model}...")
        multimodal_response = mm_client.chat_with_image(image_path, question)

        # 3. Karşılaştırma prompt'u
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

    # ─── Özel Analiz Modları ──────────────────────────────────────

    def safety_inspection(self, image_path: str, cv_result: Optional[dict] = None) -> dict:
        """Güvenlik denetimi modu — safety violations tespit eder."""
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
        """Seçilen stratejiye göre prompt oluştur."""
        if self.strategy == "direct":
            return self.prompt_engine.build_direct_prompt(cv_result, question)
        elif self.strategy == "few_shot":
            return self.prompt_engine.build_few_shot_prompt(cv_result, question)
        elif self.strategy == "cot":
            return self.prompt_engine.build_cot_prompt(cv_result, question)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def _parse_response(self, raw_response: str) -> dict:
        """LLM cevabını parse et — JSON veya text."""
        try:
            parsed = parse_json_from_text(raw_response)
            return parsed
        except (json.JSONDecodeError, ValueError):
            # JSON parse başarısız → raw text olarak döndür
            return {
                "answer": raw_response,
                "reasoning_steps": [],
                "evidence": [],
                "confidence": 0.5,
                "follow_up_questions": [],
                "parse_warning": "Response was not valid JSON, returned as raw text"
            }
