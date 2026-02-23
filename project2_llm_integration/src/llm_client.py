"""
LLM Client — Multi-provider LLM istemcisi.

4 provider destekler: DeepSeek, OpenAI, Anthropic, Google Gemini.
Her biri için:
- Chat completion (messages → response)
- Function calling / Tool use
- Streaming (opsiyonel)

Mülakat notu:
- Her provider'ın API'si farklı: message format, tool format, response format
- Unified bir abstraction yazarak provider-agnostic çalışabilirsin
- Token yönetimi kritik: GPT-4 ~$30/1M input, Claude ~$15/1M, Gemini ~$7/1M
- DeepSeek V3: OpenAI-uyumlu API, çok ucuz (~$0.27/1M input), güçlü reasoning
- Retry logic + exponential backoff production'da şart
"""

import json
import os
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


class LLMClient:
    """
    Unified LLM client — provider farkını soyutlar.

    Kullanım:
        client = LLMClient(provider="openai")
        response = client.chat(messages, tools=tools)
    """

    def __init__(self, provider: str = "deepseek", model: Optional[str] = None):
        """
        Args:
            provider: "deepseek", "openai", "anthropic", veya "gemini"
            model: Model adı. None ise provider'a göre varsayılan kullanılır.
        """
        self.provider = provider.lower()
        self._client = None
        self._model = model

        if self.provider == "deepseek":
            self._init_deepseek(model)
        elif self.provider == "openai":
            self._init_openai(model)
        elif self.provider == "anthropic":
            self._init_anthropic(model)
        elif self.provider == "gemini":
            self._init_gemini(model)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'deepseek', 'openai', 'anthropic', or 'gemini'")

    # ─── Provider Init ────────────────────────────────────────────

    def _init_deepseek(self, model: Optional[str]):
        """
        DeepSeek V3 — OpenAI-uyumlu API.

        Mülakat notu:
        - DeepSeek API, OpenAI SDK ile çalışır (base_url değiştirilerek)
        - Çok ucuz: ~$0.27/1M input, ~$1.10/1M output (GPT-4o'nun ~100x ucuzu)
        - MoE (Mixture of Experts) mimarisi — 671B parametre ama sadece 37B aktif
        - Reasoning kalitesi GPT-4 seviyesinde, özellikle kod ve matematik'te güçlü
        """
        from openai import OpenAI
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")
        self._client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com"
        )
        self._model = model or "deepseek-chat"

    def _init_openai(self, model: Optional[str]):
        from openai import OpenAI
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        self._client = OpenAI(api_key=api_key)
        self._model = model or "gpt-4o-mini"

    def _init_anthropic(self, model: Optional[str]):
        from anthropic import Anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set")
        self._client = Anthropic(api_key=api_key)
        self._model = model or "claude-sonnet-4-20250514"

    def _init_gemini(self, model: Optional[str]):
        import google.generativeai as genai
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable not set")
        genai.configure(api_key=api_key)
        self._model = model or "gemini-2.0-flash"
        self._genai = genai

    # ─── Chat (Tool'suz) ─────────────────────────────────────────

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048,
        json_mode: bool = False
    ) -> str:
        """
        Mesaj listesi gönder, string cevap al.

        Args:
            messages: [{"role": "system/user/assistant", "content": "..."}]
            temperature: 0.0-1.0 (düşük → deterministik, yüksek → yaratıcı)
            max_tokens: Maksimum yanıt token sayısı
            json_mode: True ise JSON formatında cevap zorla (OpenAI)

        Returns:
            LLM'in text yanıtı
        """
        if self.provider in ("openai", "deepseek"):
            return self._chat_openai(messages, temperature, max_tokens, json_mode)
        elif self.provider == "anthropic":
            return self._chat_anthropic(messages, temperature, max_tokens)
        elif self.provider == "gemini":
            return self._chat_gemini(messages, temperature, max_tokens)

    def _chat_openai(self, messages, temperature, max_tokens, json_mode) -> str:
        """
        OpenAI chat completion.

        Not: json_mode=True → response_format={"type": "json_object"}
        Bu, LLM'i geçerli JSON döndürmeye zorlar.
        Ama system prompt'ta JSON istediğini belirtmelisin — yoksa hata alırsın.
        """
        kwargs = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content

    def _chat_anthropic(self, messages, temperature, max_tokens) -> str:
        """
        Anthropic chat completion.

        Fark: Anthropic system prompt'u ayrı parametre olarak alır,
        messages listesinin içinde değil.
        """
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        kwargs = {
            "model": self._model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def _chat_gemini(self, messages, temperature, max_tokens) -> str:
        """
        Google Gemini chat completion.

        Fark: Gemini'de messages formatı farklı — "user" ve "model" rolleri var.
        system prompt ayrı parametre.
        """
        model = self._genai.GenerativeModel(
            model_name=self._model,
            generation_config={
                "temperature": temperature,
                "max_output_tokens": max_tokens
            }
        )

        # System prompt varsa ayır
        system_instruction = None
        gemini_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_instruction = msg["content"]
            elif msg["role"] == "user":
                gemini_messages.append({"role": "user", "parts": [msg["content"]]})
            elif msg["role"] == "assistant":
                gemini_messages.append({"role": "model", "parts": [msg["content"]]})

        if system_instruction:
            model = self._genai.GenerativeModel(
                model_name=self._model,
                system_instruction=system_instruction,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )

        chat = model.start_chat(history=gemini_messages[:-1] if len(gemini_messages) > 1 else [])
        last_message = gemini_messages[-1]["parts"][0] if gemini_messages else ""
        response = chat.send_message(last_message)
        return response.text

    # ─── Chat with Tools (Function Calling) ───────────────────────

    def chat_with_tools(
        self,
        messages: list[dict],
        tools: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> dict:
        """
        Tool'larla birlikte chat. LLM bir tool çağırmak isterse bunu döner.

        Returns:
            {
                "type": "text" | "tool_call",
                "content": "..." | None,
                "tool_calls": [{"name": "...", "arguments": {...}}] | None
            }
        """
        if self.provider in ("openai", "deepseek"):
            return self._chat_with_tools_openai(messages, tools, temperature, max_tokens)
        elif self.provider == "anthropic":
            return self._chat_with_tools_anthropic(messages, tools, temperature, max_tokens)
        else:
            # Gemini için manual tool calling (prompt-based)
            return self._chat_with_tools_manual(messages, tools, temperature, max_tokens)

    def _chat_with_tools_openai(self, messages, tools, temperature, max_tokens) -> dict:
        """
        OpenAI native function calling.

        Akış:
        1. tools parametresi ile gönder
        2. LLM tool_calls döndürürse → tool'u çalıştır → sonucu geri gönder
        3. LLM text döndürürse → final cevap
        """
        response = self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            tools=tools,
            temperature=temperature,
            max_tokens=max_tokens
        )

        choice = response.choices[0]

        if choice.message.tool_calls:
            tool_calls = []
            for tc in choice.message.tool_calls:
                tool_calls.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments)
                })
            return {
                "type": "tool_call",
                "content": choice.message.content,
                "tool_calls": tool_calls,
                "raw_message": choice.message
            }

        return {
            "type": "text",
            "content": choice.message.content,
            "tool_calls": None
        }

    def _chat_with_tools_anthropic(self, messages, tools, temperature, max_tokens) -> dict:
        """
        Anthropic native tool use.

        Fark: Anthropic tool sonucunu "tool_result" content block olarak bekler.
        """
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        kwargs = {
            "model": self._model,
            "messages": filtered_messages,
            "tools": tools,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        if system:
            kwargs["system"] = system

        response = self._client.messages.create(**kwargs)

        tool_calls = []
        text_content = ""

        for block in response.content:
            if block.type == "tool_use":
                tool_calls.append({
                    "id": block.id,
                    "name": block.name,
                    "arguments": block.input
                })
            elif block.type == "text":
                text_content += block.text

        if tool_calls:
            return {
                "type": "tool_call",
                "content": text_content or None,
                "tool_calls": tool_calls,
                "raw_response": response
            }

        return {
            "type": "text",
            "content": text_content,
            "tool_calls": None
        }

    def _chat_with_tools_manual(self, messages, tools, temperature, max_tokens) -> dict:
        """
        Manual tool calling — Gemini veya tool use desteklemeyen modeller için.
        Tool tanımlarını prompt'a gömeriz, LLM JSON ile tool çağrısı yapar.
        """
        response_text = self.chat(messages, temperature, max_tokens)

        # LLM'in tool_call JSON'ı döndürüp döndürmediğini kontrol et
        try:
            parsed = json.loads(response_text)
            if "tool_call" in parsed:
                tc = parsed["tool_call"]
                return {
                    "type": "tool_call",
                    "content": None,
                    "tool_calls": [{
                        "id": "manual_0",
                        "name": tc["name"],
                        "arguments": tc.get("arguments", {})
                    }]
                }
        except (json.JSONDecodeError, KeyError):
            pass

        return {
            "type": "text",
            "content": response_text,
            "tool_calls": None
        }

    # ─── Multi-modal (Görüntü gönderme) ──────────────────────────

    def chat_with_image(
        self,
        image_path: str,
        question: str,
        temperature: float = 0.3,
        max_tokens: int = 2048
    ) -> str:
        """
        Görüntüyü doğrudan multi-modal LLM'e gönder.
        GPT-4V / Gemini Vision karşılaştırması için kullanılır.

        Bu, CV pipeline'sız direkt analiz — karşılaştırma amacıyla.
        """
        import base64

        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode()

        if self.provider in ("openai", "deepseek"):
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            }]
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            import mimetypes
            media_type = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data
                        }
                    },
                    {"type": "text", "text": question}
                ]
            }]
            response = self._client.messages.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            return response.content[0].text

        elif self.provider == "gemini":
            import PIL.Image
            model = self._genai.GenerativeModel(self._model)
            img = PIL.Image.open(image_path)
            response = model.generate_content(
                [question, img],
                generation_config={"temperature": temperature, "max_output_tokens": max_tokens}
            )
            return response.text

    @property
    def model_name(self) -> str:
        return self._model

    @property
    def provider_name(self) -> str:
        return self.provider
