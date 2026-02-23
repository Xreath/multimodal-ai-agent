"""
Structured Output Parser — Pydantic modelleri ile LLM çıktılarını parse eder.

Neden Pydantic?
- LLM'den gelen JSON çıktıları her zaman düzgün olmayabilir
- Pydantic ile schema validation + type coercion yapıyoruz
- OpenAI'ın JSON mode'u + Pydantic = güvenilir structured output
"""

from pydantic import BaseModel, Field
from typing import Optional


class SafetyViolation(BaseModel):
    """Tespit edilen bir güvenlik ihlali."""
    description: str = Field(description="İhlalin açıklaması")
    severity: str = Field(description="Ciddiyet: low / medium / high / critical")
    affected_objects: list[str] = Field(default_factory=list, description="İlgili nesneler")
    recommendation: str = Field(description="Düzeltme önerisi")


class SceneAnalysis(BaseModel):
    """Sahne hakkında LLM'in yapısal analizi."""
    summary: str = Field(description="Sahnenin kısa özeti")
    object_relationships: list[str] = Field(
        default_factory=list,
        description="Nesneler arası ilişkiler (ör: 'kişi otobüsün yanında duruyor')"
    )
    potential_risks: list[str] = Field(
        default_factory=list,
        description="Olası riskler veya dikkat çeken durumlar"
    )
    confidence_assessment: str = Field(
        default="medium",
        description="LLM'in kendi değerlendirmesi: low / medium / high"
    )


class ReasoningResponse(BaseModel):
    """LLM'in reasoning çıktısı — soru-cevap formatında."""
    answer: str = Field(description="Ana cevap")
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Düşünce zinciri adımları (CoT)"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="CV verisinden destekleyici kanıtlar"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Güven skoru 0-1")
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Önerilen takip soruları"
    )


class ToolCallResult(BaseModel):
    """Bir tool call'un sonucu."""
    tool_name: str
    arguments: dict
    result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


def parse_json_from_text(text: str) -> dict:
    """
    LLM çıktısından JSON bloğu çıkar.
    Bazen LLM ```json ... ``` ile sarar, bazen direkt JSON döner.
    """
    import json

    text = text.strip()

    # ```json ... ``` bloğu varsa çıkar
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    return json.loads(text)
