"""
Structured Output Parser — Parses LLM outputs using Pydantic models.

Why Pydantic?
- JSON outputs from the LLM may not always be properly formatted
- We perform schema validation + type coercion with Pydantic
- OpenAI's JSON mode + Pydantic = reliable structured output
"""

from pydantic import BaseModel, Field
from typing import Optional


class SafetyViolation(BaseModel):
    """A detected safety violation."""
    description: str = Field(description="Description of the violation")
    severity: str = Field(description="Severity: low / medium / high / critical")
    affected_objects: list[str] = Field(default_factory=list, description="Related objects")
    recommendation: str = Field(description="Fix recommendation")


class SceneAnalysis(BaseModel):
    """Structured analysis of the scene by the LLM."""
    summary: str = Field(description="Short summary of the scene")
    object_relationships: list[str] = Field(
        default_factory=list,
        description="Relationships between objects (e.g. 'person is standing near the bus')"
    )
    potential_risks: list[str] = Field(
        default_factory=list,
        description="Potential risks or notable situations"
    )
    confidence_assessment: str = Field(
        default="medium",
        description="LLM's own assessment: low / medium / high"
    )


class ReasoningResponse(BaseModel):
    """LLM reasoning output — in Q&A format."""
    answer: str = Field(description="Main answer")
    reasoning_steps: list[str] = Field(
        default_factory=list,
        description="Chain-of-thought steps (CoT)"
    )
    evidence: list[str] = Field(
        default_factory=list,
        description="Supporting evidence from CV data"
    )
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="Confidence score 0-1")
    follow_up_questions: list[str] = Field(
        default_factory=list,
        description="Suggested follow-up questions"
    )


class ToolCallResult(BaseModel):
    """The result of a tool call."""
    tool_name: str
    arguments: dict
    result: Optional[str] = None
    success: bool = True
    error: Optional[str] = None


def parse_json_from_text(text: str) -> dict:
    """
    Extract JSON block from LLM output.
    Sometimes LLM wraps it with ```json ... ```, sometimes it returns JSON directly.
    """
    import json

    text = text.strip()

    # If there is a ```json ... ``` block, extract it
    if "```json" in text:
        start = text.index("```json") + 7
        end = text.index("```", start)
        text = text[start:end].strip()
    elif "```" in text:
        start = text.index("```") + 3
        end = text.index("```", start)
        text = text[start:end].strip()

    return json.loads(text)
