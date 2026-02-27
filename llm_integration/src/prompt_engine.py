"""
Prompt Engine — Optimally converts CV pipeline JSON output to an LLM prompt.

3 Strategies:
1. Direct Injection — Embed CV JSON in the system prompt
2. Few-Shot — Add example input/output pairs
3. Chain-of-Thought (CoT) — Give direct step-by-step thinking instructions

Interview note:
- Prompt engineering is one of the most critical skills for a Senior AI Engineer role
- System prompt design shapes LLM behavior
- Few-shot examples "calibrate" the LLM — especially for format consistency
- CoT improves accuracy in complex reasoning tasks (Wei et al., 2022)
"""

import json
from typing import Optional


# ─── System Prompts ─────────────────────────────────────────────────

SYSTEM_PROMPT_VISUAL_ANALYST = """You are a Visual Scene Analyst AI. You analyze computer vision pipeline outputs to answer questions about images.

Your capabilities:
- Interpret object detection results (labels, confidence scores, bounding boxes)
- Analyze instance segmentation data (object masks and areas)
- Read OCR-extracted text from the scene
- Reason about spatial relationships between objects
- Identify potential safety concerns or anomalies

Guidelines:
- Base your answers ONLY on the provided CV data — do not hallucinate objects not in the data
- When confidence scores are low (<0.5), mention the uncertainty
- Use bounding box coordinates to reason about spatial relationships (top-left origin)
- If asked about something not visible in the data, say so explicitly
- Provide structured, actionable insights when possible

Response format: Always respond in JSON with this schema:
{
  "answer": "Your main answer",
  "reasoning_steps": ["Step 1: ...", "Step 2: ..."],
  "evidence": ["CV data point supporting your answer"],
  "confidence": 0.0-1.0,
  "follow_up_questions": ["Suggested follow-up"]
}"""

SYSTEM_PROMPT_SAFETY_INSPECTOR = """You are a Safety Inspector AI that analyzes visual scenes for potential hazards and safety violations.

You receive structured CV pipeline data and must:
1. Identify any safety violations or potential hazards
2. Assess severity (low/medium/high/critical)
3. Provide actionable recommendations
4. Consider context — a person near heavy machinery is different from a person in a park

Always respond in JSON with this schema:
{
  "violations": [
    {
      "description": "What the violation is",
      "severity": "low|medium|high|critical",
      "affected_objects": ["object labels involved"],
      "recommendation": "How to fix it"
    }
  ],
  "overall_risk": "low|medium|high|critical",
  "summary": "Brief safety assessment"
}"""


# ─── Prompt Builder ──────────────────────────────────────────────────

class PromptEngine:
    """Converts CV pipeline output to an LLM prompt."""

    @staticmethod
    def format_cv_context(cv_result: dict, include_raw_coords: bool = True) -> str:
        """
        Converts CV pipeline JSON into an LLM-friendly text format.

        Why not provide JSON directly?
        → We could (and we do in JSON mode), but human-readable text
          format improves the LLM's reasoning quality.
          It also saves tokens — we only include necessary fields.
        """
        lines = []

        # Image info
        info = cv_result.get("image_info", {})
        lines.append(f"Image: {info.get('width', '?')}x{info.get('height', '?')} pixels")
        lines.append("")

        # Objects
        objects = cv_result.get("objects", [])
        if objects:
            lines.append(f"=== Detected Objects ({len(objects)}) ===")
            for i, obj in enumerate(objects, 1):
                conf_str = f"{obj['confidence']:.1%}"
                bbox = obj.get("bbox", [])
                line = f"  {i}. {obj['label']} (confidence: {conf_str})"
                if include_raw_coords and bbox:
                    line += f" [bbox: x1={bbox[0]}, y1={bbox[1]}, x2={bbox[2]}, y2={bbox[3]}]"
                lines.append(line)
            lines.append("")

        # Segments
        segments = cv_result.get("segments", [])
        if segments:
            lines.append(f"=== Segmented Regions ({len(segments)}) ===")
            for i, seg in enumerate(segments, 1):
                area = seg.get("area_pixels", "?")
                lines.append(f"  {i}. {seg['label']} (conf: {seg['confidence']:.1%}, area: {area}px)")
            lines.append("")

        # Text regions
        text_regions = cv_result.get("text_regions", [])
        if text_regions:
            lines.append(f"=== OCR Text Regions ({len(text_regions)}) ===")
            for i, tr in enumerate(text_regions, 1):
                conf_str = f"{tr['confidence']:.1%}"
                lines.append(f"  {i}. \"{tr['text']}\" (confidence: {conf_str})")
            lines.append("")

        # Processing time
        pt = cv_result.get("processing_time", {})
        if pt:
            lines.append(f"Processing: detection={pt.get('detection', '?')}s, "
                          f"segmentation={pt.get('segmentation', '?')}s, "
                          f"ocr={pt.get('ocr', '?')}s, total={pt.get('total', '?')}s")

        return "\n".join(lines)

    @staticmethod
    def build_direct_prompt(
        cv_result: dict,
        user_question: str,
        system_prompt: str = SYSTEM_PROMPT_VISUAL_ANALYST
    ) -> list[dict]:
        """
        Direct Injection strategy.
        Embeds the CV context in the system prompt and sends the user question separately.

        This is the simplest but effective method:
        - System prompt → role + rules
        - User message → CV data + question
        """
        cv_context = PromptEngine.format_cv_context(cv_result)

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"Here is the computer vision analysis of the image:\n\n{cv_context}\n\nQuestion: {user_question}"
            }
        ]
        return messages

    @staticmethod
    def build_few_shot_prompt(
        cv_result: dict,
        user_question: str,
        system_prompt: str = SYSTEM_PROMPT_VISUAL_ANALYST
    ) -> list[dict]:
        """
        Few-Shot strategy.
        "Calibrates" the LLM by adding an example CV output + ideal response pair.

        Interview note:
        - Few-shot allows the LLM to keep the output format consistent
        - 1-3 examples are generally enough (more → wasted tokens)
        - Examples must be realistic — fake data misleads the LLM
        """
        cv_context = PromptEngine.format_cv_context(cv_result)

        # Example CV data + ideal response
        example_cv = """Image: 640x480 pixels

=== Detected Objects (3) ===
  1. forklift (confidence: 92.1%) [bbox: x1=100, y1=200, x2=400, y2=450]
  2. person (confidence: 88.5%) [bbox: x1=350, y1=180, x2=420, y2=460]
  3. person (confidence: 76.3%) [bbox: x1=50, y1=300, x2=120, y2=460]

=== OCR Text Regions (1) ===
  1. "DANGER ZONE" (confidence: 94.2%)"""

        example_answer = json.dumps({
            "answer": "Yes, there is a safety concern. A person (88.5% confidence) is detected very close to a forklift, with overlapping bounding boxes suggesting they are in the forklift's operating area. The 'DANGER ZONE' text confirms this is a restricted area.",
            "reasoning_steps": [
                "Step 1: Identified 1 forklift and 2 persons in the scene",
                "Step 2: Checked spatial proximity — person #1's bbox (x1=350) overlaps with forklift's bbox (x2=400)",
                "Step 3: OCR detected 'DANGER ZONE' text, confirming hazardous area",
                "Step 4: Person #2 (x1=50) is far from forklift (x1=100), appears safe"
            ],
            "evidence": [
                "Person bbox x1=350 overlaps forklift bbox x2=400",
                "OCR text 'DANGER ZONE' with 94.2% confidence"
            ],
            "confidence": 0.85,
            "follow_up_questions": [
                "Are both persons wearing safety equipment?",
                "Is the forklift currently in motion?"
            ]
        }, indent=2)

        messages = [
            {"role": "system", "content": system_prompt},
            # Few-shot example
            {
                "role": "user",
                "content": f"Here is the computer vision analysis of the image:\n\n{example_cv}\n\nQuestion: Is there any safety concern in this scene?"
            },
            {
                "role": "assistant",
                "content": example_answer
            },
            # Actual question
            {
                "role": "user",
                "content": f"Here is the computer vision analysis of the image:\n\n{cv_context}\n\nQuestion: {user_question}"
            }
        ]
        return messages

    @staticmethod
    def build_cot_prompt(
        cv_result: dict,
        user_question: str,
        system_prompt: str = SYSTEM_PROMPT_VISUAL_ANALYST
    ) -> list[dict]:
        """
        Chain-of-Thought (CoT) strategy.
        Increases reasoning quality by giving the LLM a "think step-by-step" instruction.

        Interview note:
        - CoT was proposed by Wei et al. (2022)
        - Provides a ~15-20% accuracy increase on complex questions
        - Disadvantage: uses more tokens → slower and more expensive
        - Even "Let's think step by step" works, but a structural instruction is better
        """
        cv_context = PromptEngine.format_cv_context(cv_result)

        cot_instruction = """Before answering, think through the problem step by step:

1. OBSERVE: List all relevant objects, their positions, and confidence scores
2. RELATE: Identify spatial relationships between objects (use bbox coordinates)
3. INTERPRET: What do the OCR texts tell us about the scene?
4. REASON: Combine observations to form your answer
5. ASSESS: Rate your confidence based on detection confidence scores

Show your reasoning in the reasoning_steps field."""

        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Here is the computer vision analysis of the image:\n\n"
                    f"{cv_context}\n\n"
                    f"{cot_instruction}\n\n"
                    f"Question: {user_question}"
                )
            }
        ]
        return messages

    @staticmethod
    def build_comparison_prompt(
        cv_result: dict,
        multimodal_response: str,
        user_question: str
    ) -> list[dict]:
        """
        A prompt comparing CV pipeline output and multi-modal LLM output.
        Direct image analysis of GPT-4V/Gemini Vision vs our pipeline.
        """
        cv_context = PromptEngine.format_cv_context(cv_result)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an AI evaluation expert. Compare two different analysis approaches "
                    "for the same image and provide a balanced assessment.\n\n"
                    "Respond in JSON:\n"
                    '{"comparison": "...", "cv_pipeline_strengths": [...], '
                    '"multimodal_strengths": [...], "recommendation": "..."}'
                )
            },
            {
                "role": "user",
                "content": (
                    f"Question about the image: {user_question}\n\n"
                    f"=== Approach 1: Dedicated CV Pipeline ===\n{cv_context}\n\n"
                    f"=== Approach 2: Multi-modal LLM Direct Analysis ===\n{multimodal_response}\n\n"
                    "Compare these two approaches. Which provides better information for this question?"
                )
            }
        ]
        return messages

    @staticmethod
    def build_tool_use_system_prompt(tools_schema: list[dict]) -> str:
        """
        Create a system prompt for tool use.
        This prompt tells the LLM which tools it can use.

        Note: While OpenAI and Anthropic have native function calling,
        why use a manual tool prompt? → Educational purposes + in Gemini
        function calling works differently, so a unified approach is needed.
        """
        tools_desc = json.dumps(tools_schema, indent=2)
        return f"""You are a Visual AI Assistant with access to tools.

Available tools:
{tools_desc}

When you need to use a tool, respond with:
{{"tool_call": {{"name": "tool_name", "arguments": {{...}}}}}}

When you have enough information to answer, respond with your final answer in JSON format.
Do NOT use a tool if you already have the data needed to answer."""
