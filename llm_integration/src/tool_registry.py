"""
Tool Registry — Defines the tools available to the LLM.

Function Calling / Tool Use mechanism:
- We tell the LLM "you can use these tools" (with JSON schema)
- LLM decides which tool to call and its arguments
- We execute the tool and return the result to the LLM
- LLM interprets the result and answers the user

Interview note:
- Function calling vs Tool use: Same concept, different naming
  - OpenAI: "function calling" (older) → "tool use" (newer)
  - Anthropic: "tool use"
  - Conceptually the same: LLM → structured output → tool execution → result back
"""

import json
import sys
import os
from typing import Callable


class ToolRegistry:
    """
    Registers tools and manages schemas to be presented to the LLM.

    Flow:
    1. Register tools with register_tool()
    2. Get OpenAI format schema with get_openai_tools()
    3. Execute the tool chosen by the LLM with execute_tool()
    """

    def __init__(self):
        self._tools: dict[str, dict] = {}

    def register_tool(
        self,
        name: str,
        description: str,
        parameters: dict,
        handler: Callable
    ):
        """
        Register a new tool.

        Args:
            name: Tool name (LLM calls it by this name)
            description: Explains what it does (LLM chooses based on this)
            parameters: Parameter definition in JSON Schema format
            handler: Function to run when the tool is called
        """
        self._tools[name] = {
            "name": name,
            "description": description,
            "parameters": parameters,
            "handler": handler
        }

    def get_openai_tools(self) -> list[dict]:
        """
        Returns tool definitions in OpenAI function calling format.

        OpenAI format:
        {
          "type": "function",
          "function": {
            "name": "...",
            "description": "...",
            "parameters": { JSON Schema }
          }
        }
        """
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["parameters"]
                }
            }
            for tool in self._tools.values()
        ]

    def get_anthropic_tools(self) -> list[dict]:
        """
        Returns tool definitions in Anthropic tool use format.

        Anthropic format:
        {
          "name": "...",
          "description": "...",
          "input_schema": { JSON Schema }
        }
        """
        return [
            {
                "name": tool["name"],
                "description": tool["description"],
                "input_schema": tool["parameters"]
            }
            for tool in self._tools.values()
        ]

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Execute the tool and return the result as a string."""
        if name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {name}"})

        handler = self._tools[name]["handler"]
        try:
            result = handler(**arguments)
            if isinstance(result, dict):
                return json.dumps(result, indent=2, ensure_ascii=False)
            return str(result)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {str(e)}"})

    def list_tools(self) -> list[str]:
        return list(self._tools.keys())


# ─── Built-in Tools ───────────────────────────────────────────────────

def _analyze_image_tool(image_path: str, run_detection: bool = True,
                        run_segmentation: bool = True, run_ocr: bool = True) -> dict:
    """Run the CV pipeline as a tool."""
    # Add Project 1's path
    project1_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "project1_cv_pipeline"
    )
    if project1_path not in sys.path:
        sys.path.insert(0, project1_path)

    from src.pipeline import VisualPerceptionPipeline
    pipeline = VisualPerceptionPipeline()
    return pipeline.analyze(
        image_path,
        run_detection=run_detection,
        run_segmentation=run_segmentation,
        run_ocr=run_ocr
    )


def _calculate_tool(expression: str) -> dict:
    """Simple math calculation tool."""
    # Safe eval — math operations only
    allowed_names = {"abs": abs, "round": round, "min": min, "max": max, "sum": sum}
    try:
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return {"expression": expression, "result": result}
    except Exception as e:
        return {"expression": expression, "error": str(e)}


def _get_object_details_tool(cv_result_json: str, label: str) -> dict:
    """Get detailed information about a specific object type."""
    cv_result = json.loads(cv_result_json) if isinstance(cv_result_json, str) else cv_result_json

    # LLM might use different key names — accept them all
    objects = cv_result.get("objects") or cv_result.get("detections", [])
    segments = cv_result.get("segments") or cv_result.get("segmentations", [])

    objects = [o for o in objects if o.get("label") == label]
    segments = [s for s in segments if s.get("label") == label]

    return {
        "label": label,
        "detection_count": len(objects),
        "segmentation_count": len(segments),
        "detections": objects,
        "segments": segments,
        "avg_confidence": round(
            sum(o["confidence"] for o in objects) / len(objects), 4
        ) if objects else 0
    }


def create_default_registry() -> ToolRegistry:
    """Create a registry filled with default tools."""
    registry = ToolRegistry()

    registry.register_tool(
        name="analyze_image",
        description="Run the full CV pipeline on an image. Returns detected objects, segmentation masks, and OCR text in structured JSON format.",
        parameters={
            "type": "object",
            "properties": {
                "image_path": {
                    "type": "string",
                    "description": "Path to the image file to analyze"
                },
                "run_detection": {
                    "type": "boolean",
                    "description": "Whether to run object detection",
                    "default": True
                },
                "run_segmentation": {
                    "type": "boolean",
                    "description": "Whether to run instance segmentation",
                    "default": True
                },
                "run_ocr": {
                    "type": "boolean",
                    "description": "Whether to run OCR text extraction",
                    "default": True
                }
            },
            "required": ["image_path"]
        },
        handler=_analyze_image_tool
    )

    registry.register_tool(
        name="calculate",
        description="Evaluate a mathematical expression. Useful for computing areas, ratios, or distances from bounding box coordinates.",
        parameters={
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate (e.g., '(400-100) * (450-200)')"
                }
            },
            "required": ["expression"]
        },
        handler=_calculate_tool
    )

    registry.register_tool(
        name="get_object_details",
        description="Get detailed information about a specific object type from CV results. Returns all detections and segmentations for that label.",
        parameters={
            "type": "object",
            "properties": {
                "cv_result_json": {
                    "type": "string",
                    "description": "The CV pipeline result as JSON string"
                },
                "label": {
                    "type": "string",
                    "description": "Object label to filter (e.g., 'person', 'car', 'bus')"
                }
            },
            "required": ["cv_result_json", "label"]
        },
        handler=_get_object_details_tool
    )

    return registry
