# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a multi-modal AI agent training project for Efsora Senior AI Engineer interview preparation. The project consists of 6 sequential phases that build upon each other:

1. **CV Pipeline** (completed) - Visual Perception Engine with YOLOv8, EasyOCR, OpenCV
2. **LLM Integration** (completed) - Visual Reasoner with multi-provider LLM support
3. **Agent Architecture** (pending) - LangGraph-based multi-modal agent
4. **Fine-tuning Lab** (pending) - CV and LLM fine-tuning with LoRA/QLoRA
5. **Video Analytics Pipeline** (pending) - Real-time video processing with object tracking
6. **Production Deployment** (pending) - FastAPI, Docker, evaluation dashboard

## Development Commands

### CV Pipeline (Project 1)
```bash
cd project1_cv_pipeline
python run_pipeline.py <image_path> [options]
```
Options:
- `--output, -o`: JSON output path
- `--no-detection`: Skip object detection
- `--no-segmentation`: Skip segmentation
- `--no-ocr`: Skip OCR
- `--confidence <float>`: Confidence threshold (default: 0.25)
- `--include-masks`: Include base64 mask data in JSON
- `--max-size <int>`: Max image dimension (default: 1280)

### LLM Visual Reasoner (Project 2)
```bash
cd project2_llm_integration
python run_reasoner.py <mode> [options]
```
Modes: `standard`, `tools`, `compare`, `safety`, `interactive`

Key options:
- `--provider <name>`: deepseek (default), openai, anthropic, gemini
- `--cv-result <path>`: Pre-computed CV JSON output
- `--question <text>`: Question to ask about the image

## Architecture

### Project 1: CV Pipeline (`project1_cv_pipeline/`)

**Entry Point**: `run_pipeline.py` → `src/pipeline.py:VisualPerceptionPipeline`

Core modules:
- `src/detector.py` - YOLOv8 object detection
- `src/segmentor.py` - YOLOv8-seg instance segmentation
- `src/ocr_engine.py` - EasyOCR text extraction
- `src/preprocessor.py` - OpenCV image preprocessing
- `src/pipeline.py` - Main orchestrator producing structured JSON

Output format:
```json
{
  "image_path": "...",
  "image_info": {"width": 1280, "height": 720},
  "objects": [{"label": "car", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}],
  "segments": [{"label": "car", "mask_base64": "...", "bbox": [...]}],
  "text_regions": [{"text": "ABC 123", "bbox": [...], "confidence": 0.9}],
  "scene_description": "Detected: 2 cars, 1 bus. Text found: ABC 123.",
  "processing_time": {"detection": 0.5, "segmentation": 0.3, "ocr": 0.8, "total": 1.6}
}
```

### Project 2: LLM Integration (`project2_llm_integration/`)

**Entry Point**: `run_reasoner.py` → `src/visual_reasoner.py:VisualReasoner`

Key pattern: The VisualReasoner lazy-loads the CV pipeline to avoid circular dependencies. It orchestrates:
1. CV pipeline execution (or loads pre-computed result)
2. Prompt formatting (direct/few-shot/CoT strategies)
3. LLM inference with multi-provider support
4. Structured output parsing with Pydantic

Core modules:
- `src/llm_client.py` - Unified interface for DeepSeek, OpenAI, Anthropic, Gemini
- `src/prompt_engine.py` - Prompt strategies: direct, few_shot, cot (Chain-of-Thought)
- `src/tool_registry.py` - Function calling tools for LLM
- `src/output_parser.py` - Pydantic models: ReasoningResponse, SceneAnalysis, SafetyViolation
- `src/visual_reasoner.py` - Main orchestrator with multi-turn conversation support

**Multi-provider LLM abstraction**: Each provider has different API formats. The LLMClient unifies:
- Message formats (system prompt location differs)
- Tool calling (OpenAI vs Anthropic format)
- Multi-modal (image) input handling

### Project 3: Agent Architecture (`project3_agent_architecture/`)

**Entry Point**: `run_agent.py` → `src/graph.py:build_agent_graph()`

LangGraph-based multi-modal agent with Plan-Execute-Reflect pattern:
1. Planner node breaks user query into steps
2. Router conditionally routes to vision/reason/respond
3. Vision node runs CV pipeline (project1)
4. Reasoner node synthesizes results with LLM
5. Evaluator node scores answer quality → loop back if low
6. Memory: short-term (conversation buffer) + long-term (ChromaDB vector store)

Core modules:
- `src/state.py` - AgentState TypedDict with Annotated reducers
- `src/nodes.py` - 7 nodes: planner, router, vision, reasoner, evaluator, respond, human_approval
- `src/graph.py` - LangGraph StateGraph build, compile, visualization
- `src/memory.py` - ConversationMemory + VectorMemory (ChromaDB) + MemoryManager

CLI modes: `analyze` (image+query), `ask` (query only), `interactive` (multi-turn), `graph` (visualization), `memory-demo`

### Key Architectural Patterns

1. **Lazy Loading**: VisualReasoner loads CV pipeline only when needed via `@property cv_pipeline`
2. **Structured JSON Flow**: CV output → LLM prompt context → Structured reasoning response
3. **Provider-Agnostic Design**: Single LLMClient class supporting 4 providers with different APIs
4. **Multi-turn State**: Conversation history maintained in `_conversation_history` list
5. **Tool Calling Loop**: `analyze_with_tools()` implements ReAct-style agent loop

## Environment Setup

All projects share a single conda environment: `interview-study`

```bash
conda activate interview-study
# Dependencies are installed across all projects in this env

# For a new project, install its requirements:
cd project3_agent_architecture
pip install -r requirements.txt
cp .env.example .env  # Edit .env with API keys
```

Required API keys (set in `project2_llm_integration/.env`):
- `DEEPSEEK_API_KEY` (default provider, cheapest)
- `OPENAI_API_KEY` (optional)
- `ANTHROPIC_API_KEY` (optional)
- `GOOGLE_API_KEY` (optional)

## Testing

Current test results (from `progress.md`):
- CV Pipeline: `bus.jpg` → 6 objects, 6 segments, 9 text regions (4.189s) ✓
- DeepSeek CoT: Structured JSON with reasoning steps ✓
- Multi-turn follow-up: Context preservation ✓

## Project Progress Tracking

- `plan.md` - Full 6-phase project plan (Turkish language)
- `task_plan.md` - Detailed task breakdown
- `findings.md` - Research findings and notes
- `progress.md` - Phase completion status and test results
- Each project has `INTERVIEW_NOTES.md` with interview preparation questions

## Important Notes

1. **Language**: Project documentation uses Turkish (plan.md, task_plan.md), but code comments are English
2. **Model Downloads**: YOLOv8 models download automatically on first run (yolov8n.pt, yolov8n-seg.pt)
3. **Output Directory**: JSON outputs go to `project*/output/` directories
4. **DeepSeek vs Others**: DeepSeek V3 is the default LLM provider due to cost efficiency (~$0.27/1M input tokens)
