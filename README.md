# Multi-Modal AI Agent

A comprehensive training project and implementation of a Multi-Modal AI Agent. This repository contains a modular system that combines Computer Vision (CV), Large Language Models (LLM), and Agentic Architectures (LangGraph) to process and reason over multi-modal inputs.

## Project Structure

This repository is organized into distinct phases/projects that build upon each other:

1. **`cv_pipeline/`**: Visual Perception Engine built with YOLOv8, EasyOCR, and OpenCV.
2. **`llm_integration/`**: Visual Reasoner with multi-provider LLM support (DeepSeek, OpenAI, Anthropic, Gemini).
3. **`agent_architecture/`**: LangGraph-based multi-modal agent with Plan-Execute-Reflect patterns.
4. **`finetuning_lab/`**: Utilities for fine-tuning CV models (YOLOv8) and LLMs (LoRA/QLoRA) on custom datasets.

*(Future phases like Video Analytics and Production Deployment are under development).*

## Getting Started

### Prerequisites

All projects share a single Conda environment.

```bash
conda create -n interview-study python=3.10
conda activate interview-study
# Install dependencies for specific projects, e.g.:
cd agent_architecture
pip install -r requirements.txt
```

### Configuration

Copy `.env.example` to `.env` in the respective project directories (e.g., `llm_integration/.env`) and add your API keys.

DeepSeek V3 is the default LLM provider due to cost efficiency, but OpenAI, Anthropic, and Google Gemini are also supported.

### Running the Agent

To try out the multi-modal agent:

```bash
cd agent_architecture
# Analyze an image with a specific question
python run_agent.py analyze --image ../cv_pipeline/data/bus.jpg -q "What's in this scene?"

# Interactive chat mode
python run_agent.py interactive --image ../cv_pipeline/data/bus.jpg
```

## Documentation

For detailed architectural information, development commands, and project-specific notes, please refer to [CLAUDE.md](./CLAUDE.md).

## Language Note

All codebase comments, docstrings, and public-facing documentation are in **English**. Certain private planning and progress tracking documents in the repository are kept in Turkish for internal use.
