# Findings & Decisions

## Requirements
- Computer Vision: CNNs, ViTs, Object Detection, Segmentation, OCR, OpenCV
- LLM Proficiency: fine-tuning, advanced APIs, multi-modal integration
- Agentic Design: planning loops, tool use, memory management
- Orchestration: LangGraph, video/API/action coordination
- Production: FastAPI, Docker, evaluation metrics, scalability
- Bonus: Voice AI, LangGraph, Video Analytics, K8s, MCP, Streamlit/Gradio

## Research Findings

### Phase 1: CV Pipeline
- YOLOv8n: 3.2M param, hızlı ama düşük accuracy
- YOLOv8m: 25.9M param, daha doğru ama yavaş
- EasyOCR en yavaş bileşen (~4s) — pipeline bottleneck
- Segmentation mask piksel alanı, nesne boyutu karşılaştırması için kullanılabilir

### Phase 2: LLM Integration
- DeepSeek V3 reasoning kalitesi GPT-4o-mini seviyesinde, çok daha ucuz
- CoT prompting, karmaşık sorularda doğruluğu artırıyor (5 adım OBSERVE→RELATE→INTERPRET→REASON→ASSESS)
- Multi-turn conversation state: tüm history gönderimi basit ama token maliyeti artıyor
- Function calling: OpenAI ve DeepSeek aynı format, Anthropic farklı (input_schema vs parameters)
- JSON mode (response_format): DeepSeek ve OpenAI destekliyor — LLM'i geçerli JSON döndürmeye zorlar
- Few-shot örnekler format tutarlılığını artırıyor, 1-3 örnek yeterli

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| DeepSeek V3 varsayılan | Ucuz, OpenAI-uyumlu, güçlü reasoning |
| 3 prompt stratejisi | Farklı karmaşıklık seviyelerine göre seçilebilir |
| Provider abstraction | Runtime'da LLM değiştirilebilir |
| Pydantic output parsing | LLM çıktısını validate et, type safety |
| CV pipeline as tool | LLM'in CV pipeline'ı function calling ile çağırabilmesi |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| venv env var erişimi | .env + python-dotenv |

## Resources
- plan.md — Ana hazırlık planı
- Efsora job posting gereksinimleri plan.md Checklist bölümünde
- project1_cv_pipeline/INTERVIEW_NOTES.md — CV mülakat notları
- project2_llm_integration/INTERVIEW_NOTES.md — LLM mülakat notları

### Phase 3: Agent Architecture (LangGraph)
- LangGraph StateGraph: TypedDict state + reducer'lar (operator.add) ile birikimli veri
- Agent pattern: Plan-and-Execute + ReAct + Reflection üçlüsünü birleştirdik
- Conditional edges: router fonksiyonu string döner → mapping ile node'a yönlendirir
- Infinite loop koruması: max_iterations + evaluator auto-pass + timeout
- ChromaDB semantic search: all-MiniLM-L6-v2 embedding, cosine distance
- Memory tipleri: short-term (conversation buffer, sliding window) + long-term (vector store, RAG)
- Project arası import: importlib.util.spec_from_file_location (src/ namespace çakışması çözümü)
- LangGraph compile: graph.compile() → çalıştırılabilir, .invoke() ile baştan sona çalıştır
- Graph visualization: graph.get_graph().draw_mermaid() → Mermaid diagram otomatik

## Visual/Browser Findings
-

---
*Update this file after every 2 view/browser/search operations*
