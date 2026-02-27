# Task Plan: Efsora Senior AI Engineer Interview Preparation

## Goal
plan.md'deki 6 projeyi + mock interview'u sırasıyla uygulayarak Efsora Senior AI Engineer pozisyonuna tam hazırlanmak.

## Current Phase
Phase 4 — Fine-tuning Lab (**IN PROGRESS**)

## Phases

### Phase 1: CV Pipeline — "Visual Perception Engine" (3-4 gün)
- [x] YOLOv8 ile object detection pipeline
- [x] SAM veya YOLOv8-seg ile instance segmentation
- [x] EasyOCR veya PaddleOCR ile text extraction
- [x] Tüm çıktıları structured JSON formatında birleştir
- [x] OpenCV preprocessing (resize, crop, color space)
- [x] Mülakat sorularını çalış (YOLO, NMS, mAP, ViT vs CNN)
- **Status:** complete

### Phase 2: LLM Entegrasyonu — "Visual Reasoner" (3-4 gün)
- [x] CV JSON çıktısını LLM prompt'una optimal yerleştir (prompt_engine.py)
- [x] Function calling / Tool use mekanizması kur (tool_registry.py)
- [x] Multi-turn reasoning — görüntü hakkında Q&A (visual_reasoner.py)
- [x] Multi-provider LLM client — DeepSeek, OpenAI, Anthropic, Gemini (llm_client.py)
- [x] Prompt engineering: system prompt, few-shot, CoT (3 strateji)
- [x] Pydantic structured output parsing (output_parser.py)
- [x] Mülakat sorularını çalış (INTERVIEW_NOTES.md)
- **Status:** complete — kullanıcı çalışacak, sonra Phase 3

### Phase 3: Agent Architecture — "Multi-Modal Agent" + Voice AI (4-5 gün)
- [x] **3.1** Proje yapısı + LangGraph kurulumu
- [x] **3.2** AgentState tanımı — TypedDict ile graph state
- [x] **3.3** Planner Node — LLM kullanıcı isteğini adımlara böler
- [x] **3.4** Router Node — conditional edges ile doğru tool'a yönlendir
- [x] **3.5** Vision Node — CV pipeline'ı LangGraph node olarak entegre et
- [x] **3.6** Reasoner Node — tool sonuçlarını sentezle, final cevap üret
- [x] **3.7** Evaluator Node — cevap kalitesini değerlendir, gerekirse loop back
- [x] **3.8** Memory: short-term (conversation buffer) + long-term (ChromaDB vector store)
- [x] **3.9** Human-in-the-loop — kritik kararlarda onay mekanizması (interrupt_before)
- [x] **3.10** Voice AI: Whisper (ASR) + Edge-TTS tool'ları
- [x] **3.11** run_agent.py CLI + demo senaryoları (analyze, ask, interactive, tts, asr, voice, graph, memory-demo)
- [x] **3.12** INTERVIEW_NOTES.md — LangGraph, ReAct, memory, MCP, Voice AI mülakat notları (11 konu)
- **Status:** complete ✅ (2026-02-24)

### Phase 4: Fine-tuning Lab (3-4 gün)
- [ ] **4.1** Proje yapısı + requirements kurulumu
- [ ] **4.2** CV Part A: Custom dataset hazırlama (YOLO format, augmentation)
- [ ] **4.3** CV Part A: YOLOv8 fine-tune (güvenlik ekipmanı tespiti — hardhat/vest)
- [ ] **4.4** CV Part A: Transfer learning stratejileri (freeze, LR scheduling, early stopping)
- [ ] **4.5** CV Part A: Evaluation (mAP, precision, recall, confusion matrix)
- [ ] **4.6** LLM Part B: Instruction dataset hazırlama (visual reasoning format)
- [ ] **4.7** LLM Part B: LoRA/QLoRA fine-tune (PEFT + bitsandbytes)
- [ ] **4.8** LLM Part B: Training loop (gradient accumulation, warmup, checkpointing)
- [ ] **4.9** LLM Part B: Evaluation (perplexity, task-specific accuracy)
- [ ] **4.10** run_finetuning.py CLI + demo senaryoları
- [ ] **4.11** INTERVIEW_NOTES.md — LoRA, QLoRA, quantization, transfer learning mülakat notları
- **Status:** in_progress

### Phase 5: Video Analytics Pipeline (3-4 gün)
- [ ] Frame extraction + keyframe detection
- [ ] Object tracking (ByteTrack / DeepSORT)
- [ ] Temporal reasoning — LLM ile zaman değişim özeti
- [ ] Alert system — event-driven action trigger
- [ ] Streaming pipeline: async processing, queue management
- [ ] Mülakat sorularını çalış
- **Status:** pending

### Phase 6: Production Deployment — "Ship It" (3-4 gün)
- [ ] FastAPI RESTful API + WebSocket
- [ ] Docker multi-stage build + GPU support
- [ ] Evaluation dashboard (latency, accuracy, cost)
- [ ] Cost optimization (caching, batching, model routing)
- [ ] Streamlit/Gradio demo arayüzü
- [ ] Mülakat sorularını çalış
- **Status:** pending

### Phase 7: Mock Interview (3 round)
- [ ] Round 1: Teknik Derinlik (45 dk) — her projeden 2-3 soru
- [ ] Round 2: System Design (45 dk) — open-ended mimari soru
- [ ] Round 3: Behavioral + Code (30 dk) — hikayeleştirme + live coding
- **Status:** pending

## Key Questions
1. Her proje için hangi dataset/görüntüler kullanılacak?
2. GPU kaynağı mevcut mu? (fine-tuning ve inference için)
3. API key'ler hazır mı? (OpenAI, Anthropic, Gemini)
4. Projelerin her biri ayrı repo mu yoksa monorepo mu?

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| Faz faz ilerlenecek | Kullanıcı her fazı bitirip onayladıktan sonra sonrakine geçilecek |
| plan.md referans alınacak | Tüm proje detayları plan.md'de mevcut |
| DeepSeek V3 varsayılan LLM | Ucuz (~$0.27/1M input), OpenAI-uyumlu API, güçlü reasoning |
| 3 prompt stratejisi | Direct, Few-Shot, CoT — farklı karmaşıklık seviyelerine göre |
| Provider abstraction | Tek interface ile birden fazla LLM, runtime'da değiştirilebilir |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| venv API key erişimi | 1 | .env dosyası + python-dotenv ile çözüldü |

## Notes
- Kullanıcı her faz bittiğinde "bitti burası çalıştım" diyecek, sonra sonraki faza geçilecek
- Her fazda hem kod yazılacak hem mülakat soruları çalışılacak
- Update phase status as you progress: pending → in_progress → complete
