# Efsora Senior AI Engineer — Proje Bazlı Hazırlık Planı

## Genel Bakış

Bu plan, Efsora'nın "Senior AI Engineer" ilanındaki tüm gereksinimleri proje bazlı öğrenme ile kapsıyor. Her proje bir öncekinin üzerine biniyor ve sonunda tümü birleşerek **multi-modal agentic bir sistem** oluşturuyor.

**Toplam Süre:** ~4-5 hafta  
**Araç:** Claude Code + VS Code  
**Son Aşama:** Mock Interview  

---

## Proje 1: CV Pipeline — "Visual Perception Engine"
**Süre:** 3-4 gün  
**İlana karşılık gelen yetkinlikler:**
- Computer Vision Mastery (CNNs, ViTs, Object Detection, Segmentation, OCR)
- OpenCV proficiency

### Ne Yapacaksın
Bir görüntü alan ve içindeki tüm bilgiyi çıkaran bir CV pipeline:
1. **Object Detection** — YOLOv8 ile nesneleri tespit et
2. **Segmentation** — SAM (Segment Anything) veya YOLOv8-seg ile instance segmentation
3. **OCR** — EasyOCR veya PaddleOCR ile text extraction
4. **Output** — Tüm çıktıları structured JSON formatında üret

### Öğrenilecek Konular
- YOLOv8 architecture ve inference pipeline
- CNN vs Vision Transformer (ViT) farkları — mülakatta sorulur
- Object detection metrikleri: mAP, IoU, precision, recall
- Segmentation tipleri: semantic vs instance vs panoptic
- OCR pipeline: detection → recognition → post-processing
- OpenCV temel işlemler: resize, crop, color space, preprocessing

### Çıktı
```
input: image.jpg
output: {
  "objects": [{"label": "car", "confidence": 0.95, "bbox": [x1,y1,x2,y2]}],
  "segments": [{"label": "car", "mask": "base64..."}],
  "text_regions": [{"text": "ABC 123", "bbox": [x1,y1,x2,y2]}],
  "scene_description": "A parking lot with 3 cars and a license plate"
}
```

### Mülakat Soruları Bu Projeden
- YOLO nasıl çalışır? Anchor-based vs anchor-free farkı nedir?
- Faster R-CNN ile YOLO'nun farkı nedir? Ne zaman hangisi?
- ViT'nin CNN'den farkı nedir? Avantaj/dezavantajları?
- mAP nasıl hesaplanır?
- Non-Maximum Suppression (NMS) nedir, neden gerekli?

---

## Proje 2: LLM Entegrasyonu — "Visual Reasoner"
**Süre:** 3-4 gün  
**İlana karşılık gelen yetkinlikler:**
- LLM Proficiency (fine-tuning, advanced APIs)
- Multi-Modal Integration (CV output → LLM context)
- Prompt Engineering

### Ne Yapacaksın
Proje 1'in CV çıktısını LLM'e besleyerek reasoning yapan bir sistem:
1. **Structured Prompting** — CV JSON çıktısını LLM prompt'una optimal şekilde yerleştir
2. **Function Calling** — LLM'in CV pipeline'ı tool olarak kullanmasını sağla
3. **Multi-turn Reasoning** — Görüntü hakkında soru-cevap yapabilen bir sistem
4. **Comparison** — Aynı görüntüyü GPT-4V/Gemini Vision'a da gönderip sonuçları karşılaştır

### Öğrenilecek Konular
- OpenAI, Anthropic, Gemini API kullanımı
- Function calling / Tool use mekanizması
- Prompt engineering: system prompt, few-shot, chain-of-thought
- Structured output parsing (JSON mode, Pydantic)
- Multi-modal LLM'ler: GPT-4V, Gemini Vision, LLaVA mimarileri
- Token management ve cost optimization
- Dedicated CV model vs multi-modal LLM: ne zaman hangisi?

### Çıktı
```python
# Kullanıcı bir görüntü veriyor
result = visual_reasoner.analyze("warehouse.jpg", 
    question="Bu depoda güvenlik ihlali var mı?")

# Sistem: CV pipeline çalışır → JSON çıktı → LLM reasoning
# Response: "Evet, 2 işçi baret takmıyor (tespit edilen 5 kişiden).
#            Ayrıca forklift geçiş alanında engel var."
```

### Mülakat Soruları Bu Projeden
- CV çıktısını LLM'e nasıl besliyorsun? Neden structured JSON?
- Function calling nasıl çalışır? Tool use ile farkı?
- Ne zaman dedicated CV model, ne zaman multi-modal LLM kullanırsın?
- Token limiti aşarsa ne yaparsın? Context window management?
- Prompt injection risklerini nasıl handle edersin?

---

## Proje 3: Agent Architecture — "Multi-Modal Agent"
**Süre:** 4-5 gün  
**İlana karşılık gelen yetkinlikler:**
- Agentic Design (planning loops, tool use, memory management)
- Orchestration Frameworks (LangGraph)
- Multi-Modal Architecture & Agents
- Voice AI (ASR/TTS) — Bonus

### Ne Yapacaksın
LangGraph ile tam bir agentic sistem:
1. **Agent Loop** — Plan → Observe (vision) → Reason (LLM) → Act → Evaluate
2. **Tool Registry** — CV pipeline, web search, calculator, file ops gibi tool'lar
3. **Memory** — Short-term (conversation) + Long-term (vector store) memory
4. **Multi-step Reasoning** — Karmaşık görevleri adımlara bölen planner
5. **Human-in-the-loop** — Kritik kararlarda onay mekanizması
6. **Voice AI (Bonus)** — Whisper ile ses→text (ASR), Edge-TTS/OpenAI TTS ile text→ses (TTS) tool'ları

### Öğrenilecek Konular
- LangGraph: nodes, edges, state, conditional routing
- Agent patterns: ReAct, Plan-and-Execute, Reflection
- Tool use design: tool schema, error handling, retry logic
- Memory management: conversation buffer, summary memory, vector memory
- State management: checkpointing, rollback
- Multi-agent collaboration (supervisor, worker pattern)
- MCP (Model Context Protocol) — ilanda bonus olarak geçiyor
- Voice AI: Whisper (ASR) mimarisi, mel spectrogram, encoder-decoder yapısı
- TTS modelleri: Edge-TTS, OpenAI TTS, Bark — farkları ve kullanım alanları
- Audio preprocessing: sample rate, chunking, VAD (Voice Activity Detection)

### Mimari
```
User Query + Image/Audio
       ↓
   [Planner Node]  ← LLM decides steps
       ↓
   [Router]  → conditional edges
       ↓
   ┌──────────────────────────────┐
   │  [Vision Node] → YOLO/OCR   │
   │  [Speech Node] → Whisper/TTS│
   │  [Search Node] → Web search │
   │  [Code Node]  → Execute     │
   │  [Memory Node] → Retrieve   │
   └──────────────────────────────┘
       ↓
   [Reasoner Node]  ← synthesize results
       ↓
   [Evaluator Node] ← check quality
       ↓
   Response (or loop back to Planner)
```

### Mülakat Soruları Bu Projeden
- ReAct pattern nedir? Plan-and-Execute'dan farkı?
- LangGraph'ta state nasıl yönetilir?
- Agent loop'ta infinite loop'u nasıl önlersin?
- Memory'de ne zaman conversation buffer, ne zaman vector store?
- Multi-agent system'de agent'lar arası iletişimi nasıl yönetirsin?
- MCP nedir, ne işe yarar?
- Whisper nasıl çalışır? Encoder-decoder mimarisi nasıl?
- ASR'da mel spectrogram nedir, neden kullanılır?
- Real-time speech processing'de latency nasıl düşürülür? (streaming, chunking, VAD)
- Whisper vs Google Speech-to-Text vs Azure STT — ne zaman hangisi?

---

## Proje 4: Fine-tuning Lab
**Süre:** 3-4 gün  
**İlana karşılık gelen yetkinlikler:**
- Fine-tune LLMs and domain-specific CV models on custom datasets
- HuggingFace Transformers
- PyTorch deep familiarity

### Ne Yapacaksın
Hem CV hem LLM tarafında fine-tuning:

**Part A — CV Fine-tuning:**
1. Custom dataset hazırlama (labeling, augmentation)
2. YOLOv8'i custom bir domain'e fine-tune et (örn: güvenlik ekipmanı tespiti)
3. Transfer learning stratejileri: freeze layers, learning rate scheduling

**Part B — LLM Fine-tuning:**
1. Dataset preparation: instruction format, quality filtering
2. LoRA/QLoRA ile küçük bir model fine-tune (Mistral-7B veya Llama-3-8B)
3. HuggingFace Transformers + PEFT + bitsandbytes
4. Evaluation: perplexity, task-specific metrics

### Öğrenilecek Konular
- Transfer learning: ne zaman full fine-tune, ne zaman LoRA?
- LoRA/QLoRA: rank selection, target modules, alpha
- Quantization: fp16, bf16, int8, int4 — ne zaman hangisi?
- Dataset curation: quality > quantity
- Training dynamics: learning rate, warmup, gradient accumulation
- Evaluation methodology: train/val/test split, overfitting detection
- Distributed training: DDP, FSDP (senin zaten çalıştığın konu)

### Mülakat Soruları Bu Projeden
- LoRA nasıl çalışır? Rank ne anlama gelir?
- Full fine-tune vs LoRA vs Prompt tuning — ne zaman hangisi?
- bf16 ile fp16 farkı nedir?
- Overfitting'i nasıl tespit edip önlersin?
- FSDP nedir, DDP'den farkı ne?
- Custom dataset ne kadar büyük olmalı?

---

## Proje 5: Video Analytics Pipeline
**Süre:** 3-4 gün  
**İlana karşılık gelen yetkinlikler:**
- Video Analytics (bonus)
- Real-time video feeds
- High-throughput media processing

### Ne Yapacaksın
Proje 1-3'ü video stream'e genişlet:
1. **Frame Extraction** — Video'dan akıllı frame sampling (her frame değil, değişim olan frameler)
2. **Object Tracking** — Detection + tracking (ByteTrack veya DeepSORT)
3. **Temporal Reasoning** — Zaman içindeki değişimleri LLM'e özetlet
4. **Alert System** — Belirli durumlar tespit edildiğinde action trigger

### Öğrenilecek Konular
- Video processing: frame extraction, keyframe detection
- Object tracking: SORT, DeepSORT, ByteTrack
- Temporal reasoning: event detection, activity recognition
- Streaming pipeline: async processing, queue management
- Performance: batch processing, GPU memory management

### Mülakat Soruları Bu Projeden
- Object detection vs object tracking farkı?
- Her frame'de inference yapmak yerine ne yaparsın?
- Real-time video pipeline'da latency'yi nasıl düşürürsün?
- Video analytics'te privacy concerns nasıl handle edilir?

---

## Proje 6: Production Deployment — "Ship It"
**Süre:** 3-4 gün  
**İlana karşılık gelen yetkinlikler:**
- Deploy and maintain AI systems in production
- Scalability, cost optimization, safety
- Docker, Kubernetes (bonus)
- FastAPI backend services
- Evaluation metrics

### Ne Yapacaksın
Tüm sistemi production-ready hale getir:
1. **FastAPI** — RESTful API + WebSocket (video stream için)
2. **Docker** — Multi-stage build, GPU support
3. **Evaluation Dashboard** — Task completion, latency, accuracy, cost tracking
4. **Cost Optimization** — Caching, batching, model selection logic
5. **Streamlit/Gradio Demo** — Hızlı demo arayüzü (bonus)

### Öğrenilecek Konular
- FastAPI: async endpoints, file upload, streaming responses
- Docker: multi-stage builds, NVIDIA container toolkit
- API design: rate limiting, auth, error handling
- Cost optimization: response caching, prompt caching, model routing
- Monitoring: logging, metrics, alerting
- Evaluation framework: automated benchmarks, A/B testing

### Mülakat Soruları Bu Projeden
- Production'da model inference latency'yi nasıl düşürürsün?
- Token cost'u nasıl optimize edersin?
- Model versioning nasıl yaparsın?
- A/B testing nasıl implement edersin?
- System design: Multi-modal agent system nasıl scale edersin?

---

## Mock Interview Plan

Projeler tamamlandıktan sonra 3 aşamalı mock interview:

### Round 1: Teknik Derinlik (45 dk)
Her projeden 2-3 soru. Mimari kararları savunma. Trade-off analizi.

### Round 2: System Design (45 dk)
"Design a multi-modal AI agent for warehouse safety monitoring" gibi bir open-ended soru. Whiteboard style çözüm.

### Round 3: Behavioral + Code (30 dk)
"Tell me about a challenging CV/LLM project" — proje deneyimlerini hikayeleştirme. Kısa bir live coding challenge.

---

## Checklist — İlan Gereksinimleri Eşleştirme

| İlan Gereksinimleri | Proje |
|---|---|
| Multi-Modal Architecture & Agents | Proje 3 |
| LLMs as reasoning engines + CV models | Proje 2, 3 |
| Orchestration (video, APIs, actions) | Proje 3, 5 |
| Multi-step reasoning (text + vision) | Proje 2, 3 |
| Backend services & APIs | Proje 6 |
| Prompt engineering & CV optimization | Proje 1, 2 |
| Fine-tune LLMs and CV models | Proje 4 |
| Production deployment & reliability | Proje 6 |
| Evaluation metrics | Proje 6 |
| CNNs, ViTs, YOLO, Faster R-CNN, Segmentation, OCR | Proje 1 |
| HuggingFace, OpenAI/Anthropic/Gemini APIs | Proje 2, 4 |
| Agentic Design (planning, tool use, memory) | Proje 3 |
| **Bonus:** Voice AI (ASR/TTS) | Proje 3 ✅ |
| **Bonus:** LangGraph, LlamaIndex | Proje 3 |
| **Bonus:** Video Analytics | Proje 5 |
| **Bonus:** Docker, Kubernetes | Proje 6 |
| **Bonus:** MCP | Proje 3 |
| **Bonus:** Streamlit/Gradio | Proje 6 |

---

## Başlangıç

Proje 1'den başla. Her proje bittiğinde:
1. ✅ Kod çalışıyor mu?
2. ✅ Mülakat sorularını cevaplayabiliyor musun?
3. ✅ Mimari kararlarını savunabilir misin?

Hazır olduğunda "Proje 1 başlayalım" de — Claude Code ile adım adım yaparız.
