# Proje 3: Agent Architecture — Mülakat Notları

## 1. LangGraph Nedir? Neden LangGraph?

**LangGraph**, LangChain ekosistemindeki **graph-based agent orchestration** framework'üdür.

**Normal LangChain vs LangGraph:**
| Özellik | LangChain (Chains) | LangGraph |
|---------|-------------------|-----------|
| Akış | Doğrusal (A → B → C) | Graph (döngüler, dallanma) |
| State | Yok (her zaman baştan) | Paylaşımlı state (TypedDict) |
| Döngü | Yok | Var (evaluator → planner loop) |
| Checkpoint | Yok | Var (her adımda snapshot) |
| Human-in-the-loop | Zor | Native (interrupt_before) |

**Neden LangGraph seçtik:**
- Agent'lar **döngüsel** çalışır (plan → act → observe → reason → repeat)
- **State yönetimi** critical — her adımda bilgi birikir
- **Conditional routing** — farklı tool'lara yönlendirme
- Production-ready: checkpointing, error handling, streaming

**Mülakat cevabı:**
> "LangGraph'ı seçtim çünkü agent'lar doğrusal pipeline değil, döngüsel graph'lardır.
> Plan → Execute → Evaluate → Revise döngüsünü LangGraph'ın StateGraph'ı ile
> doğal bir şekilde modelledim. Her node state'i okur ve günceller."

---

## 2. Agent Patterns: ReAct vs Plan-and-Execute vs Reflection

### ReAct (Reason + Act)
```
Thought: "Görüntüdeki nesneleri tespit etmem lazım"
Action: analyze_image("warehouse.jpg")
Observation: "3 kişi, 2 forklift tespit edildi"
Thought: "Güvenlik ekipmanlarını kontrol edeyim"
Action: check_safety(persons)
Observation: "2 kişi baretsiz"
Answer: "2 güvenlik ihlali tespit edildi"
```
- **Basit**, tek LLM çağrısı ile tool seçimi
- **Dezavantaj:** Uzun vadeli planlama yok, her adım kısa görüşlü

### Plan-and-Execute
```
Plan: [
  1. Görüntüyü analiz et (CV pipeline)
  2. Tespit edilen kişileri incele
  3. Güvenlik ekipmanlarını kontrol et
  4. Sonuçları raporla
]
Execute: Her adımı sırayla çalıştır
```
- **Daha iyi** uzun vadeli planlama
- **Dezavantaj:** Plan değiştirilmez (statik)

### Reflection (Self-Critique)
```
Generate: "Cevap: 2 ihlal var"
Evaluate: "Skor: 0.6 — forklift alanı da kontrol edilmeli"
Revise: "Cevap: 2 kişi ihlali + 1 alan ihlali = 3 ihlal"
```
- Cevap **kalitesini artırır**
- **Dezavantaj:** Ekstra LLM çağrısı = maliyet + latency

### Biz Üçünü Birleştirdik:
```
Plan (Plan-and-Execute) → Execute with Tools (ReAct) → Evaluate (Reflection)
```

---

## 3. State Management — LangGraph'ta State Nasıl Çalışır?

```python
class AgentState(TypedDict):
    user_query: str
    plan: list[str]
    # Annotated + operator.add → listelere EKLEME semantiği
    messages: Annotated[list[dict], operator.add]
    tool_results: Annotated[list[dict], operator.add]
```

**Reducer kavramı:**
- Normal dict update: `{"messages": [new]}` → eskiyi SİLER
- `operator.add` ile: `{"messages": [new]}` → eskiye EKLER
- Redux/Flux pattern'ından esinlenilmiş

**Checkpointing:**
- Her node'dan sonra state snapshot alınır
- Hata durumunda herhangi bir noktaya geri dönülebilir (rollback)
- `MemorySaver` → bellekte, `SqliteSaver` → disk'te

**Mülakat cevabı:**
> "State'i TypedDict olarak tanımlıyorum. Birikimli veriler (mesajlar, tool sonuçları)
> için Annotated reducer'lar kullanıyorum — bu sayede her node sadece kendi
> katkısını döner, önceki veriler korunur. Production'da MemorySaver ile
> checkpoint alıyorum — bu hem debugging hem rollback imkanı sağlıyor."

---

## 4. Infinite Loop Koruması

Agent loop'larda en büyük risk: **sonsuz döngü**.

**Önlemler:**
1. `max_iterations` parametresi (varsayılan: 5)
2. `iteration_count` her adımda artırılır
3. Router, max'a ulaşılınca "respond"'a yönlendirir
4. Evaluator, son iterasyonda düşük skor bile olsa geçirir

```python
# Router'da:
if iteration_count >= max_iterations:
    return "respond"  # Zorla bitir
```

**Mülakat cevabı:**
> "Üç katmanlı koruma: 1) Max iteration hard limit, 2) Evaluator son iterasyonda
> otomatik pass, 3) Timeout mechanism. Production'da ayrıca cost limiti de
> eklenir — belirli token sayısı aşılırsa dur."

---

## 5. Memory Management — Ne Zaman Hangisi?

### Conversation Buffer (Short-term)
- **Ne:** Son N mesajı tut
- **Ne zaman:** Aktif konuşma, son context gerekli
- **Trade-off:** N büyük → daha iyi context ama daha fazla token maliyeti
- **Sliding window:** En eski mesaj düşer (system prompt hariç)

### Vector Store / RAG (Long-term)
- **Ne:** Embedding ile semantic search
- **Ne zaman:** Önceki analizleri hatırla, cross-conversation bilgi
- **Akış:** Query → Embed → Search → Top-K → Context'e ekle → LLM'e gönder
- **ChromaDB:** Lightweight, embedded (SQLite'ın vektör versiyonu)

### Summary Memory
- **Ne:** Uzun konuşmayı LLM ile özetle
- **Ne zaman:** Çok uzun konuşmalarda token tasarrufu
- **Akış:** 20+ mesaj → LLM özetle → 1 mesaj

**Mülakat cevabı:**
> "Short-term memory'de sliding window (son 20 mesaj), system prompt korunur.
> Long-term memory'de ChromaDB + sentence-transformers embedding — geçmiş
> analizleri semantic search ile bulup RAG-style context'e ekliyorum.
> Çok uzun konuşmalarda summary memory ile sıkıştırma yapılabilir."

---

## 6. Human-in-the-Loop — Ne Zaman Gerekli?

**Gerekli olduğu durumlar:**
- Güvenlik: yanlış kararın maliyeti yüksekse (silme, gönderme, ödeme)
- Etik: hassas verilerle çalışırken (kişisel bilgi, sağlık)
- Düzenleyici: compliance gereksinimleri (finans, sağlık)
- Belirsizlik: Agent emin değilse (düşük confidence)

**Gerekli olmadığı durumlar:**
- Latency kritikse (real-time sistemler)
- Karar düşük riskli ise (bilgi sorgulama)
- Tam otomatik pipeline gerekli ise (batch processing)

**LangGraph'ta iki yöntem:**
1. `interrupt_before` — graph'ı durdur, insan karar versin, sonra devam et
2. Approval node — state'e bakarak koşullu onay iste (biz bunu kullandık)

---

## 7. Conditional Routing — Graph'ta Dallanma

```python
graph.add_conditional_edges(
    "planner",          # Kaynak node
    router_function,    # State'e bakıp string dönen fonksiyon
    {
        "vision": "vision",       # Router "vision" dönerse → vision node'a git
        "reason": "reasoner",     # Router "reason" dönerse → reasoner node'a git
        "respond": "respond",     # Router "respond" dönerse → respond node'a git
    }
)
```

**Router fonksiyonu:**
- State'i okur (next_action, iteration_count, needs_human_approval)
- String döner → mapping'deki karşılık node'a gider
- if/elif/else mantığı — basit ama güçlü

---

## 8. Tool Use Design — İyi Tool Nasıl Tasarlanır?

**İyi tool özellikleri:**
1. **Tek sorumluluk:** Bir tool bir iş yapar
2. **Açık description:** LLM tool'u seçerken buna bakar
3. **Type-safe parametreler:** JSON Schema ile tanım
4. **Error handling:** Hata durumunda anlamlı mesaj döner
5. **Idempotent:** Aynı input → aynı output (mümkünse)

**Bizim tool'larımız:**
| Tool | Sorumluluk |
|------|-----------|
| vision (CV pipeline) | Görüntü analizi → nesne/segment/text |
| reasoner (LLM) | Bilgi sentezi → cevap üretme |
| evaluator (LLM) | Kalite değerlendirme → skor |

---

## 9. Graph Visualization — Debugging ve Dökümantasyon

LangGraph'ın güçlü yönlerinden biri: graph yapısını otomatik görselleştirebilirsin.

```python
graph.get_graph().draw_mermaid()  # → Mermaid diagram (string)
graph.get_graph().draw_png()       # → PNG image (pygraphviz gerekli)
```

**Neden önemli:**
- Debugging: "Agent neden bu yola gitti?"
- Dökümantasyon: Otomatik mimari diagram
- Code review: Graph yapısı bir bakışta görünür

---

## 10. MCP (Model Context Protocol) — Bonus

**MCP Nedir:**
- Anthropic'in geliştirdiği standart protokol
- LLM'lerin harici araçlara ve veri kaynaklarına erişimini standartlaştırır
- "USB for AI" — farklı tool'lar aynı arayüz ile bağlanır

**MCP vs Function Calling:**
| Özellik | Function Calling | MCP |
|---------|-----------------|-----|
| Tanım | Provider-specific | Standart protokol |
| Transport | HTTP API | JSON-RPC (stdio, HTTP) |
| Discovery | Manuel tanım | Otomatik keşif |
| Scope | Tek LLM çağrısı | Tam uygulama yaşam döngüsü |

---

## 11. Production Considerations

**Latency optimizasyonu:**
- Tool sonuçlarını cache'le (aynı görüntü → aynı CV sonucu)
- Paralel tool çağrısı (bağımsız tool'lar aynı anda)
- Streaming response (kullanıcı beklemez)

**Cost optimizasyonu:**
- Evaluation node'u sadece complex sorularda çalıştır
- Summary memory ile token sıkıştırma
- Model routing: basit sorular → küçük model, karmaşık → büyük model

**Error handling:**
- Her node try/except ile sarılı olmalı
- LangGraph retry policy: `retry_policy=RetryPolicy(max_attempts=3)`
- Graceful degradation: bir tool fail olursa diğerleriyle devam et

---

## 12. Voice AI — Whisper (ASR) + Edge-TTS

### Whisper Mimarisi

**Encoder-Decoder Transformer:**
```
Ses → Mel Spectrogram (80 filtre, 30s pencere) → Transformer Encoder → Audio Features
Audio Features + Önceki token'lar → Transformer Decoder → Sonraki token (autoregressive)
```

**Mel Spectrogram nedir?**
- Ses dalgası → STFT (Short-Time Fourier Transform) → Spectrogram
- Spectrogram → Mel ölçeği (insan kulağına uygun logaritmik frekans dağılımı)
- Sonuç: 2D matris → "görüntü" gibi işlenebilir (CNN/Transformer)

**Model boyutları:**
| Model  | Parametre | Hız  | WER (en) | Kullanım |
|--------|-----------|------|----------|----------|
| tiny   | 39M       | 32x  | ~7.6%    | Real-time demo |
| base   | 74M       | 16x  | ~5.0%    | Genel kullanım |
| small  | 244M      | 6x   | ~3.4%    | İyi denge |
| medium | 769M      | 2x   | ~2.7%    | Yüksek doğruluk |
| large  | 1550M     | 1x   | ~2.1%    | En iyi doğruluk |

### Edge-TTS vs Alternatifler

| Özellik     | Edge-TTS    | OpenAI TTS  | Bark        | gTTS        |
|-------------|-------------|-------------|-------------|-------------|
| Maliyet     | Ücretsiz    | $15/1M char | Ücretsiz    | Ücretsiz    |
| Kalite      | Neural (iyi)| Çok iyi     | İyi         | Orta        |
| Offline     | Hayır       | Hayır       | Evet        | Hayır       |
| Hız         | Hızlı       | Hızlı       | Yavaş       | Hızlı       |
| Türkçe      | Evet        | Evet        | Kısıtlı    | Evet        |

### Real-time Voice Pipeline Optimizasyonu

**Mülakat sorusu: "Latency'yi nasıl düşürürsün?"**

1. **Streaming ASR:** Whisper batch-only → alternatif: Whisper.cpp, faster-whisper (CTranslate2)
2. **VAD (Voice Activity Detection):** Sessiz kısımları atla → gereksiz inference azalt
3. **Chunking:** Ses'i 5-10s parçalara böl → paralel işle
4. **Streaming TTS:** Edge-TTS zaten streaming destekler (chunk-based)
5. **WebSocket:** HTTP yerine WebSocket → düşük overhead, bidirectional

**Latency bileşenleri:**
```
ASR: ~1-3s (base model) + Agent: ~3-8s (LLM call) + TTS: ~1-2s = Toplam: ~5-13s
```

**Mülakat cevabı:**
> "Voice pipeline'da en büyük bottleneck LLM çağrısı (~3-8s). ASR için faster-whisper
> (CTranslate2 optimizasyonu, 4x hızlı) kullanırım. VAD ile sessiz kısımları atlayarak
> gereksiz inference'ı önlerim. TTS'te Edge-TTS'in streaming API'sini kullanarak
> ilk chunk hazır olur olmaz çalmaya başlarım — böylece kullanıcı tam çıktıyı
> beklemez. WebSocket ile end-to-end streaming yapılabilir."

---

## Özet: Proje 3'ten Çıkan Anahtar Kavramlar

1. **LangGraph** → StateGraph, nodes, edges, conditional routing
2. **Agent Patterns** → ReAct + Plan-and-Execute + Reflection (hibrit)
3. **State Management** → TypedDict, reducers (operator.add), checkpointing
4. **Memory** → Short-term (buffer) + Long-term (vector store, RAG)
5. **Human-in-the-loop** → interrupt_before veya approval node
6. **Infinite Loop Protection** → max_iterations, cost limit, timeout
7. **Tool Design** → Tek sorumluluk, JSON Schema, error handling
8. **MCP** → Standart tool protokolü (USB for AI)
9. **Voice AI** → Whisper (ASR, encoder-decoder, mel spectrogram) + Edge-TTS (neural TTS)
10. **Latency Optimization** → streaming, VAD, chunking, faster-whisper
