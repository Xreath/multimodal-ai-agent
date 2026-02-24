"""
Agent Nodes — LangGraph graph'ının node'ları (iş yapan birimler).

╔══════════════════════════════════════════════════════════════════╗
║  LangGraph Node Kavramı                                         ║
║                                                                  ║
║  Node = Python fonksiyonu                                       ║
║  - Girdi: AgentState (veya bir kısmı)                          ║
║  - Çıktı: State güncellemesi (dict)                            ║
║                                                                  ║
║  Graph akışı:                                                    ║
║  START → planner → router ─┬─→ vision → reasoner → evaluator   ║
║                             ├─→ reasoner → evaluator             ║
║                             └─→ respond → END                    ║
║                                                                  ║
║  Evaluator kötü puan verirse → planner'a geri döner (loop)      ║
╚══════════════════════════════════════════════════════════════════╝

Agent Patterns (Mülakat notu):

1. ReAct (Reason + Act):
   - Düşün → Hareket et → Gözlemle → Tekrarla
   - Basit, tek LLM çağrısıyla tool seçimi
   - Bu projede: router + tool nodes

2. Plan-and-Execute:
   - Önce plan yap (tüm adımları belirle)
   - Sonra sırayla çalıştır
   - Bu projede: planner node → executor nodes

3. Reflection:
   - Cevabı üret → Değerlendir → Gerekirse düzelt
   - Bu projede: evaluator node → loop back

Biz üçünü birleştiriyoruz: Plan → Execute (ReAct) → Reflect
"""

import json
import sys
import os
from typing import Optional
from dotenv import load_dotenv

# Project path'leri
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MONOREPO_ROOT = os.path.dirname(PROJECT_ROOT)

# .env dosyalarını yükle — hem project3 hem project2'den
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
load_dotenv(os.path.join(MONOREPO_ROOT, "project2_llm_integration", ".env"))

# ─── LLM Helper ──────────────────────────────────────────────────

def _get_openai_client():
    """
    DeepSeek LLM client oluştur — OpenAI SDK ile.

    DeepSeek'in API'si OpenAI-uyumlu olduğu için doğrudan OpenAI SDK kullanıyoruz.
    Bu, project2'ye bağımlılığı kaldırır ve daha temiz bir mimari sağlar.

    Mülakat notu:
    - Birçok LLM provider (DeepSeek, Together, Groq) OpenAI-uyumlu API sunar
    - Bu sayede tek SDK (openai) ile birden fazla provider kullanılabilir
    - Sadece base_url değiştirmek yeterli
    """
    from openai import OpenAI

    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise ValueError(
            "DEEPSEEK_API_KEY environment variable not set.\n"
            "Create project3_agent_architecture/.env with:\n"
            "  DEEPSEEK_API_KEY=sk-your-key-here"
        )
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")


# DeepSeek model adı
_LLM_MODEL = "deepseek-chat"


def _call_llm(prompt: str, system_prompt: str = "", json_mode: bool = False) -> str:
    """
    Basit LLM çağrısı helper'ı.

    Tüm node'lar bu fonksiyonu kullanarak LLM'e erişir.
    Merkezi LLM erişimi → provider değişikliği tek noktadan.
    """
    client = _get_openai_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    kwargs = {
        "model": _LLM_MODEL,
        "messages": messages,
        "temperature": 0.3,
        "max_tokens": 2048,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = client.chat.completions.create(**kwargs)
    return response.choices[0].message.content


# ═══════════════════════════════════════════════════════════════════
# NODE 1: PLANNER — Görevi adımlara böler
# ═══════════════════════════════════════════════════════════════════

PLANNER_SYSTEM_PROMPT = """Sen bir görev planlayıcısın (task planner). Kullanıcının isteğini analiz edip
çalıştırılabilir adımlara böl.

Kullanılabilir yetenekler:
- vision: Görüntü analizi (nesne tespiti, segmentasyon, OCR)
- reason: Bilgiyi sentezleyip sonuç çıkar
- search: Bilgi ara (web search)
- calculate: Matematik hesaplama

Her adımı kısa ve net yaz. JSON formatında döndür:
{
  "steps": ["adım1", "adım2", ...],
  "requires_vision": true/false,
  "complexity": "simple" | "moderate" | "complex"
}"""


def planner_node(state: dict) -> dict:
    """
    PLANNER NODE — Kullanıcının isteğini analiz edip plan oluşturur.

    Bu node Plan-and-Execute pattern'ının "Plan" kısmıdır.
    LLM'i kullanarak karmaşık bir görevi adımlara böler.

    Mülakat notu:
    - Planning, agent'ın en kritik yeteneği
    - Kötü plan = kötü sonuç (garbage in, garbage out)
    - Plan complexity'ye göre uyarlanmalı: basit soru → 1 adım, karmaşık → çok adım
    - Plan'ı state'e yazarak şeffaflık sağlanır (explainability)

    Input (state'ten): user_query, image_path
    Output (state update): plan, next_action, messages
    """
    user_query = state["user_query"]
    image_path = state.get("image_path")

    print(f"\n{'='*60}")
    print(f"🧠 PLANNER NODE")
    print(f"{'='*60}")
    print(f"Sorgu: {user_query}")
    print(f"Görüntü: {image_path or 'Yok'}")

    context = f"Kullanıcı isteği: {user_query}"
    if image_path:
        context += f"\nGörüntü mevcut: {image_path}"

    raw_response = _call_llm(context, PLANNER_SYSTEM_PROMPT, json_mode=True)

    # Parse plan
    try:
        parsed = json.loads(raw_response)
        steps = parsed.get("steps", [])
        requires_vision = parsed.get("requires_vision", bool(image_path))
        complexity = parsed.get("complexity", "moderate")
    except json.JSONDecodeError:
        steps = [f"Doğrudan cevapla: {user_query}"]
        requires_vision = bool(image_path)
        complexity = "simple"

    print(f"Plan ({complexity}):")
    for i, step in enumerate(steps, 1):
        print(f"  {i}. {step}")

    # İlk action'ı belirle
    if requires_vision and image_path:
        next_action = "vision"
    else:
        next_action = "reason"

    return {
        "plan": steps,
        "current_step": 0,
        "next_action": next_action,
        "iteration_count": state.get("iteration_count", 0) + 1,
        "messages": [{
            "role": "assistant",
            "content": f"[Planner] Plan oluşturuldu ({len(steps)} adım, {complexity}): {', '.join(steps)}"
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 2: ROUTER — Conditional edge (karar noktası)
# ═══════════════════════════════════════════════════════════════════

def router_node(state: dict) -> str:
    """
    ROUTER — Conditional edge fonksiyonu.

    LangGraph'ta conditional_edges, state'e bakarak hangi node'a
    gidileceğine karar verir. Bu bir "node" değil, bir "karar fonksiyonu".

    Mülakat notu:
    - LangGraph'ta iki tip edge var:
      1. Normal edge: A → B (her zaman)
      2. Conditional edge: A → router → B veya C (state'e göre)
    - Router fonksiyonu string döner → edge mapping'de karşılığı olan node'a gider
    - Infinite loop koruması: max_iterations kontrolü şart

    Input: state
    Output: string → node adı ("vision", "reason", "respond", "human_approval")
    """
    next_action = state.get("next_action", "reason")
    iteration_count = state.get("iteration_count", 0)
    max_iterations = state.get("max_iterations", 5)
    needs_approval = state.get("needs_human_approval", False)

    print(f"\n{'='*60}")
    print(f"🔀 ROUTER NODE")
    print(f"{'='*60}")

    # Sonsuz döngü koruması
    if iteration_count >= max_iterations:
        print(f"⚠️  Max iterations ({max_iterations}) aşıldı → respond")
        return "respond"

    # Human-in-the-loop kontrolü
    if needs_approval:
        print(f"👤 Human approval gerekli → human_approval")
        return "human_approval"

    print(f"Karar: {next_action} (iteration {iteration_count}/{max_iterations})")
    return next_action


# ═══════════════════════════════════════════════════════════════════
# NODE 3: VISION — CV Pipeline çalıştırır
# ═══════════════════════════════════════════════════════════════════

def vision_node(state: dict) -> dict:
    """
    VISION NODE — Görüntüyü CV pipeline'dan geçirir.

    Project 1'deki VisualPerceptionPipeline'ı çağırır:
    - Object Detection (YOLOv8)
    - Instance Segmentation (YOLOv8-seg)
    - OCR (EasyOCR)

    Mülakat notu:
    - Bu node, agentic system'de "perception" katmanıdır
    - Agent'ın "gözü" — dünyayı algılar
    - CV pipeline lazy-loaded: sadece gerektiğinde yüklenir
    - Sonuç state'e yazılır → diğer node'lar kullanabilir

    Input (state'ten): image_path
    Output (state update): cv_result, tool_results, next_action, messages
    """
    image_path = state.get("image_path")

    print(f"\n{'='*60}")
    print(f"👁️  VISION NODE")
    print(f"{'='*60}")

    if not image_path:
        print("⚠️  Görüntü yolu yok — atlıyorum")
        return {
            "cv_result": None,
            "next_action": "reason",
            "tool_results": [{"tool": "vision", "error": "No image path provided"}],
            "messages": [{"role": "assistant", "content": "[Vision] Görüntü yolu belirtilmedi."}]
        }

    # CV Pipeline'ı yükle ve çalıştır
    # project1'in src/ dizinini import edebilmek için sys.path yönetimi:
    # 1. project1'i path'e ekle (relative import'ları çözmek için)
    # 2. sys.modules'tan project3'ün 'src' modülünü geçici kaldır
    # 3. Import yap
    # 4. Geri yükle
    project1_path = os.path.join(MONOREPO_ROOT, "project1_cv_pipeline")

    # Geçici sys.path ve modules yönetimi
    old_src_module = sys.modules.pop("src", None)
    if project1_path not in sys.path:
        sys.path.insert(0, project1_path)

    from src.pipeline import VisualPerceptionPipeline
    pipeline = VisualPerceptionPipeline()

    # Geri yükle
    if old_src_module is not None:
        sys.modules["src"] = old_src_module

    print(f"Görüntü analiz ediliyor: {image_path}")
    cv_result = pipeline.analyze(image_path)

    # Özet bilgi
    n_objects = len(cv_result.get("objects", []))
    n_segments = len(cv_result.get("segments", []))
    n_text = len(cv_result.get("text_regions", []))
    proc_time = cv_result.get("processing_time", {}).get("total", 0)

    summary = (
        f"Tespit: {n_objects} nesne, {n_segments} segment, {n_text} text bölgesi "
        f"({proc_time:.2f}s)"
    )
    print(f"✅ {summary}")

    return {
        "cv_result": cv_result,
        "next_action": "reason",
        "tool_results": [{
            "tool": "vision",
            "summary": summary,
            "objects": n_objects,
            "segments": n_segments,
            "text_regions": n_text,
            "processing_time": proc_time
        }],
        "messages": [{"role": "assistant", "content": f"[Vision] {summary}"}]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 4: REASONER — Bilgiyi sentezler, cevap üretir
# ═══════════════════════════════════════════════════════════════════

REASONER_SYSTEM_PROMPT = """Sen bir multi-modal AI analisti sin. Sana verilen bilgileri
sentezleyerek kullanıcının sorusuna kapsamlı ve doğru cevap ver.

Cevabını şu JSON formatında ver:
{
  "answer": "Ana cevap (detaylı, açıklayıcı)",
  "reasoning_steps": ["adım1", "adım2", ...],
  "confidence": 0.0-1.0,
  "evidence": ["kanıt1", "kanıt2", ...],
  "follow_up_suggestions": ["öneri1", "öneri2"]
}

Kurallar:
- CV pipeline sonuçlarını kanıt olarak kullan
- Emin olmadığın yerlerde confidence'ı düşür
- Somut sayılar ve veriler sun
- Türkçe cevap ver"""


def reasoner_node(state: dict) -> dict:
    """
    REASONER NODE — Tüm bilgiyi sentezleyip cevap üretir.

    Bu node agentic system'de "reasoning" katmanıdır.
    CV sonuçlarını, tool sonuçlarını ve konuşma geçmişini
    birleştirip anlamlı bir cevap üretir.

    Mülakat notu:
    - Reasoning = "sense-making" — ham veriyi anlama dönüştürme
    - Context window yönetimi kritik: tüm bilgiyi sığdırmak lazım
    - Chain-of-Thought (CoT) prompting reasoning kalitesini artırır
    - Evidence-based reasoning: cevabın kanıtlarını belirt

    Input (state'ten): user_query, cv_result, tool_results, plan
    Output (state update): reasoning, final_answer, next_action, messages
    """
    user_query = state["user_query"]
    cv_result = state.get("cv_result")
    tool_results = state.get("tool_results", [])
    plan = state.get("plan", [])

    print(f"\n{'='*60}")
    print(f"🤔 REASONER NODE")
    print(f"{'='*60}")

    # Context oluştur — LLM'e vereceğimiz bilgi paketi
    context_parts = [f"Kullanıcı sorusu: {user_query}"]

    if plan:
        context_parts.append(f"Plan: {', '.join(plan)}")

    if cv_result:
        # CV sonucunu özetle (tüm JSON'ı göndermek yerine — token tasarrufu)
        cv_summary = _summarize_cv_result(cv_result)
        context_parts.append(f"CV Analiz Sonucu:\n{cv_summary}")

    if tool_results:
        context_parts.append(f"Tool Sonuçları:\n{json.dumps(tool_results, indent=2, ensure_ascii=False)}")

    full_context = "\n\n".join(context_parts)
    print(f"Context uzunluğu: {len(full_context)} karakter")

    # LLM'e gönder
    raw_response = _call_llm(full_context, REASONER_SYSTEM_PROMPT, json_mode=True)

    # Parse et
    try:
        parsed = json.loads(raw_response)
        answer = parsed.get("answer", raw_response)
        reasoning_steps = parsed.get("reasoning_steps", [])
        confidence = parsed.get("confidence", 0.5)
    except json.JSONDecodeError:
        answer = raw_response
        reasoning_steps = []
        confidence = 0.5

    print(f"Cevap uzunluğu: {len(answer)} karakter")
    print(f"Confidence: {confidence}")
    print(f"Reasoning adımları: {len(reasoning_steps)}")

    return {
        "reasoning": raw_response,
        "final_answer": answer,
        "next_action": "evaluate",
        "messages": [{
            "role": "assistant",
            "content": f"[Reasoner] Confidence: {confidence} | {answer[:200]}..."
        }]
    }


def _summarize_cv_result(cv_result: dict) -> str:
    """CV sonucunu özet string'e dönüştür (token tasarrufu için)."""
    parts = []

    # Nesneler
    objects = cv_result.get("objects", [])
    if objects:
        # Nesne sayılarını grupla
        from collections import Counter
        label_counts = Counter(o["label"] for o in objects)
        obj_summary = ", ".join(f"{count}x {label}" for label, count in label_counts.items())
        parts.append(f"Tespit edilen nesneler: {obj_summary}")

        # En yüksek confidence
        max_conf = max(o["confidence"] for o in objects)
        parts.append(f"En yüksek confidence: {max_conf:.2f}")

    # Text bölgeleri
    text_regions = cv_result.get("text_regions", [])
    if text_regions:
        texts = [t["text"] for t in text_regions[:5]]  # İlk 5
        parts.append(f"Tespit edilen metinler: {', '.join(texts)}")

    # Segmentler
    segments = cv_result.get("segments", [])
    if segments:
        parts.append(f"Segmentasyon: {len(segments)} segment")

    # Görüntü bilgisi
    img_info = cv_result.get("image_info", {})
    if img_info:
        parts.append(f"Görüntü: {img_info.get('width', '?')}x{img_info.get('height', '?')}")

    return "\n".join(parts) if parts else "CV sonucu boş"


# ═══════════════════════════════════════════════════════════════════
# NODE 5: EVALUATOR — Cevap kalitesini değerlendirir
# ═══════════════════════════════════════════════════════════════════

EVALUATOR_SYSTEM_PROMPT = """Sen bir kalite değerlendirme uzmanısın. Verilen cevabı değerlendir.

JSON formatında döndür:
{
  "score": 0.0-1.0,
  "feedback": "Kısa değerlendirme",
  "pass": true/false,
  "improvement_suggestion": "Varsa iyileştirme önerisi"
}

Değerlendirme kriterleri:
- Doğruluk: Cevap soruyla uyumlu mu?
- Kanıt: CV verileri kullanılmış mı?
- Bütünlük: Soru tam cevaplanmış mı?
- Netlik: Cevap açık ve anlaşılır mı?

0.7 üstü → PASS, altı → FAIL (yeniden dene)"""


def evaluator_node(state: dict) -> dict:
    """
    EVALUATOR NODE — Reflection pattern: cevap kalitesini değerlendirir.

    Bu node "self-critique" mekanizmasıdır. Agent'ın kendi cevabını
    değerlendirip, yetersizse yeniden denemesini sağlar.

    Mülakat notu:
    - Reflection/Self-critique: LLM'in kendi çıktısını değerlendirmesi
    - Bu pattern cevap kalitesini önemli ölçüde artırır
    - Trade-off: Ekstra LLM çağrısı = daha yüksek maliyet + latency
    - Infinite loop riski: max_iterations ile sınırla
    - Production'da: basit heuristic (uzunluk, format) + LLM evaluation hibrit

    Input (state'ten): user_query, final_answer, reasoning
    Output (state update): evaluation_score, evaluation_feedback, next_action
    """
    user_query = state["user_query"]
    final_answer = state.get("final_answer", "")
    iteration = state.get("iteration_count", 0)
    max_iter = state.get("max_iterations", 5)

    print(f"\n{'='*60}")
    print(f"📊 EVALUATOR NODE")
    print(f"{'='*60}")

    eval_context = (
        f"Orijinal soru: {user_query}\n\n"
        f"Üretilen cevap: {final_answer}"
    )

    raw_response = _call_llm(eval_context, EVALUATOR_SYSTEM_PROMPT, json_mode=True)

    try:
        parsed = json.loads(raw_response)
        score = parsed.get("score", 0.5)
        feedback = parsed.get("feedback", "")
        passed = parsed.get("pass", score >= 0.7)
    except json.JSONDecodeError:
        score = 0.7
        feedback = "Değerlendirme parse edilemedi — varsayılan geçiş"
        passed = True

    print(f"Skor: {score:.2f}")
    print(f"Feedback: {feedback}")
    print(f"Geçti mi: {'✅ EVET' if passed else '❌ HAYIR'}")

    if passed or iteration >= max_iter - 1:
        next_action = "respond"
        if not passed:
            print(f"⚠️  Skor düşük ama max iteration'a ulaşıldı → respond")
    else:
        next_action = "reason"
        print(f"🔄 Yeniden deneniyor (iteration {iteration}/{max_iter})")

    return {
        "evaluation_score": score,
        "evaluation_feedback": feedback,
        "next_action": next_action,
        "messages": [{
            "role": "assistant",
            "content": f"[Evaluator] Skor: {score:.2f} — {feedback}"
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 6: RESPOND — Final cevabı oluşturur
# ═══════════════════════════════════════════════════════════════════

def respond_node(state: dict) -> dict:
    """
    RESPOND NODE — Kullanıcıya verilecek final cevabı hazırlar.

    Basit bir formatlama node'u. State'teki final_answer'ı
    kullanıcıya uygun formata getirir.

    Input (state'ten): final_answer, evaluation_score, plan, tool_results
    Output (state update): final_answer (formatted), messages
    """
    final_answer = state.get("final_answer", "Cevap üretilemedi.")
    score = state.get("evaluation_score")
    plan = state.get("plan", [])
    tool_results = state.get("tool_results", [])

    print(f"\n{'='*60}")
    print(f"💬 RESPOND NODE")
    print(f"{'='*60}")

    # Zenginleştirilmiş cevap formatı
    response_parts = [final_answer]

    if tool_results:
        tools_used = set(tr.get("tool", "unknown") for tr in tool_results)
        response_parts.append(f"\n📎 Kullanılan araçlar: {', '.join(tools_used)}")

    if score is not None:
        response_parts.append(f"📊 Güven skoru: {score:.0%}")

    formatted_answer = "\n".join(response_parts)
    print(f"Final cevap ({len(formatted_answer)} karakter)")

    return {
        "final_answer": formatted_answer,
        "messages": [{
            "role": "assistant",
            "content": formatted_answer
        }]
    }


# ═══════════════════════════════════════════════════════════════════
# NODE 7: HUMAN APPROVAL — İnsan onayı bekler
# ═══════════════════════════════════════════════════════════════════

def human_approval_node(state: dict) -> dict:
    """
    HUMAN-IN-THE-LOOP NODE — Kritik kararlarda insan onayı bekler.

    LangGraph'ta human-in-the-loop iki şekilde yapılabilir:
    1. interrupt_before/interrupt_after — graph'ı durdur, insan karar versin
    2. Approval node — state'e bakarak onay iste

    Biz burada 2. yöntemi kullanıyoruz (daha basit, daha esnek).

    Mülakat notu:
    - Human-in-the-loop neden gerekli?
      → Güvenlik: yanlış kararların maliyeti yüksekse (silme, gönderme)
      → Etik: hassas verilerle çalışırken
      → Düzenleyici: compliance gereksinimleri
    - Ne zaman kullanılMAZ?
      → Latency kritikse (real-time sistemler)
      → Karar düşük riskli ise
    - LangGraph interrupt: graph checkpointed → durdurup devam ettirilebilir
    """
    print(f"\n{'='*60}")
    print(f"👤 HUMAN APPROVAL NODE")
    print(f"{'='*60}")

    plan = state.get("plan", [])
    print(f"Plan: {plan}")
    print(f"Onay bekleniyor...")

    # CLI'da input ile onay al
    try:
        approval = input("\n✋ Bu planı onaylıyor musunuz? (e/h): ").strip().lower()
    except EOFError:
        approval = "e"  # Non-interactive modda otomatik onayla

    if approval in ("e", "evet", "y", "yes"):
        print("✅ Onaylandı — devam ediliyor")
        return {
            "needs_human_approval": False,
            "next_action": "vision" if state.get("image_path") else "reason",
            "messages": [{"role": "user", "content": "[Human] Plan onaylandı."}]
        }
    else:
        print("❌ Reddedildi — yeniden planlama")
        return {
            "needs_human_approval": False,
            "next_action": "respond",
            "final_answer": "İşlem kullanıcı tarafından iptal edildi.",
            "messages": [{"role": "user", "content": "[Human] Plan reddedildi."}]
        }
