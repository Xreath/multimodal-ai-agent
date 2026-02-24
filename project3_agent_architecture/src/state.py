"""
Agent State — LangGraph graph'ının state tanımı.

LangGraph'ta state, graph'taki tüm node'lar arasında paylaşılan veridir.
Her node state'i okur, işler ve günceller. Bu "shared blackboard" pattern'ı.

╔══════════════════════════════════════════════════════════════════╗
║  LangGraph State Kavramı                                        ║
║                                                                  ║
║  State = TypedDict (tip güvenli bir dict)                       ║
║                                                                  ║
║  Her node çalıştığında:                                         ║
║  1. State'in bir kopyasını alır (read)                          ║
║  2. İşlem yapar                                                  ║
║  3. Güncellenmiş alanları döner (partial update)                ║
║                                                                  ║
║  Önemli: Node'lar state'in TAMAMINI değil,                      ║
║  sadece değiştirdiği alanları döner → merge edilir               ║
╚══════════════════════════════════════════════════════════════════╝

Mülakat notu:
- LangGraph state'i "reducer" fonksiyonları ile günceller
- Annotated[list, operator.add] → listelerde ekleme (append) semantiği
- Bu, Redux/Flux pattern'ından esinlenmiştir
- State immutability: her node yeni değer döner, eski state korunur
- Checkpointing: state'in her adımda snapshot'ı alınabilir (rollback için)
"""

import operator
from typing import Annotated, TypedDict, Optional, Literal


class AgentState(TypedDict):
    """
    Multi-Modal Agent'ın state'i.

    Bu state, graph'taki tüm node'lar arasında paylaşılır.
    Her alan (field) bir "kanal" gibi düşünülebilir.

    Annotated[list, operator.add] açıklaması:
    - Normal dict update: {"messages": [yeni]} → eski mesajları SİLER
    - operator.add ile: {"messages": [yeni]} → eski mesajlara EKLER
    - Bu, conversation history gibi birikimli verilerde kritik
    """

    # ─── Kullanıcı Girdisi ───────────────────────────────────────
    user_query: str                          # Kullanıcının orijinal sorusu
    image_path: Optional[str]                # Analiz edilecek görüntü (opsiyonel)

    # ─── Planner Çıktısı ─────────────────────────────────────────
    plan: list[str]                          # Planner'ın oluşturduğu adım listesi
    current_step: int                        # Şu an hangi adımdayız (0-indexed)

    # ─── Tool Sonuçları ──────────────────────────────────────────
    # operator.add → her tool sonucu listeye EKLENİR (üzerine yazmaz)
    tool_results: Annotated[list[dict], operator.add]

    # ─── CV Pipeline Sonucu ──────────────────────────────────────
    cv_result: Optional[dict]                # Vision node'un ürettiği CV analizi

    # ─── Mesaj Geçmişi ───────────────────────────────────────────
    # operator.add → her mesaj birikimli olarak eklenir
    messages: Annotated[list[dict], operator.add]

    # ─── Reasoning / Final Cevap ─────────────────────────────────
    reasoning: Optional[str]                 # Reasoner'ın ürettiği analiz
    final_answer: Optional[str]              # Kullanıcıya verilecek cevap

    # ─── Kontrol Akışı ───────────────────────────────────────────
    next_action: Optional[str]               # Router'ın kararı: "vision", "reason", "respond", ...
    needs_human_approval: bool               # Human-in-the-loop flag
    iteration_count: int                     # Loop sayacı (infinite loop koruması)
    max_iterations: int                      # Maksimum loop sayısı

    # ─── Değerlendirme ───────────────────────────────────────────
    evaluation_score: Optional[float]        # Evaluator'ın verdiği kalite puanı (0-1)
    evaluation_feedback: Optional[str]       # Evaluator'ın geri bildirimi


def create_initial_state(
    user_query: str,
    image_path: Optional[str] = None,
    max_iterations: int = 5
) -> AgentState:
    """
    Yeni bir agent çalışması için başlangıç state'i oluştur.

    Args:
        user_query: Kullanıcının sorusu
        image_path: Analiz edilecek görüntü yolu
        max_iterations: Maksimum iterasyon (sonsuz döngü koruması)

    Returns:
        Başlangıç state dict'i
    """
    return {
        "user_query": user_query,
        "image_path": image_path,
        "plan": [],
        "current_step": 0,
        "tool_results": [],
        "cv_result": None,
        "messages": [{
            "role": "user",
            "content": user_query
        }],
        "reasoning": None,
        "final_answer": None,
        "next_action": None,
        "needs_human_approval": False,
        "iteration_count": 0,
        "max_iterations": max_iterations,
        "evaluation_score": None,
        "evaluation_feedback": None,
    }
