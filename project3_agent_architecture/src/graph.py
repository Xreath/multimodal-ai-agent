"""
Agent Graph — LangGraph ile node'ları birbirine bağlar.

╔══════════════════════════════════════════════════════════════════════╗
║                    MULTI-MODAL AGENT GRAPH                          ║
║                                                                      ║
║  START                                                               ║
║    │                                                                 ║
║    ▼                                                                 ║
║  ┌──────────┐                                                        ║
║  │ PLANNER  │  ← Görevi adımlara böler                              ║
║  └────┬─────┘                                                        ║
║       │                                                              ║
║       ▼                                                              ║
║  ┌──────────┐     ┌───────────────┐                                  ║
║  │ ROUTER   │────▶│ HUMAN APPROVE │ (needs_human_approval=True)      ║
║  └────┬─────┘     └───────┬───────┘                                  ║
║       │                   │                                          ║
║       ├──── "vision" ─────┼────────────────────────┐                 ║
║       │                   │                        │                 ║
║       │    ┌──────────┐   │                        │                 ║
║       │    │ VISION   │◀──┘                        │                 ║
║       │    └────┬─────┘                            │                 ║
║       │         │                                  │                 ║
║       ├──── "reason" ──────────────┐               │                 ║
║       │                            │               │                 ║
║       │         ┌──────────┐       │               │                 ║
║       │         │ REASONER │◀──────┘               │                 ║
║       │         └────┬─────┘                       │                 ║
║       │              │                             │                 ║
║       │         ┌────▼──────┐                      │                 ║
║       │         │ EVALUATOR │                      │                 ║
║       │         └────┬──────┘                      │                 ║
║       │              │                             │                 ║
║       │         score < 0.7 → loop back to ROUTER  │                 ║
║       │         score >= 0.7 ──┐                   │                 ║
║       │                       │                    │                 ║
║       ├──── "respond" ────────┼────────────────────┘                 ║
║       │                       │                                      ║
║       ▼                       ▼                                      ║
║  ┌──────────┐                                                        ║
║  │ RESPOND  │                                                        ║
║  └────┬─────┘                                                        ║
║       │                                                              ║
║       ▼                                                              ║
║     END                                                              ║
╚══════════════════════════════════════════════════════════════════════╝

LangGraph Kavramları (Mülakat notu):

1. StateGraph: State tipiyle parametrize edilmiş graf
   - Her node state'i okur ve günceller
   - State değişiklikleri reducer'lar ile merge edilir

2. add_node(name, function): Node ekleme
   - Node = Python fonksiyonu
   - name ile referans edilir

3. add_edge(A, B): A'dan B'ye her zaman git
   - Koşulsuz bağlantı

4. add_conditional_edges(A, router_fn, mapping): Koşullu yönlendirme
   - router_fn state'e bakıp string döner
   - mapping: {"string": "node_name"} eşleştirmesi

5. START / END: Özel sabitler — graf giriş ve çıkış noktaları

6. compile(): Grafiyi çalıştırılabilir hale getirir
   - Checkpointer eklenebilir (state persistance)
   - interrupt_before/after eklenebilir (human-in-the-loop)
"""

from langgraph.graph import StateGraph, START, END

from .state import AgentState
from .nodes import (
    planner_node,
    router_node,
    vision_node,
    reasoner_node,
    evaluator_node,
    respond_node,
    human_approval_node,
)


def build_agent_graph(with_memory: bool = False) -> StateGraph:
    """
    Multi-Modal Agent graph'ını oluştur ve compile et.

    Args:
        with_memory: True ise MemorySaver checkpointer ekle
                     (conversation persistence, rollback)

    Returns:
        Compiled LangGraph graph (çalıştırılmaya hazır)

    Mülakat notu:
    - Graph building pattern: declarative (ne yapılacağını tanımla, nasıl yapılacağını framework halleder)
    - Compile time vs runtime ayrımı: graph yapısı compile'da, veri runtime'da
    - Checkpointer: her adımda state snapshot'ı → hata durumunda rollback
    """

    # 1. StateGraph oluştur (state tipini belirt)
    graph = StateGraph(AgentState)

    # ─── 2. Node'ları Ekle ───────────────────────────────────────

    graph.add_node("planner", planner_node)
    graph.add_node("vision", vision_node)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("respond", respond_node)
    graph.add_node("human_approval", human_approval_node)

    # ─── 3. Edge'leri Tanımla ────────────────────────────────────

    # START → planner (her zaman planlama ile başla)
    graph.add_edge(START, "planner")

    # planner → router (conditional edge)
    # Router, state'e bakarak nereye gidileceğine karar verir
    graph.add_conditional_edges(
        "planner",
        router_node,
        {
            "vision": "vision",
            "reason": "reasoner",
            "respond": "respond",
            "human_approval": "human_approval",
        }
    )

    # vision → reasoner (görüntü analizi sonrası her zaman reasoning)
    graph.add_edge("vision", "reasoner")

    # reasoner → evaluator (her cevap değerlendirilir)
    graph.add_edge("reasoner", "evaluator")

    # evaluator → router (conditional — score'a göre loop veya respond)
    graph.add_conditional_edges(
        "evaluator",
        router_node,
        {
            "vision": "vision",
            "reason": "reasoner",
            "respond": "respond",
            "human_approval": "human_approval",
        }
    )

    # human_approval → router (onay sonrası devam)
    graph.add_conditional_edges(
        "human_approval",
        router_node,
        {
            "vision": "vision",
            "reason": "reasoner",
            "respond": "respond",
            "human_approval": "human_approval",
        }
    )

    # respond → END (final cevap verildi)
    graph.add_edge("respond", END)

    # ─── 4. Compile ──────────────────────────────────────────────

    if with_memory:
        from langgraph.checkpoint.memory import MemorySaver
        checkpointer = MemorySaver()
        compiled = graph.compile(checkpointer=checkpointer)
    else:
        compiled = graph.compile()

    return compiled


def visualize_graph(graph) -> str:
    """
    Graph'ı ASCII art olarak görselleştir.

    Mülakat notu:
    - LangGraph .get_graph().draw_mermaid() ile Mermaid diagram üretebilir
    - Bu, documentation ve debugging için çok faydalı
    - Production'da: graph yapısını otomatik dokümante et
    """
    try:
        mermaid = graph.get_graph().draw_mermaid()
        return mermaid
    except Exception:
        return "Graph visualization not available (install pygraphviz for image output)"
