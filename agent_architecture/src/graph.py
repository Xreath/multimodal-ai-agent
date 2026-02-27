"""
Agent Graph — Connects nodes together using LangGraph.

╔══════════════════════════════════════════════════════════════════════╗
║                    MULTI-MODAL AGENT GRAPH                          ║
║                                                                      ║
║  START                                                               ║
║    │                                                                 ║
║    ▼                                                                 ║
║  ┌──────────┐  (ASR if audio_path exists, else pass-through)        ║
║  │  VOICE   │  ← Whisper ASR: Audio → Text                          ║
║  └────┬─────┘                                                        ║
║       │                                                              ║
║    ┌──▼───────┐                                                      ║
║    │ PLANNER  │  ← LLM breaks down task into steps                  ║
║    └────┬─────┘                                                      ║
║         │                                                            ║
║    ┌────▼─────┐     ┌───────────────┐                               ║
║    │  ROUTER  │────▶│ HUMAN APPROVE │ (needs_human_approval=True)    ║
║    └──┬──┬──┬─┘     └───────┬───────┘                               ║
║       │  │  │               │                                       ║
║   "vision" │ "search"      ...                                      ║
║       │  │  │  "memory"                                             ║
║  ┌────▼┐ │ ┌▼──────┐ ┌──────▼──┐                                   ║
║  │VISN │ │ │SEARCH │ │ MEMORY  │                                    ║
║  │YOLO │ │ │DuckDDG│ │ChromaDB │                                    ║
║  │OCR  │ │ └───┬───┘ └────┬────┘                                    ║
║  └──┬──┘ │     │          │                                         ║
║     │    │  "reason"      │                                         ║
║     └────┴─────┴──────────┘                                         ║
║                │                                                     ║
║           ┌────▼─────┐                                              ║
║           │ REASONER │  ← Synthesizes CV + Search + Memory + Tool   ║
║           └────┬─────┘    results using LLM                         ║
║                │                                                     ║
║           ┌────▼──────┐                                             ║
║           │ EVALUATOR │  ← Evaluates answer quality                 ║
║           └────┬──────┘    score < 0.7 → loop                       ║
║                │           score >= 0.7 → respond                   ║
║           ┌────▼─────┐                                              ║
║           │  RESPOND  │  ← Formats final answer                     ║
║           └────┬─────┘                                              ║
║                │                                                     ║
║              END                                                     ║
╚══════════════════════════════════════════════════════════════════════╝

LangGraph Concepts (Interview note):

1. StateGraph: A graph parameterized by state type
   - Each node reads and updates the state
   - State changes are merged via reducers

2. add_node(name, function): Adding a node
   - Node = Python function
   - Referenced by name

3. add_edge(A, B): Always go from A to B
   - Unconditional link

4. add_conditional_edges(A, router_fn, mapping): Conditional routing
   - router_fn checks state and returns a string
   - mapping: {"string": "node_name"} mapping

5. START / END: Special constants — graph entry and exit points

6. compile(): Makes the graph executable
   - Checkpointer can be added (state persistence)
   - interrupt_before/after can be added (human-in-the-loop)
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
    voice_node,
    search_node,
    memory_node,
)


def build_agent_graph(with_memory: bool = False) -> StateGraph:
    """
    Creates and compiles the Multi-Modal Agent graph.

    Args:
        with_memory: If True, adds MemorySaver checkpointer
                     (conversation persistence, rollback)

    Returns:
        Compiled LangGraph graph (ready to run)

    Interview note:
    - Graph building pattern: declarative (define what to do, framework handles how)
    - Compile time vs runtime distinction: graph structure at compile, data at runtime
    - Checkpointer: state snapshot at each step → rollback on error
    """

    # 1. Create StateGraph (specify state type)
    graph = StateGraph(AgentState)

    # ─── 2. Add Nodes ───────────────────────────────────────

    graph.add_node("voice", voice_node)          # NEW: Speech Node (ASR)
    graph.add_node("planner", planner_node)
    graph.add_node("vision", vision_node)
    graph.add_node("search", search_node)        # NEW: Search Node (DuckDuckGo)
    graph.add_node("memory", memory_node)        # NEW: Memory Node (ChromaDB RAG)
    graph.add_node("reasoner", reasoner_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("respond", respond_node)
    graph.add_node("human_approval", human_approval_node)

    # ─── 3. Define Edges ────────────────────────────────────

    # START → voice (always, if audio_path exists, process it, otherwise pass-through)
    # Voice node transcribes if audio exists, else returns {} and continues
    graph.add_edge(START, "voice")

    # voice → planner (always — plan after transcription)
    graph.add_edge("voice", "planner")

    # Route mapping used in all conditional edges
    _route_map = {
        "vision": "vision",
        "search": "search",
        "memory": "memory",
        "reason": "reasoner",
        "respond": "respond",
        "human_approval": "human_approval",
    }

    # planner → router (conditional edge)
    graph.add_conditional_edges("planner", router_node, _route_map)

    # vision → reasoner (always reason after image analysis)
    graph.add_edge("vision", "reasoner")

    # search → reasoner (search results synthesized by reasoning)
    graph.add_edge("search", "reasoner")

    # memory → reasoner (memory context fed to reasoning)
    graph.add_edge("memory", "reasoner")

    # reasoner → evaluator (every answer is evaluated)
    graph.add_edge("reasoner", "evaluator")

    # evaluator → router (conditional — loop or respond based on score)
    graph.add_conditional_edges("evaluator", router_node, _route_map)

    # human_approval → router (continue after approval)
    graph.add_conditional_edges("human_approval", router_node, _route_map)

    # respond → END (final answer given)
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
    Visualize the graph as ASCII or Mermaid artifact.

    Interview note:
    - LangGraph can produce a Mermaid diagram via .get_graph().draw_mermaid()
    - Very useful for documentation and debugging
    - In production: auto-document the graph structure
    """
    try:
        mermaid = graph.get_graph().draw_mermaid()
        return mermaid
    except Exception:
        return "Graph visualization not available (install pygraphviz for image output)"
