"""
Memory Manager — Short-term (conversation) + Long-term (vector store) memory.

╔══════════════════════════════════════════════════════════════════╗
║  Agent Memory Types                                              ║
║                                                                  ║
║  1. Short-term Memory (Conversation Buffer)                      ║
║     - Keeps the last N messages                                  ║
║     - Simple, fast, bounded by token limit                       ║
║     - Separate for each conversation                             ║
║                                                                  ║
║  2. Long-term Memory (Vector Store — ChromaDB)                   ║
║     - Semantic search with embeddings                            ║
║     - Remembers past analyses                                    ║
║     - Cross-conversation knowledge                               ║
║                                                                  ║
║  3. Summary Memory                                               ║
║     - Compresses long conversations by summarizing               ║
║     - Token saving                                               ║
║                                                                  ║
║  Interview question: "When to use which?"                        ║
║  - Short-term: active conversation, recent context               ║
║  - Long-term: past analyses, accumulated knowledge               ║
║  - Summary: context compression in very long conversations       ║
╚══════════════════════════════════════════════════════════════════╝

Vector Store Concepts (Interview note):
- Embedding: Text → fixed size vector (384-1536 dim)
- Cosine similarity: Similarity between two vectors (0-1)
- ChromaDB: Lightweight, embedded vector DB (like SQLite but for vectors)
- Alternatives: Pinecone (cloud), Weaviate (self-hosted), FAISS (Meta, index only)
"""

import json
import os
from typing import Optional
from datetime import datetime


class ConversationMemory:
    """
    Short-term Memory — Conversation buffer.

    Keeps the last N messages. Simple but effective.
    Creates the context to be sent to the LLM.

    Interview note:
    - Window size trade-off: larger window = better context, higher token cost
    - Sliding window: oldest message drops, newest is added
    - System prompt is ALWAYS preserved (doesn't drop from window)
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[dict] = []
        self._metadata: dict = {
            "created_at": datetime.now().isoformat(),
            "turn_count": 0
        }

    def add_message(self, role: str, content: str):
        """Add a new message. If window is exceeded, delete the oldest."""
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._metadata["turn_count"] += 1

        # Sliding window — but preserve the system prompt
        if len(self._messages) > self.max_messages:
            # Preserve if the first message is the system prompt
            if self._messages[0]["role"] == "system":
                self._messages = [self._messages[0]] + self._messages[-(self.max_messages - 1):]
            else:
                self._messages = self._messages[-self.max_messages:]

    def get_messages(self) -> list[dict]:
        """Return all messages (in the format to be sent to LLM)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def get_context_summary(self) -> str:
        """Return the conversation summary as a string."""
        turns = len([m for m in self._messages if m["role"] == "user"])
        return f"Conversation: {turns} turns, {len(self._messages)} messages"

    def clear(self):
        """Clear memory."""
        self._messages = []
        self._metadata["turn_count"] = 0

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0


class VectorMemory:
    """
    Long-term Memory — ChromaDB vector store.

    Stores past analyses as embeddings.
    When a new query comes, finds similar past analyses.

    Interview note:
    - Embedding model: sentence-transformers (all-MiniLM-L6-v2, 384 dim, fast)
    - ChromaDB: can write to disk with persist_directory
    - Retrieval: get the top K most similar results using cosine similarity
    - RAG (Retrieval-Augmented Generation): retrieve → add to context → send to LLM
    - Chunk size: too small → context lost, too large → noise increases
    """

    def __init__(self, collection_name: str = "agent_memory", persist_dir: Optional[str] = None):
        """
        Args:
            collection_name: ChromaDB collection name
            persist_dir: Directory to persist to disk (None → memory only)
        """
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure_initialized(self):
        """Lazy initialization — ChromaDB loaded only when used."""
        if self._client is not None:
            return

        import chromadb

        if self._persist_dir:
            os.makedirs(self._persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=self._persist_dir)
        else:
            self._client = chromadb.Client()

        self._collection = self._client.get_or_create_collection(
            name=self._collection_name,
            metadata={"description": "Multi-modal agent analysis memory"}
        )

    def store(self, text: str, metadata: Optional[dict] = None, doc_id: Optional[str] = None):
        """
        Save information to vector store.

        Flow:
        1. Text → embedding model → vector
        2. Vector + metadata → save to ChromaDB
        3. ChromaDB creates embedding automatically (with default model)

        Args:
            text: Text to save (analysis result, answer, etc.)
            metadata: Additional info (date, question type, etc.)
            doc_id: Unique ID (None → auto)
        """
        self._ensure_initialized()

        if doc_id is None:
            doc_id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        meta = metadata or {}
        meta["stored_at"] = datetime.now().isoformat()

        self._collection.add(
            documents=[text],
            metadatas=[meta],
            ids=[doc_id]
        )

    def search(self, query: str, n_results: int = 3) -> list[dict]:
        """
        Search for similar info (semantic search).

        Flow:
        1. Query → embedding → vector
        2. Vector → ChromaDB cosine similarity search
        3. Return top n_results most similar results

        Args:
            query: Search query
            n_results: How many results to return

        Returns:
            [{"text": "...", "metadata": {...}, "distance": 0.1}, ...]
        """
        self._ensure_initialized()

        if self._collection.count() == 0:
            return []

        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, self._collection.count())
        )

        memories = []
        for i in range(len(results["documents"][0])):
            memories.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else None
            })

        return memories

    def get_relevant_context(self, query: str, n_results: int = 3) -> str:
        """
        Create RAG-style context — return search results as string.

        This creates the "past knowledge" context to be appended to the LLM prompt.
        """
        memories = self.search(query, n_results)
        if not memories:
            return ""

        context_parts = ["Relevant info from past analyses:"]
        for i, mem in enumerate(memories, 1):
            context_parts.append(f"{i}. {mem['text'][:500]}")

        return "\n".join(context_parts)

    @property
    def count(self) -> int:
        """Number of saved documents."""
        self._ensure_initialized()
        return self._collection.count()


class MemoryManager:
    """
    Unified memory manager — short-term + long-term.

    This class is the memory part of the agent's "brain".
    Conversation history (short-term) and accumulated knowledge (long-term) are managed together.

    Usage:
        memory = MemoryManager(persist_dir="./data/memory")
        memory.add_conversation_message("user", "What is in this scene?")
        memory.store_analysis("3 cars detected", {"type": "vehicle_detection"})
        context = memory.get_full_context("what do you know about vehicles?")
    """

    def __init__(self, max_conversation_messages: int = 20, persist_dir: Optional[str] = None):
        self.conversation = ConversationMemory(max_messages=max_conversation_messages)
        self.vector_store = VectorMemory(persist_dir=persist_dir)

    def add_conversation_message(self, role: str, content: str):
        """Add a conversation message."""
        self.conversation.add_message(role, content)

    def store_analysis(self, analysis_text: str, metadata: Optional[dict] = None):
        """Save the analysis result to long-term memory."""
        self.vector_store.store(analysis_text, metadata)

    def get_full_context(self, query: str) -> str:
        """
        Create context from both short and long-term memory.

        This is the agent's "recall" ability:
        1. Get recent conversation (short-term)
        2. Search relevant past info (long-term, RAG)
        3. Combine both
        """
        parts = []

        # Short-term: conversation history
        if not self.conversation.is_empty:
            parts.append(f"Conversation history:\n{self.conversation.get_context_summary()}")

        # Long-term: relevant past info (RAG)
        relevant = self.vector_store.get_relevant_context(query)
        if relevant:
            parts.append(relevant)

        return "\n\n".join(parts) if parts else ""
