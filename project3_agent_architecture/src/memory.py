"""
Memory Manager — Kısa süreli (conversation) + Uzun süreli (vector store) hafıza.

╔══════════════════════════════════════════════════════════════════╗
║  Agent Memory Tipleri                                            ║
║                                                                  ║
║  1. Short-term Memory (Conversation Buffer)                     ║
║     - Son N mesajı tut                                           ║
║     - Basit, hızlı, token limiti var                            ║
║     - Her conversation için ayrı                                 ║
║                                                                  ║
║  2. Long-term Memory (Vector Store — ChromaDB)                  ║
║     - Embedding ile semantic search                              ║
║     - Geçmiş analizleri hatırla                                  ║
║     - Cross-conversation bilgi                                   ║
║                                                                  ║
║  3. Summary Memory (özet)                                        ║
║     - Uzun conversation'ları özetleyerek sıkıştır               ║
║     - Token tasarrufu                                            ║
║                                                                  ║
║  Mülakat sorusu: "Ne zaman hangisi?"                            ║
║  - Short-term: aktif konuşma, son context                       ║
║  - Long-term: önceki analizler, bilgi birikimi                   ║
║  - Summary: çok uzun konuşmalarda context sıkıştırma            ║
╚══════════════════════════════════════════════════════════════════╝

Vector Store Kavramları (Mülakat notu):
- Embedding: Metin → sabit boyutlu vektör (384-1536 dim)
- Cosine similarity: İki vektör arası benzerlik (0-1)
- ChromaDB: Lightweight, embedded vector DB (SQLite gibi ama vektörler için)
- Alternatifler: Pinecone (cloud), Weaviate (self-hosted), FAISS (Meta, sadece index)
"""

import json
import os
from typing import Optional
from datetime import datetime


class ConversationMemory:
    """
    Short-term Memory — Conversation buffer.

    Son N mesajı tutar. Basit ama etkili.
    LLM'e gönderilecek context'i oluşturur.

    Mülakat notu:
    - Window size trade-off: büyük window = daha iyi context, daha yüksek token maliyeti
    - Sliding window: en eski mesaj düşer, en yeni eklenir
    - System prompt HER ZAMAN korunur (window'dan düşmez)
    """

    def __init__(self, max_messages: int = 20):
        self.max_messages = max_messages
        self._messages: list[dict] = []
        self._metadata: dict = {
            "created_at": datetime.now().isoformat(),
            "turn_count": 0
        }

    def add_message(self, role: str, content: str):
        """Yeni mesaj ekle. Window aşılırsa en eskiyi sil."""
        self._messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat()
        })
        self._metadata["turn_count"] += 1

        # Sliding window — ama system prompt'u koru
        if len(self._messages) > self.max_messages:
            # İlk mesaj system prompt ise koru
            if self._messages[0]["role"] == "system":
                self._messages = [self._messages[0]] + self._messages[-(self.max_messages - 1):]
            else:
                self._messages = self._messages[-self.max_messages:]

    def get_messages(self) -> list[dict]:
        """Tüm mesajları döndür (LLM'e gönderilecek format)."""
        return [{"role": m["role"], "content": m["content"]} for m in self._messages]

    def get_context_summary(self) -> str:
        """Konuşma özetini string olarak döndür."""
        turns = len([m for m in self._messages if m["role"] == "user"])
        return f"Konuşma: {turns} tur, {len(self._messages)} mesaj"

    def clear(self):
        """Hafızayı temizle."""
        self._messages = []
        self._metadata["turn_count"] = 0

    @property
    def is_empty(self) -> bool:
        return len(self._messages) == 0


class VectorMemory:
    """
    Long-term Memory — ChromaDB vector store.

    Geçmiş analizleri embedding olarak saklar.
    Yeni bir soru geldiğinde, benzer geçmiş analizleri bulur.

    Mülakat notu:
    - Embedding modeli: sentence-transformers (all-MiniLM-L6-v2, 384 dim, hızlı)
    - ChromaDB: persist_directory ile disk'e yazabilir
    - Retrieval: cosine similarity ile en benzer K sonucu getir
    - RAG (Retrieval-Augmented Generation): retrieve → context'e ekle → LLM'e gönder
    - Chunk size: çok küçük → context kaybolur, çok büyük → noise artar
    """

    def __init__(self, collection_name: str = "agent_memory", persist_dir: Optional[str] = None):
        """
        Args:
            collection_name: ChromaDB koleksiyon adı
            persist_dir: Disk'e yazılacak dizin (None → sadece bellekte)
        """
        self._collection_name = collection_name
        self._persist_dir = persist_dir
        self._client = None
        self._collection = None

    def _ensure_initialized(self):
        """Lazy initialization — ChromaDB sadece kullanılınca yüklenir."""
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
        Bilgiyi vector store'a kaydet.

        Akış:
        1. Text → embedding model → vektör
        2. Vektör + metadata → ChromaDB'ye kaydet
        3. ChromaDB embedding'i otomatik oluşturur (varsayılan model ile)

        Args:
            text: Kaydedilecek metin (analiz sonucu, cevap, vb.)
            metadata: Ek bilgiler (tarih, soru tipi, vb.)
            doc_id: Benzersiz ID (None → otomatik)
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
        Benzer bilgileri ara (semantic search).

        Akış:
        1. Query → embedding → vektör
        2. Vektör → ChromaDB cosine similarity search
        3. En benzer n_results sonucu döndür

        Args:
            query: Arama sorgusu
            n_results: Kaç sonuç döndürülecek

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
        RAG-style context oluştur — arama sonuçlarını string olarak döndür.

        Bu, LLM prompt'una eklenecek "geçmiş bilgi" context'ini oluşturur.
        """
        memories = self.search(query, n_results)
        if not memories:
            return ""

        context_parts = ["Geçmiş analizlerden ilgili bilgiler:"]
        for i, mem in enumerate(memories, 1):
            context_parts.append(f"{i}. {mem['text'][:500]}")

        return "\n".join(context_parts)

    @property
    def count(self) -> int:
        """Kayıtlı belge sayısı."""
        self._ensure_initialized()
        return self._collection.count()


class MemoryManager:
    """
    Birleşik hafıza yöneticisi — short-term + long-term.

    Bu sınıf, agent'ın "beyninin" hafıza kısmıdır.
    Konuşma geçmişi (kısa süreli) ve bilgi birikimi (uzun süreli) birlikte yönetilir.

    Kullanım:
        memory = MemoryManager(persist_dir="./data/memory")
        memory.add_conversation_message("user", "Bu sahnede ne var?")
        memory.store_analysis("3 araba tespit edildi", {"type": "vehicle_detection"})
        context = memory.get_full_context("araçlar hakkında ne biliyorsun?")
    """

    def __init__(self, max_conversation_messages: int = 20, persist_dir: Optional[str] = None):
        self.conversation = ConversationMemory(max_messages=max_conversation_messages)
        self.vector_store = VectorMemory(persist_dir=persist_dir)

    def add_conversation_message(self, role: str, content: str):
        """Konuşma mesajı ekle."""
        self.conversation.add_message(role, content)

    def store_analysis(self, analysis_text: str, metadata: Optional[dict] = None):
        """Analiz sonucunu uzun süreli hafızaya kaydet."""
        self.vector_store.store(analysis_text, metadata)

    def get_full_context(self, query: str) -> str:
        """
        Hem kısa hem uzun süreli hafızadan context oluştur.

        Bu, agent'ın "hatırlama" yeteneğidir:
        1. Son konuşmayı al (short-term)
        2. İlgili geçmiş bilgiyi ara (long-term, RAG)
        3. İkisini birleştir
        """
        parts = []

        # Short-term: konuşma geçmişi
        if not self.conversation.is_empty:
            parts.append(f"Konuşma geçmişi:\n{self.conversation.get_context_summary()}")

        # Long-term: ilgili geçmiş bilgi (RAG)
        relevant = self.vector_store.get_relevant_context(query)
        if relevant:
            parts.append(relevant)

        return "\n\n".join(parts) if parts else ""
