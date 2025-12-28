from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class EmbeddingsClient(Protocol):
    """
    Interface for embedding providers (Gemini today, replaceable later).
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


class VectorStore(Protocol):
    """
    Interface for vector stores (Chroma today, replaceable later).
    """

    def query(
        self,
        query_embeddings: list[list[float]],
        n_results: int,
        include: list[str],
        where: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        ...


@dataclass(frozen=True)
class RetrievedChunk:
    """
    A single retrieved text chunk with optional metadata and distance score.
    """
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    distance: float | None = None


@dataclass(frozen=True)
class RetrievalConfig:
    """
    Retrieval settings for the RAG retriever.
    """
    top_k: int = 6
    include: tuple[str, ...] = ("documents", "metadatas", "distances")
    where: dict[str, Any] | None = None


class JiraIssueRetriever:
    """
    Retrieves the most relevant Jira issue chunks from the vector store.
    """

    def __init__(
        self,
        embedder: EmbeddingsClient,
        store: VectorStore,
        config: RetrievalConfig | None = None,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._config = config or RetrievalConfig()

    # This is a function to retrieve top-k relevant chunks given a natural-language query.
    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        query_text = (query_text or "").strip()
        if not query_text:
            return []

        query_vec = self._embedder.embed_texts([query_text])[0]

        result = self._store.query(
            query_embeddings=[query_vec],
            n_results=self._config.top_k,
            include=list(self._config.include),
            where=self._config.where,
        )

        # Chroma returns lists-of-lists: ids[0], documents[0], ...
        ids = (result.get("ids") or [[]])[0]
        docs = (result.get("documents") or [[]])[0]
        metas = (result.get("metadatas") or [[]])[0]
        dists = (result.get("distances") or [[]])[0] if "distances" in result else []

        chunks: list[RetrievedChunk] = []
        for i, chunk_id in enumerate(ids):
            text = docs[i] if i < len(docs) else ""
            meta = metas[i] if i < len(metas) and metas[i] is not None else {}
            dist = dists[i] if i < len(dists) else None

            chunks.append(
                RetrievedChunk(
                    chunk_id=str(chunk_id),
                    text=str(text or ""),
                    metadata=dict(meta) if isinstance(meta, dict) else {},
                    distance=float(dist) if dist is not None else None,
                )
            )

        return chunks

    # This is a function to turn retrieved chunks into a single context string for prompting.
    def format_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No relevant historical tickets found in the index."

        lines: list[str] = ["## Retrieved historical tickets (top matches)"]
        for rank, ch in enumerate(chunks, start=1):
            issue_key = ch.metadata.get("issue_key", "UNKNOWN")
            status = ch.metadata.get("status", "UNKNOWN")
            priority = ch.metadata.get("priority", "UNKNOWN")
            distance = f"{ch.distance:.4f}" if ch.distance is not None else "N/A"

            lines.append(
                f"\n### Match {rank}: {issue_key} (status={status}, priority={priority}, distance={distance})\n"
                f"{ch.text.strip()}\n"
            )
        return "\n".join(lines)
