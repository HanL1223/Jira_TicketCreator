from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class EmbeddingsClient(Protocol):
    """
    Interface for embedding providers (Gemini today, replaceable later).
    """

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class GeminiEmbeddingsConfig:
    """
    Configuration for Gemini embedding model.
    """
    api_key: str
    model: str = "models/text-embedding-004"


class GeminiEmbeddingsClient:
    """
    Gemini-based embeddings client with a small surface area.
    """

    def __init__(self, config: GeminiEmbeddingsConfig) -> None:
        self._config = config

        # Lazy import keeps package optional until you actually embed.
        try:
            import google.generativeai as genai  # type: ignore
        except ImportError as e:
            raise ImportError(
                "Missing dependency: google-generativeai"
            ) from e

        genai.configure(api_key=config.api_key)
        self._genai = genai

    # This is a function to embed multiple texts into vectors using Gemini embedding model.
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for t in texts:
            # Gemini embedding response format depends on library version; handle the common cases.
            resp = self._genai.embed_content(model=self._config.model, content=t)
            emb = resp.get("embedding") if isinstance(resp, dict) else getattr(resp, "embedding", None)
            if emb is None:
                raise RuntimeError("Gemini embed response missing 'embedding'")
            vectors.append(list(emb))
        return vectors
