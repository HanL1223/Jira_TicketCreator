from dataclasses import dataclass
from typing import Protocol,Optional
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log
)

logger = logging.getLogger(__name__)



class EmbeddingsClient(Protocol):
    """Interface for embedding providers"""
    def embed_text(self,texts:list[str]) -> list[list[float]]:
        ...


@dataclass(frozen=True)
class GeminiEmbeddingConfig:
    """Configuration for Gemini embedding model."""
    api_key:str
    model:str ="models/text-embedding-004"
    batch_size:int = 100

class GeminiEmbeddingClient:
    """Gemini embeddings client"""
    def __init__(self,config :GeminiEmbeddingConfig) -> None:
        self._config = config

        try:
            from google import genai
            from google.genai import types
        except ImportError as e:
            raise ImportError("Missing dependency:google-genai, Install using uv add google-genai") from e 
        self._genai = genai
        self._types = types
        self._client = genai.Client(api_key=config.api_key)


    def _extract_embedding(self,result) -> list[list[float]]:
        if hasattr(result, "embeddings") and result.embeddings:
            return [list(e.values) for e in result.embeddings]
        if hasattr(result, "embedding") and result.embedding and hasattr(result.embedding, "values"):
            return [list(result.embedding.values)]
        if isinstance(result, dict):
            if "embeddings" in result:
                out = []
                for e in result["embeddings"]:
                    if isinstance(e, dict) and "values" in e:
                        out.append(list(e["values"]))
                if out:
                    return out
            if "embedding" in result:
                emb = result["embedding"]
                if isinstance(emb, dict) and "values" in emb:
                    return [list(emb["values"])]
        raise RuntimeError(
            f"Cannot extract embedding(s) from response. "
            f"Response type: {type(result)}, repr: {repr(result)[:200]}"
        )


    def _embed_batch(self,batch_texts:list[str])-> list[list[float]]:
        try:
            from google.genai import types

            config = types.EmbedContentConfig(task_type="RETRIEVAL_DOCUMENT")



            result = self._client.models.embed_content(
                model = self._config.model,
                contents=batch_texts,
                config = config
            )

            vectors = self._extract_embedding(result)
            #For each text we should except 1 embedded vector
            if len(vectors) != len(batch_texts):
                raise RuntimeError(
                f"Embedding count mismatch: got {len(vectors)} vectors for {len(batch_texts)} texts"
            )

            return vectors
        except Exception as e:
            logger.error(
                f"Embedding batch failed. Model: {self._config.model}, Error: {str(e)}"
            )
            
            error_str = str(e).lower()
            if "api key" in error_str or "authentication" in error_str:
                raise RuntimeError(
                    "Invalid API key. Check GOOGLE_API_KEY environment variable."
                ) from e
            elif "quota" in error_str or "rate limit" in error_str:
                raise RuntimeError(
                    "API quota exceeded or rate limited. Consider adding delays or upgrading quota."
                ) from e
            else:
                raise RuntimeError(f"Gemini embedding API error: {str(e)}") from e
            


        

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((ConnectionError, TimeoutError)),
        before_sleep=before_sleep_log(logger, logging.WARNING)
    )
    def embed_texts(self,texts:list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors:list [list[float]] = []

        for i in range (0,len(texts),self._config.batch_size):
            batch = texts[i:i +self._config.batch_size]
            batch_vectors = self._embed_batch(batch)
            vectors.extend(batch_vectors)
        return vectors
    






