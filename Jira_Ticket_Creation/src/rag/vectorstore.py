from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class ChromaConfig:
    """Configuration for ChromaDB persistence and collection selection"""
    persist_dir:Path
    collection_name:str = "jira_issues"

class ChromaVectorStore:
    """
    ChromaDB wrapper for collection operations (add/query/get/count).
    """

    def __init__(self,config:ChromaConfig) -> None:
        self._config = config

        try:
            import chromadb
        except ImportError as e:
            raise ImportError("missing dependency, install using uv add chromadb") from e
        self._client = chromadb.PersistentClient(path=str(config.persist_dir))
        self._collection = self._client.get_or_create_collection(name=config.collection_name)

        
    #Function to add document & embedsding into chroma collection
    def add(self,
            ids:list[str],
            documents:list[str],
            embeddings:list[list[float]],
            metadatas:list[dict[str,Any]] | None = None ) -> None:
        
        if metadatas is None:
            metadatas = [{} for _ in ids]
        
        self._collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )

    #Function to run similarity search against collection using query embeddings
    #It returns a Chroma-like dict

    def query(
            self,
            query_embeddings:list[list[float]],
            n_results:int,
            include:list[str],
            where:dict[str,Any] | None = None
    ) -> dict[str,Any]:
        return self._collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            include=include,
            where=where,
        )
    
     # This is a function to return the number of items stored in the collection.
    def count(self) -> int:
        return int(self._collection.count())

    # This is a function to fetch a small sample of stored documents/metadatas for sanity checking.
    def peek(self, limit: int = 3) -> dict[str, Any]:
        # IMPORTANT: For .get(), "include" must NOT contain "ids" (Chroma returns ids by default).
        return self._collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )
