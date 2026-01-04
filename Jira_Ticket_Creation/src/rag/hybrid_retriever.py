import logging
from dataclasses import dataclass
from typing import Any, Protocol

from rag.bm25_retriever import BM25Index, BM25Result

from rag.retriever import RetrievedChunk


LOGGER = logging.getLogger(__name__)
class DenseRetriever(Protocol):
    def retrieve(self,query_text:str) -> list[RetrievedChunk]:
        """
        Retrieve relevant chunks using dense embeddings.
        """
        ...
@dataclass(frozen=True)
class HybridConfig:
    """
    Configuration for hybrid retrieval.
    
    Attributes:
        top_k: Final number of results to return
        
        dense_weight: Weight for dense retrieval (0.0 to 1.0)
            - Higher = trust dense more
            - Lower = trust BM25 more
            - 0.5 = equal weight
            
        bm25_weight: Weight for BM25 retrieval (0.0 to 1.0)
            - Together with dense_weight, determines final ranking
            - Don't need to sum to 1.0 (we normalize)
            
        rrf_k: RRF constant (default 60)
            - Higher k = more even scoring across ranks
            - Lower k = stronger preference for top ranks
            - 60 is the standard value from the original paper
            
        dense_top_k: How many results to fetch from dense (before fusion)
            - Fetch more than final top_k for better fusion
            - Rule of thumb: 2-3x final top_k
            
        bm25_top_k: How many results to fetch from BM25 (before fusion)
            - Same logic as dense_top_k
    """
    top_k: int = 5
    dense_weight: float = 0.5
    bm25_weight: float = 0.5
    rrf_k: int = 60
    #Provide more candidate for each retriever before fusion
    dense_top_k: int = 15
    bm25_top_k: int = 15

class HybridRetriever:
    """
     Hybrid retriever combining dense embeddings and BM25 keyword search.
    
    This is the main class you'll use in your pipeline.
    
    ARCHITECTURE:
    -------------
        User Query
            │
            ├──────────────────┬─────────────────┐
            ▼                  ▼                 │
        Dense Retriever    BM25 Index           │
        (embeddings)       (keywords)           │
            │                  │                 │
            ▼                  ▼                 │
        Ranked List A      Ranked List B        │
            │                  │                 │
            └────────┬─────────┘                 │
                     ▼                           │
            Reciprocal Rank Fusion               │
                     │                           │
                     ▼                           │
            Final Ranked Results ◄───────────────┘
    
    USAGE:
    ------
    # Setup (in your bootstrap script)
    hybrid = HybridRetriever(
        dense_retriever=your_existing_retriever,
        bm25_index=bm25_index,
        config=HybridConfig(top_k=5)
    )
    
    # Query (in your generation pipeline)
    results = hybrid.retrieve("Appriss ELT pipeline issue")
    """

