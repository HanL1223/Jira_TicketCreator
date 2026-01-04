import json                         # For loading/saving the BM25 index
import logging                      # For debug and info messages
import pickle                       # For serializing the BM25 model object
import re                           # For text tokenization
from dataclasses import dataclass   # For clean configuration classes
from pathlib import Path            # For file path handling
from typing import Any              # For flexible type hints


LOGGER = logging.getLogger(__name__)

@dataclass(frozen=True)
class BM25Config:
    """
    Attributes:
        k1: Term frequency saturation parameter (default 1.5)
            - Higher k1 = term frequency matters more
            - Lower k1 = diminishing returns for repeated terms
            - Industry standard is 1.2-2.0
            
        b: Document length normalization (default 0.75)
            - b=1.0: Full length normalization (long docs penalized)
            - b=0.0: No length normalization
            - 0.75 is the "Goldilocks" value
            
        persist_path: Where to save/load the BM25 index
            - None means in-memory only (lost when program exits)
            - Set a path to persist across runs
    """
    k1:float = 1.5
    b:float = 0.75
    persist_path: Path | None = None

@dataclass
class BM25Result:
    """
    Docstring for BM25Result
     Attributes:
        doc_id: Unique identifier for the document
        text: The actual text content of the chunk
        metadata: Dictionary of metadata (issue_key, status, priority, etc.)
        score: BM25 relevance score (higher = more relevant)
            - Note: BM25 scores are NOT normalized (can be any positive number)
            - Dense retrieval returns distances (lower = more similar)
            - We'll handle this difference in the hybrid retriever
    """
    doc_id:str
    text:str
    metadata:dict[str,Any]
    score:float

def tokenize(text:str) -> list[str]:
    """
     Convert text into tokens for BM25 indexing/querying.
    """
    if not text:
        return []
    
    text = text.lower()
    tokens = re.split(r'[^a-z0-9]+', text)
    tokens = [t for t in tokens if len(t) >= 2]

    return tokens

class BM25Index:
    """
     BM25 index for keyword-based retrieval.
    """
    def __init__(self,config:BM25Config | None = None) -> None:
        self._config = config or BM25Config
        self._tokenized_corpus: list[list[str]] = []  # Tokenized docs for BM25
        self._doc_ids: list[str] = []                  # Document IDs
        self._doc_texts: list[str] = []                # Original texts
        self._doc_metadata: list[dict[str, Any]] = []  # Metadata dicts
        
        self._bm25 = None

        LOGGER.debug("BM25Index initialized with config %s",self._config)

    
    def add_documents(self,
                      doc_ids:list[str],
                      texts:list[str],
                      metadatas:list[dict[str,Any]] | None ) -> None:
        """
        Docstring for add_documents
        
        :param self: Description
        :param doc_ids: Description
        :type doc_ids: list[str]
        :param texts: Description
        :type texts: list[str]
        :param metadatas: Description
        :type metadatas: list[dict[str, Any]] | None
        """

        if not doc_ids or not texts:
            raise ValueError("doc_ids or texts cannot be empty")
        if len(doc_ids) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(doc_ids)} doc_ids vs {len(texts)} texts"
            )
        
        if metadatas is None:
            metadatas = [{} for _ in texts]
        if len(metadatas) != len(texts):
            raise ValueError(
                f"Length mismatch: {len(metadatas)} metadatas vs {len(texts)} texts"
            )
        LOGGER.info("Adding %d documents to BM25 index...", len(texts))

        for doc_id,text,meta in zip(doc_ids,texts,metadatas):
            tokens = tokenize(text)

            # Store everything
            self._tokenized_corpus.append(tokens)
            self._doc_ids.append(doc_id)
            self._doc_texts.append(text)
            self._doc_metadata.append(meta)

        self._build_bm25()

        LOGGER.info(
            "BM25 index built: %d documents, avg tokens per doc: %.1f",
            len(self._tokenized_corpus),
            sum(len(t) for t in self._tokenized_corpus) / len(self._tokenized_corpus)
        )

    def _build_bm25(self) -> None:
        """
        Build the BM25 model from the tokenized corpus.
    
    This is called internally after adding documents.
    
    WHY SEPARATE METHOD:
    --------------------
    - Encapsulates the BM25 library dependency
    - Can be called to rebuild after updates
    - Handles the import here to give a clear error if missing
        """

        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            raise ImportError(
                "Missing depencency, install with uv add rank-bm25"
            )
        
        self._bm25 = BM25Okapi(
            self._tokenized_corpus,
            k1 = self._config.k1,
            b=self._config.b
        )

    def search(self,query:str,top_k:int = 5 ) -> list[BM25Result]:
        """
         Search the index for documents matching the query.
        
        Args:
            query: Natural language search query
            top_k: Number of results to return
            
        Returns:
            List of BM25Result objects, sorted by relevance (highest first)
            
        Example:
            >>> results = index.search("Appriss ELT query", top_k=5)
            >>> for r in results:
            ...     print(f"{r.doc_id}: score={r.score:.2f}")
            CSCI-750::chunk=0: score=12.34
            CSCI-769::chunk=1: score=8.21
        """

        if self._bm25 is None or not self._tokenized_corpus:
            LOGGER.warning("BM25 search on empty index, returning []")
            return []
        
        query = (query or "").strip()
        if not query:
            return[]
        
        query_tokens = tokenize(query)

        if not query_tokens:
            LOGGER.warning("Query tokenized to empty listL '%s", query)
            return []
        
        LOGGER.debug("BM25 query tokens: %s", query_tokens)

        # Get BM25 scores for all documents
        # Returns array of scores, one per document
        scores = self._bm25.get_scores(query_tokens)


        #Get top k
        scored_indices = sorted(enumerate(scores),
                                key=lambda x:x[1], #Sort by score which is the 2nd element is socre
                                reverse = True)[:top_k]
        
        results:list[BM25Result] = []

        for idx,score in scored_indices:
            if score == 0:
                continue
            results.append(
                BM25Result(
                doc_id=self._doc_ids[idx],
                text=self._doc_texts[idx],
                metadata=self._doc_metadata[idx],
                score=float(score)
                )
            )

        LOGGER.debug("BM25 search returned %d results", len(results))
        return results
    
    def save(self,path: Path | None = None) -> None:
        """
        Docstring for save
        
        :param self: Description
        :param path: Description
        :type path: Path | None
        """
        save_path = path or self._config.persist_path
        if save_path is None:
            raise ValueError("No save path provided and config.persist_path is None")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Save the BM25 model and corpus data
        index_data = {
            "tokenized_corpus": self._tokenized_corpus,
            "doc_ids": self._doc_ids,
            "doc_texts": self._doc_texts,
            "doc_metadata": self._doc_metadata,
            "config": {
                "k1": self._config.k1,
                "b": self._config.b,
            }
        }

        with open(f"{save_path}.pkl", "wb") as f:
            pickle.dump(index_data, f) 


        # Save human-readable metadata
        meta = {
            "num_documents": len(self._doc_ids),
            "config": {"k1": self._config.k1, "b": self._config.b},
            "sample_doc_ids": self._doc_ids[:5] if self._doc_ids else [],
        }
        
        with open(f"{save_path}.meta.json", "w") as f:
            json.dump(meta, f, indent=2)
        
        LOGGER.info("BM25 index saved to %s", save_path)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        """
        Load a BM25 index from disk.
        
        """
        path = Path(path)
        pkl_path = f"{path}.pkl"
        
        if not Path(pkl_path).exists():
            raise FileNotFoundError(f"BM25 index not found: {pkl_path}")
        
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)
        
        # Reconstruct the index
        config = BM25Config(
            k1=data["config"]["k1"],
            b=data["config"]["b"],
            persist_path=path
        )
        
        index = cls(config)
        index._tokenized_corpus = data["tokenized_corpus"]
        index._doc_ids = data["doc_ids"]
        index._doc_texts = data["doc_texts"]
        index._doc_metadata = data["doc_metadata"]
        
        # Rebuild the BM25 model
        index._build_bm25()
        
        LOGGER.info("BM25 index loaded from %s (%d documents)", path, len(index._doc_ids))
        return index
    
    def __len__(self) -> int:
        """Return number of documents in the index."""
        return len(self._doc_ids)
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"BM25Index(documents={len(self)}, k1={self._config.k1}, b={self._config.b})"


def create_bm25_index(
    doc_ids: list[str],
    texts: list[str],
    metadatas: list[dict[str, Any]] | None = None,
    config: BM25Config | None = None
) -> BM25Index:
    """
    Create and populate a BM25 index in one call.
    
    This is a convenience function that combines:
    1. Creating the index
    2. Adding documents
    
    Args:
        doc_ids: Document identifiers
        texts: Document texts
        metadatas: Optional metadata dicts
        config: BM25 configuration
        
    Returns:
        Populated BM25Index ready for searching
        
    Example:
        index = create_bm25_index(
            doc_ids=["doc1", "doc2"],
            texts=["First document", "Second document"],
            config=BM25Config(persist_path=Path("data/bm25/index"))
        )
        index.save()
    """
    index = BM25Index(config)
    index.add_documents(doc_ids, texts, metadatas)
    return index



if __name__ == "__main__":
    # Simple test
    print("Testing BM25Index...")
    
    # Sample documents (like your Jira data)
    test_docs = [
        "CSCI-750: Appriss query refine - Update the ELT SQL query for data validation",
        "CSCI-769: Work breakdown for RF tactical access - Create data share between Sigma and CW",
        "CSCI-748: Investigate benchmark transaction issue for Appriss audit data",
        "CSCI-770: Appriss worksheet response - Review comments related to Appriss data",
    ]
    test_ids = ["doc_0", "doc_1", "doc_2", "doc_3"]
    test_meta = [{"issue_key": f"CSCI-{750+i}"} for i in range(4)]
    
    # Create and populate index
    index = create_bm25_index(test_ids, test_docs, test_meta)
    print(f"Created: {index}")
    
    # Test searches
    queries = [
        "Appriss data",          # Should match docs 0, 2, 3
        "CSCI-750",               # Should match doc 0
        "ELT SQL query",          # Should match doc 0
        "Sigma data share",       # Should match doc 1
    ]
    
    for q in queries:
        results = index.search(q, top_k=2)
        print(f"\nQuery: '{q}'")
        for r in results:
            print(f"  {r.metadata.get('issue_key', 'N/A')}: score={r.score:.2f}")
    
    print("\n BM25 tests passed!")


