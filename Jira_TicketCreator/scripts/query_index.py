import argparse
import logging
import os
from pathlib import Path

from jira_formatter.rag.embeddings import GeminiEmbeddingsClient, GeminiEmbeddingsConfig
from jira_formatter.rag.retriever import JiraIssueRetriever, RetrievalConfig
from jira_formatter.rag.vectorstore import ChromaConfig, ChromaVectorStore

LOGGER = logging.getLogger(__name__)


# This is a function to configure structured logging for CLI scripts.
def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# This is a function to parse CLI arguments for testing retrieval.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Query ChromaDB index and print top matches.")
    p.add_argument("--persist-dir", default="data/chromadb")
    p.add_argument("--collection", default="jira_issues")
    p.add_argument("--top-k", type=int, default=6)
    p.add_argument("--query", required=True, help="Natural-language query to retrieve similar tickets.")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. In PowerShell run: $env:GOOGLE_API_KEY='...'"
        )

    embedder = GeminiEmbeddingsClient(GeminiEmbeddingsConfig(api_key=api_key))
    store = ChromaVectorStore(
        ChromaConfig(persist_dir=Path(args.persist_dir), collection_name=args.collection)
    )

    retriever = JiraIssueRetriever(
        embedder=embedder,
        store=store,
        config=RetrievalConfig(top_k=args.top_k),
    )

    LOGGER.info("Collection count=%d", store.count())

    chunks = retriever.retrieve(args.query)
    print(retriever.format_context(chunks))


if __name__ == "__main__":
    main()
