from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from jira_formatter.core.datasource import SourceConfig, create_issue_source
from jira_formatter.rag.embeddings import GeminiEmbeddingsClient, GeminiEmbeddingsConfig
from jira_formatter.rag.indexer import IndexingConfig, JiraIssueIndexer
from jira_formatter.rag.vectorstore import ChromaConfig, ChromaVectorStore


LOGGER = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# This is a function to configure structured logging for CLI scripts.
def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )


# This is a function to parse CLI arguments for indexing your dataset into ChromaDB.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Bootstrap ChromaDB with Jira issues (RAG index).")
    p.add_argument("--input-kind", choices=["jsonl", "csv"], default="jsonl")
    p.add_argument("--input", required=True, help="Path to data/processed jira_issues.jsonl or trimmed csv.")
    p.add_argument("--persist-dir", default="data/chromadb", help="ChromaDB persistent directory.")
    p.add_argument("--collection", default="jira_issues", help="Chroma collection name.")
    p.add_argument("--chunk-size", type=int, default=1200)
    p.add_argument("--chunk-overlap", type=int, default=200)
    p.add_argument("--max-issues", type=int, default=0, help="0 means no limit.")
    p.add_argument("--log-level", default="INFO")
    p.add_argument(
    "--persist-dir",
    default=str(PROJECT_ROOT / "data" / "chromadb"),
    help="ChromaDB persistent directory."
)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is required to build embeddings.")

    source = create_issue_source(SourceConfig(kind=args.input_kind, path=Path(args.input)))
    embedder = GeminiEmbeddingsClient(GeminiEmbeddingsConfig(api_key=api_key))
    store = ChromaVectorStore(
        ChromaConfig(persist_dir=Path(args.persist_dir), collection_name=args.collection)
    )

    max_issues = None if args.max_issues == 0 else args.max_issues
    indexer = JiraIssueIndexer(
        source=source,
        embedder=embedder,
        store=store,
        config=IndexingConfig(
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            max_issues=max_issues,
        ),
    )

    stats = indexer.run()
    LOGGER.info("Indexing complete: %s", stats)


if __name__ == "__main__":
    main()
