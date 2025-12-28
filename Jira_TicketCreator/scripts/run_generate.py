from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

from jira_formatter.llm.gemini import GeminiClient, GeminiConfig
from jira_formatter.pipelines.generate_ticket import TicketGenerator
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


# This is a function to parse CLI arguments for generating a Jira card from a natural-language request.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate Jira card using RAG + multi-agent refinement.")
    p.add_argument("--persist-dir", default="data/chromadb", help="ChromaDB persistent directory.")
    p.add_argument("--collection", default="jira_issues", help="Chroma collection name.")
    p.add_argument("--top-k", type=int, default=6, help="Number of retrieved chunks.")
    p.add_argument("--query", required=True, help="User request (what you want the Jira card for).")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--no-rag", action="store_true", help="Disable retrieval augmentation.")
    p.add_argument("--no-agents", action="store_true", help="Disable multi-agent refinement loop.")
    p.add_argument("--model", default="gemini-2.5-flash-lite-preview-09-2025", help="Gemini model id.")

    return p.parse_args()


# This is a function to resolve GOOGLE_API_KEY from environment variables.
def get_api_key() -> str:
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is required.\n"
            "PowerShell example: $env:GOOGLE_API_KEY='YOUR_KEY'\n"
            "Then rerun the command."
        )
    return api_key


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    api_key = get_api_key()

    store = ChromaVectorStore(
        ChromaConfig(
            persist_dir=Path(args.persist_dir),
            collection_name=args.collection,
        )
    )
    LOGGER.info("Collection count=%s", store.count())

    embedder = GeminiEmbeddingsClient(GeminiEmbeddingsConfig(api_key=api_key))
    retriever = JiraIssueRetriever(
        embedder=embedder,
        store=store,
        config=RetrievalConfig(top_k=args.top_k),
    )

    llm = GeminiClient(GeminiConfig(api_key=api_key, model=args.model))
    generator = TicketGenerator(
        retriever=retriever,
        llm=llm,
        config=None if (not args.no_rag and not args.no_agents) else None,
    )

    # Apply flags without creating a new config class just for CLI.
    if args.no_rag or args.no_agents:
        from jira_formatter.pipelines.generate_ticket import GenerationConfig

        generator = TicketGenerator(
            retriever=retriever,
            llm=llm,
            config=GenerationConfig(use_rag=not args.no_rag, use_agents=not args.no_agents),
        )

    result = generator.generate(args.query)

    print("\n" + "=" * 80)
    print("JIRA CARD OUTPUT")
    print("=" * 80)
    print(result.jira_markdown)
    print("\n" + "=" * 80)
    print(f"RAG used: {result.rag_context_used} | retrieved_chunks={result.retrieved_chunks}")
    print("=" * 80)


if __name__ == "__main__":
    main()
