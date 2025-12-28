import argparse
from pathlib import Path

from jira_formatter.rag.vectorstore import ChromaConfig, ChromaVectorStore


# This is a function to parse CLI arguments for inspecting an existing ChromaDB index.
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Peek into ChromaDB to confirm index contents.")
    p.add_argument("--persist-dir", type=str, required=True, help="Chroma persist directory.")
    p.add_argument("--collection", type=str, default="jira_issues", help="Collection name.")
    p.add_argument("--limit", type=int, default=3, help="How many items to preview.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    store = ChromaVectorStore(
        ChromaConfig(persist_dir=Path(args.persist_dir), collection_name=args.collection)
    )

    print(f"Collection: {args.collection}")
    print(f"Persist dir: {args.persist_dir}")
    print(f"Count: {store.count()}")

    sample = store.peek(limit=args.limit)
    print("Peek sample:")
    print(sample)


if __name__ == "__main__":
    main()
