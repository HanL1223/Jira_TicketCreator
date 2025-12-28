from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from typing import Any

from jira_formatter.core.datasource import IssueSource
from jira_formatter.core.text_utils import build_issue_text
from jira_formatter.rag.chunking import chunk_text
from jira_formatter.rag.embeddings import EmbeddingsClient
from jira_formatter.rag.vectorstore import ChromaVectorStore

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class IndexingConfig:
    """
    Controls how issues are converted to chunks and indexed.
    """
    chunk_size: int = 1200
    chunk_overlap: int = 200
    max_issues: int | None = None  # useful for dev runs


class JiraIssueIndexer:
    """
    Orchestrates: source -> text -> chunks -> embeddings -> vector store.
    """

    def __init__(
        self,
        *,
        source: IssueSource,
        embedder: EmbeddingsClient,
        store: ChromaVectorStore,
        config: IndexingConfig | None = None,
    ) -> None:
        self._source = source
        self._embedder = embedder
        self._store = store
        self._config = config or IndexingConfig()

    # This is a function to index all issues from the source into the vector store.
    def run(self) -> dict[str, Any]:
        ids: list[str] = []
        docs: list[str] = []
        metas: list[dict[str, Any]] = []

        issue_count = 0
        chunk_count = 0

        for issue in self._source.iter_issues():
            if self._config.max_issues is not None and issue_count >= self._config.max_issues:
                break

            issue_text = build_issue_text(
                summary=issue.summary,
                description=issue.description,
                acceptance_criteria=issue.acceptance_criteria,
                comments=issue.comments,
            )

            chunks = chunk_text(
                issue_text,
                chunk_size=self._config.chunk_size,
                chunk_overlap=self._config.chunk_overlap,
            )

            for i, ch in enumerate(chunks):
                chunk_id = _stable_chunk_id(issue.issue_key, i, ch)
                ids.append(chunk_id)
                docs.append(ch)
                metas.append(
                    {
                        "issue_key": issue.issue_key,
                        "project_key": issue.project_key,
                        "issue_type": issue.issue_type,
                        "status": issue.status,
                        "priority": issue.priority,
                        "chunk_index": i,
                    }
                )

            issue_count += 1
            chunk_count += len(chunks)

        if not docs:
            LOGGER.warning("No documents to index. Check your input dataset and parsing.")
            return {"issues_indexed": 0, "chunks_indexed": 0}

        embeddings = self._embedder.embed_texts(docs)
        self._store.upsert(ids=ids, documents=docs, embeddings=embeddings, metadatas=metas)

        return {"issues_indexed": issue_count, "chunks_indexed": chunk_count}


# This is a function to create stable chunk IDs so repeated indexing is idempotent.
def _stable_chunk_id(issue_key: str, chunk_index: int, chunk_text: str) -> str:
    h = hashlib.sha256(chunk_text.encode("utf-8")).hexdigest()[:16]
    return f"{issue_key}::chunk={chunk_index}::{h}"
