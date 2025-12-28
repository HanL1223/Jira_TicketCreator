from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class JiraIssue:
    """
    Represents a normalized Jira issue record used across ingestion, indexing, retrieval, and prompting.
    This should be identical to output of script/prepare_dataset.py 
    """

    issue_key: str
    project_key: str
    issue_type: str
    status: str
    summary: str
    description: str | None = None
    acceptance_criteria: str | None = None
    comments: list[str] | None = None
    labels: list[str] | None = None
    priority: str | None = None
    created: datetime | None = None
    updated: datetime | None = None
    raw: dict[str, Any] | None = None  # optional: keep raw row for traceability



