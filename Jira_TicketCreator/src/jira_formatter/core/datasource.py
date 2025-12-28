from __future__ import annotations

import csv
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Iterator

from jira_formatter.core.models import JiraIssue


@dataclass(frozen=True)
class SourceConfig:
    """
    Configuration for a Jira issue data source.
    """
    kind: str  # "jsonl" | "csv" | "jira_api" (Phase 2)
    path: Path | None = None

class IssueSource(ABC):
    @abstractmethod
    def iter_issues(self) -> Iterator[JiraIssue]:
        """Yield JiraIssue objects from the data source."""
        raise NotImplementedError

# This is a function to parse Jira-export date strings safely (best-effort).
def _parse_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    value = value.strip()
    # Handle common Jira export formats, expand as needed
    for fmt in ("%d/%m/%Y %H:%M", "%d/%b/%y %I:%M %p", "%Y-%m-%dT%H:%M:%S.%f%z"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


# This is a function to normalize "labels" into a list[str] regardless of export format.
def _coerce_labels(row: dict) -> list[str] | None:
    labels = row.get("Labels") or row.get("labels")
    if not labels:
        return None
    if isinstance(labels, list):
        return [str(x).strip() for x in labels if str(x).strip()]
    # Jira CSV exports often store labels as "a b c" or "a, b, c"
    s = str(labels).strip()
    if not s:
        return None
    if "," in s:
        return [p.strip() for p in s.split(",") if p.strip()]
    return [p.strip() for p in s.split() if p.strip()]


# This is a function to extract comment fields from your Jira export (which can appear as Comment, Comment.1, ...).
def _coerce_comments(row: dict) -> list[str] | None:
    candidates: list[str] = []
    for k, v in row.items():
        if not k:
            continue
        if str(k).strip().lower().startswith("comment") and v:
            s = str(v).strip()
            if s:
                candidates.append(s)
    return candidates or None


class JsonlIssueSource(IssueSource):
    def __init__(self, path: Path) -> None:
        self.path = path

    def iter_issues(self) -> Iterator[JiraIssue]:
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)

                yield JiraIssue(
                    issue_key=str(row.get("issue_key") or row.get("Issue key") or "").strip(),
                    project_key=str(row.get("project_key") or row.get("Project key") or "").strip(),
                    issue_type=str(row.get("issue_type") or row.get("Issue Type") or "").strip(),
                    status=str(row.get("status") or row.get("Status") or "").strip(),
                    summary=str(row.get("summary") or row.get("Summary") or "").strip(),
                    description=row.get("description") or row.get("Description"),
                    acceptance_criteria=row.get("acceptance_criteria") or row.get("Custom field (Acceptance Criteria)"),
                    comments=row.get("comments") or _coerce_comments(row),
                    labels=_coerce_labels(row),
                    priority=row.get("priority") or row.get("Priority"),
                    created=_parse_dt(row.get("created") or row.get("Created")),
                    updated=_parse_dt(row.get("updated") or row.get("Updated")),
                    raw=row,
                )

class CsvIssueSource(IssueSource):
    """
    Reads normalized issues from a CSV file (useful for quick experiments).
    """

    def __init__(self, path: Path) -> None:
        self._path = path

    # This is a function to stream issues from a CSV file using DictReader.
    def iter_issues(self) -> Iterator[JiraIssue]:
        with self._path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                yield JiraIssue(
                    issue_key=str(row.get("Issue key") or row.get("issue_key") or "").strip(),
                    project_key=str(row.get("Project key") or row.get("project_key") or "").strip(),
                    issue_type=str(row.get("Issue Type") or row.get("issue_type") or "").strip(),
                    status=str(row.get("Status") or row.get("status") or "").strip(),
                    summary=str(row.get("Summary") or row.get("summary") or "").strip(),
                    description=row.get("Description") or row.get("description"),
                    acceptance_criteria=row.get("Custom field (Acceptance Criteria)") or row.get("acceptance_criteria"),
                    comments=_coerce_comments(row),
                    labels=_coerce_labels(row),
                    priority=row.get("Priority") or row.get("priority"),
                    created=_parse_dt(row.get("Created") or row.get("created")),
                    updated=_parse_dt(row.get("Updated") or row.get("updated")),
                    raw=row,
                )

# This is a factory method that returns the correct source implementation for the configured input.
def create_issue_source(config: SourceConfig) -> IssueSource:
    kind = config.kind.lower().strip()
    if kind in {"jsonl", "jsonlines"}:
        if not config.path:
            raise ValueError("jsonl source requires a path")
        return JsonlIssueSource(config.path)

    if kind in {"csv"}:
        if not config.path:
            raise ValueError("csv source requires a path")
        return CsvIssueSource(config.path)

    if kind in {"jira_api"}:
        raise NotImplementedError(
            "jira_api source is not implemented yet. Keep the interface; implement later via Jira REST API."
        )

    raise ValueError(f"Unknown source kind: {config.kind}")