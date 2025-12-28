import argparse
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


LOGGER = logging.getLogger("jira_formatter.prepare_dataset")

DEFAULT_REQUIRED_COLUMNS = [
    "Issue key",
    "Summary",
    "Issue Type",
    "Status",
    "Project key",
    "Project name",
    "Priority",
    "Created",
    "Updated",
    "Labels",
    "Description",
    "Custom field (Acceptance Criteria)",
]

@dataclass
class PreparedIssue:
    issue_key: str
    project_key: Optional[str]
    project_name: Optional[str]
    issue_type: Optional[str]
    status: Optional[str]
    priority: Optional[str]
    created: Optional[str]
    updated: Optional[str]
    labels: List[str]
    summary: str
    description: str
    acceptance_criteria: str
    comments: str
    text: str
    source: Dict[str, Any]


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

def _safe_str(value: Any) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value)

def _normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

_ACCOUNT_ID_PATTERN = re.compile(r"\[~accountid:[^\]]+\]")
_IMAGE_PATTERN = re.compile(r"!\S+?\|[^!]*!")  # Jira image macro variants
_SMART_LINK_PATTERN = re.compile(r"\|smart-link\]", re.IGNORECASE)

def _redact_jira_tokens(text: str) -> str:
    # Redact account ids but keep readability
    text = _ACCOUNT_ID_PATTERN.sub("@user", text)
    # Remove common image macro blocks
    text = _IMAGE_PATTERN.sub("", text)
    # Remove smart-link suffix noise
    text = _SMART_LINK_PATTERN.sub("]", text)
    return text

def _parse_labels(raw: str) -> List[str]:
    raw = _normalize_whitespace(_safe_str(raw))
    if not raw:
        return []
    # Jira export often uses space-separated labels, sometimes comma-separated
    tokens = re.split(r"[\s,]+", raw)
    return [t for t in (tok.strip() for tok in tokens) if t]


def _extract_comment_bodies(row:pd.Series, comment_cols:List[str]) -> List[str]:
    bodies: List[str] = []
    for col in comment_cols:
        raw = _safe_str(row.get(col, ""))
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split(";", 2)
        if len(parts) == 3:
            body = parts[2]
        else:
            body = raw
        body = _normalize_whitespace(_redact_jira_tokens(body))
        if body:
            bodies.append(body)
    return bodies


def _build_text(summary: str, description: str, acceptance_criteria: str, comments: str) -> str:
    sections: List[Tuple[str, str]] = [
        ("Summary", summary),
        ("Description", description),
        ("Acceptance Criteria", acceptance_criteria),
        ("Comments", comments),
    ]

    chunks: List[str] = []
    for title, content in sections:
        content = _normalize_whitespace(content)
        if not content:
            continue
        chunks.append(f"{title}\n{content}")

    return _normalize_whitespace("\n\n---\n\n".join(chunks))


def load_csv(csv_path:Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path,encoding='utf-8-sig',encoding_errors='replace')
    return df


def validate_columns(df: pd.DataFrame, required: List[str]) -> None:
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

#Prepare csv into ready to use list of issues(ticket)
def prepare_issues(df:pd.DataFrame,source_name:str) -> List[PreparedIssue]:
    comment_cols = [c for c in df.columns if c.startswith("Comment")]
    prepared:List[PreparedIssue] = []

    for idx,row in df.iterrows():
        issue_key = _normalize_whitespace(_safe_str(row.get("Issue key")))
        summary = _normalize_whitespace(_safe_str(row.get("Summary")))

        if not issue_key or not summary:
            continue #REVIEW - Should this retrun error? these 2 are not affecting Model output

        description = _normalize_whitespace(_redact_jira_tokens(_safe_str(row.get("Description"))))
        acceptance_criteria = _normalize_whitespace(
            _redact_jira_tokens(_safe_str(row.get("Custom field (Acceptance Criteria)")))
        )

        comment_bodies = _extract_comment_bodies(row,comment_cols)
        comments = _normalize_whitespace("\n\n".join(comment_bodies))

        text = _build_text(summary, description, acceptance_criteria, comments)
        labels = _parse_labels(_safe_str(row.get("Labels")))

        prepared.append(
            PreparedIssue(
                issue_key=issue_key,
                project_key=_normalize_whitespace(_safe_str(row.get("Project key"))) or None,
                project_name=_normalize_whitespace(_safe_str(row.get("Project name"))) or None,
                issue_type=_normalize_whitespace(_safe_str(row.get("Issue Type"))) or None,
                status=_normalize_whitespace(_safe_str(row.get("Status"))) or None,
                priority=_normalize_whitespace(_safe_str(row.get("Priority"))) or None,
                created=_normalize_whitespace(_safe_str(row.get("Created"))) or None,
                updated=_normalize_whitespace(_safe_str(row.get("Updated"))) or None,
                labels=labels,
                summary=summary,
                description=description,
                acceptance_criteria=acceptance_criteria,
                comments=comments,
                text=text,
                source={"type": "csv", "name": source_name, "row_index": int(idx)},
            )
        )
     # Deduplicate by issue_key (keep most recent by Updated when possible)
    by_key: Dict[str, PreparedIssue] = {}
    for item in prepared:
        existing = by_key.get(item.issue_key)
        if existing is None:
            by_key[item.issue_key] = item
            continue

        # Fallback: prefer record with longer text if timestamps aren't comparable
        if (item.updated or "") >= (existing.updated or ""):
            by_key[item.issue_key] = item
        elif len(item.text) > len(existing.text):
            by_key[item.issue_key] = item

    return list(by_key.values())



#Prepare Json line
def write_jsonl(items: List[PreparedIssue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for it in items:
            payload = {
                "issue_key": it.issue_key,
                "project_key": it.project_key,
                "project_name": it.project_name,
                "issue_type": it.issue_type,
                "status": it.status,
                "priority": it.priority,
                "created": it.created,
                "updated": it.updated,
                "labels": it.labels,
                "summary": it.summary,
                "description": it.description,
                "acceptance_criteria": it.acceptance_criteria,
                "comments": it.comments,
                "text": it.text,
                "source": it.source,
            }
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


#Prepared csv with only required columns
def write_min_csv(items: List[PreparedIssue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        {
            "issue_key": it.issue_key,
            "project_key": it.project_key,
            "project_name": it.project_name,
            "issue_type": it.issue_type,
            "status": it.status,
            "priority": it.priority,
            "created": it.created,
            "updated": it.updated,
            "labels": " ".join(it.labels),
            "summary": it.summary,
            "description": it.description,
            "acceptance_criteria": it.acceptance_criteria,
            "comments": it.comments,
        }
        for it in items
    ]
    pd.DataFrame(rows).to_csv(out_path, index=False, encoding="utf-8")

def write_stats(items: List[PreparedIssue], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def _count(key_fn):
        out: Dict[str, int] = {}
        for it in items:
            k = key_fn(it) or "UNKNOWN"
            out[k] = out.get(k, 0) + 1
        return dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))

    stats = {
        "total_prepared": len(items),
        "by_project_key": _count(lambda x: x.project_key),
        "by_issue_type": _count(lambda x: x.issue_type),
        "by_status": _count(lambda x: x.status),
    }

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare Jira export CSV into processed dataset for RAG indexing.")
    parser.add_argument("--input", required=True, help="Path to Jira CSV export")
    parser.add_argument("--out-dir", default="data/processed", help="Output directory (default: data/processed)")
    parser.add_argument("--log-level", default="INFO", help="Logging level (default: INFO)")
    args = parser.parse_args()

    setup_logging(args.log_level)

    csv_path = Path(args.input).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser()

    LOGGER.info("Loading CSV: %s", csv_path)
    df = load_csv(csv_path)

    validate_columns(df, DEFAULT_REQUIRED_COLUMNS)

    LOGGER.info("Preparing issues (rows=%d, cols=%d)", df.shape[0], df.shape[1])
    items = prepare_issues(df, source_name=csv_path.name)

    jsonl_path = out_dir / "jira_issues.jsonl"
    min_csv_path = out_dir / "jira_issues_min.csv"
    stats_path = out_dir / "stats.json"

    LOGGER.info("Writing jsonl: %s", jsonl_path)
    write_jsonl(items, jsonl_path)

    LOGGER.info("Writing trimmed csv: %s", min_csv_path)
    write_min_csv(items, min_csv_path)

    LOGGER.info("Writing stats: %s", stats_path)
    write_stats(items, stats_path)

    LOGGER.info("Done. Prepared issues: %d", len(items))


if __name__ == "__main__":
    main()