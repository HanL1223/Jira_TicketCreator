import re


# This is a function to normalize whitespace so the index receives clean, consistent text.
def normalize_whitespace(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def build_issue_text(
    *,
    summary: str,
    description: str | None,
    acceptance_criteria: str | None,
    comments: list[str] | None,
) -> str:
    """Assemble an issue body from components.

    Sections are only included when non-empty. The resulting text is run
    through `normalize_whitespace` for consistent spacing.
    """
    parts: list[str] = []
    parts.append(f"Summary:\n{summary.strip()}")

    if description and description.strip():
        parts.append(f"Description:\n{description.strip()}")

    if acceptance_criteria and acceptance_criteria.strip():
        parts.append(f"Acceptance Criteria:\n{acceptance_criteria.strip()}")

    if comments:
        non_empty = [c.strip() for c in comments if c and c.strip()]
        if non_empty:
            # Keep comments short-ish; retrieval benefits from recent decisions and context.
            joined = "\n---\n".join(non_empty[:30])
            parts.append(f"Comments:\n{joined}")

    return normalize_whitespace("\n\n".join(parts))