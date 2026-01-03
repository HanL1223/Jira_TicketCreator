"""
Reason of this file to have a structured text format simliar to Jira so that the model can output simliar result
# Just concatenate everything
text = f"{issue.summary} {issue.description} {issue.acceptance_criteria}"
# Result: "Configure Snowflake integration We need to switch to key pair auth for security - [ ] Works in DEV - [ ] Works in SIT"
#  No structure, hard to parse, words blur together
"""

import re

def normalise_whitespace(text: str) -> str:
    text = text.replace("\r\n","\n").replace("\r","\n")
    text = re.sub(r"[ \t]+"," ",text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def build_issue_text(
        *, #We expect Keyword arg only
        summary:str,
        description:str|None,
        acceptance_criteria:str|None,
        comments:list[str] |None
) -> str:
    parts:list[str] = []
    parts.append(f"Summary:\n{summary.strip()}")
    if description and description.strip():
            parts.append(f"Description:\n{description.strip()}")

    if acceptance_criteria and acceptance_criteria.strip():
        parts.append(f"Acceptance Criteria:\n{acceptance_criteria.strip()}")
    if comments:
         not_empty = [c.strip() for c in comments if c and c.strip()]
         if not_empty:
              # Keep comments relative short(limit to first 15 as the comment order are new to old); retrieval benefits from recent decisions and context.
              joined = "\n---\n".join(not_empty[:15])
              parts.append(f"Comments:\n{joined}")
    return normalise_whitespace("\n\n".join(parts))



    

    