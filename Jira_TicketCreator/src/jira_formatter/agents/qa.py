from __future__ import annotations

from jira_formatter.agents.contracts import AgentInput, AgentOutput
from jira_formatter.llm.gemini import GeminiClient


class QaAgent:
    """
    QA agent: writes acceptance criteria as checkboxes (measurable).
    """

    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    # This is a function to generate measurable Jira acceptance criteria checkboxes.
    def run(self, inp: AgentInput) -> AgentOutput:
        prompt = f"""
You are a QA Lead. Write Jira Acceptance Criteria using Jira checkbox markdown.

User request:
{inp.user_request}

Relevant historical context:
{inp.rag_context}

Rules:
- 1–10 items
- Each must be specific, measurable, testable
- Use Jira checkbox format: - [] ...
Return ONLY the checkbox list items (no heading).
""".strip()

        text = self._llm.generate(prompt)
        return AgentOutput(agent_name="qa", content=text)
