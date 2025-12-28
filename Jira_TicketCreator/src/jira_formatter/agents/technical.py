from __future__ import annotations

from jira_formatter.agents.contracts import AgentInput, AgentOutput
from jira_formatter.llm.gemini import GeminiClient


class TechnicalAgent:
    """
    Technical agent: proposes concrete, actionable implementation steps.
    """

    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    # This is a function to draft concrete engineering steps (actionable tasks).
    def run(self, inp: AgentInput) -> AgentOutput:
        prompt = f"""
You are a Staff Engineer. Draft a numbered Jira "Steps" section.

User request:
{inp.user_request}

Relevant historical context:
{inp.rag_context}

Rules:
- 2-12 steps
- Each step should be concrete and testable
- Prefer verbs like "Implement", "Add", "Configure", "Validate"
Return ONLY the numbered list items (no heading).
""".strip()

        text = self._llm.generate(prompt)
        return AgentOutput(agent_name="technical", content=text)
