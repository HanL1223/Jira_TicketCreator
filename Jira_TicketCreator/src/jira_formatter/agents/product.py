from jira_formatter.agents.contracts import AgentInput, AgentOutput
from jira_formatter.llm.gemini import GeminiClient


class ProductAgent:
    """
    Product agent: focuses on business goal, user value, and clear overview.
    """

    def __init__(self, llm: GeminiClient) -> None:
        self._llm = llm

    # This is a function to write a product-focused overview and user-centric steps outline.
    def run(self, inp: AgentInput) -> AgentOutput:
        prompt = f"""
You are a senior Product Owner. Write a crisp Jira card "Overview" section only.

User request:
{inp.user_request}

Relevant historical context:
{inp.rag_context}

Rules:
- 2–4 sentences
- Focus on goal and business value
- No implementation detail
Return ONLY the Overview text (no headings).
""".strip()

        text = self._llm.generate(prompt)
        return AgentOutput(agent_name="product", content=text)
