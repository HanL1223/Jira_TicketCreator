from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class AgentInput:
    """
    Shared input passed to all agents.
    """
    user_request: str
    rag_context: str

@dataclass(frozen=True)
class AgentOutput:
    """
    Docstring for AgentOutput
    """
    agent_name:str
    content:str


class Agent(Protocol):
    """
    Agent interface (Product/Technical/QA).
    """

    def run(self, inp: AgentInput) -> AgentOutput:
        ...