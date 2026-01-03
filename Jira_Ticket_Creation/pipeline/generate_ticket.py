
from src.agents.orchestrator import AgentConfig, MultiAgentOrchestrator
from src.llm.gemini import GeminiClient
from src.llm.prompts import PromptBundle, default_prompts
from src.rag.retriever import JiraIssueRetriever

from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationConfig:
    """
    Controls generation behavior for the end-to-end pipeline.
    """
    use_rag: bool = True
    use_agents: bool = True

@dataclass(frozen = True)
class TicketGenerationResult:
    """
    Output of the ticket generation pipeline
    """

    jira_markdown:str
    rag_context_used:bool
    retrieved_chunks:int

class TicketGenerator:
    """
    End-to-end generator: retrieval -> draft -> multi-agent refinement -> final Jira markdown.
    """

    def __init__(
            self,
            *,
            retriever:JiraIssueRetriever,
            llm:GeminiClient,
            prompts:PromptBundle | None = None,
            config:GenerationConfig | None = None
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._prompts = prompts or default_prompts()
        self._config = config or GenerationConfig()
        self._orchestrator = MultiAgentOrchestrator(
            llm=self._llm,
            prompts=self._prompts,
            config=AgentConfig(enabled=self._config.use_agents, passes=1),
        )

    """
    Function to generate a Jira card from a user request using RAG + multi-agent refinement.
    """

    def generate(self, user_request: str) -> TicketGenerationResult:
        user_request = (user_request or "").strip()
        if not user_request:
            return TicketGenerationResult(
                jira_markdown="",
                rag_context_used=False,
                retrieved_chunks=0,
            )

        rag_context = "No relevant historical tickets found in the index."
        retrieved_count = 0

        if self._config.use_rag:
            chunks = self._retriever.retrieve(user_request)
            retrieved_count = len(chunks)
            rag_context = self._retriever.format_context(chunks)

        draft_prompt = self._prompts.draft.format(
            user_request=user_request,
            rag_context=rag_context,
        )

        draft = self._llm.generate(draft_prompt, system_prompt=self._prompts.system).strip()

        final_text = draft
        if self._config.use_agents:
            final_text = self._orchestrator.refine(user_request=user_request, draft=draft)

        return TicketGenerationResult(
            jira_markdown=final_text.strip(),
            rag_context_used=self._config.use_rag,
            retrieved_chunks=retrieved_count,
        )