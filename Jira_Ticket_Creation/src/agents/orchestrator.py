from dataclasses import dataclass
from typing import Iterable

from src.llm.gemini import GeminiClient
from src.llm.prompts import PromptBundle, default_prompts

@dataclass(frozen = True)
class AgentConfig:
    """
    Configuration for the multi-agent refinement loop
    """
    enabled:bool = True
    passes:int = 1

class MultiAgentOrchestrator:
    """
    Coordinats multi agent refinement
    """

    def __init__(
            self,
            llm:GeminiClient,
            prompts:PromptBundle | None = None,
            config: AgentConfig | None = None
    ) -> None:
        self._llm = llm
        self._prompts = prompts
        self._config = config
    
    # Function to a draft Jira card using product/tech/qa passes.
    def refine(self,*,user_request:str,draft:str) -> str:
        if not self._config.enabled:
            return draft
        user_request = (user_request or "").strip()
        draft = (draft or "").strip()
        if not user_request or not draft:
            return draft
        refined = draft

        for _ in range(max(1,self._config.passes)):
            refined = self._run_agent(
                agent_prompt=self._prompts.product_refine,
                user_request=user_request,
                draft=refined,
            )
            refined = self._run_agent(
                agent_prompt=self._prompts.technical_refine,
                user_request=user_request,
                draft=refined,
            )
            refined = self._run_agent(
                agent_prompt=self._prompts.qa_refine,
                user_request=user_request,
                draft=refined,
            )
        return self._ensure_structure(refined)
    def _run_agent(self, *, agent_prompt: str, user_request: str, draft: str) -> str:
        prompt = agent_prompt.format(user_request=user_request, draft=draft)
        out = self._llm.generate(prompt, system_prompt=self._prompts.system)
        out = (out or "").strip()
        return out if out else draft
    # This is a function to ensure the Jira card keeps the required headings and checkbox format.
    def _ensure_structure(self, text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t

        # Minimal "guardrails" only (avoid overly rewriting model output).
        # Ensure checkbox marker is Jira standard: "- [ ]"
        t = t.replace("- []", "- [ ]").replace("-[ ]", "- [ ]")

        # Ensure headings exist (light-touch)
        required = ["## Overview", "## Steps", "## Acceptance Criteria"]
        if all(h in t for h in required):
            return t

        # If headings got damaged, do a minimal repair by re-asking the LLM.
        repair_prompt = (
            "Fix formatting to EXACTLY match the required structure.\n\n"
            "Required structure:\n"
            "## Overview\n...\n\n"
            "## Steps\n1. ...\n\n"
            "## Acceptance Criteria\n- [ ] ...\n\n"
            "Text to fix:\n"
            f"{t}"
        )
        repaired = self._llm.generate(repair_prompt, system_prompt=self._prompts.system)
        repaired = (repaired or "").strip()
        return repaired if repaired else t
