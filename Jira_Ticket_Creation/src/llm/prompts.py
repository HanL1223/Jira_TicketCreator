from dataclasses import dataclass
from pathlib import Path
import yaml

@dataclass(frozen=True)
class PromptBundle:
    """
    Holds prompt templates used across the project

    Keeping prompt in code for POC 
    Phase 2: this should be migrated to config/prompt.yaml
    """
    system:str
    draft:str
    product_refine:str
    technical_refine:str
    qa_refine:str


def load_prompts_from_yaml(path: Path) -> PromptBundle:
    with path.open() as f:
        data = yaml.safe_load(f)
    
    return PromptBundle(
        system=data["system"],
        draft=data["draft"],
        product_refine=data["product_refine"],
        technical_refine=data["technical_refine"],
        qa_refine=data["qa_refine"]
    )

#Retrun default prompt templates for project
def default_prompts() ->PromptBundle:
    return PromptBundle(
        system=(
            "You are an expert Product Owner and Delivery Lead.\n"
            "You create Jira cards that are clear, actionable, and testable.\n"
            "Output MUST be valid Jira markdown.\n"
            "Follow the exact card structure:\n"
            "## Overview\n"
            "...\n\n"
            "## Steps\n"
            "1. ...\n\n"
            "## Acceptance Criteria\n"
            "- [ ] ...\n"
        ),
        draft=(
            "Create a Jira card from the user request.\n\n"
            "User request:\n{user_request}\n\n"
            "Relevant historical tickets (for style and alignment):\n{rag_context}\n\n"
            "Rules:\n"
            "- Output EXACTLY three sections: Overview, Steps, Acceptance Criteria.\n"
            "- Steps must be numbered.\n"
            "- Acceptance Criteria must be checkboxes using '- []'.\n"
            "- Keep it concise but complete.\n"
            "- Do not mention that you used retrieval.\n"
        ),
        product_refine=(
            "You are the PRODUCT agent.\n"
            "Improve the Jira card to maximize business value clarity.\n"
            "Ensure Overview explains 'what' + 'why' in 2–4 sentences.\n"
            "Ensure Steps reflect value delivery and stakeholder outcomes.\n\n"
            "User request:\n{user_request}\n\n"
            "Draft Jira card:\n{draft}\n\n"
            "Return the improved Jira card in the SAME required structure."
        ),
        technical_refine=(
            "You are the TECHNICAL agent.\n"
            "Improve the Jira card to make implementation steps concrete.\n"
            "Add specific engineering steps (interfaces, data, validation, logging, tests) where relevant.\n"
            "Do NOT invent systems; keep it generic if uncertain.\n\n"
            "User request:\n{user_request}\n\n"
            "Draft Jira card:\n{draft}\n\n"
            "Return the improved Jira card in the SAME required structure."
        ),
        qa_refine=(
            "You are the QA agent.\n"
            "Improve Acceptance Criteria so every item is specific, measurable, and testable.\n"
            "Use '- [ ]' checkbox format.\n"
            "Avoid vague words like 'appropriate', 'correct', 'nice'.\n\n"
            "User request:\n{user_request}\n\n"
            "Draft Jira card:\n{draft}\n\n"
            "Return the improved Jira card in the SAME required structure."
        ),
    )