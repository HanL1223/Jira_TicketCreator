from pathlib import Path
from Create_project_scructure import create_structure


PROJECT_STRUCTURE = {
    "src": {
        "jira_formatter": {
            "__init__.py": None,
            "core": [
                "__init__.py",
                "models.py",          # Pydantic dataclasses: IssueRecord, JiraCard, RetrievalDoc, AgentResult
                "exceptions.py",      # Domain errors
                "config.py",          # Typed settings (env + defaults)
                "logging.py",         # Logging config (no prints)
                "constants.py",       # Field names, allowed issue types, etc.
                "utils.py",           # Small utilities only (hashing, jsonl io)
            ],

            "llm": [
                "__init__.py",
                "gemini.py",          # Gemini client wrapper (retry, rate-limit, cost tracking hooks)
                "prompts.py",         # Prompt templates + builders
                "schemas.py",         # Output schemas + structured output parsing/validation
                "safety.py",          # Guardrails: formatting constraints, truncation rules
            ],

            "rag": [
                "__init__.py",
                "embeddings.py",      # Gemini embedding adapter (batch + caching)
                "vectorstore.py",     # Chroma wrapper (add/query/persist, metadata strategy)
                "retriever.py",       # Retrieve + filter + optional rerank hook
                "indexer.py",         # Build index from processed dataset
                "chunking.py",        # Ticket -> chunks (title/desc/ac/comments) and doc assembly
            ],

            "agents": [
                "__init__.py",
                "orchestrator.py",    # Majority voting + conflict resolution + final formatting
                "technical.py",
                "product.py",
                "qa.py",
                "contracts.py",       # Agent I/O contracts + scoring rubric (critical for tests)
            ],

            "ingestion": [
                "__init__.py",
                "base.py",            # IssueSource interface
                "factory.py",         # Factory method: CSVSource now, JiraAPISource later
                "csv_source.py",      # Read Jira export CSV -> IssueRecord stream
                "jira_api_source.py", # Stub or partial impl (future)
            ],

            "preprocessing": [
                "__init__.py",
                "cleaners.py",        # Wiki->md, de-dup fields, whitespace, PII redaction option
                "normalizers.py",     # Field normalization: summary/desc/ac/comments unify
                "filters.py",         # Drop low-signal tickets, missing fields, etc.
                "dataset_builder.py", # Build processed jsonl + stats from ingestion
            ],

            "pipelines": [
                "__init__.py",
                "build_corpus.py",    # ingestion -> preprocessing -> processed jsonl
                "build_index.py",     # processed jsonl -> chromadb
                "generate_ticket.py", # user input -> retrieve -> agents -> jira markdown
                "evaluate.py",        # offline eval harness + regression checks
            ],
        }
    },

    "scripts": [
        "learn_step_by_step.py",     # Tutorial runner / guided examples (kept)
        "prepare_dataset.py",        # NEW: raw csv -> processed jsonl (calls preprocessing)
        "bootstrap_rag.py",          # Seeds Chroma with processed corpus
        "run_generate.py",           # NEW: CLI-like generation runner
        "run_eval.py",               # NEW: evaluation runner
        "test_system.py",            # End-to-end smoke test
    ],

    "data": {
        "raw": [],                   # NEW: Jira exports / api dumps
        "processed": [],             # NEW: normalized jsonl + stats
        "chromadb": []
    },

    "tests": {
        "unit": [
            "test_csv_source.py",
            "test_cleaners.py",
            "test_chunking.py",
            "test_vectorstore.py",
            "test_agents_contracts.py",
        ],
        "integration": [
            "test_build_corpus.py",
            "test_index_retrieve.py",
            "test_generate_ticket.py",
        ],
        "fixtures": []
    },

    "configs": [
        "app.yaml",                  # paths, model ids, chroma settings, retrieval params
        "prompts.yaml",              # prompt templates versioned outside code
        "eval.yaml",                 # eval set + thresholds
        ".env.example",
    ],

    "docs": [
        "architecture.md",
        "data_prep.md",
        "rag_design.md",
        "evaluation.md",
    ]
}



if __name__ == "__main__":
    project_name = "Jira_TicketCreator"
    root = Path(project_name)
    root.mkdir(exist_ok=True)

    create_structure(root, PROJECT_STRUCTURE)

    print(f"Project '{project_name}' created successfully.")