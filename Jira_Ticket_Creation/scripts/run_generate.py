#!/usr/bin/env python3
"""
================================================================================
B6 · run_generate.py – CLI for End-to-End Jira Ticket Generation
================================================================================

WHAT THIS SCRIPT DOES
---------------------
This is the "command centre" that wires together everything you've built in the
RAG tutorial series:

    A5 (embeddings.py)      → Converts text to vectors
    A6 (vectorstore.py)     → Stores/queries vectors in ChromaDB  
    A8 (bootstrap_rag.py)   → Indexed your Jira issues (already done!)
    B1 (retriever.py)       → Retrieves relevant chunks
    B3 (gemini.py)          → LLM generation client
    B4 (orchestrator.py)    → Multi-agent refinement
    B5 (generate_ticket.py) → Orchestrates the full pipeline

This CLI script:
1. Accepts a natural-language request (e.g., "create a ticket for dim_customer")
2. Retrieves relevant historical Jira tickets from your indexed vector store
3. Drafts a new Jira ticket using Gemini + RAG context
4. Optionally refines the draft via multi-agent critique
5. Outputs professional Jira markdown ready to paste

ARCHITECTURE FLOW
-----------------
    ┌─────────────────┐
    │  User Request   │  "create a ticket for data modelling for dim_customer"
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Embedder      │  Converts query → vector (A5)
    │ (GeminiEmbed)   │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Vector Store   │  Similarity search in ChromaDB (A6)
    │   (ChromaDB)    │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │   Retriever     │  Returns top-k relevant chunks (B1)
    │(JiraIssueRetr)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  LLM Client     │  Gemini generates draft ticket (B3)
    │ (GeminiClient)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  Multi-Agent    │  Optional refinement pass (B4)
    │ (Orchestrator)  │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ Final Jira Card │  Professional markdown output
    └─────────────────┘

USAGE EXAMPLES
--------------
# Basic usage (full pipeline: RAG + agents)
python scripts/run_generate.py --query "create a ticket for data modelling for dim_customer"

# Fast mode (skip multi-agent refinement)
python scripts/run_generate.py --query "create a ticket for dim_customer" --no-agents

# Generic LLM only (no RAG context)
python scripts/run_generate.py --query "create a ticket for dim_customer" --no-rag

# Minimal mode (no RAG, no agents - just raw Gemini)
python scripts/run_generate.py --query "create a ticket for dim_customer" --no-rag --no-agents

# Custom retrieval settings
python scripts/run_generate.py --query "..." --top-k 10 --collection jira_issues_v2

# Debug mode
python scripts/run_generate.py --query "..." --log-level DEBUG

PREREQUISITES
-------------
1. GOOGLE_API_KEY environment variable set
2. ChromaDB indexed (run bootstrap_rag.py first)
3. Dependencies installed: google-genai, chromadb, tenacity

================================================================================
"""

# =============================================================================
# SECTION 1: IMPORTS
# =============================================================================
# We use __future__ annotations for Python 3.9+ type hint compatibility.
# This allows us to use `list[str]` instead of `List[str]` everywhere.

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# PATH SETUP (Critical!)
# -----------------------------------------------------------------------------
# Scripts live in /scripts but our modules live in /src.
# We need to add the project root to sys.path so Python can find our imports.
#
# Path(__file__)           → /path/to/Jira_Ticket_Creation/scripts/run_generate.py
# .resolve()               → Absolute path
# .parents[1]              → Go up 2 levels: scripts → Jira_Ticket_Creation
#
# This is the same pattern used in bootstrap_rag.py (A8).

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# Now we can import from src.* and pipeline.*
# Notice: we import from src.rag, src.llm, and pipeline - the modules you built!

from src.rag.embeddings import GeminiEmbeddingClient, GeminiEmbeddingConfig
from src.rag.vectorstore import ChromaConfig, ChromaVectorStore
from src.rag.retriever import JiraIssueRetriever, RetrievalConfig
from src.llm.gemini import GeminiClient, GeminiConfig
from pipeline.generate_ticket import TicketGenerator, GenerationConfig

# =============================================================================
# SECTION 2: LOGGING SETUP
# =============================================================================
# We create a module-level logger. The actual log level is set at runtime
# based on the --log-level argument.

LOGGER = logging.getLogger(__name__)


def setup_logging(level: str) -> None:
    """
    Configure logging with a clean, readable format.
    
    Args:
        level: Log level string (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    
    WHY THIS EXISTS:
    ----------------
    Good logging is essential for debugging RAG pipelines. When something
    goes wrong (bad embeddings, empty retrieval, API errors), you need
    visibility into what's happening at each step.
    
    The format includes:
    - Timestamp: When did this happen?
    - Level: How serious is it?
    - Name: Which module logged this?
    - Message: What happened?
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )


# =============================================================================
# SECTION 3: ARGUMENT PARSING
# =============================================================================
# We define all CLI arguments here. This follows the same pattern as
# bootstrap_rag.py but with generation-specific options.

def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the generation pipeline.
    
    ARGUMENT DESIGN PHILOSOPHY:
    ---------------------------
    - Required args: Only --query (the user's request)
    - Optional args: Everything else has sensible defaults
    - Flags: --no-rag and --no-agents disable features
    
    This design lets users run with minimal friction:
        python run_generate.py --query "my request"
    
    While power users can customize everything.
    """
    parser = argparse.ArgumentParser(
        description="Generate Jira tickets using RAG + multi-agent refinement.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES:
  %(prog)s --query "create a ticket for data modelling for dim_customer"
  %(prog)s --query "..." --no-agents    # Skip multi-agent refinement
  %(prog)s --query "..." --no-rag       # Skip RAG retrieval
  %(prog)s --query "..." --top-k 10     # Retrieve more context
        """,
    )
    
    # -------------------------------------------------------------------------
    # REQUIRED ARGUMENT
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--query", "-q",
        required=True,
        help="Natural language request describing the ticket to generate. "
             "Example: 'create a ticket for data modelling for dim_customer'",
    )
    
    # -------------------------------------------------------------------------
    # VECTOR STORE OPTIONS
    # -------------------------------------------------------------------------
    # These control where/how we retrieve context from ChromaDB.
    
    parser.add_argument(
        "--persist-dir",
        default=str(PROJECT_ROOT / "data" / "chromadb"),
        help="Path to ChromaDB persistent directory. "
             "Default: data/chromadb (same as bootstrap_rag.py)",
    )
    
    parser.add_argument(
        "--collection",
        default="jira_issues",
        help="ChromaDB collection name to query. "
             "Default: jira_issues (same as bootstrap_rag.py)",
    )
    
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=6,
        help="Number of similar chunks to retrieve. "
             "More = richer context but slower/more tokens. "
             "Default: 6",
    )
    
    # -------------------------------------------------------------------------
    # MODEL OPTIONS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--model",
        default="gemini-2.0-flash-exp",
        help="Gemini model to use for generation. "
             "Default: gemini-2.0-flash-exp (fast and capable)",
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.2,
        help="Generation temperature (0.0-1.0). "
             "Lower = more focused/deterministic. "
             "Default: 0.2",
    )
    
    # -------------------------------------------------------------------------
    # FEATURE FLAGS
    # -------------------------------------------------------------------------
    # These let users disable parts of the pipeline for speed or debugging.
    
    parser.add_argument(
        "--no-rag",
        action="store_true",
        help="Disable RAG retrieval. The LLM will generate without "
             "historical context (useful for comparison/debugging).",
    )
    
    parser.add_argument(
        "--no-agents",
        action="store_true",
        help="Disable multi-agent refinement. The draft will be returned "
             "as-is without critique/revision passes (faster).",
    )
    
    # -------------------------------------------------------------------------
    # OUTPUT OPTIONS
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity. DEBUG shows all retrieval details. "
             "Default: INFO",
    )
    
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Print the retrieved RAG context before the generated ticket.",
    )
    
    return parser.parse_args()


# =============================================================================
# SECTION 4: COMPONENT FACTORY FUNCTIONS
# =============================================================================
# These functions create the individual components of our pipeline.
# Breaking them into separate functions makes the code testable and readable.

def create_embedder(api_key: str) -> GeminiEmbeddingClient:
    """
    Create the embedding client (A5).
    
    WHY WE NEED THIS:
    -----------------
    The retriever needs to convert the user's query into a vector so it can
    find similar chunks in ChromaDB. We use the same embedding model that
    was used during indexing (text-embedding-004) for consistency.
    
    Args:
        api_key: Google API key for Gemini
        
    Returns:
        Configured GeminiEmbeddingClient
    """
    LOGGER.debug("Creating embedding client...")
    return GeminiEmbeddingClient(
        GeminiEmbeddingConfig(api_key=api_key)
    )


def create_vector_store(persist_dir: str, collection_name: str) -> ChromaVectorStore:
    """
    Create the vector store connection (A6).
    
    WHY WE NEED THIS:
    -----------------
    ChromaDB stores our indexed Jira issues. We connect to the same database
    that bootstrap_rag.py created. The collection must exist and contain
    embedded chunks from the indexing step.
    
    Args:
        persist_dir: Path to ChromaDB directory
        collection_name: Name of the collection to query
        
    Returns:
        Configured ChromaVectorStore
    """
    LOGGER.debug("Connecting to ChromaDB at %s...", persist_dir)
    store = ChromaVectorStore(
        ChromaConfig(
            persist_dir=Path(persist_dir),
            collection_name=collection_name,
        )
    )
    
    # Sanity check: verify the collection has data
    count = store.count()
    if count == 0:
        LOGGER.warning(
            "Collection '%s' is empty! Did you run bootstrap_rag.py first?",
            collection_name,
        )
    else:
        LOGGER.info("Connected to collection '%s' with %d chunks", collection_name, count)
    
    return store


def create_retriever(
    embedder: GeminiEmbeddingClient,
    store: ChromaVectorStore,
    top_k: int,
) -> JiraIssueRetriever:
    """
    Create the retriever (B1).
    
    WHY WE NEED THIS:
    -----------------
    The retriever is the bridge between user queries and stored knowledge.
    It:
    1. Embeds the query using the same model as indexing
    2. Runs similarity search in ChromaDB
    3. Returns the top-k most relevant chunks
    
    Args:
        embedder: Embedding client for query vectorization
        store: Vector store to search
        top_k: Number of results to retrieve
        
    Returns:
        Configured JiraIssueRetriever
    """
    LOGGER.debug("Creating retriever with top_k=%d...", top_k)
    return JiraIssueRetriever(
        embedder=embedder,
        store=store,
        config=RetrievalConfig(top_k=top_k),
    )


def create_llm_client(api_key: str, model: str, temperature: float) -> GeminiClient:
    """
    Create the LLM generation client (B3).
    
    WHY WE NEED THIS:
    -----------------
    The LLM is the "brain" that takes the user request + retrieved context
    and generates a coherent Jira ticket. We use Gemini for its:
    - Strong instruction following
    - Good structured output (markdown)
    - Fast inference
    
    Args:
        api_key: Google API key
        model: Gemini model name
        temperature: Generation temperature
        
    Returns:
        Configured GeminiClient
    """
    LOGGER.debug("Creating LLM client with model=%s, temp=%.2f...", model, temperature)
    return GeminiClient(
        GeminiConfig(
            api_key=api_key,
            model=model,
            temperature=temperature,
        )
    )


def create_generator(
    retriever: JiraIssueRetriever,
    llm: GeminiClient,
    use_rag: bool,
    use_agents: bool,
) -> TicketGenerator:
    """
    Create the ticket generator (B5).
    
    WHY WE NEED THIS:
    -----------------
    The TicketGenerator orchestrates the full pipeline:
    1. Retrieval (if use_rag=True)
    2. Draft generation
    3. Multi-agent refinement (if use_agents=True)
    
    It's the high-level API that hides all the complexity.
    
    Args:
        retriever: For fetching relevant context
        llm: For text generation
        use_rag: Whether to use retrieval
        use_agents: Whether to use multi-agent refinement
        
    Returns:
        Configured TicketGenerator
    """
    LOGGER.debug(
        "Creating generator with use_rag=%s, use_agents=%s...",
        use_rag, use_agents,
    )
    return TicketGenerator(
        retriever=retriever,
        llm=llm,
        config=GenerationConfig(
            use_rag=use_rag,
            use_agents=use_agents,
        ),
    )


# =============================================================================
# SECTION 5: OUTPUT FORMATTING
# =============================================================================
# Clean, readable output is important for CLI tools.

def print_header(text: str) -> None:
    """Print a section header with box-drawing characters."""
    width = max(60, len(text) + 4)
    print("\n" + "═" * width)
    print(f"  {text}")
    print("═" * width)


def print_result(result, query: str, show_context: bool) -> None:
    """
    Print the generation result in a readable format.
    
    Args:
        result: TicketGenerationResult from generator
        query: Original user query
        show_context: Whether to print retrieved context
    """
    # Print mode summary
    print_header("Generation Complete")
    
    mode_parts = []
    if result.rag_context_used:
        mode_parts.append(f"RAG ({result.retrieved_chunks} chunks)")
    else:
        mode_parts.append("No RAG")
    
    # Note: We can't easily detect if agents were used from the result,
    # but we show chunk count which indicates RAG was active
    mode_str = " + ".join(mode_parts) if mode_parts else "Direct LLM"
    print(f"Mode: {mode_str}")
    print(f"Query: {query[:80]}{'...' if len(query) > 80 else ''}")
    
    # Print the generated ticket
    print_header("Generated Jira Ticket")
    print(result.jira_markdown)
    print("\n" + "─" * 60)


# =============================================================================
# SECTION 6: MAIN FUNCTION
# =============================================================================
# This is where everything comes together.

def main() -> None:
    """
    Main entry point for the generation CLI.
    
    FLOW:
    -----
    1. Parse arguments
    2. Validate environment (API key)
    3. Create components (embedder → store → retriever → llm → generator)
    4. Run generation
    5. Print results
    
    ERROR HANDLING:
    ---------------
    We catch and report common errors:
    - Missing API key
    - Empty collection
    - API failures
    """
    # Step 1: Parse arguments
    args = parse_args()
    setup_logging(args.log_level)
    
    LOGGER.info("Starting Jira ticket generation...")
    LOGGER.debug("Arguments: %s", args)
    
    # Step 2: Validate API key
    api_key = os.environ.get("GOOGLE_API_KEY", "").strip()
    if not api_key:
        print("\n❌ ERROR: GOOGLE_API_KEY environment variable is required.")
        print("   Set it with: export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)
    
    LOGGER.debug("API key found (length: %d)", len(api_key))
    
    # Step 3: Create components
    # -------------------------
    # We build the pipeline from bottom to top:
    #   embedder → store → retriever → llm → generator
    #
    # Each component depends on the ones before it.
    
    try:
        # 3a. Create embedder (A5)
        embedder = create_embedder(api_key)
        
        # 3b. Create vector store connection (A6)
        store = create_vector_store(args.persist_dir, args.collection)
        
        # 3c. Create retriever (B1)
        retriever = create_retriever(embedder, store, args.top_k)
        
        # 3d. Create LLM client (B3)
        llm = create_llm_client(api_key, args.model, args.temperature)
        
        # 3e. Create generator (B5) - this also creates the orchestrator (B4)
        generator = create_generator(
            retriever=retriever,
            llm=llm,
            use_rag=not args.no_rag,
            use_agents=not args.no_agents,
        )
        
    except Exception as e:
        LOGGER.error("Failed to initialize pipeline: %s", e)
        print(f"\n❌ Initialization error: {e}")
        sys.exit(1)
    
    # Step 4: Run generation
    # ----------------------
    # This is where the magic happens! The generator:
    # 1. Retrieves relevant chunks (if RAG enabled)
    # 2. Drafts a ticket with Gemini
    # 3. Refines via multi-agent critique (if agents enabled)
    
    try:
        LOGGER.info("Generating ticket for query: %s", args.query[:50] + "...")
        
        # Show a spinner or progress message for long operations
        if not args.no_rag:
            print("\n🔍 Retrieving relevant context from vector store...")
        if not args.no_agents:
            print("🤖 Running multi-agent refinement (this may take a moment)...")
        else:
            print("⚡ Generating ticket (fast mode, no agent refinement)...")
        
        result = generator.generate(args.query)
        
    except Exception as e:
        LOGGER.error("Generation failed: %s", e, exc_info=True)
        print(f"\n❌ Generation error: {e}")
        sys.exit(1)
    
    # Step 5: Print results
    # ---------------------
    if not result.jira_markdown.strip():
        print("\n⚠️  Warning: Generated empty ticket. Try a different query.")
    else:
        print_result(result, args.query, args.show_context)
    
    LOGGER.info("Done!")


# =============================================================================
# SECTION 7: SCRIPT ENTRY POINT
# =============================================================================
# This guard ensures the script only runs when executed directly,
# not when imported as a module.

if __name__ == "__main__":
    main()


# =============================================================================
# TUTORIAL REVIEW QUESTIONS
# =============================================================================
# Test your understanding of this script:
#
# 1. PATH SETUP: Why do we need `sys.path.insert(0, str(PROJECT_ROOT))`?
#    What would happen if we removed this line?
#
# 2. COMPONENT WIRING: In what order do we create components? Why does
#    the retriever need both an embedder AND a store?
#
# 3. FEATURE FLAGS: What's the difference between --no-rag and --no-agents?
#    When would you use each one?
#
# 4. ERROR HANDLING: What happens if the ChromaDB collection is empty?
#    How does the script handle missing API keys?
#
# 5. GENERATION FLOW: Trace the path of a user query through all the
#    components. What transformations happen at each step?
#
# =============================================================================
# COMPLETION CHECKLIST
# =============================================================================
# After this tutorial, you should be able to:
#
# [ ] Run the full RAG generation pipeline from the command line
# [ ] Understand how all the pieces (A5-A8, B1-B5) connect together
# [ ] Customize retrieval (top-k, collection) and generation (model, temp)
# [ ] Debug pipeline issues using --log-level DEBUG
# [ ] Compare RAG vs non-RAG output using --no-rag flag
# [ ] Speed up iteration using --no-agents flag
#
# =============================================================================
# NEXT STEPS
# =============================================================================
# Now that you have a working end-to-end pipeline:
#
# 1. EXPERIMENT: Try different queries and see how RAG context affects output
# 2. TUNE: Adjust top-k and temperature to find optimal settings
# 3. EVALUATE: Compare --no-rag vs full pipeline for quality
# 4. EXTEND: Add new features like --output-file or --format json
#
# Congratulations on completing the RAG tutorial series! 🎉
# =============================================================================