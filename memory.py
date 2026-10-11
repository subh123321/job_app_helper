"""
memory.py — ChromaDB memory layer for the job application agent
===============================================================
Drop this file in the same folder as job_application_agent.py.
Import it at the top of your agent file:

    from memory import get_cached_research, save_research_to_memory, memory_stats

What it does:
- Stores company research as vector embeddings in a local ChromaDB
- Before each web search, checks if we already researched this company
- Cache hit = instant load, skips DDGS entirely
- After 50+ companies, your DB becomes a valuable asset of company intel

Install:
    pip install chromadb sentence-transformers
"""

import re
from datetime import date
from typing import Optional

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

CHROMA_DIR   = "chroma_memory"   # local folder where ChromaDB stores data
COLLECTION   = "company_research"
EMBED_MODEL  = "all-MiniLM-L6-v2"  # tiny, fast, free — 80MB download once
CACHE_DAYS   = 30                   # re-research a company after 30 days

# ---------------------------------------------------------------------------
# SETUP — runs once on import
# ---------------------------------------------------------------------------

_embedder = None

_client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(anonymized_telemetry=False),
)

_collection = _client.get_or_create_collection(
    name=COLLECTION,
    metadata={"hnsw:space": "cosine"},
)

print(f"[MEMORY] ChromaDB ready — {_collection.count()} companies in memory")


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _make_id(startup_name: str) -> str:
    """Sanitize startup name into a stable ChromaDB document ID."""
    return re.sub(r"[^\w]", "_", startup_name.lower()).strip("_")


def _embed(text: str) -> list[float]:
    """Embed a string using the sentence transformer."""
    global _embedder

    if _embedder is None:
        print("[MEMORY] Loading embedding model...")
        _embedder = SentenceTransformer(EMBED_MODEL)

    return _embedder.encode(text, normalize_embeddings=True).tolist()


def _is_stale(researched_at: str) -> bool:
    """Return True if the cached research is older than CACHE_DAYS."""
    try:
        cached_date = date.fromisoformat(researched_at)
        delta = (date.today() - cached_date).days
        return delta > CACHE_DAYS
    except Exception:
        return True


# ---------------------------------------------------------------------------
# PUBLIC API
# ---------------------------------------------------------------------------

def get_cached_research(startup_name: str) -> Optional[dict]:
    """
    Check ChromaDB for existing research on this company.

    Returns a dict with keys:
        company_research, tech_stack, researched_at
    or None if not found / stale.

    Usage in your agent:
        cached = get_cached_research("Hugging Face")
        if cached:
            return cached   # skip web search entirely
    """
    doc_id = _make_id(startup_name)

    try:
        result = _collection.get(
            ids=[doc_id],
            include=["documents", "metadatas"],
        )
    except Exception as e:
        print(f"[MEMORY] ChromaDB get failed: {e}")
        return None

    if not result["ids"]:
        print(f"[MEMORY] Cache MISS — {startup_name}")
        return None

    metadata = result["metadatas"][0]
    researched_at = metadata.get("researched_at", "2000-01-01")

    if _is_stale(researched_at):
        print(f"[MEMORY] Cache STALE ({researched_at}) — will re-research {startup_name}")
        return None

    print(f"[MEMORY] Cache HIT — {startup_name} (researched {researched_at})")
    return {
        "company_research": metadata.get("company_research", ""),
        "tech_stack":       metadata.get("tech_stack", ""),
        "researched_at":    researched_at,
    }


def save_research_to_memory(
    startup_name: str,
    company_research: str,
    tech_stack: str,
) -> None:
    """
    Save company research to ChromaDB as a vector embedding.

    The document text (used for semantic search) is the full research summary.
    Metadata stores the raw fields for retrieval.

    Usage in your agent:
        save_research_to_memory(
            startup_name="Hugging Face",
            company_research=state["company_research"],
            tech_stack=state["tech_stack"],
        )
    """
    doc_id = _make_id(startup_name)
    research_str = company_research if isinstance(company_research, str) else str(company_research)
    tech_str     = tech_stack if isinstance(tech_stack, str) else str(tech_stack)
    doc_text     = f"{startup_name}\n{research_str}\n{tech_str}"

    metadata = {
        "startup_name":     startup_name,
        "company_research": research_str[:2000],  # ChromaDB metadata limit
        "tech_stack":       tech_str[:500],
        "researched_at":    date.today().isoformat(),
    }

    try:
        embedding = _embed(doc_text)
        _collection.upsert(
            ids=[doc_id],
            embeddings=[embedding],
            documents=[doc_text],
            metadatas=[metadata],
        )
        print(f"[MEMORY] Saved {startup_name} to ChromaDB")
    except Exception as e:
        print(f"[MEMORY] Save failed for {startup_name}: {e}")


def find_similar_companies(
    query: str,
    n_results: int = 5,
    exclude: list[str] = None,
) -> list[dict]:
    """
    Semantic search — find companies in memory similar to a query.

    Useful for:
    - "find companies similar to Hugging Face"
    - "find companies that use RAG pipelines"
    - auto-suggesting new targets based on ones you liked

    Returns list of dicts: [{startup_name, company_research, score}, ...]
    """
    if _collection.count() == 0:
        print("[MEMORY] No companies in memory yet.")
        return []

    exclude_ids = [_make_id(n) for n in (exclude or [])]

    try:
        embedding = _embed(query)
        results = _collection.query(
            query_embeddings=[embedding],
            n_results=min(n_results + len(exclude_ids), _collection.count()),
            include=["metadatas", "distances"],
        )
    except Exception as e:
        print(f"[MEMORY] Similarity search failed: {e}")
        return []

    output = []
    for meta, dist in zip(results["metadatas"][0], results["distances"][0]):
        doc_id = _make_id(meta.get("startup_name", ""))
        if doc_id in exclude_ids:
            continue
        output.append({
            "startup_name":     meta.get("startup_name", ""),
            "company_research": meta.get("company_research", ""),
            "tech_stack":       meta.get("tech_stack", ""),
            "similarity_score": round(1 - dist, 3),   # cosine: 1=identical
        })
        if len(output) >= n_results:
            break

    return output


def memory_stats() -> dict:
    """
    Print a summary of what's in ChromaDB memory.
    Call this any time to see your knowledge base grow.
    """
    count = _collection.count()
    stats = {"total_companies": count}

    if count > 0:
        all_data  = _collection.get(include=["metadatas"])
        names     = [m.get("startup_name", "?") for m in all_data["metadatas"]]
        dates     = [m.get("researched_at", "?") for m in all_data["metadatas"]]
        stats["companies"] = sorted(names)
        stats["oldest"]    = min(dates)
        stats["newest"]    = max(dates)

    print(f"\n[MEMORY STATS] {count} companies in ChromaDB")
    if count > 0:
        print(f"  Oldest: {stats['oldest']}  |  Newest: {stats['newest']}")
        print(f"  Companies: {', '.join(stats['companies'][:10])}")
        if count > 10:
            print(f"  ... and {count - 10} more")

    return stats
