"""
Job Application Agent — v2 with ChromaDB Memory
================================================
LangGraph multi-agent pipeline for researching startups,
assessing fit, and drafting cold emails + LinkedIn notes.

NEW in v2:
- ChromaDB memory layer caches all company research locally
- Cache hit = instant load, skips web search entirely
- find_similar_companies() for semantic startup discovery
- Research stays fresh for 30 days, then auto-refreshes

FREE SETUP (no paid API needed):
- LLM: Groq free tier (llama-3.3-70b-versatile) — 14,400 req/day
- Search: DuckDuckGo (ddgs) — no API key
- Memory: ChromaDB local + sentence-transformers — free, runs offline
"""
from dotenv import load_dotenv
from pathlib import Path
load_dotenv(Path(__file__).parent / ".env")

import json
import os
import re
import sys
import time
import concurrent.futures
from datetime import date
from typing import Annotated


def _enable_utf8_console() -> None:
    """Avoid Windows cp1252 crashes when printing Unicode status text."""
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")


_enable_utf8_console()

from ddgs import DDGS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI

# Memory layer — ChromaDB + sentence-transformers
from memory import get_cached_research, save_research_to_memory, find_similar_companies, memory_stats

# ---------------------------------------------------------------------------
# CONFIG — edit these before running
# ---------------------------------------------------------------------------

# Get a free Groq key at https://console.groq.com (takes 2 minutes, no card)
#GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY") 
# Update this once; all agents read from this constant
MY_SKILLS = """
Skills: Python, LangChain, LangGraph, RAG pipelines, ChromaDB,
FastAPI, Groq API, LangGraph agent systems, sentence-transformers,
deep learning (CNNs, RNNs, LSTMs), Docker basics, W&B, DDGS search
Background: 3rd year IIT Roorkee student (Mechanical/Production Engineering)
Projects: Built RAG pipelines, LangGraph multi-agent systems, FastAPI services
GitHub: github.com/[yourname]
"""

OUTPUT_DIR = "applications"

# ---------------------------------------------------------------------------
# LLM SETUP
# ---------------------------------------------------------------------------

#llm = ChatGroq(
  #  api_key=GROQ_API_KEY,
  #  model="llama-3.3-70b-versatile",   # Best free model on Groq
   # temperature=0.3,
    #max_tokens=1024,
#)
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    api_key=GOOGLE_API_KEY,
    temperature=0.3,
    max_tokens=1024,

)

# ---------------------------------------------------------------------------
# STATE
# ---------------------------------------------------------------------------

class JobApplicationState(TypedDict):
    messages: Annotated[list, add_messages]
    startup_name: str
    company_research: str
    tech_stack: str
    fit_assessment: str
    fit_score: int
    cold_email: str
    linkedin_note: str
    action: str
    error: str          # NEW: captures any node-level errors


def _extract_text(response) -> str:
    """Safely extract plain text from an LLM response (supports strings, AIMessage, and list content blocks)."""
    if hasattr(response, "text") and isinstance(response.text, str) and response.text.strip():
        return response.text.strip()
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
            else:
                parts.append(str(part))
        return "".join(parts).strip()
    return str(content).strip()


# ---------------------------------------------------------------------------
# NODES
# ---------------------------------------------------------------------------

def company_researcher(state: JobApplicationState) -> dict:
    """
    Search for company info, recent news, tech stack.
    Checks ChromaDB memory first — only hits the web on a cache miss.
    """
    name = state["startup_name"]
    print(f"\n[RESEARCHER] Investigating {name}...")

    # ── Step 1: Check memory cache ──────────────────────────────────────────
    cached = get_cached_research(name)
    if cached:
        print(f"[RESEARCHER] Loaded from memory — skipping web search")
        return {
            "company_research": cached["company_research"],
            "tech_stack":       cached["tech_stack"],
            "error":            "",
        }

    # ── Step 2: Cache miss — search the web ─────────────────────────────────
    try:
        with DDGS() as ddgs:
            company_info = list(ddgs.text(f"{name} AI startup product", max_results=3))
            tech_info    = list(ddgs.text(f"{name} tech stack engineering blog", max_results=2))
        time.sleep(1)
    except Exception as e:
        print(f"[RESEARCHER] Search failed: {e}")
        return {
            "company_research": f"Search failed: {e}",
            "tech_stack":       "Unknown",
            "error":            str(e),
        }

    company_str = "\n".join(r.get("body", "") for r in company_info) or "No results found."
    tech_str    = "\n".join(r.get("body", "") for r in tech_info)    or "No results found."

    try:
        research_response = llm.invoke([
            SystemMessage(content=(
                "Summarize this company in 3–4 sentences: what they do, their core product, "
                "recent news, funding stage, and approximate team size if mentioned."
            )),
            HumanMessage(content=company_str),
        ])

        tech_response = llm.invoke([
            SystemMessage(content=(
                "Extract the tech stack from this text: programming languages, ML/AI frameworks, "
                "vector databases, cloud infra, and any LLM tooling mentioned. "
                "Format as a short bullet list. If nothing found, say 'Not publicly disclosed'."
            )),
            HumanMessage(content=tech_str),
        ])
    except Exception as e:
        print(f"[RESEARCHER] LLM call failed: {e}")
        return {
            "company_research": company_str[:500],
            "tech_stack":       "Unknown",
            "error":            str(e),
        }

    company_research = _extract_text(research_response)
    tech_stack       = _extract_text(tech_response)

    # ── Step 3: Save to memory for next time ────────────────────────────────
    save_research_to_memory(
        startup_name=name,
        company_research=company_research,
        tech_stack=tech_stack,
    )

    return {
        "company_research": company_research,
        "tech_stack":       tech_stack,
        "error":            "",
    }


def fit_assessor(state: JobApplicationState) -> dict:
    """Assess how well the candidate's skills fit this company."""
    name = state["startup_name"]
    print(f"\n[FIT ASSESSOR] Assessing fit for {name}...")

    try:
        response = llm.invoke([
            SystemMessage(content="""You are a brutally honest career advisor. 
Assess the fit between a candidate and a company.

Output format (use exactly these labels):
SCORE: <integer 1-10>
REASONING: <2-3 sentences on overall fit>
MATCHING_SKILLS: <comma-separated list of matching skills>
GAPS: <comma-separated list of missing skills>"""),
            HumanMessage(content=f"""
Company: {name}
What they do: {state['company_research']}
Their tech stack: {state['tech_stack']}

Candidate profile:
{MY_SKILLS}
"""),
        ])
    except Exception as e:
        print(f"[FIT ASSESSOR] LLM call failed: {e}")
        return {"fit_assessment": f"Assessment failed: {e}", "fit_score": 0, "error": str(e)}

    content = _extract_text(response)

    # Robust score extraction — handles "SCORE: 8", "SCORE: 8/10", "SCORE: 8 out of 10"
    score = 5  # safe default
    match = re.search(r"SCORE:\s*(\d+)", content)
    if match:
        score = min(10, max(1, int(match.group(1))))

    print(f"[FIT ASSESSOR] Score: {score}/10")
    return {"fit_assessment": content, "fit_score": score, "error": ""}


def email_writer(state: JobApplicationState) -> dict:
    """Write a personalized cold email."""
    name = state["startup_name"]
    print(f"\n[EMAIL WRITER] Drafting email for {name}...")

    try:
        response = llm.invoke([
            SystemMessage(content="""Write a cold email for an ML internship application.
Rules:
- Under 150 words total
- First sentence must be a specific, non-generic hook about their product or a recent news item
- Mention IIT Roorkee
- Reference one specific project (RAG pipeline OR LangGraph agent) with context
- End with one clear, low-friction ask (15-minute call or async response)
- Never write 'I am passionate about AI' or any generic opener
- Sign off as: [Your Name] | IIT Roorkee | github.com/[yourname]"""),
            HumanMessage(content=f"""
Company: {name}
What they do: {state['company_research']}
Their tech stack: {state['tech_stack']}
Fit analysis: {state['fit_assessment']}
"""),
        ])
    except Exception as e:
        print(f"[EMAIL WRITER] LLM call failed: {e}")
        return {"cold_email": f"Email generation failed: {e}", "error": str(e)}

    return {"cold_email": _extract_text(response), "error": ""}


def linkedin_writer(state: JobApplicationState) -> dict:
    """Write a LinkedIn connection note (300 char limit)."""
    print(f"\n[LINKEDIN WRITER] Writing connection note for {state['startup_name']}...")

    try:
        response = llm.invoke([
            SystemMessage(content=(
                "Write a LinkedIn connection request note. "
                "Hard limit: 300 characters including spaces. "
                "Must reference one specific thing about their product or tech. "
                "No 'I admire your company' or 'I would love to connect'. "
                "End with a brief value prop (e.g., 'Built X with LangGraph — would love to discuss')."
            )),
            HumanMessage(content=(
                f"Company: {state['startup_name']}\n"
                f"What they do: {state['company_research'][:400]}\n"
                f"Tech: {state['tech_stack'][:200]}"
            )),
        ])
    except Exception as e:
        print(f"[LINKEDIN WRITER] LLM call failed: {e}")
        return {"linkedin_note": f"LinkedIn note generation failed: {e}", "error": str(e)}

    note = _extract_text(response)
    if len(note) > 300:
        note = note[:297] + "..."

    return {"linkedin_note": note, "error": ""}


def report_writer(state: JobApplicationState) -> dict:
    """
    Save the full application package to applications/{startup_name}.json.
    Runs after linkedin_writer (or after skip_node for low-fit companies).
    """
    name = state["startup_name"]
    print(f"\n[REPORT WRITER] Saving application package for {name}...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Sanitize filename: lowercase, spaces → underscores, strip special chars
    safe_name = re.sub(r"[^\w]", "_", name.lower()).strip("_")
    filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.json")

    package = {
        "startup":          name,
        "fit_score":        state.get("fit_score", 0),
        "company_summary":  state.get("company_research", ""),
        "tech_stack":       state.get("tech_stack", ""),
        "fit_assessment":   state.get("fit_assessment", ""),
        "cold_email":       state.get("cold_email", ""),
        "linkedin_note":    state.get("linkedin_note", ""),
        "researched_at":    date.today().isoformat(),
        "skipped":          state.get("fit_score", 0) < 5,
        "error":            state.get("error", ""),
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(package, f, indent=2, ensure_ascii=False)

    print(f"[REPORT WRITER] Saved → {filepath}")
    return {"messages": [AIMessage(content=f"Saved application package to {filepath}")]}


def skip_node(state: JobApplicationState) -> dict:
    """Low fit — generate a skip message and still save the record."""
    score = state["fit_score"]
    name  = state["startup_name"]
    print(f"\n[SKIP] {name} scored {score}/10 — skipping email drafts.")
    return {
        "cold_email":   f"[SKIPPED — fit score {score}/10 below threshold]",
        "linkedin_note": "[SKIPPED]",
        "messages": [AIMessage(content=f"Low fit ({score}/10). Not applying to {name}.")],
    }


def application_supervisor(state: JobApplicationState) -> dict:
    """Route to next agent based on what is done and fit score."""
    has_research = bool(state.get("company_research"))
    has_fit      = bool(state.get("fit_assessment"))
    has_email    = bool(state.get("cold_email"))
    has_linkedin = bool(state.get("linkedin_note"))
    has_report   = any(
        msg.content.startswith("Saved application")
        for msg in state.get("messages", [])
        if isinstance(msg, AIMessage)
    )
    score = state.get("fit_score", 0)

    if not has_research:
        action = "research"
    elif not has_fit:
        action = "assess"
    elif score < 5 and not has_email:
        action = "skip"           # Skip email/linkedin but still save
    elif score >= 5 and not has_email:
        action = "email"
    elif score >= 5 and not has_linkedin:
        action = "linkedin"
    elif not has_report:
        action = "report"         # Always save — even skipped applications
    else:
        action = "FINISH"

    print(f"\n[SUPERVISOR] → {action}")
    return {"action": action}


# ---------------------------------------------------------------------------
# BUILD THE GRAPH
# ---------------------------------------------------------------------------

app_workflow = StateGraph(JobApplicationState)

app_workflow.add_node("supervisor", application_supervisor)
app_workflow.add_node("research",   company_researcher)
app_workflow.add_node("assess",     fit_assessor)
app_workflow.add_node("email",      email_writer)
app_workflow.add_node("linkedin",   linkedin_writer)
app_workflow.add_node("skip",       skip_node)
app_workflow.add_node("report",     report_writer)      # NEW node

app_workflow.set_entry_point("supervisor")

app_workflow.add_conditional_edges(
    "supervisor",
    lambda s: s["action"],
    {
        "research": "research",
        "assess":   "assess",
        "email":    "email",
        "linkedin": "linkedin",
        "skip":     "skip",
        "report":   "report",
        "FINISH":   END,
    },
)

# Every worker loops back to supervisor
for node in ["research", "assess", "email", "linkedin", "skip", "report"]:
    app_workflow.add_edge(node, "supervisor")

job_agent = app_workflow.compile()


# ---------------------------------------------------------------------------
# SINGLE-STARTUP RUNNER
# ---------------------------------------------------------------------------

def apply_to_startup(startup_name: str) -> dict:
    """Run the full agent pipeline for one startup."""
    initial = {
        "messages":         [HumanMessage(content=f"Research and apply to {startup_name}")],
        "startup_name":     startup_name,
        "company_research": "",
        "tech_stack":       "",
        "fit_assessment":   "",
        "fit_score":        0,
        "cold_email":       "",
        "linkedin_note":    "",
        "action":           "",
        "error":            "",
    }

    print(f"\n{'='*60}")
    print(f"JOB APPLICATION AGENT: {startup_name}")
    print("=" * 60)

    final = job_agent.invoke(initial)

    print(f"\n{'='*60}")
    print(f"FIT SCORE: {final['fit_score']}/10")
    print(f"\nCOLD EMAIL:\n{final['cold_email']}")
    print(f"\nLINKEDIN NOTE:\n{final['linkedin_note']}")

    return final


# ---------------------------------------------------------------------------
# IMPROVEMENT 1 — BATCH PROCESSING
# ---------------------------------------------------------------------------

def _saved_application_is_usable(filepath: str) -> bool:
    """Return False for old failed runs so they get regenerated."""
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False

    error = str(data.get("error", "")).strip()
    fit_assessment = str(data.get("fit_assessment", "")).lower()

    if error:
        return False
    if "invalid api key" in fit_assessment or "assessment failed" in fit_assessment:
        return False
    if data.get("fit_score", 0) == 0 and data.get("skipped") is True:
        return False

    return True


def apply_to_list(
    startups: list[str],
    max_workers: int = 3,    # 3 parallel threads — safe for Groq free tier
    skip_existing: bool = True,
) -> list[dict]:
    """
    Run the agent on a list of startups in parallel.
    Returns a list of result dicts sorted by fit_score descending.

    Args:
        startups:      List of startup names.
        max_workers:   Parallel threads. Keep ≤3 on Groq free tier to avoid 429s.
        skip_existing: If True, skips startups that already have a saved JSON.
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Deduplicate input
    seen = set()
    unique_startups = []
    for s in startups:
        key = s.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_startups.append(s.strip())

    # Optionally skip already-processed startups
    if skip_existing:
        to_process = []
        for s in unique_startups:
            safe_name = re.sub(r"[^\w]", "_", s.lower()).strip("_")
            filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
            if os.path.exists(filepath) and _saved_application_is_usable(filepath):
                print(f"[BATCH] Skipping {s} — already processed.")
            else:
                if os.path.exists(filepath):
                    print(f"[BATCH] Reprocessing {s} — existing file is from a failed run.")
                to_process.append(s)
    else:
        to_process = unique_startups

    if not to_process:
        print("[BATCH] All startups already processed. Loading from disk...")
    else:
        print(f"\n[BATCH] Processing {len(to_process)} startups with {max_workers} workers...\n")

    results = []
    errors  = []

    # Run in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_startup = {
            executor.submit(apply_to_startup, name): name
            for name in to_process
        }
        for future in concurrent.futures.as_completed(future_to_startup):
            name = future_to_startup[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"[BATCH ERROR] {name}: {e}")
                errors.append({"startup_name": name, "fit_score": 0, "error": str(e)})

    # Load existing results from disk (for skip_existing=True runs)
    if skip_existing:
        for s in unique_startups:
            safe_name = re.sub(r"[^\w]", "_", s.lower()).strip("_")
            filepath = os.path.join(OUTPUT_DIR, f"{safe_name}.json")
            if os.path.exists(filepath) and s not in to_process:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Remap to match state keys
                    results.append({
                        "startup_name":     data["startup"],
                        "fit_score":        data["fit_score"],
                        "company_research": data["company_summary"],
                        "cold_email":       data["cold_email"],
                        "linkedin_note":    data["linkedin_note"],
                    })

    results.extend(errors)

    # Sort by fit_score descending
    results.sort(key=lambda x: x.get("fit_score", 0), reverse=True)

    # Print summary table
    print(f"\n{'='*60}")
    print(f"BATCH COMPLETE — {len(results)} startups ranked")
    print("=" * 60)
    print(f"{'Rank':<5} {'Company':<30} {'Score':>5}")
    print("-" * 42)
    for i, r in enumerate(results, 1):
        name  = r.get("startup_name", r.get("startup", "Unknown"))
        score = r.get("fit_score", 0)
        flag  = " ✓" if score >= 7 else (" ~" if score >= 5 else " ✗")
        print(f"{i:<5} {name:<30} {score:>4}/10{flag}")
    print("-" * 42)
    print(f"Apply (≥7):  {sum(1 for r in results if r.get('fit_score', 0) >= 7)}")
    print(f"Consider (5-6): {sum(1 for r in results if 5 <= r.get('fit_score', 0) <= 6)}")
    print(f"Skip (<5):   {sum(1 for r in results if r.get('fit_score', 0) < 5)}")

    return results


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Quick single test
    # result = apply_to_startup("Hugging Face")

    # Batch test — your actual 20-startup sprint
    MY_TARGET_STARTUPS = [
        "Hugging Face",
        "Weaviate",
        "Mistral AI",
        "Cohere",
        "LangChain",
        "Weights & Biases",
        "Replicate",
        "Together AI",
        "Anyscale",
        "Modal Labs",
        "Chroma",
        "Qdrant",
        "Unstructured",
        "LlamaIndex",
        "Gradient AI",
        "Axflow",
        "Galileo",
        "Arize AI",
        "Vellum AI",
        "Scale AI",
    ]

    ranked = apply_to_list(MY_TARGET_STARTUPS, max_workers=2)


# ---------------------------------------------------------------------------
# BONUS — Memory-powered discovery
# ---------------------------------------------------------------------------

def find_more_like(startup_name: str, n: int = 5) -> list[dict]:
    """
    Find companies in your memory that are similar to a given startup.

    After researching 50+ companies, use this to discover patterns:
        find_more_like("Hugging Face")  →  similar AI infra companies
        find_more_like("Weaviate")      →  similar vector DB companies

    Great for building your next batch list from what's already working.
    """
    # Load the research for the reference company
    cached = get_cached_research(startup_name)
    if not cached:
        print(f"[DISCOVERY] {startup_name} not in memory yet. Research it first.")
        return []

    query   = f"{startup_name} {cached['company_research'][:300]}"
    similar = find_similar_companies(query, n_results=n, exclude=[startup_name])

    print(f"\n[DISCOVERY] Companies similar to {startup_name}:")
    for i, c in enumerate(similar, 1):
        print(f"  {i}. {c['startup_name']} (similarity: {c['similarity_score']})")

    return similar


def show_memory():
    """Print a summary of everything in your ChromaDB knowledge base."""
    memory_stats()
