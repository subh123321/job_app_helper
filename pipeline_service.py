"""
pipeline_service.py — Dynamic multi-agent pipeline service for JobApp Helper
=============================================================================
Provides profile management, dynamic LLM configuration, synchronous and
streaming (SSE) pipeline execution, and archive retrieval.
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import date
from pathlib import Path
from typing import AsyncGenerator, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load .env from project root
ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

from ddgs import DDGS
from langchain_core.messages import HumanMessage, SystemMessage
from memory import (
    find_similar_companies,
    get_cached_research,
    memory_stats,
    save_research_to_memory,
)

OUTPUT_DIR = ROOT_DIR / "applications"
PROFILES_FILE = ROOT_DIR / "profiles.json"


# ---------------------------------------------------------------------------
# DATA MODELS
# ---------------------------------------------------------------------------

class CandidateProfile(BaseModel):
    id: str = Field(default="default", description="Profile unique ID")
    name: str = Field(default="Subhra", description="Candidate name")
    target_role: str = Field(
        default="AI / Machine Learning Engineer Intern",
        description="Target job position",
    )
    background: str = Field(
        default="3rd year IIT Roorkee student (Mechanical/Production Engineering)",
        description="College, year, and degree or background",
    )
    skills: str = Field(
        default="Python, LangChain, LangGraph, RAG pipelines, ChromaDB, FastAPI, Groq API, sentence-transformers, deep learning, Docker basics, W&B",
        description="Comma-separated skills",
    )
    projects: str = Field(
        default="Built enterprise RAG pipelines, LangGraph multi-agent systems, and FastAPI microservices",
        description="Key projects or achievements",
    )
    github: str = Field(default="github.com/subhra", description="GitHub URL")
    linkedin: str = Field(default="linkedin.com/in/subhra", description="LinkedIn URL")
    contact_email: str = Field(default="subhra@example.com", description="Contact email")
    tone: str = Field(
        default="confident, concise, and technically grounded",
        description="Tone of cold emails",
    )


class LLMConfig(BaseModel):
    provider: str = Field(default="gemini", description="gemini or groq")
    api_key: Optional[str] = Field(default=None, description="Optional custom key")
    model: Optional[str] = Field(default=None, description="Model identifier")
    temperature: float = Field(default=0.3, description="LLM temperature")


class PipelineOptions(BaseModel):
    startup_name: str
    profile: Optional[CandidateProfile] = None
    llm_config: Optional[LLMConfig] = None
    force_refresh: bool = Field(default=False, description="Bypass memory cache if True")


class ApplicationPackage(BaseModel):
    startup: str
    fit_score: int
    company_summary: str
    tech_stack: str
    fit_assessment: str
    matching_skills: List[str] = []
    skill_gaps: List[str] = []
    cold_email: str
    email_subject: str
    linkedin_note: str
    researched_at: str
    skipped: bool = False
    source: str = "web"  # "cache" or "web"
    error: str = ""


# ---------------------------------------------------------------------------
# HELPERS
# ---------------------------------------------------------------------------

def _extract_text(response) -> str:
    """Safely extract string text from various LLM response types."""
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


def get_llm(config: Optional[LLMConfig] = None):
    """Instantiate the appropriate LLM client."""
    config = config or LLMConfig()
    provider = (config.provider or "gemini").lower()

    if provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        key = config.api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY is not configured in .env or settings.")
        model_name = config.model or "gemini-3.8-flash"
        return ChatGoogleGenerativeAI(
            model=model_name,
            api_key=key,
            temperature=config.temperature,
            max_tokens=1024,
        )
    elif provider == "groq":
        from langchain_groq import ChatGroq
        key = config.api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("GROQ_API_KEY is not configured in .env or settings.")
        model_name = config.model or "llama-3.3-70b-versatile"
        return ChatGroq(
            api_key=key,
            model=model_name,
            temperature=config.temperature,
            max_tokens=1024,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def test_llm_connection(config: LLMConfig) -> Dict[str, str]:
    """Test LLM credentials with a quick ping."""
    t0 = time.time()
    llm = get_llm(config)
    res = llm.invoke([HumanMessage(content="Respond with just the single word: OK")])
    elapsed_ms = int((time.time() - t0) * 1000)
    text = _extract_text(res)
    return {
        "status": "success",
        "message": f"Connection verified in {elapsed_ms}ms!",
        "response": text,
        "provider": config.provider,
        "model": config.model or ("gemini-3.8-flash" if config.provider == "gemini" else "llama-3.3-70b-versatile"),
    }


# ---------------------------------------------------------------------------
# PROFILE STORAGE
# ---------------------------------------------------------------------------

def load_profiles() -> List[CandidateProfile]:
    """Load profiles from disk or return default."""
    if not PROFILES_FILE.exists():
        default_prof = CandidateProfile()
        save_profiles([default_prof])
        return [default_prof]

    try:
        with open(PROFILES_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return [CandidateProfile(**p) for p in data]
    except Exception:
        default_prof = CandidateProfile()
        return [default_prof]


def save_profiles(profiles: List[CandidateProfile]) -> None:
    """Save profiles to disk."""
    with open(PROFILES_FILE, "w", encoding="utf-8") as f:
        json.dump([p.model_dump() for p in profiles], f, indent=2, ensure_ascii=False)


def get_default_profile() -> CandidateProfile:
    """Return current primary candidate profile."""
    profs = load_profiles()
    return profs[0] if profs else CandidateProfile()


# ---------------------------------------------------------------------------
# APPLICATIONS ARCHIVE
# ---------------------------------------------------------------------------

def list_saved_applications() -> List[Dict]:
    """List all saved application JSON files with parsed metrics."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = []

    for path in OUTPUT_DIR.glob("*.json"):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            slug = path.stem
            startup = data.get("startup", slug.replace("_", " ").title())
            fit_score = data.get("fit_score", 0)
            skipped = data.get("skipped", False)
            error = data.get("error", "")

            status = "completed"
            if error:
                status = "error"
            elif skipped:
                status = "skipped"

            results.append({
                "slug": slug,
                "startup": startup,
                "fit_score": fit_score,
                "company_summary": data.get("company_summary", ""),
                "tech_stack": data.get("tech_stack", ""),
                "fit_assessment": data.get("fit_assessment", ""),
                "cold_email": data.get("cold_email", ""),
                "linkedin_note": data.get("linkedin_note", ""),
                "researched_at": data.get("researched_at", ""),
                "skipped": skipped,
                "status": status,
                "error": error,
            })
        except Exception as e:
            continue

    # Sort: highest fit score first, then newest
    results.sort(key=lambda x: (x.get("fit_score", 0), x.get("researched_at", "")), reverse=True)
    return results


def get_application(slug: str) -> Optional[Dict]:
    """Get single application details."""
    filepath = OUTPUT_DIR / f"{slug}.json"
    if not filepath.exists():
        return None
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def delete_application(slug: str) -> bool:
    """Delete an application record."""
    filepath = OUTPUT_DIR / f"{slug}.json"
    if filepath.exists():
        filepath.unlink()
        return True
    return False


def save_application_record(package: ApplicationPackage) -> str:
    """Save application package to applications/{slug}.json."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^\w]", "_", package.startup.lower()).strip("_")
    filepath = OUTPUT_DIR / f"{safe_name}.json"

    data = {
        "startup": package.startup,
        "fit_score": package.fit_score,
        "company_summary": package.company_summary,
        "tech_stack": package.tech_stack,
        "fit_assessment": package.fit_assessment,
        "matching_skills": package.matching_skills,
        "skill_gaps": package.skill_gaps,
        "cold_email": package.cold_email,
        "email_subject": package.email_subject,
        "linkedin_note": package.linkedin_note,
        "researched_at": package.researched_at,
        "skipped": package.skipped,
        "error": package.error,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return safe_name


# ---------------------------------------------------------------------------
# PIPELINE GENERATOR (SSE-STREAMABLE)
# ---------------------------------------------------------------------------

async def run_pipeline_stream(
    startup_name: str,
    profile: Optional[CandidateProfile] = None,
    llm_config: Optional[LLMConfig] = None,
    force_refresh: bool = False,
) -> AsyncGenerator[Dict, None]:
    """
    Execute multi-agent job application pipeline and yield real-time SSE progress events.
    """
    profile = profile or get_default_profile()
    name = startup_name.strip()
    researched_at = date.today().isoformat()

    if not name:
        yield {"event": "error", "message": "Startup name cannot be blank."}
        return

    # Initialize LLM
    try:
        llm = get_llm(llm_config)
    except Exception as e:
        yield {"event": "error", "message": f"LLM initialization failed: {str(e)}"}
        return

    # -----------------------------------------------------------------------
    # Step 1: ChromaDB Memory Cache Check
    # -----------------------------------------------------------------------
    yield {
        "event": "step",
        "step": "memory",
        "step_num": 1,
        "title": "Checking Memory Cache",
        "message": f"Scanning ChromaDB vector memory for '{name}'...",
        "progress": 15,
    }

    company_research = ""
    tech_stack = ""
    cache_hit = False

    if not force_refresh:
        cached = get_cached_research(name)
        if cached and cached.get("company_research"):
            cache_hit = True
            company_research = str(cached["company_research"])
            tech_stack = str(cached.get("tech_stack", "Not publicly disclosed"))
            yield {
                "event": "log",
                "message": f"⚡ Cache Hit: Found research for '{name}' in ChromaDB (from {cached.get('researched_at')}). Skipping web search.",
            }

    # -----------------------------------------------------------------------
    # Step 2: Web Search via DuckDuckGo (if cache miss)
    # -----------------------------------------------------------------------
    if not cache_hit:
        yield {
            "event": "step",
            "step": "search",
            "step_num": 2,
            "title": "Web Intelligence",
            "message": f"Searching DuckDuckGo for '{name}' startup details & tech stack...",
            "progress": 35,
        }

        try:
            with DDGS() as ddgs:
                info_results = list(ddgs.text(f"{name} AI startup product overview", max_results=3))
                tech_results = list(ddgs.text(f"{name} tech stack engineering blog architecture", max_results=2))

            company_raw = "\n".join(r.get("body", "") for r in info_results) or f"{name} is an AI startup."
            tech_raw = "\n".join(r.get("body", "") for r in tech_results) or "No public tech blog."

            yield {
                "event": "log",
                "message": f"Found {len(info_results)} product sources and {len(tech_results)} engineering sources.",
            }

            # Summarize company
            yield {
                "event": "step",
                "step": "analyze",
                "step_num": 2,
                "title": "Synthesizing Intel",
                "message": f"Synthesizing product and tech profile for {name} with LLM...",
                "progress": 50,
            }

            res_summary = llm.invoke([
                SystemMessage(content=(
                    "Summarize this company in 3-4 sentences: what they do, core product value, "
                    "recent news/achievements, and funding stage/team size if available."
                )),
                HumanMessage(content=company_raw),
            ])
            company_research = _extract_text(res_summary)

            res_tech = llm.invoke([
                SystemMessage(content=(
                    "Extract the tech stack from this text: programming languages, ML/AI frameworks, "
                    "vector databases, cloud infra, and LLM tooling. Format as a concise bulleted list. "
                    "If nothing is found, output 'Not publicly disclosed'."
                )),
                HumanMessage(content=tech_raw),
            ])
            tech_stack = _extract_text(res_tech)

            # Save to ChromaDB
            try:
                save_research_to_memory(
                    startup_name=name,
                    company_research=company_research,
                    tech_stack=tech_stack,
                )
                yield {
                    "event": "log",
                    "message": f"💾 Saved vector embedding for '{name}' to ChromaDB memory.",
                }
            except Exception as mem_err:
                yield {
                    "event": "log",
                    "message": f"⚠️ Memory save notice: {mem_err}",
                }

        except Exception as search_err:
            yield {
                "event": "log",
                "message": f"Web search fallback notice: {search_err}. Generating best-effort assessment.",
            }
            company_research = f"{name} is an AI technology company."
            tech_stack = "Python, Machine Learning, Cloud Services"

    # -----------------------------------------------------------------------
    # Step 3: Fit Assessor
    # -----------------------------------------------------------------------
    yield {
        "event": "step",
        "step": "assess",
        "step_num": 3,
        "title": "Fit Assessment",
        "message": f"Evaluating alignment between {profile.name}'s skills and {name}...",
        "progress": 68,
    }

    fit_prompt = f"""You are a brutally honest AI career advisor.
Assess candidate-company alignment.

Candidate Profile:
- Name: {profile.name}
- Target Role: {profile.target_role}
- Background: {profile.background}
- Skills: {profile.skills}
- Projects: {profile.projects}

Target Company:
- Company: {name}
- What they do: {company_research}
- Their Tech Stack: {tech_stack}

OUTPUT FORMAT (Use exactly these headers):
SCORE: <integer 1 to 10>
REASONING: <2-3 sentences on true fit and feasibility>
MATCHING_SKILLS: <comma-separated list of candidate skills that directly match>
GAPS: <comma-separated list of missing skills or domain gaps>"""

    try:
        assess_res = llm.invoke([
            SystemMessage(content="You assess technical fit for tech roles objectively."),
            HumanMessage(content=fit_prompt),
        ])
        assess_text = _extract_text(assess_res)
    except Exception as e:
        assess_text = f"SCORE: 5\nREASONING: Automated assessment fallback: {e}\nMATCHING_SKILLS: Python, Machine Learning\nGAPS: Cloud Infra"

    # Extract score
    score = 5
    score_match = re.search(r"SCORE:\s*(\d+)", assess_text, re.IGNORECASE)
    if score_match:
        score = min(10, max(1, int(score_match.group(1))))

    # Extract matching skills & gaps
    matching_skills = []
    skill_gaps = []

    m_match = re.search(r"MATCHING_SKILLS:\s*(.*?)(?=\nGAPS:|$)", assess_text, re.DOTALL | re.IGNORECASE)
    if m_match:
        matching_skills = [s.strip() for s in m_match.group(1).split(",") if s.strip()]

    g_match = re.search(r"GAPS:\s*(.*?)$", assess_text, re.DOTALL | re.IGNORECASE)
    if g_match:
        skill_gaps = [s.strip() for s in g_match.group(1).split(",") if s.strip()]

    yield {
        "event": "log",
        "message": f"📊 Fit Assessment complete: Score {score}/10 ({'Strong Match' if score >= 7 else 'Moderate Fit' if score >= 5 else 'Low Match'}).",
    }

    # -----------------------------------------------------------------------
    # Step 4: Outreach Writer (Cold Email & LinkedIn Note)
    # -----------------------------------------------------------------------
    cold_email = ""
    email_subject = f"Exploring {profile.target_role} Opportunities at {name}"
    linkedin_note = ""
    skipped = score < 5

    if skipped:
        yield {
            "event": "step",
            "step": "outreach",
            "step_num": 4,
            "title": "Low Fit - Skipped Outreach",
            "message": f"Fit score ({score}/10) is below threshold (5). Outreach drafts skipped.",
            "progress": 85,
        }
        cold_email = f"[SKIPPED — Candidate fit score of {score}/10 is below application threshold]"
        linkedin_note = "[SKIPPED]"
    else:
        yield {
            "event": "step",
            "step": "outreach",
            "step_num": 4,
            "title": "Drafting Outreach",
            "message": f"Writing tailored cold email and LinkedIn connection note for {name}...",
            "progress": 85,
        }

        # Email draft
        email_prompt = f"""Write an impactful cold email for an internship/job outreach.
Rules:
- Under 140 words total
- First sentence MUST reference a specific, non-generic insight about {name}'s product or recent engineering work
- Candidate Name: {profile.name}
- Candidate Background: {profile.background}
- Mention candidate's relevant project work: {profile.projects}
- Tone: {profile.tone}
- End with one frictionless call to action (e.g. 15-min chat or feedback)
- Never use generic clichés like 'I am writing to express interest' or 'I am passionate about AI'
- Sign off cleanly as:
  Best,
  {profile.name}
  {profile.github} | {profile.linkedin} | {profile.contact_email}

Company Context: {company_research}
Tech Stack: {tech_stack}
Matching Skills: {', '.join(matching_skills)}"""

        try:
            email_res = llm.invoke([
                SystemMessage(content="You are an elite tech outreach copywriter."),
                HumanMessage(content=email_prompt),
            ])
            cold_email = _extract_text(email_res)
        except Exception as e:
            cold_email = f"Hi {name} team,\n\nI admire your work in AI and wanted to share how my background in {profile.background} and projects ({profile.projects}) could contribute.\n\nBest,\n{profile.name}"

        # Subject line
        try:
            subject_res = llm.invoke([
                SystemMessage(content="Generate a single punchy email subject line under 8 words. Return ONLY the subject line."),
                HumanMessage(content=f"Company: {name}\nRole: {profile.target_role}\nCandidate Project: {profile.projects}"),
            ])
            extracted_sub = _extract_text(subject_res).strip('"\'')
            if extracted_sub:
                email_subject = extracted_sub
        except Exception:
            pass

        # LinkedIn note
        linkedin_prompt = f"""Write a LinkedIn connection request note for {name}.
STRICT HARD LIMIT: 290 characters total including spaces and punctuation.
Must mention a specific technical topic from their product/stack and a quick value prop from candidate ({profile.projects}).
No generic pleasantries.

Company Context: {company_research[:300]}
Tech Stack: {tech_stack[:150]}"""

        try:
            li_res = llm.invoke([
                SystemMessage(content="You write compact high-conversion LinkedIn invite notes."),
                HumanMessage(content=linkedin_prompt),
            ])
            linkedin_note = _extract_text(li_res).strip()
            if len(linkedin_note) > 300:
                linkedin_note = linkedin_note[:297] + "..."
        except Exception:
            linkedin_note = f"Hi {name} team — impressed by your product! Built AI pipelines with Python & LangGraph and would love to connect."

    # -----------------------------------------------------------------------
    # Step 5: Save Package
    # -----------------------------------------------------------------------
    yield {
        "event": "step",
        "step": "save",
        "step_num": 5,
        "title": "Finalizing Record",
        "message": f"Saving application package to local archive...",
        "progress": 95,
    }

    pkg = ApplicationPackage(
        startup=name,
        fit_score=score,
        company_summary=company_research,
        tech_stack=tech_stack,
        fit_assessment=assess_text,
        matching_skills=matching_skills,
        skill_gaps=skill_gaps,
        cold_email=cold_email,
        email_subject=email_subject,
        linkedin_note=linkedin_note,
        researched_at=researched_at,
        skipped=skipped,
        source="cache" if cache_hit else "web",
    )

    slug = save_application_record(pkg)

    yield {
        "event": "complete",
        "slug": slug,
        "data": pkg.model_dump(),
        "progress": 100,
        "message": f"Application package for {name} generated and saved successfully!",
    }


def run_pipeline_sync(
    startup_name: str,
    profile: Optional[CandidateProfile] = None,
    llm_config: Optional[LLMConfig] = None,
    force_refresh: bool = False,
) -> ApplicationPackage:
    """Synchronous pipeline helper."""
    import asyncio
    async def _run():
        final_pkg = None
        async for evt in run_pipeline_stream(startup_name, profile, llm_config, force_refresh):
            if evt.get("event") == "complete":
                final_pkg = ApplicationPackage(**evt["data"])
            elif evt.get("event") == "error":
                raise RuntimeError(evt.get("message"))
        return final_pkg
    return asyncio.run(_run())
