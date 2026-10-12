"""
server.py — FastAPI Backend for JobApp Helper Prototype
========================================================
Run with:
    python server.py
or:
    uvicorn server:app --reload --port 8000
"""

import sys

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

import json
import os
from pathlib import Path
from typing import List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / ".env")

import memory
import pipeline_service
from pipeline_service import (
    ApplicationPackage,
    CandidateProfile,
    LLMConfig,
    PipelineOptions,
    delete_application,
    get_application,
    get_default_profile,
    list_saved_applications,
    load_profiles,
    run_pipeline_stream,
    run_pipeline_sync,
    save_application_record,
    save_profiles,
    test_llm_connection,
)

app = FastAPI(
    title="JobApp Helper API",
    description="Multi-agent autonomous job application and outreach studio",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ensure static directories exist
STATIC_DIR = ROOT_DIR / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "css").mkdir(parents=True, exist_ok=True)
(STATIC_DIR / "js").mkdir(parents=True, exist_ok=True)

# Mount static folder
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# ---------------------------------------------------------------------------
# ROOT & HEALTH
# ---------------------------------------------------------------------------

@app.get("/", response_class=FileResponse)
async def serve_index():
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not yet created.")
    return FileResponse(index_file)


@app.get("/api/health")
async def health_check():
    stats = memory.memory_stats()
    has_gemini = bool(os.getenv("GEMINI_API_KEY"))
    has_groq = bool(os.getenv("GROQ_API_KEY"))

    return {
        "status": "online",
        "chroma_memory_count": stats.get("total_companies", 0),
        "providers_available": {
            "gemini": has_gemini,
            "groq": has_groq,
        },
        "default_provider": "gemini" if has_gemini else ("groq" if has_groq else "none"),
    }


# ---------------------------------------------------------------------------
# CANDIDATE PROFILES
# ---------------------------------------------------------------------------

@app.get("/api/profiles", response_model=List[CandidateProfile])
async def get_profiles():
    return load_profiles()


@app.post("/api/profiles", response_model=CandidateProfile)
async def create_or_update_profile(profile: CandidateProfile):
    profiles = load_profiles()
    found = False
    for i, p in enumerate(profiles):
        if p.id == profile.id:
            profiles[i] = profile
            found = True
            break
    if not found:
        profiles.insert(0, profile)

    save_profiles(profiles)
    return profile


@app.delete("/api/profiles/{profile_id}")
async def remove_profile(profile_id: str):
    profiles = load_profiles()
    if len(profiles) <= 1:
        raise HTTPException(status_code=400, detail="Cannot delete the only remaining profile.")

    filtered = [p for p in profiles if p.id != profile_id]
    if len(filtered) == len(profiles):
        raise HTTPException(status_code=404, detail="Profile not found.")

    save_profiles(filtered)
    return {"status": "deleted", "id": profile_id}


# ---------------------------------------------------------------------------
# SAVED APPLICATIONS ARCHIVE
# ---------------------------------------------------------------------------

@app.get("/api/applications")
async def get_applications(
    q: Optional[str] = Query(None, description="Search query"),
    min_score: Optional[int] = Query(None, description="Minimum fit score"),
    status: Optional[str] = Query(None, description="Status filter: all, high, medium, skipped"),
):
    apps = list_saved_applications()

    if q:
        query_str = q.lower()
        apps = [
            a for a in apps
            if query_str in a["startup"].lower()
            or query_str in a.get("tech_stack", "").lower()
            or query_str in a.get("company_summary", "").lower()
        ]

    if min_score is not None:
        apps = [a for a in apps if a.get("fit_score", 0) >= min_score]

    if status and status != "all":
        if status == "high":
            apps = [a for a in apps if a.get("fit_score", 0) >= 7 and not a.get("error")]
        elif status == "medium":
            apps = [a for a in apps if 5 <= a.get("fit_score", 0) < 7 and not a.get("error")]
        elif status == "skipped":
            apps = [a for a in apps if a.get("skipped") or a.get("fit_score", 0) < 5 or a.get("error")]

    return {
        "total": len(apps),
        "applications": apps,
    }


@app.get("/api/applications/{slug}")
async def get_single_application(slug: str):
    data = get_application(slug)
    if not data:
        raise HTTPException(status_code=404, detail=f"Application '{slug}' not found.")
    return data


@app.delete("/api/applications/{slug}")
async def delete_single_application(slug: str):
    deleted = delete_application(slug)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Application '{slug}' not found.")
    return {"status": "deleted", "slug": slug}


class SaveEditsRequest(BaseModel):
    cold_email: Optional[str] = None
    email_subject: Optional[str] = None
    linkedin_note: Optional[str] = None


@app.post("/api/applications/{slug}/save")
async def save_application_edits(slug: str, edits: SaveEditsRequest):
    data = get_application(slug)
    if not data:
        raise HTTPException(status_code=404, detail=f"Application '{slug}' not found.")

    if edits.cold_email is not None:
        data["cold_email"] = edits.cold_email
    if edits.email_subject is not None:
        data["email_subject"] = edits.email_subject
    if edits.linkedin_note is not None:
        data["linkedin_note"] = edits.linkedin_note

    filepath = pipeline_service.OUTPUT_DIR / f"{slug}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {"status": "updated", "slug": slug, "data": data}


# ---------------------------------------------------------------------------
# PIPELINE GENERATION (SYNCHRONOUS & SSE STREAMING)
# ---------------------------------------------------------------------------

@app.post("/api/generate", response_model=ApplicationPackage)
async def generate_single(options: PipelineOptions):
    """Synchronous generation endpoint."""
    try:
        pkg = run_pipeline_sync(
            startup_name=options.startup_name,
            profile=options.profile,
            llm_config=options.llm_config,
            force_refresh=options.force_refresh,
        )
        return pkg
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/generate-stream")
async def generate_stream_post(options: PipelineOptions):
    """Server-Sent Events streaming endpoint for live step-by-step progress."""
    async def event_generator():
        try:
            async for evt in run_pipeline_stream(
                startup_name=options.startup_name,
                profile=options.profile,
                llm_config=options.llm_config,
                force_refresh=options.force_refresh,
            ):
                payload = json.dumps(evt)
                yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"event": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/api/generate-stream")
async def generate_stream_get(
    startup: str = Query(..., description="Startup name"),
    force_refresh: bool = Query(False, description="Bypass cache"),
    profile_id: Optional[str] = Query(None, description="Profile ID"),
):
    """Convenience GET SSE endpoint for easy browser EventSource."""
    profiles = load_profiles()
    active_profile = profiles[0] if profiles else get_default_profile()
    if profile_id:
        for p in profiles:
            if p.id == profile_id:
                active_profile = p
                break

    async def event_generator():
        try:
            async for evt in run_pipeline_stream(
                startup_name=startup,
                profile=active_profile,
                force_refresh=force_refresh,
            ):
                payload = json.dumps(evt)
                yield f"data: {payload}\n\n"
        except Exception as e:
            err_payload = json.dumps({"event": "error", "message": str(e)})
            yield f"data: {err_payload}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


class BatchRequest(BaseModel):
    startups: List[str]
    profile: Optional[CandidateProfile] = None
    llm_config: Optional[LLMConfig] = None
    force_refresh: bool = False


@app.post("/api/batch-generate-stream")
async def batch_generate_stream(batch: BatchRequest):
    """Stream progress across a batch of startups sequentially."""
    startups = [s.strip() for s in batch.startups if s.strip()]
    if not startups:
        raise HTTPException(status_code=400, detail="Empty startup list.")

    async def batch_generator():
        total = len(startups)
        yield f"data: {json.dumps({'event': 'batch_start', 'total': total})}\n\n"

        for idx, startup in enumerate(startups, start=1):
            yield f"data: {json.dumps({'event': 'batch_item_start', 'startup': startup, 'index': idx, 'total': total})}\n\n"
            try:
                async for evt in run_pipeline_stream(
                    startup_name=startup,
                    profile=batch.profile,
                    llm_config=batch.llm_config,
                    force_refresh=batch.force_refresh,
                ):
                    yield f"data: {json.dumps(evt)}\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'event': 'error', 'startup': startup, 'message': str(e)})}\n\n"

        yield f"data: {json.dumps({'event': 'batch_complete', 'total': total})}\n\n"

    return StreamingResponse(batch_generator(), media_type="text/event-stream")


# ---------------------------------------------------------------------------
# DISCOVERY & CHROMADB MEMORY
# ---------------------------------------------------------------------------

@app.get("/api/discovery/stats")
async def get_discovery_stats():
    return memory.memory_stats()


@app.get("/api/discovery/similar")
async def get_similar_companies(
    query: str = Query(..., description="Query or startup name to search"),
    n: int = Query(5, description="Number of results"),
    exclude: Optional[str] = Query(None, description="Comma-separated company names to exclude"),
):
    exclude_list = [x.strip() for x in exclude.split(",")] if exclude else []
    results = memory.find_similar_companies(query=query, n_results=n, exclude=exclude_list)
    return {
        "query": query,
        "count": len(results),
        "results": results,
    }


# ---------------------------------------------------------------------------
# SETTINGS & CREDENTIALS
# ---------------------------------------------------------------------------

@app.post("/api/settings/test")
async def test_credentials(config: LLMConfig):
    try:
        return test_llm_connection(config)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/settings/models")
async def get_available_models():
    return {
        "gemini": [
            {"id": "gemini-3.8-flash", "name": "Gemini 3.8 Flash (Recommended - Fastest & High Quality)", "default": True},
            {"id": "gemini-3.5-flash", "name": "Gemini 3.5 Flash"},
        ],
        "groq": [
            {"id": "llama-3.3-70b-versatile", "name": "Llama 3.3 70B Versatile (Free - 14k req/day)", "default": True},
            {"id": "llama-3.1-8b-instant", "name": "Llama 3.1 8B Instant (Ultra-fast)"},
        ],
    }


# ---------------------------------------------------------------------------
# ENTRYPOINT
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", 8000))
    print("\n" + "=" * 60)
    print(f"[SERVER] JobApp Helper Server Starting on http://{host}:{port}")
    print("=" * 60 + "\n")
    uvicorn.run("server:app", host=host, port=port, reload=False)
