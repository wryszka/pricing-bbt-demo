"""Admin endpoints — demo reset, status, AI response cache toggle."""
import asyncio
import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server import ai_cache
from server.audit import log_audit_event
from server.config import get_catalog, get_schema, get_workspace_client, get_workspace_host

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

RESET_JOB_NAME = "v1 — Demo reset (landing page button)"


def _find_job_id(w, name: str) -> int | None:
    try:
        for j in w.jobs.list(name=name, limit=25):
            return j.job_id
    except Exception: pass
    try:
        for j in w.jobs.list(limit=100):
            if (j.settings.name or "").endswith(name):
                return j.job_id
    except Exception: pass
    return None


@router.post("/reset-demo")
async def reset_demo() -> dict:
    """Fire the demo_reset job — single click to put the workbench back
    into clean demo state. Returns the job run ids so the UI can link
    to the workspace run page."""
    w = get_workspace_client()
    job_id = await asyncio.to_thread(_find_job_id, w, RESET_JOB_NAME)
    if not job_id:
        raise HTTPException(500,
            f"Job '{RESET_JOB_NAME}' not found. Deploy the bundle with `databricks bundle deploy`.")

    try:
        run = await asyncio.to_thread(
            w.jobs.run_now,
            job_id=job_id,
            job_parameters={"catalog_name": get_catalog(), "schema_name": get_schema()},
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to trigger demo reset: {e}")

    run_id = getattr(run, "run_id", None)
    host   = get_workspace_host()

    await log_audit_event(
        event_type="demo_reset_triggered",
        entity_type="workbench",
        entity_id="all",
        details={"job_id": job_id, "run_id": run_id, "source": "landing_page_button"},
    )
    return {
        "job_id":       job_id,
        "run_id":       run_id,
        "run_page_url": f"{host}/jobs/{job_id}/runs/{run_id}" if host and run_id else None,
    }


class AiModeRequest(BaseModel):
    mode: str  # "live" or "cached"


@router.get("/ai-mode")
async def get_ai_mode() -> dict:
    """Return the current AI response mode + cached-entry summary."""
    return {
        "mode":         ai_cache.get_mode(),
        "entries":      len(ai_cache.list_entries()),
        "modes":        ["live", "cached"],
        "description": {
            "live":   "Always call the real serving endpoint.",
            "cached": "Try the on-volume cache first; on miss call live and write the response back so repeats are instant.",
        },
    }


@router.post("/ai-mode")
async def set_ai_mode(req: AiModeRequest) -> dict:
    """Flip the global AI response mode. Persists to a UC Volume so a new
    replica picks up the same setting."""
    try:
        new_mode = ai_cache.set_mode(req.mode)
    except ValueError as e:
        raise HTTPException(400, str(e))
    await log_audit_event(
        event_type="ai_mode_changed",
        entity_type="config",
        entity_id="ai_response_mode",
        details={"mode": new_mode},
    )
    return {"mode": new_mode, "entries": len(ai_cache.list_entries())}


@router.get("/ai-cache")
async def list_ai_cache() -> dict:
    return {"mode": ai_cache.get_mode(), "entries": ai_cache.list_entries()}


@router.delete("/ai-cache")
async def clear_ai_cache() -> dict:
    """Remove every cached response. Use after a model rebuild so cached
    answers can be re-recorded against the new champion."""
    n = ai_cache.clear_cache()
    await log_audit_event(
        event_type="ai_cache_cleared",
        entity_type="config",
        entity_id="ai_response_mode",
        details={"removed": n},
    )
    return {"removed": n}
