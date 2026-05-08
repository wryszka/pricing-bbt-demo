"""Admin endpoints — demo reset, status."""
import asyncio
import logging

from fastapi import APIRouter, HTTPException

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
