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


# ---------------------------------------------------------------------------
# Cost controls — pause every always-on compute resource the workbench
# leans on, so the demo can sit idle without burning budget. Restart with
# `POST /api/admin/wake` (or just `POST /api/live-pricing/start`).
# ---------------------------------------------------------------------------

_LAKEBASE_INSTANCES = ["motor-pricing-online-store"]
_SERVING_ENDPOINTS  = ["motor_pricing_scorer", "pricing_chat_agent", "pricing_governance_agent"]


def _set_lakebase_stopped(name: str, stopped: bool) -> dict:
    import requests as _rq
    w = get_workspace_client()
    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()
    try:
        resp = _rq.patch(
            f"{host}/api/2.0/database/instances/{name}?update_mask=stopped",
            headers={**token, "Content-Type": "application/json"},
            json={"stopped": stopped}, timeout=20,
        )
        ok = resp.status_code in (200, 204)
        return {"name": name, "stopped": stopped, "ok": ok, "status": resp.status_code}
    except Exception as e:
        return {"name": name, "stopped": stopped, "ok": False, "error": str(e)[:200]}


@router.post("/sleep")
async def sleep_all() -> dict:
    """Pause every always-on resource the workbench owns: deletes the live
    pricing serving endpoint and pauses the motor Lakebase instance. The
    three agent endpoints already scale to zero on idle, so no action is
    needed for them — the next quote or chat warms them naturally."""
    import asyncio
    w = get_workspace_client()
    deleted = []
    for ep in ["motor_pricing_scorer"]:
        try:
            await asyncio.to_thread(w.serving_endpoints.delete, ep)
            deleted.append(ep)
        except Exception as e:
            logger.info("sleep_all: %s already absent or delete failed: %s", ep, e)
    lakebase = []
    for name in _LAKEBASE_INSTANCES:
        lakebase.append(await asyncio.to_thread(_set_lakebase_stopped, name, True))
    await log_audit_event(
        event_type="workbench_sleep",
        entity_type="config",
        entity_id="all_compute",
        details={"endpoints_deleted": deleted, "lakebase": lakebase},
    )
    return {"endpoints_deleted": deleted, "lakebase": lakebase,
            "hint": "POST /api/live-pricing/start when ready to demo."}


@router.post("/wake")
async def wake_lakebase() -> dict:
    """Resume the Lakebase instance — convenience for the operator before
    a demo. Does not redeploy the serving endpoint; use /api/live-pricing/start
    for that."""
    import asyncio
    out = []
    for name in _LAKEBASE_INSTANCES:
        out.append(await asyncio.to_thread(_set_lakebase_stopped, name, False))
    return {"lakebase": out}


@router.get("/cost-status")
async def cost_status() -> dict:
    """Quick check: which workbench resources are currently consuming compute."""
    import asyncio, requests as _rq
    w = get_workspace_client()
    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()

    def _serving(ep: str) -> dict:
        try:
            d = w.serving_endpoints.get(ep)
            se = (d.config.served_entities or [None])[0] if d.config else None
            return {
                "name":          ep,
                "ready":         str(d.state.ready) if d.state else None,
                "scale_to_zero": getattr(se, "scale_to_zero_enabled", None) if se else None,
                "workload":      getattr(se, "workload_size", None) if se else None,
            }
        except Exception as e:
            return {"name": ep, "error": str(e)[:120]}

    def _lakebase(name: str) -> dict:
        try:
            resp = _rq.get(
                f"{host}/api/2.0/database/instances/{name}",
                headers=token, timeout=20,
            )
            d = resp.json()
            return {"name": name, "state": d.get("state"),
                    "stopped": d.get("stopped"), "capacity": d.get("capacity")}
        except Exception as e:
            return {"name": name, "error": str(e)[:120]}

    endpoints = await asyncio.gather(*(asyncio.to_thread(_serving, ep) for ep in _SERVING_ENDPOINTS))
    lakebase  = await asyncio.gather(*(asyncio.to_thread(_lakebase, name) for name in _LAKEBASE_INSTANCES))
    return {"endpoints": list(endpoints), "lakebase": list(lakebase)}
