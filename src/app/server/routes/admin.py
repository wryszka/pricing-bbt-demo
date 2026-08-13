"""Admin endpoints — demo reset, status, AI response cache toggle."""
import asyncio
import json
import logging
import os

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server import ai_cache
from server.audit import log_audit_event
from server.config import get_catalog, get_schema, get_workspace_client, get_workspace_host, get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/admin", tags=["admin"])

RESET_JOB_NAME = "v1 — Demo reset (landing page button)"


def _require_admin(action: str) -> None:
    """Gate destructive, everyone-affecting actions (demo reset, sleep, cache
    clear) behind an allowlist so a random viewer can't wipe a live demo for all
    users. ADMIN_USERS is a comma-separated env of emails. If it's unset the
    guard is permissive (preserves single-presenter behaviour) but logs a warning
    — set ADMIN_USERS in app.yaml to enforce for a shared demo."""
    allow = [u.strip().lower() for u in os.getenv("ADMIN_USERS", "").split(",") if u.strip()]
    if not allow:
        logger.warning("ADMIN_USERS not set — '%s' allowed for any user (set ADMIN_USERS to restrict)", action)
        return
    user = (get_current_user() or "").lower()
    if user not in allow:
        raise HTTPException(403, f"'{action}' is admin-only. {user or 'unknown user'} is not in ADMIN_USERS.")


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


@router.get("/reset-demo/status")
async def reset_demo_status(run_id: int) -> dict:
    """Poll the demo_reset job run. Returns life-cycle / result + the
    notebook's exit payload (including the ai-cache warm outcome) once
    the run terminates."""
    w = get_workspace_client()
    try:
        run = await asyncio.to_thread(w.jobs.get_run, run_id=run_id)
    except Exception as e:
        raise HTTPException(500, f"Could not fetch run {run_id}: {e}")

    state  = run.state
    life   = str(state.life_cycle_state).split(".")[-1] if state and state.life_cycle_state else None
    result = str(state.result_state).split(".")[-1] if state and state.result_state else None

    exit_payload: dict | None = None
    try:
        for t in (run.tasks or []):
            if t.run_id:
                out = await asyncio.to_thread(w.jobs.get_run_output, run_id=t.run_id)
                if out.notebook_output and out.notebook_output.result:
                    try:
                        exit_payload = json.loads(out.notebook_output.result)
                    except Exception:
                        exit_payload = {"raw": out.notebook_output.result}
                    break
    except Exception as e:
        logger.info("could not read reset run %s output yet: %s", run_id, e)

    return {
        "run_id":        run_id,
        "life_cycle":    life,                # PENDING / RUNNING / TERMINATED / ...
        "result":        result,              # SUCCESS / FAILED / CANCELED / None
        "state_message": state.state_message if state else None,
        "summary":       exit_payload,        # populated once notebook exits
    }


@router.post("/reset-demo")
async def reset_demo() -> dict:
    """Fire the demo_reset job — single click to put the workbench back
    into clean demo state. Returns the job run ids so the UI can link
    to the workspace run page."""
    _require_admin("reset-demo")
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
    replica picks up the same setting. Admin-gated — it changes the mode for
    every viewer, so a random user shouldn't flip it mid-demo."""
    _require_admin("set-ai-mode")
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
    _require_admin("clear-ai-cache")
    n = ai_cache.clear_cache()
    await log_audit_event(
        event_type="ai_cache_cleared",
        entity_type="config",
        entity_id="ai_response_mode",
        details={"removed": n},
    )
    return {"removed": n}


# ---------------------------------------------------------------------------
# Warm-up — populate the cache with canonical demo questions so the first
# click in a recorded demo run lands instantly. Called by /reset-demo at the
# end of the data-reset job so the cache rebuilds against fresh champions.
# ---------------------------------------------------------------------------

# Curated list — covers the buttons most likely to be clicked during a live
# demo. New entries: add (endpoint, question, custom_inputs).
_WARMUP_PROMPTS: list[dict] = [
    # Bias investigator — every protected attribute the UI exposes
    {"endpoint": "pricing_chat_agent",
     "question": "Brief: is there a director_gender bias signal in the live champions? Headline finding only.",
     "custom_inputs": {"persona": "bias_investigator", "mode": "live", "protected_attribute": "director_gender"}},
    {"endpoint": "pricing_chat_agent",
     "question": "Brief: is there a postcode_demographic bias signal in the live champions?",
     "custom_inputs": {"persona": "bias_investigator", "mode": "live", "protected_attribute": "postcode_demographic"}},
    {"endpoint": "pricing_chat_agent",
     "question": "Brief: is there an ethnicity_proxy bias signal in the live champions?",
     "custom_inputs": {"persona": "bias_investigator", "mode": "live", "protected_attribute": "ethnicity_proxy"}},
    {"endpoint": "pricing_chat_agent",
     "question": "Brief: is there a director_age_band bias signal in the live champions?",
     "custom_inputs": {"persona": "bias_investigator", "mode": "live", "protected_attribute": "director_age_band"}},
    # Governance agent — one canonical lookup per family
    *[{"endpoint": "pricing_governance_agent",
       "question": f"Brief: which pack governs the latest {fam} champion?",
       "custom_inputs": {"pack_id": ""}}
      for fam in ("freq_glm", "sev_glm", "demand_gbm", "fraud_gbm",
                  "freq_glm_motor", "sev_glm_motor",
                  "demand_gbm_motor", "fraud_gbm_motor")],
    # Impact explainer — one rolling summary
    {"endpoint": "pricing_chat_agent",
     "question": "Why did premiums change in the latest data update?",
     "custom_inputs": {"persona": "explain"}},
    # Multi-agent fan-out — warming each leg individually so the supervisor's
    # parallel fan-out also lands instantly from cache.
    {"endpoint": "pricing_governance_agent",
     "question": "For freq_glm_motor v4: which pack defends it, is there a director_gender disparity in its predictions, and why did premiums move on the last data refresh?",
     "custom_inputs": {"pack_id": ""}},
    {"endpoint": "pricing_chat_agent",
     "question": "For freq_glm_motor v4: which pack defends it, is there a director_gender disparity in its predictions, and why did premiums move on the last data refresh?",
     "custom_inputs": {"persona": "bias_investigator", "mode": "live"}},
    {"endpoint": "pricing_chat_agent",
     "question": "For freq_glm_motor v4: which pack defends it, is there a director_gender disparity in its predictions, and why did premiums move on the last data refresh?",
     "custom_inputs": {"persona": "explain"}},
]


@router.post("/ai-cache/warm")
async def warm_ai_cache(clear_first: bool = False, keep_cached: bool = True) -> dict:
    """Fire the curated canonical questions once so the cache holds an entry
    for each. Flips into `cached` mode for the warm-up so `invoke_agent`
    writes the responses back.

    By default (`keep_cached=true`) the endpoint leaves the workbench in
    `cached` mode when it returns — that's the point of warming. Pass
    `keep_cached=false` if you want the mode restored to whatever it was
    before the call. `clear_first=true` wipes the cache before warming."""
    from server.agent_client import invoke_agent

    if clear_first:
        ai_cache.clear_cache()

    prior_mode = ai_cache.get_mode()
    ai_cache.set_mode("cached")
    try:
        results: list[dict] = []
        for p in _WARMUP_PROMPTS:
            try:
                r = await invoke_agent(
                    endpoint_name=p["endpoint"],
                    question=p["question"],
                    custom_inputs=p["custom_inputs"],
                    timeout=300,
                )
                results.append({
                    "endpoint": p["endpoint"],
                    "persona":  (p["custom_inputs"] or {}).get("persona"),
                    "ok":       bool(r.get("ok")),
                    "cached":   bool(r.get("cached")),  # true if it was already cached
                    "error":    (r.get("error") or "")[:200],
                })
            except Exception as e:
                logger.warning("warm failed for %s: %s", p["endpoint"], e)
                results.append({"endpoint": p["endpoint"], "ok": False, "error": str(e)[:200]})
    finally:
        if not keep_cached:
            ai_cache.set_mode(prior_mode)

    ok_count    = sum(1 for r in results if r["ok"])
    fail_count  = len(results) - ok_count
    final_mode  = ai_cache.get_mode()
    await log_audit_event(
        event_type="ai_cache_warmed",
        entity_type="config",
        entity_id="ai_response_mode",
        details={"ok": ok_count, "failed": fail_count, "clear_first": clear_first,
                 "prior_mode": prior_mode, "final_mode": final_mode},
    )
    return {
        "ok":              ok_count,
        "failed":          fail_count,
        "total":           len(results),
        "entries_in_cache": len(ai_cache.list_entries()),
        "prior_mode":      prior_mode,
        "final_mode":      final_mode,
        "results":         results,
    }


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
    _require_admin("sleep")
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
