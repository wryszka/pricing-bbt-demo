"""Compare & Test — candidate-vs-champion scoring via a Databricks Job.

The tab calls:
  1. POST /api/compare/run     — trigger the compare_scoring job
  2. GET  /api/compare/runs/:id — poll job status; on SUCCESS the notebook
                                   exit payload carries the cache_key
  3. GET  /api/compare/cache/:key — fetch the full result payload
                                     from the compare_results Delta table
  4. GET  /api/compare/scenarios  — the canned scenario list

All scoring happens inside the notebook (which loads models via
`mlflow.pyfunc.load_model`); the app only orchestrates.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import fqn, get_catalog, get_current_user, get_schema, get_workspace_client, get_workspace_host
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/compare", tags=["compare"])

COMPARE_JOB_NAME = "v1 — Compare & test models"
VALID_FAMILIES  = {"freq_glm", "sev_glm", "demand_gbm", "fraud_gbm"}

# Canned what-if scenarios — commercial-book semantics. Paired with the
# `apply_scenario` implementation in compare_score.py.
SCENARIOS: list[dict[str, Any]] = [
    {"id": "none",                       "label": "Baseline (no perturbation)",
     "description": "Score each model on the current portfolio, unmodified.",
     "applies_to": ["freq_glm", "sev_glm", "demand_gbm", "fraud_gbm"]},
    {"id": "flood_plus_1",               "label": "Flood risk +1 on coastal postcodes",
     "description": "Flood-zone rating climbs by 1 tier in coastal postcodes; composite location risk +10%.",
     "applies_to": ["freq_glm", "sev_glm", "fraud_gbm"]},
    {"id": "london_claims_surge_20pct",  "label": "London E-postcode claim frequency +20%",
     "description": "Bumps claim_count_5y by 20% in E/SE/EC postcode sectors — re-score to see downstream impact.",
     "applies_to": ["fraud_gbm", "freq_glm"]},
    {"id": "industry_mix_up",            "label": "Industry mix shifts up",
     "description": "30% of policies move one industry risk tier higher.",
     "applies_to": ["freq_glm", "sev_glm", "demand_gbm", "fraud_gbm"]},
    {"id": "competitor_a_minus_5pct",    "label": "Competitor A drops rates 5%",
     "description": "competitor_a_min_rate × 0.95; market_median_rate × 0.975.",
     "applies_to": ["demand_gbm"]},
]


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

@router.get("/scenarios")
async def list_scenarios(family: str | None = None) -> dict:
    out = [s for s in SCENARIOS if (family is None or family in s["applies_to"])]
    return {"scenarios": out, "families": sorted(VALID_FAMILIES)}


# ---------------------------------------------------------------------------
# Job lookup helper
# ---------------------------------------------------------------------------

def _find_job_id(w) -> int | None:
    """Exact-match first, then suffix-match — the bundle adds a
    `[dev <user>]` prefix to every job name, so exact lookup misses."""
    try:
        for j in w.jobs.list(name=COMPARE_JOB_NAME, limit=25):
            return j.job_id
    except Exception as e:
        logger.warning("jobs.list(name=%r) failed: %s", COMPARE_JOB_NAME, e)
    try:
        for j in w.jobs.list(limit=100):
            if (j.settings.name or "").endswith(COMPARE_JOB_NAME):
                return j.job_id
    except Exception as e:
        logger.warning("jobs.list suffix-match fallback failed: %s", e)
    return None


# ---------------------------------------------------------------------------
# Trigger
# ---------------------------------------------------------------------------

class RunRequest(BaseModel):
    family: str
    versions: list[str]
    portfolio_size: int = 5000
    scenario_id: str = "none"


async def _find_cached_compare(family: str, versions: list, scenario: str,
                               portfolio_size: int) -> dict | None:
    """In cached mode, find the most recent compare_results row matching the
    full request signature. Returns the cache_key + metadata so the caller
    can return a synthetic "already done" response and skip the 2-min job."""
    # Versions stored as comma-joined string in compare_results.
    version_str = ",".join(str(v) for v in versions)
    try:
        rows = await execute_query(f"""
            SELECT cache_key, requested_by, generated_at
            FROM {fqn('compare_results')}
            WHERE family = '{family}'
              AND versions = '{version_str}'
              AND scenario = '{scenario}'
              AND portfolio_size = {int(portfolio_size)}
            ORDER BY generated_at DESC
            LIMIT 1
        """)
    except Exception as e:
        logger.info("cache lookup failed (compare_results not queryable yet): %s", e)
        return None
    if not rows:
        return None
    r = rows[0]
    return {
        "cache_key":    r["cache_key"],
        "requested_by": r.get("requested_by"),
        "generated_at": str(r.get("generated_at", "")),
    }


@router.post("/run")
async def trigger_run(req: RunRequest) -> dict:
    if req.family not in VALID_FAMILIES:
        raise HTTPException(400, f"family must be one of {sorted(VALID_FAMILIES)}")
    if not 2 <= len(req.versions) <= 5:
        raise HTTPException(400, "versions must list 2-5 entries")
    scenario = req.scenario_id or "none"
    if scenario not in {s["id"] for s in SCENARIOS}:
        raise HTTPException(400, f"unknown scenario_id '{scenario}'")

    # Cache mode: if a prior identical run exists in compare_results, reuse
    # its cache_key. Returns a synthetic "completed" response so the UI skips
    # straight to /api/compare/cache/{key} and renders instantly.
    from server import ai_cache
    if ai_cache.get_mode() == "cached":
        hit = await _find_cached_compare(req.family, req.versions, scenario, req.portfolio_size)
        if hit:
            return {
                "job_id":         None,
                "job_run_id":     None,
                "run_page_url":   None,
                "family":         req.family,
                "versions":       req.versions,
                "scenario_id":    scenario,
                "portfolio_size": req.portfolio_size,
                "cached":         True,
                "cache_key":      hit["cache_key"],
                "life_cycle":     "TERMINATED",
                "result":         "SUCCESS",
                "generated_at":   hit["generated_at"],
                "requested_by":   hit["requested_by"],
            }

    user = get_current_user()
    w = get_workspace_client()
    job_id = await asyncio.to_thread(_find_job_id, w)
    if not job_id:
        raise HTTPException(500,
            f"Job '{COMPARE_JOB_NAME}' not found. Deploy the bundle with `databricks bundle deploy`.")

    try:
        run = await asyncio.to_thread(
            w.jobs.run_now,
            job_id=job_id,
            job_parameters={
                "catalog_name":   get_catalog(),
                "schema_name":    get_schema(),
                "model_family":   req.family,
                "versions":       ",".join(str(v) for v in req.versions),
                "portfolio_size": str(req.portfolio_size),
                "scenario_id":    scenario,
                "requested_by":   user,
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to trigger compare job: {e}")

    run_id = run.run_id if hasattr(run, "run_id") else (run.get("run_id") if isinstance(run, dict) else None)
    host = get_workspace_host()

    await log_audit_event(
        event_type="compare_run_requested",
        entity_type="model",
        entity_id=req.family,
        entity_version=",".join(str(v) for v in req.versions),
        user_id=user,
        details={"job_id": job_id, "job_run_id": run_id, "scenario": scenario,
                 "portfolio_size": req.portfolio_size, "requested_by": user},
    )

    return {
        "job_id":       job_id,
        "job_run_id":   run_id,
        "run_page_url": f"{host}/jobs/{job_id}/runs/{run_id}" if host and run_id else None,
        "family":       req.family,
        "versions":     req.versions,
        "scenario_id":  scenario,
        "portfolio_size": req.portfolio_size,
        "cached":       False,
    }


# ---------------------------------------------------------------------------
# Poll status + surface cache_key
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}")
async def run_status(run_id: int) -> dict:
    w = get_workspace_client()
    try:
        run = await asyncio.to_thread(w.jobs.get_run, run_id=run_id)
    except Exception as e:
        raise HTTPException(500, f"Could not fetch run {run_id}: {e}")

    state   = run.state
    life    = str(state.life_cycle_state).split(".")[-1] if state else None
    result  = str(state.result_state).split(".")[-1] if state and state.result_state else None

    exit_payload: dict | None = None
    try:
        for t in run.tasks or []:
            if t.task_key == "score" and t.run_id:
                out = await asyncio.to_thread(w.jobs.get_run_output, run_id=t.run_id)
                if out.notebook_output and out.notebook_output.result:
                    try:
                        exit_payload = json.loads(out.notebook_output.result)
                    except Exception:
                        exit_payload = {"raw": out.notebook_output.result}
                break
    except Exception as e:
        logger.warning("extract compare run %s output failed: %s", run_id, e)

    return {
        "run_id":        run_id,
        "life_cycle":    life,
        "result":        result,
        "state_message": state.state_message if state else None,
        "summary":       exit_payload,
    }


# ---------------------------------------------------------------------------
# Cache lookup — hydrate the full payload after the job finishes
# ---------------------------------------------------------------------------

@router.get("/cache/{cache_key}")
async def get_cached(cache_key: str) -> dict:
    rows = await execute_query(f"""
        SELECT cache_key, family, versions, scenario, portfolio_size,
               score_summary, pair_shifts, segment_rows, outlier_rows,
               holdout_metrics, explain_diff, review,
               requested_by, generated_at
        FROM {fqn('compare_results')}
        WHERE cache_key = '{cache_key}'
        ORDER BY generated_at DESC
        LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"no cached result for {cache_key}")
    row = rows[0]

    def _parse(v):
        if v is None or v == "":
            return None
        if isinstance(v, (list, dict)):
            return v
        try:
            return json.loads(v)
        except Exception:
            return v

    return {
        "cache_key":       row["cache_key"],
        "family":          row["family"],
        "versions":        (row["versions"] or "").split(","),
        "scenario":        row["scenario"],
        "portfolio_size":  row["portfolio_size"],
        "score_summary":   _parse(row["score_summary"]),
        "pair_shifts":     _parse(row["pair_shifts"]),
        "segment_rows":    _parse(row["segment_rows"]),
        "outlier_rows":    _parse(row["outlier_rows"]),
        "holdout_metrics": _parse(row["holdout_metrics"]),
        "explain_diff":    _parse(row["explain_diff"]),
        "review":          _parse(row["review"]),
        "requested_by":    row["requested_by"],
        "generated_at":    str(row.get("generated_at", "")),
    }


# ---------------------------------------------------------------------------
# History — recent compare runs so the tab can show them
# ---------------------------------------------------------------------------

@router.get("/history")
async def recent_runs(limit: int = 10) -> dict:
    limit = max(1, min(50, int(limit)))
    try:
        rows = await execute_query(f"""
            SELECT cache_key, family, versions, scenario, portfolio_size,
                   requested_by, generated_at
            FROM {fqn('compare_results')}
            ORDER BY generated_at DESC
            LIMIT {limit}
        """)
    except Exception as e:
        logger.info("compare_results not queryable yet: %s", e)
        return {"runs": []}
    return {"runs": rows}
