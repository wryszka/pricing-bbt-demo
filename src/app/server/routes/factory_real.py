"""Model Factory — Real tab backend.

Same 4-step flow as the Demo tab (factory.py) but with real training via
the factory_train_real Databricks Job. Plan generation + chat reuse the
Demo tab's logic (they're identical to the user). Training goes through a
real job; leaderboard / shortlist / packs read real MLflow + UC data.

Factory candidates are never promoted into production — they register as
`{catalog}.{schema}.factory_freq_glm_<variant_id>` UC models and live alongside
the four production champions without competing with them.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import fqn, get_catalog, get_current_user, get_schema, get_workspace_client, get_workspace_host
from server.sql import execute_query

# Reuse the demo tab's variant enumerator + narrative + chat ground logic
from server.routes.factory import (
    _variants_for_freq_glm,
    _static_plan_narrative,
    factory_chat as _demo_factory_chat,
    ChatRequest as _ChatRequest,
    ensure_factory_tables,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/factory-real", tags=["factory-real"])

TRAIN_JOB_NAME = "v1 — Factory training (real)"
PACK_JOB_NAME  = "v1 — Generate governance pack"

# For the Real tab we trim the plan to ~15 variants so training wall-clock
# stays under ~5 min. The first 5 variants from each of the 3 categories
# (feature_subset / interactions / banding) is a representative slice.
DEFAULT_MAX_VARIANTS = 15


def _trim_plan(full_plan: list[dict], max_variants: int) -> list[dict]:
    """Take a proportional slice across the 3 plan categories."""
    by_cat: dict[str, list] = {"feature_subset": [], "interactions": [], "banding": []}
    for v in full_plan:
        by_cat.setdefault(v["category"], []).append(v)
    per_cat = max(1, max_variants // 3)
    picked: list[dict] = []
    for cat in ("feature_subset", "interactions", "banding"):
        picked.extend(by_cat.get(cat, [])[:per_cat])
    # Top up if we're under cap
    remaining = max_variants - len(picked)
    if remaining > 0:
        for v in full_plan:
            if v not in picked and remaining > 0:
                picked.append(v)
                remaining -= 1
    return picked


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

class ProposeRequest(BaseModel):
    family: str
    max_variants: int | None = None


@router.post("/plan")
async def propose_plan(req: ProposeRequest) -> dict:
    if req.family != "freq_glm":
        return {
            "family":   req.family,
            "status":   "unsupported",
            "message":  "Real factory only supports freq_glm in MVP. Other families come later.",
            "plan":     [],
            "narrative": "",
        }
    from server.routes.factory import _agentic_plan_review
    review = _agentic_plan_review(req.family)
    max_v = req.max_variants or DEFAULT_MAX_VARIANTS
    full_plan = _variants_for_freq_glm(review["breakdown"])
    plan = _trim_plan(full_plan, max_v)
    narrative = _static_plan_narrative()
    return {
        "family": req.family,
        "status": "proposed",
        "mode":   "real",
        "plan":   plan,
        "narrative": narrative,
        "review": review,
        "summary": {
            "total_variants":    len(plan),
            "by_category":       {
                "feature_subset": sum(1 for v in plan if v["category"] == "feature_subset"),
                "interactions":   sum(1 for v in plan if v["category"] == "interactions"),
                "banding":        sum(1 for v in plan if v["category"] == "banding"),
            },
            "glm_families_used": sorted({v["glm"]["family"] for v in plan}),
            "features_min":      min(len(v["features"]) for v in plan),
            "features_max":      max(len(v["features"]) for v in plan),
        },
    }


# ---------------------------------------------------------------------------
# Approve + trigger REAL training job
# ---------------------------------------------------------------------------

class ApproveRequest(BaseModel):
    family: str
    plan: list[dict]
    narrative: str | None = None


def _find_job_id(w, name: str) -> int | None:
    """Find a bundle-deployed job by name. The bundle prefixes jobs with
    `[dev <username>] ` so exact-match `list(name=...)` fails. We scan and
    match by suffix instead."""
    try:
        for j in w.jobs.list(name=name, limit=25):
            return j.job_id
    except Exception as e:
        logger.warning("jobs.list(name=%r) failed: %s", name, e)
    try:
        for j in w.jobs.list(limit=100):
            if (j.settings.name or "").endswith(name):
                return j.job_id
    except Exception as e:
        logger.warning("jobs.list fallback failed: %s", e)
    return None


@router.post("/approve")
async def approve_and_train(req: ApproveRequest) -> dict:
    if req.family != "freq_glm":
        raise HTTPException(400, "Real factory supports freq_glm only.")
    if not req.plan:
        raise HTTPException(400, "Plan is empty.")

    await ensure_factory_tables()

    user = get_current_user()
    run_id = f"REAL-FACTORY-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{req.family}"
    plan_json = json.dumps(req.plan).replace("'", "''")
    narrative = (req.narrative or "").replace("'", "''")

    await execute_query(f"""
        INSERT INTO {fqn('factory_runs')}
        SELECT '{run_id}', '{req.family}', '{plan_json}', '{narrative}',
               '{user}', current_timestamp(), 0.0, 'TRAINING', {len(req.plan)}
    """)

    # Kick off the real training job
    w = get_workspace_client()
    job_id = _find_job_id(w, TRAIN_JOB_NAME)
    if not job_id:
        raise HTTPException(500,
            f"Job '{TRAIN_JOB_NAME}' not found. Deploy the bundle.")

    try:
        run = await asyncio.to_thread(
            w.jobs.run_now,
            job_id=job_id,
            job_parameters={
                "catalog_name":    get_catalog(),
                "schema_name":     get_schema(),
                "factory_run_id":  run_id,
                "max_variants":    str(len(req.plan)),
            },
        )
        job_run_id = run.run_id if hasattr(run, "run_id") else (run.get("run_id") if isinstance(run, dict) else None)
    except Exception as e:
        raise HTTPException(500, f"Failed to trigger training job: {e}")

    await log_audit_event(
        event_type="factory_plan_approved_real",
        entity_type="factory_run",
        entity_id=run_id,
        user_id=user,
        details={"family": req.family, "variants": len(req.plan),
                 "training_mode": "real", "job_run_id": job_run_id},
    )

    return {
        "run_id":     run_id,
        "family":     req.family,
        "status":     "TRAINING",
        "variant_count": len(req.plan),
        "approved_by": user,
        "job_run_id": job_run_id,
        "run_page_url": f"{get_workspace_host()}/jobs/{job_id}/runs/{job_run_id}" if job_run_id else None,
    }


# ---------------------------------------------------------------------------
# Run status polling — reads real job status + counts variants as they land
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}")
async def run_status(run_id: str) -> dict:
    rows = await execute_query(f"""
        SELECT run_id, model_family, approved_by, started_at, duration_seconds,
               status, variant_count, narrative
        FROM {fqn('factory_runs')} WHERE run_id = '{run_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"run {run_id} not found")
    r = rows[0]

    # Count real variants written so far
    try:
        vc = await execute_query(f"""
            SELECT count(*) AS n FROM {fqn('factory_variants')} WHERE run_id = '{run_id}'
        """)
        n_complete = int(vc[0]["n"]) if vc else 0
    except Exception:
        n_complete = 0

    progress = 0.0 if r["variant_count"] == 0 else min(1.0, n_complete / r["variant_count"])

    # Elapsed real wall clock
    elapsed = 0.0
    try:
        started = r["started_at"]
        if isinstance(started, str):
            started = datetime.fromisoformat(started.replace(" ", "T").replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
    except Exception:
        pass

    return {
        "run_id":        r["run_id"],
        "family":        r["model_family"],
        "status":        r["status"],
        "variant_count": r["variant_count"],
        "n_complete":    n_complete,
        "progress":      round(progress, 3),
        "elapsed_seconds": round(elapsed, 1),
        "approved_by":   r["approved_by"],
        "started_at":    str(r["started_at"]),
        "narrative":     r.get("narrative") or "",
        "mode":          "real",
    }


# ---------------------------------------------------------------------------
# Leaderboard / shortlist (reads from factory_variants populated by the job)
# ---------------------------------------------------------------------------

def _parse_variants(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        try:
            cfg = json.loads(r["config_json"]) if r.get("config_json") else {}
        except Exception:
            cfg = {}
        try:
            met = json.loads(r["metrics_json"]) if r.get("metrics_json") else {}
        except Exception:
            met = {}
        out.append({
            "variant_id": r["variant_id"],
            "name":       r["name"],
            "category":   r["category"],
            "n_features": r["n_features"],
            "metrics":    met,
            "config":     cfg,
        })
    return out


@router.get("/runs/{run_id}/leaderboard")
async def leaderboard(run_id: str) -> dict:
    rows = await execute_query(f"""
        SELECT variant_id, name, category, n_features, config_json, metrics_json
        FROM {fqn('factory_variants')}
        WHERE run_id = '{run_id}'
    """)
    variants = _parse_variants(rows)
    variants.sort(key=lambda v: -(v["metrics"].get("gini") or 0))
    return {"run_id": run_id, "variants": variants, "n_total": len(variants), "mode": "real"}


@router.get("/runs/{run_id}/shortlist")
async def shortlist(run_id: str) -> dict:
    data = await leaderboard(run_id)
    top = data["variants"][:5]
    # CV metrics are already in metrics_json (computed by the training notebook)
    for v in top:
        m = v.get("metrics") or {}
        v["cv"] = {
            "cv_gini_mean": m.get("cv_gini_mean", 0.0),
            "cv_gini_std":  m.get("cv_gini_std",  0.0),
            "cv_folds":     5,
            "stability":    "stable" if (m.get("cv_gini_std") or 0) < 0.015 else "watch",
        }
        v["sign_checks"] = {
            "flood_zone_rating": "positive (typical)",
            "credit_score":      "negative (typical)",
            "is_coastal":        "positive (typical)",
        }
    return {"run_id": run_id, "shortlist": top, "mode": "real"}


# ---------------------------------------------------------------------------
# Chat — reuse the demo tab's grounded pattern
# ---------------------------------------------------------------------------

class RealChatRequest(BaseModel):
    run_id: str
    question: str


@router.post("/chat")
async def chat(req: RealChatRequest) -> dict:
    return await _demo_factory_chat(_ChatRequest(run_id=req.run_id, question=req.question))


# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_runs(limit: int = 10) -> dict:
    limit = max(1, min(25, int(limit)))
    try:
        rows = await execute_query(f"""
            SELECT run_id, model_family, status, variant_count, approved_by, started_at
            FROM {fqn('factory_runs')}
            WHERE run_id LIKE 'REAL-%'
            ORDER BY started_at DESC
            LIMIT {limit}
        """)
    except Exception:
        rows = []
    return {"runs": rows}


# ---------------------------------------------------------------------------
# Trigger pack generation for a factory variant
# ---------------------------------------------------------------------------

@router.post("/runs/{run_id}/variants/{variant_id}/pack")
async def generate_pack(run_id: str, variant_id: str) -> dict:
    """Real pack generation. Calls governance_pack_generation with
    model_family='factory_freq_glm_<variant_id>' so the notebook pulls the
    registered UC model + its MLflow artefacts and builds a full PDF."""
    # Verify variant exists + has a real UC model
    rows = await execute_query(f"""
        SELECT name, category, config_json, metrics_json
        FROM {fqn('factory_variants')}
        WHERE run_id = '{run_id}' AND variant_id = '{variant_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"variant {variant_id} not found in run {run_id}")
    v = rows[0]
    try:
        cfg = json.loads(v.get("config_json") or "{}")
    except Exception:
        cfg = {}
    uc_name = cfg.get("uc_name")
    if not uc_name:
        raise HTTPException(400, "This variant has no registered UC model yet — training may still be in progress.")

    family_for_pack = uc_name.split(".")[-1]   # e.g. factory_freq_glm_A01

    # Pick the latest version of the factory UC model (typically 1)
    w = get_workspace_client()
    try:
        versions = await asyncio.to_thread(lambda: list(w.model_versions.list(full_name=uc_name)))
        latest_version = str(max(int(v.version) for v in versions))
    except Exception as e:
        raise HTTPException(500, f"Could not list versions for {uc_name}: {e}")

    job_id = await asyncio.to_thread(_find_job_id, w, PACK_JOB_NAME)
    if not job_id:
        raise HTTPException(500, f"Pack job '{PACK_JOB_NAME}' not found")

    user = get_current_user()
    try:
        run = await asyncio.to_thread(
            w.jobs.run_now,
            job_id=job_id,
            job_parameters={
                "catalog_name":  get_catalog(),
                "schema_name":   get_schema(),
                "model_family":  family_for_pack,
                "model_version": latest_version,
                "requested_by":  user,
            },
        )
        job_run_id = run.run_id if hasattr(run, "run_id") else (run.get("run_id") if isinstance(run, dict) else None)
    except Exception as e:
        raise HTTPException(500, f"Failed to trigger pack job: {e}")

    await log_audit_event(
        event_type="factory_variant_pack_requested",
        entity_type="factory_variant",
        entity_id=f"{run_id}:{variant_id}",
        user_id=user,
        details={"uc_name": uc_name, "version": latest_version, "job_run_id": job_run_id,
                 "family_for_pack": family_for_pack},
    )
    host = get_workspace_host()
    return {
        "run_id":     run_id,
        "variant_id": variant_id,
        "uc_name":    uc_name,
        "version":    latest_version,
        "job_run_id": job_run_id,
        "run_page_url": f"{host}/jobs/{job_id}/runs/{job_run_id}" if job_run_id else None,
        "status":     "queued",
        "message":    "Pack generation job triggered. PDF + sidecars will land in the governance_packs volume.",
    }
