"""Pricing Engine routes.

Live scoring goes through the `pricing_scorer` Model Serving endpoint —
one unified endpoint that bakes in the 4 current champions and returns
all predictions in a single round-trip. The app holds no model code; it
only applies rating-engine arithmetic on top of the endpoint's output.

Historical non-champion scoring stays on the `compare_scoring` job
(batch). Multi-version quote runs against historical versions return a
"needs batch" marker with a link to the Compare & Test flow."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
from datetime import datetime, timezone, date
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import fqn, get_current_user
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/pricing", tags=["pricing"])

FAMILIES = ("freq_glm", "sev_glm", "demand_gbm", "fraud_gbm")
SCORER_ENDPOINT = "pricing_scorer"
COMPARE_JOB_NAME = "v1 — Compare & test models"
HISTORICAL_JOB_NAME = "v1 — Historical quote score (any release)"


# ---------------------------------------------------------------------------
# Pricing engine releases — monthly rate-book snapshots
# ---------------------------------------------------------------------------

def _coerce_release(r: dict) -> dict:
    """SQL API returns everything as strings — nothing to coerce structurally
    here, but strip known-None fields and pass through."""
    return dict(r)


@router.get("/releases")
async def list_releases() -> dict:
    try:
        rows = await execute_query(f"""
            SELECT release_id, display_name, cast(effective_date as string) as effective_date,
                   status, freq_glm_version, sev_glm_version, demand_gbm_version,
                   fraud_gbm_version, rating_engine_version, approved_by, narrative
            FROM {fqn('pricing_engine_releases')}
            ORDER BY effective_date DESC
        """)
    except Exception as e:
        raise HTTPException(500, f"releases query failed: {e}")
    return {"releases": [_coerce_release(r) for r in rows]}


@router.get("/releases/current")
async def current_release() -> dict:
    rows = await execute_query(f"""
        SELECT release_id, display_name, cast(effective_date as string) as effective_date,
               status, freq_glm_version, sev_glm_version, demand_gbm_version,
               fraud_gbm_version, rating_engine_version, approved_by, narrative
        FROM {fqn('pricing_engine_releases')}
        WHERE status = 'champion' ORDER BY effective_date DESC LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, "no pricing engine release marked as champion")
    return _coerce_release(rows[0])


@router.get("/releases/{release_id}")
async def get_release(release_id: str) -> dict:
    rows = await execute_query(f"""
        SELECT release_id, display_name, cast(effective_date as string) as effective_date,
               status, freq_glm_version, sev_glm_version, demand_gbm_version,
               fraud_gbm_version, rating_engine_version, approved_by, narrative
        FROM {fqn('pricing_engine_releases')}
        WHERE release_id = '{release_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"release '{release_id}' not found")
    return _coerce_release(rows[0])


class CompareReleasesRequest(BaseModel):
    release_id:       str                     # historical release to compare against current
    portfolio_size:   int = 2000
    scenario_id:      str = "none"


@router.post("/compare-release")
async def compare_release(req: CompareReleasesRequest) -> dict:
    """Queue a Compare & Test batch job comparing the current champion release
    to the specified historical release (family-by-family)."""
    # Look up both releases
    cur  = await current_release()
    hist = await get_release(req.release_id)

    from server.config import get_workspace_client
    w = get_workspace_client()

    # Locate the compare_scoring job (suffix-match — bundle prefixes names)
    def _find_job() -> int | None:
        try:
            for j in w.jobs.list(name=COMPARE_JOB_NAME, limit=25):
                return j.job_id
        except Exception:
            pass
        try:
            for j in w.jobs.list(limit=100):
                if (j.settings.name or "").endswith(COMPARE_JOB_NAME):
                    return j.job_id
        except Exception:
            pass
        return None
    job_id = await asyncio.to_thread(_find_job)
    if job_id is None:
        raise HTTPException(500, f"Compare & Test job '{COMPARE_JOB_NAME}' not found")

    # Fire ONE compare job per family so each produces its own result row.
    # We parallelise by kicking them all off at once; the UI polls them.
    user = get_current_user()

    def _trigger(family: str, cur_v: str, hist_v: str):
        return w.jobs.run_now(
            job_id=job_id,
            job_parameters={
                "catalog_name":   get_catalog_name(),
                "schema_name":    get_schema_name(),
                "model_family":   family,
                "versions":       f"{cur_v},{hist_v}",
                "portfolio_size": str(req.portfolio_size),
                "scenario_id":    req.scenario_id,
                "requested_by":   f"release-compare:{req.release_id}:{user}",
            },
        )

    triggered = {}
    for fam in FAMILIES:
        try:
            run = await asyncio.to_thread(
                _trigger, fam,
                cur[f"{fam}_version"], hist[f"{fam}_version"],
            )
            triggered[fam] = getattr(run, "run_id", None)
        except Exception as e:
            triggered[fam] = {"error": str(e)[:200]}

    host = get_workspace_host()
    await log_audit_event(
        event_type="release_comparison_triggered",
        entity_type="pricing_engine_release",
        entity_id=req.release_id,
        details={"current_release":  cur["release_id"],
                 "versus_release":   hist["release_id"],
                 "job_id":           job_id,
                 "family_run_ids":   triggered,
                 "portfolio_size":   req.portfolio_size,
                 "scenario_id":      req.scenario_id},
    )

    return {
        "current":      cur,
        "versus":       hist,
        "job_id":       job_id,
        "family_runs":  triggered,
        "run_page_urls": {
            fam: (f"{host}/jobs/{job_id}/runs/{rid}"
                  if host and isinstance(rid, int) else None)
            for fam, rid in triggered.items()
        },
    }


def get_catalog_name():
    from server.config import get_catalog
    return get_catalog()


def get_schema_name():
    from server.config import get_schema
    return get_schema()


def get_workspace_host():
    from server.config import get_workspace_host as _h
    return _h()


# ---------------------------------------------------------------------------
# Historical release scoring — score a single quote on any (older) release
# ---------------------------------------------------------------------------

def _find_historical_job(w) -> int | None:
    """Exact-match, then suffix-match (bundle prefixes names with `[dev X]`).
    Sync helper — callers must wrap with asyncio.to_thread inside async routes."""
    try:
        for j in w.jobs.list(name=HISTORICAL_JOB_NAME, limit=25):
            return j.job_id
    except Exception: pass
    try:
        for j in w.jobs.list(limit=100):
            if (j.settings.name or "").endswith(HISTORICAL_JOB_NAME):
                return j.job_id
    except Exception: pass
    return None


class HistoricalScoreRequest(BaseModel):
    features: dict[str, Any]
    label:    str | None = None


@router.post("/releases/{release_id}/score-quote")
async def score_quote_on_release(release_id: str, req: HistoricalScoreRequest) -> dict:
    """Score a single quote against any (historical) release. Fires the
    `historical_quote_score` job which downloads that release's pinned
    model versions from UC and scores on an ephemeral cluster. Returns
    the job run ids so the UI can poll."""
    # Look up release first so we catch bad IDs before firing a job
    rel = await get_release(release_id)  # raises 404 if missing

    from server.config import get_workspace_client
    w = get_workspace_client()
    job_id = await asyncio.to_thread(_find_historical_job, w)
    if job_id is None:
        raise HTTPException(500, f"job '{HISTORICAL_JOB_NAME}' not found — run `databricks bundle deploy`")

    user = get_current_user()
    feat_json = json.dumps(req.features or {})

    try:
        run = await asyncio.to_thread(
            w.jobs.run_now,
            job_id=job_id,
            job_parameters={
                "catalog_name":  get_catalog_name(),
                "schema_name":   get_schema_name(),
                "release_id":    release_id,
                "features_json": feat_json,
                "run_label":     req.label or f"ui:{user}",
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to trigger historical score job: {e}")

    run_id = getattr(run, "run_id", None)
    host   = get_workspace_host()

    await log_audit_event(
        event_type="historical_quote_score_triggered",
        entity_type="pricing_engine_release",
        entity_id=release_id,
        details={"job_id": job_id, "run_id": run_id, "label": req.label,
                 "features_preview": {k: str(v)[:60] for k, v in list((req.features or {}).items())[:8]}},
    )

    return {
        "release":      rel,
        "job_id":       job_id,
        "run_id":       run_id,
        "run_page_url": f"{host}/jobs/{job_id}/runs/{run_id}" if host and run_id else None,
    }


@router.get("/historical-score/{run_id}")
async def historical_score_status(run_id: int) -> dict:
    """Poll a historical-score run. When the run SUCCEEDS, pulls the
    notebook-output JSON (the result dict from dbutils.notebook.exit)
    and returns it inline so the UI has a single endpoint to watch."""
    from server.config import get_workspace_client
    w = get_workspace_client()
    try:
        r = await asyncio.to_thread(w.jobs.get_run, run_id=run_id)
    except Exception as e:
        raise HTTPException(500, f"Could not fetch run {run_id}: {e}")

    state = r.state.life_cycle_state if r.state else None
    result_state = r.state.result_state if r.state else None
    out = {
        "run_id":       run_id,
        "state":        str(state).split(".")[-1] if state else "UNKNOWN",
        "result_state": str(result_state).split(".")[-1] if result_state else None,
    }

    if str(state).endswith("TERMINATED") and str(result_state).endswith("SUCCESS"):
        # Pull the notebook result (the JSON our notebook exits with)
        try:
            task_run_id = r.tasks[0].run_id if r.tasks else run_id
            output = await asyncio.to_thread(w.jobs.get_run_output, run_id=task_run_id)
            raw = getattr(output, "notebook_output", None)
            text = raw.result if raw and raw.result else None
            if text:
                try:
                    out["result"] = json.loads(text)
                except Exception:
                    out["result_raw"] = text[:2000]
        except Exception as e:
            logger.warning("notebook output fetch failed: %s", e)
            out["fetch_error"] = str(e)[:200]

    return out


def _num(v, default=0.0):
    try: return float(v) if v is not None else default
    except (TypeError, ValueError): return default


# ---------------------------------------------------------------------------
# Serving-endpoint scoring — the live pricing runtime
# ---------------------------------------------------------------------------

async def _score_via_inference_logs(policy_id: str) -> dict | None:
    """Fallback when the live `pricing_scorer` endpoint isn't deployed on
    this workspace (e.g. dev). Returns the most recent batch-scored row
    from `inference_logs` so the MTA / Quote Runner still has model
    predictions to drive the rating engine."""
    try:
        rows = await execute_query(f"""
            SELECT freq_pred, sev_pred, demand_pred, fraud_pred,
                   base_premium, fraud_loading, demand_adj,
                   technical_premium,
                   freq_version, sev_version, demand_version, fraud_version
            FROM {fqn('inference_logs')}
            WHERE policy_id = '{policy_id}'
            ORDER BY scored_at DESC
            LIMIT 1
        """)
    except Exception as e:
        logger.info("inference_logs lookup for %s failed: %s", policy_id, e)
        return None
    if not rows:
        return None
    r = rows[0]
    return {
        "freq_pred":   _num(r.get("freq_pred")),
        "sev_pred":    _num(r.get("sev_pred")),
        "demand_pred": _num(r.get("demand_pred")),
        "fraud_pred":  _num(r.get("fraud_pred")),
        "fraud_load":  _num(r.get("fraud_loading")),
        "demand_adj":  _num(r.get("demand_adj")),
        "technical_premium": _num(r.get("technical_premium")),
        # We synthesise final_premium from technical + the in-process
        # rating-engine config later; the endpoint normally returns it
        # directly. Leave None to force the recompute path.
        "final_premium":         None,
        "rating_engine_version": None,
        "_source": "inference_logs",
    }


async def _score_via_endpoint(policy_id: str) -> dict | None:
    """Call the unified `pricing_scorer` Model Serving endpoint with a
    policy_id. The endpoint resolves features against UPT (online store at
    serving time), runs the 4 champions and applies the baked rating-engine
    business rules — returning final_premium plus every intermediate value.
    Returns None if the endpoint is unavailable so the caller can show a
    clear 'offline' response."""
    import asyncio, requests as _rq
    from server.config import get_workspace_client

    def _blocking() -> dict | None:
        try:
            w = get_workspace_client()
            host  = w.config.host.rstrip("/")
            token = w.config._header_factory()
            resp = _rq.post(
                f"{host}/serving-endpoints/{SCORER_ENDPOINT}/invocations",
                headers={**token, "Content-Type": "application/json"},
                json={"dataframe_records": [{"policy_id": policy_id}]},
                timeout=120,
            )
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            logger.warning("pricing_scorer endpoint call failed: %s", str(e)[:200])
            return None
        preds = data.get("predictions") or data.get("outputs") or data
        # serving wraps result as list of dicts (records) or dict of lists
        if isinstance(preds, list) and preds:
            row = preds[0]
        elif isinstance(preds, dict):
            # dict-of-lists shape
            try:
                row = {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}
            except Exception:
                row = {}
        else:
            row = {}
        return row or None

    return await asyncio.to_thread(_blocking)


# ---------------------------------------------------------------------------
# Rating engine config
# ---------------------------------------------------------------------------

def _coerce_config(cfg: dict) -> dict:
    numeric = ["expense_loading_pct", "commission_bp",
               "fraud_loading_pct", "fraud_loading_threshold",
               "demand_adj_pct", "demand_adj_threshold_lo",
               "demand_adj_threshold_hi", "min_premium", "max_premium"]
    out = dict(cfg)
    for k in numeric:
        if k in out and out[k] is not None:
            try: out[k] = float(out[k])
            except (TypeError, ValueError): pass
    if "commission_bp" in out and out["commission_bp"] is not None:
        try: out["commission_bp"] = int(float(out["commission_bp"]))
        except (TypeError, ValueError): pass
    return out


async def _current_config() -> dict:
    rows = await execute_query(f"""
        SELECT * FROM {fqn('rating_engine_config')}
        WHERE status = 'champion' ORDER BY effective_date DESC LIMIT 1
    """)
    if rows:
        return rows[0]
    return {
        "version": "bootstrap", "status": "champion",
        "expense_loading_pct": 20.0, "commission_bp": 1500,
        "fraud_loading_pct": 5.0, "fraud_loading_threshold": 0.25,
        "demand_adj_pct": 2.0, "demand_adj_threshold_lo": 0.40,
        "demand_adj_threshold_hi": 0.75,
        "min_premium": 150.0, "max_premium": 250_000.0,
        "narrative": "(no rating_engine_config rows found — using defaults)",
    }


async def _config_row_by_version(version: str) -> dict | None:
    rows = await execute_query(f"""
        SELECT * FROM {fqn('rating_engine_config')}
        WHERE version = '{version}' LIMIT 1
    """)
    return rows[0] if rows else None


@router.get("/rating-config/current")
async def get_rating_config_current() -> dict:
    try: return _coerce_config(await _current_config())
    except Exception as e: raise HTTPException(500, f"rating-config query failed: {e}")


@router.get("/rating-config/history")
async def get_rating_config_history() -> dict:
    try:
        rows = await execute_query(f"""
            SELECT version, status, cast(effective_date as string) as effective_date,
                   expense_loading_pct, commission_bp, fraud_loading_pct,
                   fraud_loading_threshold, demand_adj_pct,
                   min_premium, max_premium, approved_by, narrative
            FROM {fqn('rating_engine_config')}
            ORDER BY effective_date DESC
        """)
    except Exception as e:
        raise HTTPException(500, f"rating-config history failed: {e}")
    return {"versions": [_coerce_config(r) for r in rows]}


# ---------------------------------------------------------------------------
# Model versions (UI picker)
# ---------------------------------------------------------------------------

import threading as _threading
_ALIAS_CACHE: dict[tuple[str, str], tuple[float, str | None]] = {}
_ALIAS_TTL_S = 30.0
_ALIAS_LOCK = _threading.Lock()  # the cache is read/written from the thread pool


def _resolve_alias_to_version(family: str, alias: str) -> str | None:
    """Sync; cached for 30s. Aliases only change on promote/rollback so this
    is safe — and on those events we explicitly invalidate (see _bust_alias_cache).
    Caller is responsible for wrapping in asyncio.to_thread inside async routes."""
    import time
    key = (family, alias)
    now = time.time()
    with _ALIAS_LOCK:
        cached = _ALIAS_CACHE.get(key)
    if cached and now < cached[0]:
        return cached[1]
    from server.config import get_workspace_client, get_catalog, get_schema
    try:
        w = get_workspace_client()
        mv = w.model_versions.get_by_alias(
            full_name=f"{get_catalog()}.{get_schema()}.{family}",
            alias=alias,
        )
        version = str(mv.version) if mv else None
    except Exception as e:
        logger.info("alias %s for %s missing: %s", alias, family, e)
        version = None
    with _ALIAS_LOCK:
        _ALIAS_CACHE[key] = (now + _ALIAS_TTL_S, version)
    return version


def _bust_alias_cache() -> None:
    """Called by deployment.py after promote/rollback so the next /status reads
    the fresh alias instead of the cached version."""
    with _ALIAS_LOCK:
        _ALIAS_CACHE.clear()


@router.get("/status")
async def pricing_status() -> dict:
    """Reports champion aliases AND the pricing_scorer endpoint state. UI
    gates the Run button on `ready`, which is True only when the endpoint
    is READY (not provisioning, not down)."""
    # Resolve all 4 champion aliases in parallel against the thread pool —
    # cache hits return instantly; cache misses go to UC concurrently.
    champions_list = await asyncio.gather(*[
        asyncio.to_thread(_resolve_alias_to_version, fam, "champion") for fam in FAMILIES
    ])
    champions = dict(zip(FAMILIES, champions_list))
    endpoint_state = None
    endpoint_ready = False
    try:
        from server.config import get_workspace_client
        w = get_workspace_client()
        ep = await asyncio.to_thread(w.serving_endpoints.get, SCORER_ENDPOINT)
        endpoint_state = str(ep.state.ready).split(".")[-1] if ep.state and ep.state.ready else "UNKNOWN"
        endpoint_ready = "READY" in str(endpoint_state)
    except Exception as e:
        logger.info("pricing_scorer endpoint lookup failed: %s", e)
    return {
        "champions":       champions,
        "endpoint":        SCORER_ENDPOINT,
        "endpoint_state":  endpoint_state,
        "ready":           endpoint_ready,
    }


@router.get("/model-versions")
async def list_model_versions() -> dict:
    from server.config import get_workspace_client, get_catalog, get_schema
    w = get_workspace_client()
    cat, sch = get_catalog(), get_schema()

    # Fan out: list versions + resolve aliases for all 4 families in parallel.
    # Each family's sync work (one MLflow list + 2 cached alias reads) runs
    # entirely on a thread so the event loop stays free.
    def _family_block(fam: str) -> dict:
        full_name = f"{cat}.{sch}.{fam}"
        champion = _resolve_alias_to_version(fam, "champion")
        previous = _resolve_alias_to_version(fam, "previous_champion")
        try:
            vs = list(w.model_versions.list(full_name=full_name))
            versions = [
                {
                    "version":    str(v.version),
                    "run_id":     v.run_id,
                    "created_at": v.created_at,
                    "is_champion":          str(v.version) == champion,
                    "is_previous_champion": str(v.version) == previous,
                }
                for v in sorted(vs, key=lambda x: int(x.version), reverse=True)
            ]
        except Exception as e:
            logger.warning("list versions for %s failed: %s", fam, e)
            versions = []
        return {
            "family":            fam,
            "champion":          champion,
            "previous_champion": previous,
            "versions":          versions,
        }

    blocks = await asyncio.gather(*[asyncio.to_thread(_family_block, f) for f in FAMILIES])
    return dict(zip(FAMILIES, blocks))


# ---------------------------------------------------------------------------
# Rating engine application
# ---------------------------------------------------------------------------

def _apply_rating_engine(cfg: dict, freq, sev, fraud, demand) -> dict:
    base = float(freq) * float(sev)
    fraud_trigger = _num(cfg.get("fraud_loading_threshold"), 0.25)
    fraud_pct     = _num(cfg.get("fraud_loading_pct"), 0.0)
    fraud_loading = base * (fraud_pct / 100.0) if (fraud or 0) > fraud_trigger else 0.0

    dlo = _num(cfg.get("demand_adj_threshold_lo"), 0.40)
    dhi = _num(cfg.get("demand_adj_threshold_hi"), 0.75)
    adj_pct = _num(cfg.get("demand_adj_pct"), 0.0)
    if demand is None: demand_adj = 0.0
    elif demand < dlo: demand_adj = base * (adj_pct / 100.0)
    elif demand > dhi: demand_adj = -base * (adj_pct / 100.0)
    else:               demand_adj = 0.0

    technical = base + fraud_loading + demand_adj
    expense   = technical * _num(cfg.get("expense_loading_pct"), 0) / 100.0
    with_exp  = technical + expense
    commission = with_exp * _num(cfg.get("commission_bp"), 0) / 10_000.0
    gross      = with_exp + commission
    gross      = max(_num(cfg.get("min_premium"), 0),
                     min(_num(cfg.get("max_premium"), 1e12), gross))
    return {
        "base_premium":      round(base, 2),
        "fraud_loading":     round(fraud_loading, 2),
        "demand_adj":        round(demand_adj, 2),
        "technical_premium": round(technical, 2),
        "expense_loading":   round(expense, 2),
        "commission":        round(commission, 2),
        "gross_premium":     round(gross, 2),
    }


# ---------------------------------------------------------------------------
# Quote Runner
# ---------------------------------------------------------------------------

class QuoteRunRequest(BaseModel):
    features: dict[str, Any]
    policy_id: str | None = None
    model_versions: dict[str, list[str]] | None = None
    rating_engine_version: str | None = None
    label: str | None = None


async def _run_quote(features: dict, policy_id: str | None,
                     versions: dict[str, list[str]], cfg: dict) -> list[dict]:
    """Champion combos go through the live `pricing_scorer` endpoint, which
    needs a policy_id to look up features against UPT. Non-champion combos
    return a `needs_batch` marker — those cost a batch job to score."""
    import itertools

    champions = {fam: _resolve_alias_to_version(fam, "champion") for fam in FAMILIES}
    if not all(champions.values()):
        raise HTTPException(500, f"missing champion alias for one of: {FAMILIES}")

    # Resolve version lists — default to champion
    resolved: dict[str, list[str]] = {}
    for fam in FAMILIES:
        vs = (versions or {}).get(fam) or []
        resolved[fam] = vs or [champions[fam]]

    # Endpoint needs a policy_id (FeatureLookup key). The Quote Runner usually
    # has one — fall back to features["policy_id"] if the caller didn't pass it.
    pid = (policy_id or features.get("policy_id") or "").strip().upper() or None

    champion_row = None
    if pid and all(champions[f] in resolved[f] for f in FAMILIES):
        champion_row = await _score_via_endpoint(pid)
        if not champion_row:
            # No live endpoint on this workspace — fall back to the last
            # batch-scored row from inference_logs so the rating engine
            # still has freq/sev/demand/fraud predictions to work with.
            champion_row = await _score_via_inference_logs(pid)

    out = []
    for vf, vs_, vd, vfr in itertools.product(
        resolved["freq_glm"], resolved["sev_glm"],
        resolved["demand_gbm"], resolved["fraud_gbm"],
    ):
        is_champion_combo = (
            vf  == champions["freq_glm"]  and vs_ == champions["sev_glm"] and
            vd  == champions["demand_gbm"] and vfr == champions["fraud_gbm"])
        if is_champion_combo:
            if champion_row:
                freq   = _num(champion_row.get("freq_pred"))
                sev    = _num(champion_row.get("sev_pred"))
                demand = _num(champion_row.get("demand_pred"))
                fraud  = _num(champion_row.get("fraud_pred"))
                # The endpoint has already applied rules with its baked rating
                # engine config. If the caller asked for a different rating
                # engine version, recompute against `cfg` instead — same freq /
                # sev / demand / fraud predictions, different rules.
                endpoint_re = champion_row.get("rating_engine_version")
                if endpoint_re and endpoint_re == cfg.get("version"):
                    technical = _num(champion_row.get("technical_premium"))
                    fraud_load = _num(champion_row.get("fraud_load"))
                    demand_adj = _num(champion_row.get("demand_adj"))
                    final_premium = _num(champion_row.get("final_premium"))
                    base = float(freq) * float(sev)
                    expense    = base * _num(cfg.get("expense_loading_pct"), 0) / 100.0
                    commission = (base + expense) * _num(cfg.get("commission_bp"), 0) / 10_000.0
                    price_buildup = {
                        "base_premium":      round(base, 2),
                        "fraud_loading":     round(fraud_load, 2),
                        "demand_adj":        round(demand_adj, 2),
                        "technical_premium": round(technical, 2),
                        "expense_loading":   round(expense, 2),
                        "commission":        round(commission, 2),
                        "gross_premium":     round(final_premium, 2),
                    }
                else:
                    price_buildup = _apply_rating_engine(cfg, freq, sev, fraud, demand)

                out.append({
                    "model_versions": {"freq_glm": vf, "sev_glm": vs_,
                                        "demand_gbm": vd, "fraud_gbm": vfr},
                    "predictions": {
                        "freq_pred":   round(freq, 6),
                        "sev_pred":    round(sev, 2),
                        "demand_pred": round(demand, 6),
                        "fraud_pred":  round(fraud, 6),
                    },
                    "price_buildup": price_buildup,
                    "source":        "live_endpoint",
                })
            elif not pid:
                out.append({
                    "model_versions": {"freq_glm": vf, "sev_glm": vs_,
                                        "demand_gbm": vd, "fraud_gbm": vfr},
                    "predictions":    None,
                    "price_buildup":  None,
                    "source":         "policy_id_required",
                    "note":           (f"The {SCORER_ENDPOINT} endpoint needs a policy_id "
                                        f"to resolve features. Pass `policy_id` in the request."),
                })
            else:
                out.append({
                    "model_versions": {"freq_glm": vf, "sev_glm": vs_,
                                        "demand_gbm": vd, "fraud_gbm": vfr},
                    "predictions":    None,
                    "price_buildup":  None,
                    "source":         "endpoint_unavailable",
                    "note":           (f"The {SCORER_ENDPOINT} serving endpoint isn't responding "
                                        f"(cold start or down). Try again in ~60s."),
                })
        else:
            out.append({
                "model_versions": {"freq_glm": vf, "sev_glm": vs_,
                                    "demand_gbm": vd, "fraud_gbm": vfr},
                "predictions":    None,
                "price_buildup":  None,
                "source":         "needs_batch",
                "note": ("Historical (non-champion) versions are not on the live endpoint. "
                         "Use Compare & Test to score these — ~2 min batch run."),
            })
    return out


@router.post("/quote/run")
async def run_quote(req: QuoteRunRequest) -> dict:
    cfg = (await _config_row_by_version(req.rating_engine_version)
           if req.rating_engine_version else await _current_config())
    if not cfg:
        raise HTTPException(404, f"rating_engine_config version '{req.rating_engine_version}' not found")
    cfg = _coerce_config(cfg)

    results = await _run_quote(req.features, req.policy_id, req.model_versions or {}, cfg)

    await log_audit_event(
        event_type="quote_run",
        entity_type="quote",
        entity_id=(req.label or "adhoc"),
        details={
            "features_preview":      {k: str(v)[:60] for k, v in list(req.features.items())[:8]},
            "n_combinations":        len(results),
            "rating_engine_version": cfg.get("version"),
            "model_versions":        req.model_versions,
            "scoring_engine":        "deterministic_formula_v1",
        },
    )

    return {
        "rating_engine":  {
            "version":                 cfg.get("version"),
            "status":                  cfg.get("status"),
            "expense_loading_pct":     cfg.get("expense_loading_pct"),
            "commission_bp":           cfg.get("commission_bp"),
            "fraud_loading_pct":       cfg.get("fraud_loading_pct"),
            "fraud_loading_threshold": cfg.get("fraud_loading_threshold"),
            "demand_adj_pct":          cfg.get("demand_adj_pct"),
            "min_premium":             cfg.get("min_premium"),
            "max_premium":             cfg.get("max_premium"),
        },
        "quotes":         results,
        "n_combinations": len(results),
        "scored_at":      datetime.now(timezone.utc).isoformat(),
        "scoring_engine": "deterministic_formula_v1",
    }


# ---------------------------------------------------------------------------
# Policy context — used by the MTA flow to surface inception + release-of-record
# ---------------------------------------------------------------------------

async def _release_at_or_before(d: date) -> dict | None:
    """The pricing release in force on date `d`. Most recent release whose
    effective_date <= d. Falls back to the earliest release if d predates all
    releases (so we always have something to price on)."""
    rows = await execute_query(f"""
        SELECT release_id, display_name, cast(effective_date as string) AS effective_date,
               status, freq_glm_version, sev_glm_version, demand_gbm_version,
               fraud_gbm_version, rating_engine_version
        FROM {fqn('pricing_engine_releases')}
        WHERE effective_date <= DATE'{d.isoformat()}'
        ORDER BY effective_date DESC LIMIT 1
    """)
    if rows:
        return dict(rows[0])
    fallback = await execute_query(f"""
        SELECT release_id, display_name, cast(effective_date as string) AS effective_date,
               status, freq_glm_version, sev_glm_version, demand_gbm_version,
               fraud_gbm_version, rating_engine_version
        FROM {fqn('pricing_engine_releases')}
        ORDER BY effective_date ASC LIMIT 1
    """)
    return dict(fallback[0]) if fallback else None


def _release_versions(rel: dict) -> dict:
    return {
        "freq_glm":      rel.get("freq_glm_version"),
        "sev_glm":       rel.get("sev_glm_version"),
        "demand_gbm":    rel.get("demand_gbm_version"),
        "fraud_gbm":     rel.get("fraud_gbm_version"),
        "rating_engine": rel.get("rating_engine_version"),
    }


@router.get("/policy-context/{policy_id}")
async def policy_context(policy_id: str) -> dict:
    """Return the policy's inception/renewal dates plus the pricing release
    in force at inception (release-of-record) and today's live champion. Used
    by the MTA flow so the UI can show 'this policy was bound on Feb 2026,
    we'll re-price using that release; the live release shown for reference'."""
    policy_id = policy_id.strip().upper()
    rows = await execute_query(f"""
        SELECT policy_id,
               cast(inception_date AS string) AS inception_date,
               cast(renewal_date   AS string) AS renewal_date,
               current_premium, industry_risk_tier, region, construction_type
        FROM {fqn('unified_pricing_table_live')}
        WHERE policy_id = '{policy_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"policy {policy_id} not found")
    p = rows[0]

    try:
        inception = date.fromisoformat(str(p["inception_date"])[:10])
    except Exception:
        inception = date.today()

    inception_release = await _release_at_or_before(inception) or {}
    cur_rows = await execute_query(f"""
        SELECT release_id, display_name, cast(effective_date as string) AS effective_date,
               status, freq_glm_version, sev_glm_version, demand_gbm_version,
               fraud_gbm_version, rating_engine_version
        FROM {fqn('pricing_engine_releases')}
        WHERE status = 'champion' ORDER BY effective_date DESC LIMIT 1
    """)
    current = dict(cur_rows[0]) if cur_rows else {}

    return {
        "policy_id":         policy_id,
        "inception_date":    p.get("inception_date"),
        "renewal_date":      p.get("renewal_date"),
        "current_premium":   p.get("current_premium"),
        "industry_risk_tier":p.get("industry_risk_tier"),
        "region":            p.get("region"),
        "construction_type": p.get("construction_type"),
        "inception_release": {
            **inception_release,
            "model_versions": _release_versions(inception_release),
        } if inception_release else None,
        "current_release": {
            **current,
            "model_versions": _release_versions(current),
        } if current else None,
    }


# ---------------------------------------------------------------------------
# MTA — repricing for mid-term adjustments
# ---------------------------------------------------------------------------

class MtaRequest(BaseModel):
    policy_id: str
    changes:   dict[str, Any]
    effective_date: str | None = None
    reason: str | None = None


def _release_calibration_factor(release_eff: date | None, current_eff: date | None) -> float:
    """Premium multiplier applied to the live-engine quote so a historical
    release prices ~1.2% higher per month older. Mirrors the recalibration
    drift narrative — each refit bringing rates down. Returns 1.0 if the
    release IS current (or dates missing)."""
    if not release_eff or not current_eff:
        return 1.0
    months = max(0.0, (current_eff.year - release_eff.year) * 12
                       + (current_eff.month - release_eff.month)
                       + (current_eff.day - release_eff.day) / 30.0)
    return round(1.0 + 0.012 * months, 4)


def _apply_calibration(quote: dict, factor: float) -> dict:
    """Return a copy of a quote with the price_buildup multiplied by `factor`.
    Predictions stay the same (we're emulating model recalibration on the
    rating side, deterministically)."""
    if not quote.get("price_buildup"):
        return quote
    pb = dict(quote["price_buildup"])
    for k, v in list(pb.items()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            pb[k] = round(v * factor, 2)
    out = dict(quote)
    out["price_buildup"] = pb
    return out


@router.post("/mta/simulate")
async def simulate_mta(req: MtaRequest) -> dict:
    policy_id = req.policy_id.strip().upper()
    rows = await execute_query(f"""
        SELECT * FROM {fqn('unified_pricing_table_live')}
        WHERE policy_id = '{policy_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"policy {policy_id} not found")
    policy = rows[0]

    before = dict(policy)
    after  = dict(policy)
    after.update(req.changes or {})

    # Live-engine quote (current champion). With the FeatureLookup-based
    # scorer the endpoint reads features from UPT by policy_id — the in-memory
    # `after` modifications won't be reflected. We score `before` on the
    # endpoint then apply a sum-insured-ratio heuristic to derive `after`,
    # which captures the dominant linear effect of MTA changes (capacity,
    # contents, liability) on technical premium.
    cfg = _coerce_config(await _current_config())
    quotes_before = await _run_quote(before, policy_id, {}, cfg)
    q_before = quotes_before[0]
    if not q_before.get("price_buildup"):
        raise HTTPException(
            503,
            "Could not price this policy — no live scorer endpoint and no "
            "prior inference_logs row to fall back on. Run a batch scoring "
            "job first, or deploy the pricing_scorer endpoint.",
        )

    si_before = _num(before.get("sum_insured"))
    si_after  = _num(after.get("sum_insured"))
    mta_factor = (si_after / si_before) if si_before > 0 and si_after > 0 else 1.0
    q_after = _apply_calibration(q_before, mta_factor)
    quotes_after = [q_after]

    try:
        eff = date.fromisoformat(req.effective_date) if req.effective_date else date.today()
    except Exception:
        eff = date.today()
    inception_raw = policy.get("inception_date") or policy.get("policy_inception_date")
    renewal_raw   = policy.get("renewal_date")   or policy.get("policy_renewal_date")
    try:
        inception = date.fromisoformat(str(inception_raw)[:10]) if inception_raw else eff
    except Exception:
        inception = eff
    try:
        renewal = date.fromisoformat(str(renewal_raw)[:10]) if renewal_raw else date(eff.year + 1, eff.month, min(28, eff.day))
    except Exception:
        renewal = date(eff.year + 1, eff.month, min(28, eff.day))

    term_days      = max(1, (renewal - inception).days)
    remaining_days = max(0, min(term_days, (renewal - eff).days))
    remaining_frac = remaining_days / term_days if term_days > 0 else 0

    # Resolve release-of-record vs current champion. The release-of-record is
    # the release in force at inception — the consistent re-pricing target.
    inception_release = await _release_at_or_before(inception) or {}
    cur_rows = await execute_query(f"""
        SELECT release_id, display_name, cast(effective_date as string) AS effective_date,
               status, freq_glm_version, sev_glm_version, demand_gbm_version,
               fraud_gbm_version, rating_engine_version
        FROM {fqn('pricing_engine_releases')}
        WHERE status = 'champion' ORDER BY effective_date DESC LIMIT 1
    """)
    current_release = dict(cur_rows[0]) if cur_rows else {}

    # Calibration factor for the inception release (1.0 if same as current)
    try:
        i_eff = date.fromisoformat(str(inception_release.get("effective_date"))[:10])
    except Exception:
        i_eff = None
    try:
        c_eff = date.fromisoformat(str(current_release.get("effective_date"))[:10])
    except Exception:
        c_eff = None
    factor = _release_calibration_factor(i_eff, c_eff)

    # Two re-prices: on the policy's release-of-record, and on today's live release.
    on_inception_before = _apply_calibration(q_before, factor)
    on_inception_after  = _apply_calibration(q_after,  factor)
    on_current_before   = q_before
    on_current_after    = q_after

    def _block(b, a):
        full_delta = (a["price_buildup"]["gross_premium"] - b["price_buildup"]["gross_premium"]) if (a.get("price_buildup") and b.get("price_buildup")) else 0
        return {
            "before":         b,
            "after":          a,
            "full_delta":     round(full_delta, 2),
            "prorated_delta": round(full_delta * remaining_frac, 2),
        }

    on_inception = _block(on_inception_before, on_inception_after)
    on_current   = _block(on_current_before,   on_current_after)

    user = get_current_user()
    version_chain = dict(q_after["model_versions"])
    version_chain["rating_engine"] = cfg.get("version")

    await log_audit_event(
        event_type="mta_simulated",
        entity_type="policy",
        entity_id=policy_id,
        details={
            "effective_date":     eff.isoformat(),
            "reason":             (req.reason or "")[:200],
            "changes":            {k: str(v)[:80] for k, v in (req.changes or {}).items()},
            "inception_release":  inception_release.get("release_id"),
            "current_release":    current_release.get("release_id"),
            "calibration_factor": factor,
            "before_premium":     q_before["price_buildup"]["gross_premium"],
            "after_premium":      q_after["price_buildup"]["gross_premium"],
            "full_delta":         on_current["full_delta"],
            "prorated_delta":     on_current["prorated_delta"],
            "term_days":          term_days,
            "remaining_days":     remaining_days,
            "version_chain":      version_chain,
        },
    )

    return {
        "policy_id":         policy_id,
        "inception_date":    inception.isoformat(),
        "renewal_date":      renewal.isoformat(),
        "effective_date":    eff.isoformat(),
        "reason":            req.reason,
        "term_days":         term_days,
        "remaining_days":    remaining_days,
        "remaining_frac":    round(remaining_frac, 3),
        "calibration_factor": factor,
        "inception_release": {
            **inception_release,
            "model_versions": _release_versions(inception_release),
        } if inception_release else None,
        "current_release": {
            **current_release,
            "model_versions": _release_versions(current_release),
        } if current_release else None,
        "on_inception_release": on_inception,
        "on_current_release":   on_current,
        # Backwards-compat fields used by older clients
        "before":         q_before,
        "after":          q_after,
        "full_delta":     on_current["full_delta"],
        "prorated_delta": on_current["prorated_delta"],
        "version_chain":  version_chain,
        "user":           user,
    }
