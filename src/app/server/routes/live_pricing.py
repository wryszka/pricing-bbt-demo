"""Live Pricing System — backend route.

Bring the live pricing demo up and down on demand. The live demo is a
small ON/OFF stack:

  Lakebase online store + CONTINUOUS publish from UPT
    → pricing_scorer Model Serving endpoint (route_optimised, no scale-zero)
      → live_pricing_metrics for the load-test chart

The route exposes:

  GET  /api/live-pricing/status           current state of the stack
  POST /api/live-pricing/start            fire provision job
  POST /api/live-pricing/stop             fire teardown job
  POST /api/live-pricing/quote            single low-latency quote
  POST /api/live-pricing/claim            file claim + MERGE UPT inline
  GET  /api/live-pricing/claim/{run_id}   poll an async refresh (kept for
                                          parity with other long-running flows)
  POST /api/live-pricing/load-test/start  fire load-test job
  POST /api/live-pricing/load-test/stop   cancel an in-flight load test
  GET  /api/live-pricing/load-test/metrics?since=<iso8601>
                                          per-second QPS/p50/p95/p99 chart data

All sync SDK calls are wrapped in `asyncio.to_thread` to keep the single
uvicorn worker non-blocking under multi-user load.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import (
    fqn, get_catalog, get_schema, get_current_user,
    get_workspace_client, get_workspace_host,
)
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/live-pricing", tags=["live-pricing"])

ENDPOINT_NAME       = "pricing_scorer"
ONLINE_STORE_NAME   = "pricing-upt-online-store-live"
PROVISION_JOB_NAME  = "v1 — Live pricing: provision (online store + endpoint + warm-up)"
TEARDOWN_JOB_NAME   = "v1 — Live pricing: teardown (delete endpoint + online store)"
LOAD_TEST_JOB_NAME  = "v1 — Live pricing: load test (sustained QPS against scorer)"
REFRESH_JOB_NAME    = "v1 — Live pricing: file claim + refresh UPT"


def _find_job_by_name(name: str) -> int | None:
    """Exact-match first, then suffix-match — bundle prefixes job names with
    `[dev whoami]` in the dev target. Sync helper; wrap in asyncio.to_thread."""
    w = get_workspace_client()
    try:
        for j in w.jobs.list(name=name, limit=25):
            return j.job_id
    except Exception:
        pass
    try:
        for j in w.jobs.list(limit=200):
            if (j.settings.name or "").endswith(name):
                return j.job_id
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

@router.get("/status")
async def status() -> dict:
    """Snapshot of the live pricing stack — feeds the on/off header in the UI.

    State machine:
      off       — endpoint absent or NOT_DEPLOYED
      starting  — config update in progress, or endpoint exists but not READY
      on        — endpoint READY and config_update is none
      stopping  — teardown job is running (best-effort signal)
    """
    def _endpoint_state() -> dict:
        try:
            w = get_workspace_client()
            ep = w.serving_endpoints.get(ENDPOINT_NAME)
            ready  = str(getattr(ep.state, "ready", "")).split(".")[-1]
            update = str(getattr(ep.state, "config_update", "")).split(".")[-1]
            return {"present": True, "ready": ready, "config_update": update}
        except Exception as e:
            return {"present": False, "ready": None, "config_update": None,
                    "error": str(e)[:200]}

    def _online_store_state() -> dict:
        try:
            w = get_workspace_client()
            store = w.feature_store.get_online_store(ONLINE_STORE_NAME)
            return {"present": True,
                    "name":    ONLINE_STORE_NAME,
                    "state":   str(getattr(store, "state", "")).split(".")[-1],
                    "capacity": str(getattr(store, "capacity", ""))}
        except Exception as e:
            return {"present": False, "name": ONLINE_STORE_NAME,
                    "error": str(e)[:200]}

    ep_state, store_state = await asyncio.gather(
        asyncio.to_thread(_endpoint_state),
        asyncio.to_thread(_online_store_state),
    )

    endpoint_ready = ep_state["present"] and ep_state.get("ready") == "READY" and \
                     (ep_state.get("config_update") in (None, "", "NOT_UPDATING"))
    store_ready    = store_state["present"] and (store_state.get("state") or "") == "AVAILABLE"

    if endpoint_ready and store_ready:
        state = "on"
    elif ep_state.get("config_update") == "UPDATE_FAILED":
        state = "error"
    elif ep_state["present"] or store_state["present"]:
        state = "starting"
    else:
        state = "off"

    return {
        "state":         state,
        "endpoint":      {"name": ENDPOINT_NAME, **ep_state},
        "online_store":  store_state,
        "metrics_table": fqn("live_pricing_metrics"),
    }


# ---------------------------------------------------------------------------
# Start / stop
# ---------------------------------------------------------------------------

async def _trigger_job(job_name: str, params: dict) -> dict:
    job_id = await asyncio.to_thread(_find_job_by_name, job_name)
    if job_id is None:
        raise HTTPException(500, f"job '{job_name}' not found — run `databricks bundle deploy`")

    def _run():
        w = get_workspace_client()
        return w.jobs.run_now(job_id=job_id, job_parameters=params)

    try:
        run = await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(500, f"failed to trigger {job_name}: {e}")

    run_id = getattr(run, "run_id", None)
    host   = get_workspace_host()
    return {
        "job_id":       job_id,
        "run_id":       run_id,
        "run_page_url": f"{host}/jobs/{job_id}/runs/{run_id}" if host and run_id else None,
    }


@router.post("/start")
async def start() -> dict:
    user = get_current_user()
    triggered = await _trigger_job(PROVISION_JOB_NAME, {
        "catalog_name":      get_catalog(),
        "schema_name":       get_schema(),
        "online_store_name": ONLINE_STORE_NAME,
        "endpoint_name":     ENDPOINT_NAME,
    })
    await log_audit_event(
        event_type="live_pricing_start_requested",
        entity_type="endpoint",
        entity_id=ENDPOINT_NAME,
        details={"job_id": triggered["job_id"], "run_id": triggered["run_id"], "user": user},
    )
    return {"state": "starting", **triggered}


@router.post("/stop")
async def stop() -> dict:
    user = get_current_user()
    triggered = await _trigger_job(TEARDOWN_JOB_NAME, {
        "catalog_name":      get_catalog(),
        "schema_name":       get_schema(),
        "online_store_name": ONLINE_STORE_NAME,
        "endpoint_name":     ENDPOINT_NAME,
    })
    await log_audit_event(
        event_type="live_pricing_stop_requested",
        entity_type="endpoint",
        entity_id=ENDPOINT_NAME,
        details={"job_id": triggered["job_id"], "run_id": triggered["run_id"], "user": user},
    )
    return {"state": "stopping", **triggered}


# ---------------------------------------------------------------------------
# Single quote
# ---------------------------------------------------------------------------

class QuoteRequest(BaseModel):
    policy_id: str


def _write_metric_blocking(source: str, policy_id: str, latency_ms: float,
                            final_premium: float | None, status_code: int,
                            run_id: str) -> None:
    """Synchronous metric write — wrap with asyncio.to_thread."""
    fp = "NULL" if final_premium is None else str(final_premium)
    sql = f"""
        INSERT INTO {fqn('live_pricing_metrics')}
          (ts, source, policy_id, latency_ms, final_premium, status_code, run_id)
        VALUES (current_timestamp(), '{source}',
                '{policy_id.replace("'", "''")}', {latency_ms},
                {fp}, {int(status_code)}, '{run_id}')
    """
    # execute_query is async in this project — use a sync escape hatch
    from server.config import get_workspace_client as _wc
    w = _wc()
    from server.config import get_warehouse_id
    warehouse_id = get_warehouse_id()
    try:
        w.statement_execution.execute_statement(
            warehouse_id=warehouse_id, statement=sql, wait_timeout="0s",
        )
    except Exception as e:
        logger.warning("live_pricing metric write failed: %s", str(e)[:200])


@router.post("/quote")
async def quote(req: QuoteRequest) -> dict:
    """Time a single quote against the live endpoint and return the full
    pricing breakdown — final_premium plus every intermediate value the
    endpoint computes (freq/sev/demand/fraud, technical, fraud_load, etc.).
    Writes a row to `live_pricing_metrics` with source='single_quote' so the
    chart history persists across navigations."""
    pid = req.policy_id.strip().upper()
    if not pid:
        raise HTTPException(400, "policy_id required")

    def _call() -> tuple[float, int, dict | None]:
        import requests as _rq
        w = get_workspace_client()
        host  = w.config.host.rstrip("/")
        token = w.config._header_factory()
        t0 = time.perf_counter()
        try:
            resp = _rq.post(
                f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations",
                headers={**token, "Content-Type": "application/json"},
                json={"dataframe_records": [{"policy_id": pid}]},
                timeout=30,
            )
        except Exception as e:
            return (time.perf_counter() - t0) * 1000.0, 0, {"error": str(e)[:300]}
        dt = (time.perf_counter() - t0) * 1000.0
        try:
            data = resp.json()
        except Exception:
            data = None
        return dt, resp.status_code, data

    latency_ms, status_code, data = await asyncio.to_thread(_call)

    row: dict[str, Any] = {}
    if status_code == 200 and isinstance(data, dict):
        preds = data.get("predictions") or data.get("outputs") or data
        if isinstance(preds, list) and preds:
            row = preds[0] or {}
        elif isinstance(preds, dict):
            row = {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}

    final_premium = row.get("final_premium")
    try:
        fp_num = float(final_premium) if final_premium is not None else None
    except (TypeError, ValueError):
        fp_num = None

    await asyncio.to_thread(
        _write_metric_blocking, "single_quote", pid, latency_ms, fp_num,
        status_code, "",
    )

    if status_code != 200:
        return {
            "ok":          False,
            "policy_id":   pid,
            "latency_ms":  round(latency_ms, 1),
            "status_code": status_code,
            "error":       (data or {}).get("error")
                            if isinstance(data, dict) else f"HTTP {status_code}",
        }

    return {
        "ok":           True,
        "policy_id":    pid,
        "latency_ms":   round(latency_ms, 1),
        "status_code":  status_code,
        "result":       row,
    }


# ---------------------------------------------------------------------------
# Claim filing — inline INSERT + UPT MERGE for snappy demo
# ---------------------------------------------------------------------------

class ClaimRequest(BaseModel):
    policy_id:    str
    claim_amount: float
    claim_type:   str = "ACCIDENTAL_DAMAGE"


_PERIL_MAP = {
    "ACCIDENTAL_DAMAGE": "Other",
    "FIRE":              "Fire",
    "FLOOD":             "Flood",
    "THEFT":             "Theft",
    "STORM":             "Storm",
    "SUBSIDENCE":        "Subsidence",
    "WATER":             "Escape of Water",
}


@router.post("/claim")
async def file_claim(req: ClaimRequest, background_tasks: BackgroundTasks) -> dict:
    """File a synthetic claim and merge claim aggregates into UPT inline so
    the next quote against this policy reflects the new claim. Continuous
    online sync pushes the change into Lakebase within ~5-15s — that's the
    update-speed metric the demo is showcasing."""
    pid          = req.policy_id.strip().upper()
    claim_amount = float(req.claim_amount)
    claim_type   = req.claim_type.upper()
    peril        = _PERIL_MAP.get(claim_type, "Other")

    if not pid or claim_amount <= 0:
        raise HTTPException(400, "policy_id and positive claim_amount required")

    rows = await execute_query(f"""
        SELECT policy_id FROM {fqn('unified_pricing_table_live')}
        WHERE policy_id = '{pid}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"policy {pid} not found")

    claim_id  = f"CLM-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
    loss_date = datetime.now(timezone.utc).date().isoformat()
    paid      = int(claim_amount * 0.5)

    t0 = time.perf_counter()
    await execute_query(f"""
        INSERT INTO {fqn('internal_claims_history')}
          (claim_id, policy_id, peril, incurred_amount, paid_amount, reserve,
           loss_date, status)
        VALUES ('{claim_id}', '{pid}', '{peril}',
                {int(claim_amount)}, {paid}, {int(claim_amount) - paid},
                '{loss_date}', 'Open')
    """)
    claim_write_ms = (time.perf_counter() - t0) * 1000.0

    t0 = time.perf_counter()
    await execute_query(f"""
        MERGE INTO {fqn('unified_pricing_table_live')} target
        USING (
            SELECT policy_id,
                   COUNT(*) AS claim_count_5y,
                   SUM(incurred_amount) AS total_incurred_5y,
                   SUM(paid_amount) AS total_paid_5y,
                   SUM(CASE WHEN status='Open' THEN 1 ELSE 0 END) AS open_claims_count,
                   COUNT(DISTINCT peril) AS distinct_perils
            FROM {fqn('internal_claims_history')}
            WHERE policy_id = '{pid}'
            GROUP BY policy_id
        ) src
        ON target.policy_id = src.policy_id
        WHEN MATCHED THEN UPDATE SET
            target.claim_count_5y    = src.claim_count_5y,
            target.total_incurred_5y = src.total_incurred_5y,
            target.total_paid_5y     = src.total_paid_5y,
            target.open_claims_count = src.open_claims_count,
            target.distinct_perils   = src.distinct_perils,
            target.loss_ratio_5y     = ROUND(src.total_incurred_5y /
                                             (target.current_premium * 5), 3)
    """)
    upt_merge_ms = (time.perf_counter() - t0) * 1000.0

    user = get_current_user()
    await log_audit_event(
        event_type="live_pricing_claim_filed",
        entity_type="policy",
        entity_id=pid,
        details={
            "claim_id":       claim_id,
            "claim_type":     claim_type,
            "peril":          peril,
            "claim_amount":   claim_amount,
            "claim_write_ms": round(claim_write_ms, 1),
            "upt_merge_ms":   round(upt_merge_ms, 1),
            "user":           user,
            "publish_mode":   "CONTINUOUS",
        },
    )

    return {
        "ok":             True,
        "claim_id":       claim_id,
        "policy_id":      pid,
        "claim_amount":   claim_amount,
        "peril":          peril,
        "claim_write_ms": round(claim_write_ms, 1),
        "upt_merge_ms":   round(upt_merge_ms, 1),
        "total_ms":       round(claim_write_ms + upt_merge_ms, 1),
        "filed_at":       datetime.now(timezone.utc).isoformat(),
    }


@router.get("/claim/{run_id}")
async def claim_run_status(run_id: int) -> dict:
    """Status of an async refresh job (kept for parity with long-running
    flows even though the inline path is the demo default)."""
    def _get():
        w = get_workspace_client()
        return w.jobs.get_run(run_id=run_id)
    try:
        run = await asyncio.to_thread(_get)
    except Exception as e:
        raise HTTPException(404, f"run {run_id} not found: {e}")
    return {
        "run_id":       run_id,
        "state":        str(getattr(getattr(run, "state", None), "life_cycle_state", "")),
        "result_state": str(getattr(getattr(run, "state", None), "result_state", "")),
        "run_page_url": f"{get_workspace_host()}/jobs/{run.job_id}/runs/{run_id}"
                         if get_workspace_host() else None,
    }


# ---------------------------------------------------------------------------
# Load test
# ---------------------------------------------------------------------------

class LoadTestRequest(BaseModel):
    target_qps:       int = 100
    duration_seconds: int = 60
    concurrency:      int = 50


@router.post("/load-test/start")
async def start_load_test(req: LoadTestRequest) -> dict:
    user   = get_current_user()
    run_id = f"loadtest_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    triggered = await _trigger_job(LOAD_TEST_JOB_NAME, {
        "catalog_name":     get_catalog(),
        "schema_name":      get_schema(),
        "endpoint_name":    ENDPOINT_NAME,
        "target_qps":       str(req.target_qps),
        "duration_seconds": str(req.duration_seconds),
        "concurrency":      str(req.concurrency),
        "run_id":           run_id,
    })
    await log_audit_event(
        event_type="live_pricing_load_test_started",
        entity_type="endpoint",
        entity_id=ENDPOINT_NAME,
        details={"job_id": triggered["job_id"], "run_id": triggered["run_id"],
                 "load_test_run_id": run_id, "target_qps": req.target_qps,
                 "duration_seconds": req.duration_seconds, "user": user},
    )
    return {"load_test_run_id": run_id, **triggered}


@router.post("/load-test/stop")
async def stop_load_test(run_id: int) -> dict:
    """Cancel an in-flight load test by Databricks job run id."""
    def _cancel():
        w = get_workspace_client()
        w.jobs.cancel_run(run_id=run_id)
    try:
        await asyncio.to_thread(_cancel)
    except Exception as e:
        raise HTTPException(500, f"cancel failed: {e}")
    return {"ok": True, "run_id": run_id}


@router.get("/load-test/metrics")
async def load_test_metrics(since: str | None = None,
                              run_id: str | None = None) -> dict:
    """Per-second QPS/p50/p95/p99 from `live_pricing_load_test_summary`.
    Optional `since` (ISO 8601) and `run_id` filters."""
    where = []
    if since:
        where.append(f"ts >= TIMESTAMP'{since}'")
    if run_id:
        where.append(f"run_id = '{run_id.replace(chr(39), chr(39)+chr(39))}'")
    where_sql = ("WHERE " + " AND ".join(where)) if where else ""

    table = fqn("live_pricing_load_test_summary")
    try:
        rows = await execute_query(f"""
            SELECT cast(ts as string) as ts, run_id, qps,
                   p50_ms, p95_ms, p99_ms, error_pct
            FROM {table}
            {where_sql}
            ORDER BY ts ASC
            LIMIT 5000
        """)
    except Exception as e:
        # Table may not exist until first load-test run
        if "TABLE_OR_VIEW_NOT_FOUND" in str(e) or "not found" in str(e).lower():
            return {"rows": [], "table_ready": False}
        raise

    return {"rows": rows, "table_ready": True}
