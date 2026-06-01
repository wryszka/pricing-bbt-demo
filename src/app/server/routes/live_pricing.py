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
from collections import deque
from concurrent.futures import ThreadPoolExecutor
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

ENDPOINT_NAME       = "motor_pricing_scorer"
ONLINE_STORE_NAME   = "motor-pricing-online-store"
UPT_TABLE_NAME      = "unified_motor_table_live"
TELEMATICS_TABLE    = "motor_telematics_aggregate"
RUNTIME_STATE_TABLE = "live_motor_runtime_state"
METRICS_TABLE_NAME  = "live_motor_metrics"
PROVISION_JOB_NAME  = "Motor live serving: provision (Lakebase + endpoint reconcile)"
TEARDOWN_JOB_NAME   = "Motor live serving: teardown (delete endpoint)"
LOAD_TEST_JOB_NAME  = "v1 — Live pricing: load test (sustained QPS against scorer)"


def _find_job_by_name(name: str) -> int | None:
    """Exact-match first, then suffix-match — bundle prefixes job names with
    `[dev whoami]` in the dev target. Sync helper; wrap in asyncio.to_thread."""
    w = get_workspace_client()
    try:
        for j in w.jobs.list(name=name, limit=25):
            return j.job_id
    except Exception as e:
        logger.warning("jobs.list(name=...) failed: %s", str(e)[:200])
    # The jobs.list() API caps `limit` at 100 per request — use the SDK's
    # built-in pagination by not passing limit so it reads all pages.
    try:
        for j in w.jobs.list():
            settings = getattr(j, "settings", None)
            jname = getattr(settings, "name", None) if settings else None
            if jname and jname.endswith(name):
                return j.job_id
    except Exception as e:
        logger.warning("jobs.list() iter failed: %s", str(e)[:200])
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
                    "error":   str(e)[:200]}

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
        "metrics_table": fqn(METRICS_TABLE_NAME),
    }


# ---------------------------------------------------------------------------
# Policy profile — driver / vehicle / live telematics, for the external
# quote UI and the black-box panel.
# ---------------------------------------------------------------------------

@router.get("/policy/{policy_id}")
async def policy_profile(policy_id: str) -> dict:
    """Display-friendly snapshot of a motor policy: driver, vehicle, and the
    current live telematics signal. Powers the /quote and /blackbox pages."""
    pid = policy_id.strip().upper()
    rows = await execute_query(f"""
        SELECT policy_id, driver_age, gender, marital_status, occupation_class,
               license_years_held, no_claims_years, postcode_area, region,
               vehicle_make, vehicle_model, vehicle_year, vehicle_value,
               vehicle_group, annual_mileage, parking_overnight, current_premium,
               renewal_date, behaviour_score, avg_speed_mph, night_driving_pct,
               recent_speeding_events, recent_curfew_breaches,
               recent_harsh_braking_30d, telematics_recent_event_count
        FROM {fqn(UPT_TABLE_NAME)}
        WHERE policy_id = '{pid}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"policy {pid} not found")
    r = rows[0]

    def _i(v, d=0):
        try: return int(float(v))
        except Exception: return d
    def _f(v, d=0.0):
        try: return float(v)
        except Exception: return d

    return {
        "policy_id":   r.get("policy_id"),
        "driver": {
            "age":            _i(r.get("driver_age")),
            "gender":         r.get("gender"),
            "marital_status": r.get("marital_status"),
            "occupation":     r.get("occupation_class"),
            "license_years":  _i(r.get("license_years_held")),
            "no_claims_years":_i(r.get("no_claims_years")),
            "postcode_area":  r.get("postcode_area"),
            "region":         r.get("region"),
        },
        "vehicle": {
            "make":    r.get("vehicle_make"),
            "model":   r.get("vehicle_model"),
            "year":    _i(r.get("vehicle_year")),
            "value":   _f(r.get("vehicle_value")),
            "group":   _i(r.get("vehicle_group")),
            "mileage": _i(r.get("annual_mileage")),
            "parking": r.get("parking_overnight"),
        },
        "telematics": {
            "behaviour_score":        _i(r.get("behaviour_score")),
            "avg_speed_mph":          _f(r.get("avg_speed_mph")),
            "night_driving_pct":      _f(r.get("night_driving_pct")),
            "recent_speeding_events": _i(r.get("recent_speeding_events")),
            "recent_curfew_breaches": _i(r.get("recent_curfew_breaches")),
            "recent_harsh_braking_30d": _i(r.get("recent_harsh_braking_30d")),
            "recent_event_count":     _i(r.get("telematics_recent_event_count")),
        },
        "current_premium": _f(r.get("current_premium")),
        "renewal_date":    str(r.get("renewal_date") or ""),
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


def _resume_lakebase(name: str) -> dict:
    """Lakebase instances are paused when the demo isn't running. Flip
    `stopped=false` so the provision job's online-store check finds the
    instance back in AVAILABLE state."""
    import requests as _rq
    w = get_workspace_client()
    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()
    try:
        resp = _rq.patch(
            f"{host}/api/2.0/database/instances/{name}?update_mask=stopped",
            headers={**token, "Content-Type": "application/json"},
            json={"stopped": False}, timeout=20,
        )
        if resp.status_code in (200, 204):
            return {"resumed": True, "name": name}
        return {"resumed": False, "name": name, "status": resp.status_code, "body": resp.text[:200]}
    except Exception as e:
        return {"resumed": False, "name": name, "error": str(e)[:200]}


@router.post("/start")
async def start() -> dict:
    user = get_current_user()
    # Resume the Lakebase instance first — it warms in the background
    # (~60s) in parallel with the provision job, which also waits for
    # AVAILABLE before proceeding to publish.
    lakebase = await asyncio.to_thread(_resume_lakebase, ONLINE_STORE_NAME)
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
        details={"job_id": triggered["job_id"], "run_id": triggered["run_id"],
                 "user": user, "lakebase": lakebase},
    )
    return {"state": "starting", "lakebase": lakebase, **triggered}


def _stop_lakebase(name: str) -> dict:
    import requests as _rq
    w = get_workspace_client()
    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()
    try:
        resp = _rq.patch(
            f"{host}/api/2.0/database/instances/{name}?update_mask=stopped",
            headers={**token, "Content-Type": "application/json"},
            json={"stopped": True}, timeout=20,
        )
        if resp.status_code in (200, 204):
            return {"stopped": True, "name": name}
        return {"stopped": False, "name": name, "status": resp.status_code, "body": resp.text[:200]}
    except Exception as e:
        return {"stopped": False, "name": name, "error": str(e)[:200]}


@router.post("/stop")
async def stop() -> dict:
    """Soft stop — fires the motor_teardown job which deletes the Model
    Serving endpoint, then pauses the Lakebase online store so neither is
    burning compute. A subsequent /start resumes the Lakebase instance and
    rebuilds the endpoint container (~60-90s warm-up)."""
    user = get_current_user()
    triggered = await _trigger_job(TEARDOWN_JOB_NAME, {
        "catalog_name":  get_catalog(),
        "schema_name":   get_schema(),
        "endpoint_name": ENDPOINT_NAME,
    })
    lakebase = await asyncio.to_thread(_stop_lakebase, ONLINE_STORE_NAME)
    await log_audit_event(
        event_type="live_pricing_stop_requested",
        entity_type="endpoint",
        entity_id=ENDPOINT_NAME,
        details={"job_id": triggered["job_id"], "run_id": triggered["run_id"],
                 "user": user, "lakebase": lakebase},
    )
    return {"state": "stopping", "lakebase": lakebase, **triggered}


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
        INSERT INTO {fqn(METRICS_TABLE_NAME)}
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
# Telematics event — mutates the policy's telematics aggregate inline so the
# next quote returns a different premium driven by real model signal.
# ---------------------------------------------------------------------------

class TelematicsEventRequest(BaseModel):
    policy_id:               str
    speeding_event:          bool = True
    curfew_breach:           bool = True
    behaviour_score_delta:   int  = -8
    harsh_braking_delta:     int  = 1


async def _get_publish_pipeline_id() -> str | None:
    """Pull the persisted Lakebase publish pipeline id (set by provision)."""
    try:
        rows = await execute_query(f"""
            SELECT value FROM {fqn(RUNTIME_STATE_TABLE)}
            WHERE key = 'publish_pipeline_id' LIMIT 1
        """)
        if rows and rows[0].get("value"):
            return rows[0]["value"]
    except Exception as e:
        logger.warning("could not read publish_pipeline_id: %s", str(e)[:200])
    return None


@router.post("/telematics-event")
async def telematics_event(req: TelematicsEventRequest,
                            background_tasks: BackgroundTasks) -> dict:
    """Simulate a telematics black-box event landing for a policy: increments
    the speeding/curfew counters, drops behaviour_score, MERGEs the change
    into the motor UPT, and triggers a Lakebase SNAPSHOT refresh so the next
    quote against this policy sees the new feature values."""
    pid = req.policy_id.strip().upper()
    if not pid:
        raise HTTPException(400, "policy_id required")

    # Verify policy exists + capture before-state
    before_rows = await execute_query(f"""
        SELECT behaviour_score, recent_speeding_events, recent_curfew_breaches,
               recent_harsh_braking_30d, telematics_recent_event_count
        FROM {fqn(UPT_TABLE_NAME)}
        WHERE policy_id = '{pid}' LIMIT 1
    """)
    if not before_rows:
        raise HTTPException(404, f"policy {pid} not found")
    before = before_rows[0]

    sp_inc = 1 if req.speeding_event else 0
    cb_inc = 1 if req.curfew_breach  else 0
    hb_inc = max(0, int(req.harsh_braking_delta))
    bs_dec = max(0, -int(req.behaviour_score_delta))  # delta is negative; convert to positive subtract

    event_id  = f"TLM-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"

    # 1. Update the telematics_aggregate source-of-truth table
    t0 = time.perf_counter()
    await execute_query(f"""
        UPDATE {fqn(TELEMATICS_TABLE)} SET
            recent_speeding_events    = recent_speeding_events    + {sp_inc},
            recent_curfew_breaches    = recent_curfew_breaches    + {cb_inc},
            recent_harsh_braking_30d  = recent_harsh_braking_30d  + {hb_inc},
            behaviour_score           = GREATEST(0, LEAST(100, behaviour_score - {bs_dec}))
        WHERE policy_id = '{pid}'
    """)
    telematics_write_ms = (time.perf_counter() - t0) * 1000.0

    # 2. MERGE the updated telematics row into UPT (so Lakebase has fresh data)
    t0 = time.perf_counter()
    await execute_query(f"""
        MERGE INTO {fqn(UPT_TABLE_NAME)} target
        USING (
            SELECT policy_id, behaviour_score, recent_speeding_events,
                   recent_curfew_breaches, recent_harsh_braking_30d,
                   recent_speeding_events + recent_curfew_breaches +
                   recent_harsh_braking_30d AS telematics_recent_event_count
            FROM {fqn(TELEMATICS_TABLE)}
            WHERE policy_id = '{pid}'
        ) src
        ON target.policy_id = src.policy_id
        WHEN MATCHED THEN UPDATE SET
            target.behaviour_score               = src.behaviour_score,
            target.recent_speeding_events        = src.recent_speeding_events,
            target.recent_curfew_breaches        = src.recent_curfew_breaches,
            target.recent_harsh_braking_30d      = src.recent_harsh_braking_30d,
            target.telematics_recent_event_count = src.telematics_recent_event_count
    """)
    upt_merge_ms = (time.perf_counter() - t0) * 1000.0

    # Backwards-compat naming for the UI which still reads claim_write_ms.
    claim_write_ms = telematics_write_ms

    # Capture after-state
    after_rows = await execute_query(f"""
        SELECT behaviour_score, recent_speeding_events, recent_curfew_breaches,
               recent_harsh_braking_30d, telematics_recent_event_count
        FROM {fqn(UPT_TABLE_NAME)}
        WHERE policy_id = '{pid}' LIMIT 1
    """)
    after = after_rows[0] if after_rows else {}

    # Trigger a Lakebase SNAPSHOT refresh. The publish_table call at provision
    # time stood up a DLT pipeline that runs SNAPSHOT on demand. start_update
    # kicks off a fresh run; polling waits up to 60s for it to complete so the
    # next quote sees the new feature row in Lakebase.
    pipeline_id = await _get_publish_pipeline_id()
    refresh: dict[str, Any] = {"pipeline_id": pipeline_id, "triggered": False,
                               "completed": False, "duration_ms": None}
    if pipeline_id:
        def _trigger_and_wait():
            w = get_workspace_client()
            t0 = time.perf_counter()
            try:
                upd = w.pipelines.start_update(pipeline_id=pipeline_id, full_refresh=False)
                update_id = getattr(upd, "update_id", None)
            except Exception as e:
                return {"triggered": False, "error": str(e)[:200]}
            # Poll for completion. Lakebase SNAPSHOT publish on a 500k-row
            # source typically lands in 50-70s; cap at 180s so the demo card
            # reports `completed` instead of an inaccurate `RUNNING (62s)`.
            deadline = time.perf_counter() + 180.0
            terminal = {"COMPLETED", "FAILED", "CANCELED"}
            state = None
            while time.perf_counter() < deadline:
                try:
                    info = w.pipelines.get_update(pipeline_id=pipeline_id, update_id=update_id)
                    state = str(getattr(info.update, "state", "")).split(".")[-1]
                except Exception as e:
                    state = f"poll-error:{str(e)[:80]}"
                if state in terminal:
                    break
                time.sleep(2.0)
            return {"triggered": True, "update_id": update_id,
                    "completed": state == "COMPLETED", "state": state,
                    "duration_ms": round((time.perf_counter() - t0) * 1000.0, 1)}
        refresh.update(await asyncio.to_thread(_trigger_and_wait))

    user = get_current_user()
    await log_audit_event(
        event_type="live_motor_telematics_event",
        entity_type="policy",
        entity_id=pid,
        details={
            "event_id":             event_id,
            "speeding_event":       req.speeding_event,
            "curfew_breach":        req.curfew_breach,
            "behaviour_score_delta": req.behaviour_score_delta,
            "before":               before,
            "after":                after,
            "telematics_write_ms":  round(telematics_write_ms, 1),
            "upt_merge_ms":         round(upt_merge_ms, 1),
            "user":                 user,
            "publish_mode":         "SNAPSHOT",
            "online_refresh":       refresh,
        },
    )

    return {
        "ok":              True,
        "event_id":        event_id,
        "policy_id":       pid,
        "before":          before,
        "after":           after,
        # legacy names so the existing UI still works without changes
        "claim_id":        event_id,
        "claim_amount":    0,
        "peril":           "Telematics event",
        "claim_write_ms":  round(claim_write_ms, 1),
        "upt_merge_ms":    round(upt_merge_ms, 1),
        "online_refresh":  refresh,
        "total_ms":        round(claim_write_ms + upt_merge_ms +
                                 (refresh.get("duration_ms") or 0), 1),
        "filed_at":        datetime.now(timezone.utc).isoformat(),
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
# Live quote stream — app-driven continuous load with in-memory rolling
# metrics. Unlike the job-based load test (which has ~1-2 min startup lag),
# this fires quotes from the app process the instant you hit start and
# exposes live QPS/latency that the UI can poll sub-second. Modest QPS
# (the app container is small) — it's a "watch it flow live" view, not a
# max-throughput test.
# ---------------------------------------------------------------------------

_STREAM_MAX_DURATION_S = 180.0   # hard failsafe — a stream can never run longer


class _LiveStream:
    def __init__(self) -> None:
        self.task: asyncio.Task | None = None
        self.running = False
        self.epoch = 0           # bumped on every start/stop; stale workers self-exit
        self.target_qps = 0
        self.started_at = 0.0
        self.total = 0
        self.errors = 0
        # rolling samples: (completion_ts, latency_ms, ok)
        self.samples: deque = deque(maxlen=4000)
        self._policy_ids: list[str] = []

    def snapshot(self, window_s: float = 5.0) -> dict:
        now = time.time()
        recent = [(ts, lat, ok) for (ts, lat, ok) in self.samples if now - ts <= window_s]
        lats = sorted(lat for (_, lat, ok) in recent if ok)
        n = len(recent)
        ok_n = len(lats)
        def pct(p: float) -> float:
            if not lats:
                return 0.0
            i = min(len(lats) - 1, int(round(p * (len(lats) - 1))))
            return lats[i]
        qps = round(n / window_s, 1) if n else 0.0
        err_pct = round(100.0 * (n - ok_n) / n, 1) if n else 0.0
        return {
            "running":     self.running,
            "target_qps":  self.target_qps,
            "qps":         qps,
            "p50_ms":      round(pct(0.50), 1),
            "p95_ms":      round(pct(0.95), 1),
            "p99_ms":      round(pct(0.99), 1),
            "error_pct":   err_pct,
            "total":       self.total,
            "errors":      self.errors,
            "uptime_s":    round(now - self.started_at, 1) if self.started_at else 0,
            # small recent series for a sparkline (last ~60 completions)
            "recent":      [round(lat, 1) for (_, lat, ok) in list(self.samples)[-60:]],
        }


_stream = _LiveStream()

# Dedicated thread pool for firing quotes. asyncio.to_thread uses the loop's
# default executor (~min(32, cpu+4) — only ~5-6 threads on a small app
# container), which caps achievable QPS far below target. A bigger pool keeps
# more requests in flight (requests releases the GIL during socket I/O), so the
# app-driven stream can approach the slider target instead of stalling ~30 QPS.
_stream_executor = ThreadPoolExecutor(max_workers=96, thread_name_prefix="qstream")


async def _stream_fire_one(pid: str, url: str, headers: dict, session) -> None:
    def _call():
        t0 = time.perf_counter()
        try:
            r = session.post(url, headers=headers,
                             json={"dataframe_records": [{"policy_id": pid}]}, timeout=15)
            return (time.perf_counter() - t0) * 1000.0, r.status_code == 200
        except Exception:
            return (time.perf_counter() - t0) * 1000.0, False
    loop = asyncio.get_running_loop()
    lat, ok = await loop.run_in_executor(_stream_executor, _call)
    _stream.samples.append((time.time(), lat, ok))
    _stream.total += 1
    if not ok:
        _stream.errors += 1


async def _stream_worker(target_qps: int, my_epoch: int) -> None:
    """Pace quotes at ~target_qps. Self-terminates the instant the epoch
    changes (any start/stop bumps it), if running is cleared, or once the
    hard max-duration failsafe elapses — so an orphaned loop can never keep
    firing after stop."""
    import requests as _rq
    w = get_workspace_client()
    host  = w.config.host.rstrip("/")
    headers = {**w.config._header_factory(), "Content-Type": "application/json"}
    url = f"{host}/serving-endpoints/{ENDPOINT_NAME}/invocations"
    # Reuse one keep-alive session so we measure steady-state latency, not a
    # fresh TLS handshake on every call (which inflated the numbers well above
    # the server-side latency Databricks reports).
    session = _rq.Session()
    if not _stream._policy_ids:
        try:
            rows = await execute_query(
                f"SELECT policy_id FROM {fqn(UPT_TABLE_NAME)} ORDER BY rand() LIMIT 2000")
            _stream._policy_ids = [r["policy_id"] for r in rows] or ["POL-MOTOR-00000001"]
        except Exception:
            _stream._policy_ids = ["POL-MOTOR-00000001"]
    # Allow enough in-flight requests to actually hit target_qps given the
    # per-request latency (concurrency ≈ qps × latency). Capped to the thread
    # pool size so we don't queue behind it.
    sem = asyncio.Semaphore(max(8, min(90, target_qps)))
    period = 1.0 / max(1, target_qps)
    deadline = time.time() + _STREAM_MAX_DURATION_S
    i = 0
    try:
        while (_stream.running and _stream.epoch == my_epoch
               and time.time() < deadline):
            pid = _stream._policy_ids[i % len(_stream._policy_ids)]
            i += 1
            async def _guarded(p=pid):
                async with sem:
                    # Drop late completions from a superseded epoch.
                    if _stream.epoch != my_epoch:
                        return
                    await _stream_fire_one(p, url, headers, session)
            asyncio.create_task(_guarded())
            await asyncio.sleep(period)
    except asyncio.CancelledError:
        pass
    finally:
        try: session.close()
        except Exception: pass
        # If this was the active epoch, mark stopped on natural/deadline exit.
        if _stream.epoch == my_epoch:
            _stream.running = False


class LiveStreamRequest(BaseModel):
    target_qps: int = 25


@router.post("/stream/start")
async def stream_start(req: LiveStreamRequest) -> dict:
    # Always invalidate any prior worker (bump epoch + cancel) before starting,
    # so a stuck/orphaned loop from an earlier start can't survive.
    _stream.epoch += 1
    if _stream.task:
        _stream.task.cancel()
        _stream.task = None
    my_epoch = _stream.epoch
    _stream.running    = True
    _stream.target_qps = max(1, min(100, req.target_qps))
    _stream.started_at = time.time()
    _stream.total = 0
    _stream.errors = 0
    _stream.samples.clear()
    _stream.task = asyncio.create_task(_stream_worker(_stream.target_qps, my_epoch))
    return {"running": True, "target_qps": _stream.target_qps,
            "max_duration_s": _STREAM_MAX_DURATION_S}


@router.post("/stream/stop")
async def stream_stop() -> dict:
    # Bump epoch so ANY in-flight worker (even an orphan) self-exits, clear
    # the flag, and cancel the tracked task.
    _stream.epoch += 1
    _stream.running = False
    if _stream.task:
        _stream.task.cancel()
        _stream.task = None
    return {"running": False, "total": _stream.total, "errors": _stream.errors}


@router.get("/stream/metrics")
async def stream_metrics() -> dict:
    return _stream.snapshot()


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
