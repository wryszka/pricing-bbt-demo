"""Engine bridge + telemetry for the agentic-distribution tools.

Prices come from `motor_pricing_scorer_direct` — the plain pyfunc endpoint that
takes a full 28-feature vector and returns the rating breakdown. It is the same
engine the consumer quote journey (`/quote`) prices against.

Why not the route-optimized `motor_pricing_scorer`? That endpoint is a
FeatureLookup model: you pass a `policy_id` and it hydrates the features from
the Lakebase online store. Perfect for repricing the existing book at high QPS
(the Live Pricing System and load tester use it for exactly that) — but an
agent-channel prospect has no policy with us yet, so there is nothing to look
up. Passing a raw feature vector to it fails schema enforcement, correctly.
Serving a brand-new risk means the direct endpoint.

Trade-off worth knowing on stage: `_direct` is scale-to-zero, so the first call
after an idle period pays a cold start (tens of seconds) and then settles to
sub-second. Warm it before a live demo.

Every tool call is logged to `mcp_tool_calls` so the telemetry view can answer
the question the business actually cares about: which agents are calling us,
where do they drop out, and what converts.
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from typing import Any

import requests

from server.config import (
    fqn, get_workspace_client, get_warehouse_id, reset_workspace_client,
)
from server.routes.live_pricing import DIRECT_ENDPOINT

logger = logging.getLogger(__name__)

TOOL_CALLS_TABLE = "mcp_tool_calls"

# Keep-alive session so repeated agent calls don't pay TCP+TLS setup each time.
_session: requests.Session | None = None


def _get_session() -> requests.Session:
    global _session
    if _session is None:
        s = requests.Session()
        s.mount("https://", requests.adapters.HTTPAdapter(
            pool_connections=8, pool_maxsize=32, max_retries=0))
        _session = s
    return _session


def _price_blocking(features: dict[str, Any]) -> tuple[float, int, dict | None, str]:
    """Score one risk against the direct endpoint.

    Retries once on a 401/403 — a long-running app can hold a stale token —
    and rebuilds the cached client before that retry.
    """
    records = [features]
    dt = 0.0

    for attempt in (1, 2):
        t0 = time.perf_counter()
        try:
            w = get_workspace_client()
            host = w.config.host.rstrip("/")
            token = w.config._header_factory()
            resp = _get_session().post(
                f"{host}/serving-endpoints/{DIRECT_ENDPOINT}/invocations",
                headers={**token, "Content-Type": "application/json"},
                json={"dataframe_records": records},
                timeout=90,   # scale-to-zero: first call may cold-start
            )
            dt = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                return dt, 200, resp.json(), DIRECT_ENDPOINT
            try:
                body = resp.json()
            except Exception:
                body = {"error": resp.text[:300]}
            if attempt == 1 and resp.status_code in (401, 403):
                logger.warning("mcp: auth rejected, rebuilding client and retrying")
                reset_workspace_client()
                continue
            logger.warning("mcp: pricing query failed %s: %s",
                           resp.status_code, str(body)[:300])
            return dt, resp.status_code, body, DIRECT_ENDPOINT
        except Exception as e:
            dt = (time.perf_counter() - t0) * 1000.0
            logger.warning("mcp: pricing query error: %s", str(e)[:200])
            if attempt == 1:
                reset_workspace_client()
                continue
            return dt, 0, {"error": str(e)[:300]}, DIRECT_ENDPOINT

    return dt, 0, {"error": "pricing engine unreachable"}, DIRECT_ENDPOINT


def _unwrap(body: dict | None) -> dict[str, Any]:
    """Pull the single prediction row out of whichever shape came back."""
    if not isinstance(body, dict):
        return {}
    preds = body.get("predictions") or body.get("outputs") or body
    if isinstance(preds, list) and preds:
        return preds[0] or {}
    if isinstance(preds, dict):
        return {k: (v[0] if isinstance(v, list) else v) for k, v in preds.items()}
    return {}


async def price_risk(features: dict[str, Any]) -> dict[str, Any]:
    """Score a risk against the live engine. Never fabricates a premium."""
    latency_ms, status, body, engine = await asyncio.to_thread(
        _price_blocking, features)
    row = _unwrap(body) if status == 200 else {}
    ok = status == 200 and row.get("final_premium") is not None

    result: dict[str, Any] = {
        "ok": ok,
        "engine": engine,
        "latency_ms": round(latency_ms, 1),
        "status_code": status,
    }
    if ok:
        result["breakdown"] = row
        result["annual_premium"] = round(float(row["final_premium"]), 2)
        result["monthly_premium"] = round(float(row["final_premium"]) / 12, 2)
    else:
        err = body.get("error") if isinstance(body, dict) else None
        result["error"] = err or f"pricing engine returned HTTP {status}"
        if status in (0, 503):
            result["detail"] = ("pricing engine is warming up — "
                                "start the Live Pricing System or retry shortly")
    return result


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

_TOOL_CALLS_DDL = f"""
CREATE TABLE IF NOT EXISTS {{table}} (
  ts             TIMESTAMP,
  session_id     STRING,
  agent_id       STRING,
  surface        STRING,
  tool           STRING,
  ok             BOOLEAN,
  latency_ms     DOUBLE,
  annual_premium DOUBLE,
  fields_supplied INT,
  detail         STRING
) USING DELTA
COMMENT 'Agent-facing tool calls against the motor pricing services (MCP + broker chat).'
"""

_ensured = False


async def ensure_tool_calls_table() -> None:
    global _ensured
    if _ensured:
        return
    from server.sql import execute_query
    try:
        await execute_query(_TOOL_CALLS_DDL.format(table=fqn(TOOL_CALLS_TABLE)))
        _ensured = True
        logger.info("mcp_tool_calls table ready")
    except Exception:
        logger.exception("could not ensure mcp_tool_calls — telemetry will retry")


def _sql_str(v: Any) -> str:
    if v is None:
        return "NULL"
    return "'" + str(v).replace("'", "''")[:400] + "'"


def _sql_num(v: Any) -> str:
    if v is None:
        return "NULL"
    try:
        return str(float(v))
    except (TypeError, ValueError):
        return "NULL"


async def log_tool_call(*, session_id: str, agent_id: str, surface: str, tool: str,
                        ok: bool, latency_ms: float | None = None,
                        annual_premium: float | None = None,
                        fields_supplied: int | None = None,
                        detail: str | None = None) -> None:
    """Fire-and-forget telemetry write. Never raises into the tool path — a
    telemetry outage must not break a live demo."""
    await ensure_tool_calls_table()
    sql = f"""
        INSERT INTO {fqn(TOOL_CALLS_TABLE)}
          (ts, session_id, agent_id, surface, tool, ok, latency_ms,
           annual_premium, fields_supplied, detail)
        VALUES (current_timestamp(), {_sql_str(session_id)}, {_sql_str(agent_id)},
                {_sql_str(surface)}, {_sql_str(tool)}, {str(bool(ok)).lower()},
                {_sql_num(latency_ms)}, {_sql_num(annual_premium)},
                {int(fields_supplied) if fields_supplied is not None else 'NULL'},
                {_sql_str(detail)})
    """
    try:
        w = get_workspace_client()
        await asyncio.to_thread(
            lambda: w.statement_execution.execute_statement(
                warehouse_id=get_warehouse_id(), statement=sql, wait_timeout="0s"))
    except Exception as e:
        logger.warning("tool-call telemetry write failed: %s", str(e)[:200])


def new_session_id() -> str:
    return uuid.uuid4().hex[:12]
