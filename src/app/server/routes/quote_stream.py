"""Quote Stream routes — transaction lookup, replay, analytics, payload export.

Sits on top of:
- quotes            (one row per transaction, flattened)
- quote_payload_sales       (JSON from sales channel)
- quote_payload_engine_request      (JSON to rating engine)
- quote_payload_engine_response     (JSON from rating engine)
"""

import asyncio
import hashlib
import json
import logging
import random
import re
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import fqn, get_catalog, get_schema, get_workspace_client
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/quote-stream", tags=["quote-stream"])

# Current "live" model version — used by the simulated replay
CURRENT_MODEL = "pricing_v7.1_glm"

# Strict transaction_id guard — prevents SQL injection via the path parameter
_TX_ID_RE = re.compile(r"^TX-[A-Z0-9\-]{1,40}$")


def _validate_tx(tx_id: str) -> str:
    if not _TX_ID_RE.match(tx_id):
        raise HTTPException(status_code=400, detail="Invalid transaction_id format")
    return tx_id


# ---------------------------------------------------------------------------
# 1. Recent transactions (with outlier flagging)
# ---------------------------------------------------------------------------

@router.get("/recent")
async def list_recent(limit: int = 50):
    """Return recent transactions that have captured JSON payloads (picker list)."""
    limit = max(1, min(limit, 200))
    try:
        rows = await execute_query(f"""
            SELECT transaction_id, company_name, postcode, region, sic_description,
                   gross_premium, quote_status, is_outlier, model_version,
                   CAST(created_at AS STRING) AS created_at
            FROM {fqn('quotes')}
            WHERE has_payload = true
            ORDER BY is_outlier DESC, created_at DESC
            LIMIT {limit}
        """)
        return rows
    except Exception as e:
        logger.warning("list_recent failed: %s", e)
        return []


# ---------------------------------------------------------------------------
# 2. Transaction detail — silver row + three JSON payloads
# ---------------------------------------------------------------------------

@router.get("/{tx_id}")
async def get_transaction(tx_id: str):
    tx_id = _validate_tx(tx_id)
    try:
        meta = await execute_query(f"""
            SELECT *, CAST(created_at AS STRING) AS created_at_str
            FROM {fqn('quotes')}
            WHERE transaction_id = '{tx_id}'
            LIMIT 1
        """)
        if not meta:
            raise HTTPException(status_code=404, detail=f"No quote found for {tx_id}")

        payloads = {"sales": None, "engine_request": None, "engine_response": None}

        for key, table in [
            ("sales",            "quote_payload_sales"),
            ("engine_request",   "quote_payload_engine_request"),
            ("engine_response",  "quote_payload_engine_response"),
        ]:
            rows = await execute_query(f"""
                SELECT payload
                FROM {fqn(table)}
                WHERE transaction_id = '{tx_id}'
                LIMIT 1
            """)
            if rows:
                try:
                    payloads[key] = json.loads(rows[0]["payload"])
                except Exception:
                    payloads[key] = {"_parse_error": rows[0]["payload"][:300]}

        return {"meta": meta[0], "payloads": payloads}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("get_transaction failed for %s", tx_id)
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# 3. Replay — simulated re-call of the rating engine
# ---------------------------------------------------------------------------

class ReplayRequest(BaseModel):
    pass  # body reserved for future overrides


@router.post("/{tx_id}/replay")
async def replay(tx_id: str, _req: ReplayRequest | None = None):
    """Deterministic simulated replay. Outliers re-price sanely (demonstrating
    that today's model would have caught it); regular quotes drift within ±1.5%."""
    tx_id = _validate_tx(tx_id)

    meta = await execute_query(f"""
        SELECT gross_premium, model_version, is_outlier
        FROM {fqn('quotes')}
        WHERE transaction_id = '{tx_id}'
        LIMIT 1
    """)
    if not meta:
        raise HTTPException(status_code=404, detail=f"No quote found for {tx_id}")

    resp_rows = await execute_query(f"""
        SELECT payload
        FROM {fqn('quote_payload_engine_response')}
        WHERE transaction_id = '{tx_id}'
        LIMIT 1
    """)
    stored_response = None
    if resp_rows:
        try:
            stored_response = json.loads(resp_rows[0]["payload"])
        except Exception:
            stored_response = None

    m = meta[0]
    stored_premium = float(m.get("gross_premium") or 0.0)
    stored_model = m.get("model_version") or "unknown"
    is_outlier = bool(m.get("is_outlier"))

    seed = int(hashlib.md5(tx_id.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    if is_outlier:
        pricing = (stored_response or {}).get("pricing", {}) or {}
        honest_base = (float(pricing.get("base_building_premium")  or 0)
                     + float(pricing.get("base_contents_premium")  or 0)
                     + float(pricing.get("base_liability_premium") or 0))
        net = max(honest_base * 1.1, 1500.0) * rng.uniform(0.95, 1.05)
        notes = ("The current rating engine prices this risk within the expected "
                 "peer-group band. The stored anomaly was not reproduced — "
                 "investigate upstream (sales channel payload, API mapping, or "
                 "factor overrides) for the source of the blow-up.")
    else:
        drift = rng.uniform(-0.015, 0.015)
        net = stored_premium / 1.12 * (1 + drift) if stored_premium else 0.0
        notes = "Price reproduces within normal model drift (±1.5%)."

    ipt = round(net * 0.12, 2)
    gross = round(net + ipt, 2)
    delta_pct = ((gross - stored_premium) / stored_premium * 100.0) if stored_premium else 0.0

    return {
        "transaction_id":   tx_id,
        "stored_premium":   stored_premium,
        "stored_model":     stored_model,
        "replay_premium":   gross,
        "replay_model":     CURRENT_MODEL,
        "delta_pct":        delta_pct,
        "notes":            notes,
        "is_outlier":       is_outlier,
        "replay_response": {
            "quote_reference":       f"Q-REPLAY-{seed:08X}",
            "sales_transaction_id":  tx_id,
            "model_version":         CURRENT_MODEL,
            "timestamp":             datetime.now(timezone.utc).isoformat(),
            "pricing": {
                "net_premium":    round(net, 2),
                "ipt":            ipt,
                "gross_premium":  gross,
            },
            "decision": {"status": "QUOTED", "notes": notes},
            "_simulated": True,
        },
    }


# ---------------------------------------------------------------------------
# 4. Save payload to UC volume
# ---------------------------------------------------------------------------

class SavePayloadRequest(BaseModel):
    payload: dict
    kind: str  # "sales" | "engine_request" | "engine_response"


@router.post("/{tx_id}/save")
async def save_payload(tx_id: str, req: SavePayloadRequest):
    tx_id = _validate_tx(tx_id)
    if req.kind not in ("sales", "engine_request", "engine_response"):
        raise HTTPException(status_code=400, detail="invalid kind")
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    catalog, schema = get_catalog(), get_schema()
    path = f"/Volumes/{catalog}/{schema}/saved_payloads/{tx_id}_{req.kind}_{ts}.json"
    try:
        w = get_workspace_client()
        content = json.dumps(req.payload, indent=2).encode("utf-8")
        await asyncio.to_thread(
            w.files.upload, file_path=path, contents=content, overwrite=True,
        )
        return {"saved_to": path, "bytes": len(content)}
    except Exception as e:
        logger.exception("save_payload failed")
        raise HTTPException(status_code=500, detail=str(e)[:200])


# ---------------------------------------------------------------------------
# 5. Analytics: summary, outliers, funnel, distribution
# ---------------------------------------------------------------------------

@router.get("/analytics/summary")
async def analytics_summary():
    """Analytics are scoped to the captured live-stream subset (has_payload=true).
    That's the set of transactions an operator can actually investigate and replay."""
    try:
        rows = await execute_query(f"""
            SELECT
                COUNT(*)                                              AS total_transactions,
                COUNT_IF(quote_status = 'BOUND')                      AS bound,
                COUNT_IF(quote_status = 'QUOTED')                     AS quoted_not_bound,
                COUNT_IF(quote_status = 'ABANDONED')                  AS abandoned,
                COUNT_IF(is_outlier)                                  AS outliers,
                ROUND(AVG(CASE WHEN quote_status <> 'ABANDONED'
                                AND NOT is_outlier THEN gross_premium END), 2)  AS avg_premium,
                ROUND(PERCENTILE(gross_premium, 0.95), 2)             AS p95_premium
            FROM {fqn('quotes')}
            WHERE has_payload = true
        """)
        return rows[0] if rows else {}
    except Exception as e:
        logger.warning("analytics_summary failed: %s", e)
        return {}


@router.get("/analytics/outliers")
async def analytics_outliers(limit: int = 20):
    limit = max(1, min(limit, 100))
    try:
        rows = await execute_query(f"""
            WITH peers AS (
                SELECT region, construction_type,
                       PERCENTILE(gross_premium, 0.99) AS p99
                FROM {fqn('quotes')}
                WHERE quote_status <> 'ABANDONED' AND has_payload = true
                GROUP BY region, construction_type
            )
            SELECT q.transaction_id, q.company_name, q.region, q.construction_type,
                   ROUND(q.gross_premium, 2) AS gross_premium,
                   ROUND(p.p99, 2)           AS peer_p99,
                   ROUND(q.gross_premium / NULLIF(p.p99, 0), 2) AS vs_peer_p99,
                   q.model_version, q.is_outlier
            FROM {fqn('quotes')} q
            JOIN peers p USING (region, construction_type)
            WHERE q.has_payload = true
              AND q.gross_premium > p.p99 * 3
            ORDER BY q.gross_premium DESC
            LIMIT {limit}
        """)
        return rows
    except Exception as e:
        logger.warning("analytics_outliers failed: %s", e)
        return []


@router.get("/analytics/funnel")
async def analytics_funnel():
    try:
        rows = await execute_query(f"""
            SELECT channel,
                   COUNT(*)                                    AS started,
                   COUNT_IF(quote_status <> 'ABANDONED')       AS priced,
                   COUNT_IF(quote_status = 'BOUND')            AS bound,
                   ROUND(
                     1.0 - (COUNT_IF(quote_status <> 'ABANDONED') * 1.0 / COUNT(*)),
                     3
                   ) AS dropout_rate,
                   ROUND(
                     COUNT_IF(quote_status = 'BOUND') * 1.0 / COUNT(*),
                     3
                   ) AS bind_rate
            FROM {fqn('quotes')}
            WHERE has_payload = true
            GROUP BY channel
            ORDER BY started DESC
        """)
        return rows
    except Exception as e:
        logger.warning("analytics_funnel failed: %s", e)
        return []


@router.get("/analytics/distribution")
async def analytics_distribution():
    """Premium box-plot data by region, excluding outliers and abandoned."""
    try:
        rows = await execute_query(f"""
            SELECT region,
                   ROUND(PERCENTILE(gross_premium, 0.25), 0) AS p25,
                   ROUND(PERCENTILE(gross_premium, 0.50), 0) AS p50,
                   ROUND(PERCENTILE(gross_premium, 0.75), 0) AS p75,
                   ROUND(PERCENTILE(gross_premium, 0.95), 0) AS p95,
                   ROUND(MIN(gross_premium), 0)              AS min_val,
                   ROUND(MAX(gross_premium), 0)              AS max_val,
                   COUNT(*)                                  AS n
            FROM {fqn('quotes')}
            WHERE has_payload = true
              AND quote_status <> 'ABANDONED' AND NOT is_outlier
              AND gross_premium < 500000
            GROUP BY region
            ORDER BY p50 DESC
        """)
        return rows
    except Exception as e:
        logger.warning("analytics_distribution failed: %s", e)
        return []
