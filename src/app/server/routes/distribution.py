"""Agentic-distribution telemetry — the business view of the agent channel.

Answers the question a distribution director actually asks: who is calling our
services, where do journeys fall over, and what converts. Reads
`mcp_tool_calls`, which every tool call on both surfaces writes to.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter

from server.config import fqn
from server.mcp_engine import TOOL_CALLS_TABLE, ensure_tool_calls_table
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/distribution", tags=["distribution"])


@router.get("/telemetry")
async def telemetry(hours: int = 24) -> dict:
    """Tool-call activity over the last `hours`. Degrades to empty lists rather
    than erroring — an empty table before the first demo run is normal."""
    await ensure_tool_calls_table()
    table = fqn(TOOL_CALLS_TABLE)
    window = f"ts >= current_timestamp() - INTERVAL {int(hours)} HOURS"

    async def _q(sql: str) -> list[dict]:
        try:
            return await execute_query(sql)
        except Exception as e:
            logger.warning("telemetry query failed: %s", str(e)[:200])
            return []

    by_tool = await _q(f"""
        SELECT tool,
               COUNT(*)                                   AS calls,
               SUM(CASE WHEN ok THEN 1 ELSE 0 END)        AS ok_calls,
               ROUND(AVG(latency_ms), 1)                  AS avg_latency_ms,
               ROUND(PERCENTILE(latency_ms, 0.5), 1)      AS p50_latency_ms
        FROM {table} WHERE {window}
        GROUP BY tool ORDER BY calls DESC
    """)

    by_surface = await _q(f"""
        SELECT surface,
               COUNT(DISTINCT session_id) AS sessions,
               COUNT(*)                   AS calls
        FROM {table} WHERE {window}
        GROUP BY surface ORDER BY calls DESC
    """)

    agents = await _q(f"""
        SELECT agent_id,
               COUNT(DISTINCT session_id) AS sessions,
               COUNT(*)                   AS calls,
               MAX(ts)                    AS last_seen
        FROM {table} WHERE {window}
        GROUP BY agent_id ORDER BY calls DESC LIMIT 12
    """)

    funnel = await _q(f"""
        WITH s AS (
          SELECT session_id,
                 MAX(CASE WHEN tool = 'get_quote_requirements' THEN 1 ELSE 0 END) AS discovered,
                 MAX(CASE WHEN tool = 'check_answers'          THEN 1 ELSE 0 END) AS checked,
                 MAX(CASE WHEN tool = 'price_motor_risk' AND NOT ok THEN 1 ELSE 0 END) AS attempted,
                 MAX(CASE WHEN tool = 'price_motor_risk' AND ok     THEN 1 ELSE 0 END) AS priced,
                 MAX(CASE WHEN tool = 'explain_price'           THEN 1 ELSE 0 END) AS explained
          FROM {table} WHERE {window} GROUP BY session_id
        )
        SELECT COUNT(*)          AS sessions,
               SUM(discovered)   AS discovered,
               SUM(checked)      AS checked,
               SUM(attempted)    AS incomplete_attempts,
               SUM(priced)       AS priced,
               SUM(explained)    AS explained
        FROM s
    """)

    premiums = await _q(f"""
        SELECT COUNT(*)                          AS quotes,
               ROUND(AVG(annual_premium), 2)     AS avg_premium,
               ROUND(MIN(annual_premium), 2)     AS min_premium,
               ROUND(MAX(annual_premium), 2)     AS max_premium,
               ROUND(AVG(fields_supplied), 1)    AS avg_fields_supplied
        FROM {table}
        WHERE {window} AND tool = 'price_motor_risk' AND ok
    """)

    recent = await _q(f"""
        SELECT ts, session_id, agent_id, surface, tool, ok,
               ROUND(latency_ms, 1) AS latency_ms, annual_premium
        FROM {table} WHERE {window}
        ORDER BY ts DESC LIMIT 40
    """)

    return {
        "window_hours": hours,
        "by_tool":    by_tool,
        "by_surface": by_surface,
        "agents":     agents,
        "funnel":     (funnel[0] if funnel else {}),
        "premiums":   (premiums[0] if premiums else {}),
        "recent":     recent,
    }
