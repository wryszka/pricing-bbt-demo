"""Governance dashboard routes — unified view of all audit, lineage, and DQ data."""

import logging

from fastapi import APIRouter

from server.config import fqn, get_workspace_host
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/governance", tags=["governance"])


@router.get("/summary")
async def governance_summary():
    """Aggregate governance data across all systems."""
    host = get_workspace_host()

    # Audit events by type
    events_by_type = []
    try:
        events_by_type = await execute_query(f"""
            SELECT event_type, entity_type,
                   COUNT(*) AS event_count,
                   MAX(timestamp) AS last_occurrence,
                   COUNT(DISTINCT user_id) AS unique_users
            FROM {fqn('audit_log')}
            GROUP BY event_type, entity_type
            ORDER BY event_count DESC
        """)
    except Exception:
        pass

    # Recent activity
    recent = []
    try:
        recent = await execute_query(f"""
            SELECT event_id, event_type, entity_type, entity_id,
                   user_id, timestamp, source
            FROM {fqn('audit_log')}
            ORDER BY timestamp DESC LIMIT 20
        """)
    except Exception:
        pass

    # DQ pass rates
    dq = []
    try:
        for ds, raw, silver in [
            ("Market Pricing", "raw_market_pricing_benchmark", "silver_market_pricing_benchmark"),
            ("Geospatial Hazard", "raw_geospatial_hazard_enrichment", "silver_geospatial_hazard_enrichment"),
            ("Credit Bureau", "raw_credit_bureau_summary", "silver_credit_bureau_summary"),
        ]:
            r = await execute_query(f"SELECT count(*) as cnt FROM {fqn(raw)}")
            s = await execute_query(f"SELECT count(*) as cnt FROM {fqn(silver)}")
            raw_cnt = int(r[0]["cnt"]) if r else 0
            silver_cnt = int(s[0]["cnt"]) if s else 0
            dq.append({
                "dataset": ds,
                "raw_rows": raw_cnt,
                "silver_rows": silver_cnt,
                "dropped": raw_cnt - silver_cnt,
                "pass_rate": round(silver_cnt / raw_cnt * 100, 1) if raw_cnt else 0,
            })
    except Exception:
        pass

    # Delta history for gold table
    lineage = []
    try:
        lineage = await execute_query(f"""
            SELECT version, timestamp, operation, userName
            FROM (DESCRIBE HISTORY {fqn('unified_pricing_table_live')} LIMIT 10)
            ORDER BY version DESC
        """)
    except Exception:
        pass

    # Agent usage
    agent_events = []
    try:
        agent_events = await execute_query(f"""
            SELECT event_id, user_id, timestamp, details
            FROM {fqn('audit_log')}
            WHERE event_type = 'agent_recommendation'
            ORDER BY timestamp DESC LIMIT 10
        """)
    except Exception:
        pass

    return {
        "events_by_type": events_by_type,
        "recent_activity": recent,
        "data_quality": dq,
        "delta_lineage": lineage,
        "agent_usage": agent_events,
        "workspace_host": host,
    }
