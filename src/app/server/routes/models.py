"""Model Factory review and approval routes.

Provides endpoints for:
1. Listing factory runs
2. Viewing the leaderboard for a run
3. Viewing model details (metrics, relativities, importances)
4. Recording actuary approve/reject decisions
5. Viewing the audit trail for a run
"""

import logging
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import fqn, get_current_user
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/models", tags=["models"])


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ModelDecisionRequest(BaseModel):
    decision: str  # "APPROVED", "REJECTED", "DEFERRED"
    reviewer_notes: str = ""
    conditions: str = ""


# ---------------------------------------------------------------------------
# Ensure tables exist
# ---------------------------------------------------------------------------

async def ensure_model_factory_tables():
    """Create model factory tables if they don't exist yet."""
    await execute_query(f"""
        CREATE TABLE IF NOT EXISTS {fqn('mf_actuary_decisions')} (
            decision_id STRING,
            factory_run_id STRING,
            model_config_id STRING,
            mlflow_run_id STRING,
            decision STRING,
            reviewer STRING,
            reviewer_notes STRING,
            decided_at STRING,
            regulatory_sign_off BOOLEAN,
            conditions STRING
        )
    """)
    await execute_query(f"""
        CREATE TABLE IF NOT EXISTS {fqn('mf_audit_log')} (
            event_id STRING,
            factory_run_id STRING,
            event_type STRING,
            event_timestamp STRING,
            actor STRING,
            details_json STRING,
            mlflow_run_id STRING,
            upt_table_version BIGINT
        )
    """)


# ---------------------------------------------------------------------------
# 1. List factory runs
# ---------------------------------------------------------------------------

@router.get("/runs")
async def list_factory_runs():
    """List all model factory runs with summary stats."""
    try:
        runs = await execute_query(f"""
            SELECT
                p.factory_run_id,
                p.created_at as started_at,
                count(DISTINCT p.model_config_id) as models_planned,
                count(DISTINCT l.model_config_id) as models_trained,
                count(DISTINCT CASE WHEN l.status = 'SUCCESS' THEN l.model_config_id END) as models_succeeded,
                count(DISTINCT CASE WHEN l.status = 'FAILED' THEN l.model_config_id END) as models_failed,
                count(DISTINCT d.model_config_id) as models_decided,
                count(DISTINCT CASE WHEN d.decision = 'APPROVED' THEN d.model_config_id END) as models_approved
            FROM {fqn('mf_training_plan')} p
            LEFT JOIN {fqn('mf_training_log')} l
                ON p.factory_run_id = l.factory_run_id AND p.model_config_id = l.model_config_id
            LEFT JOIN {fqn('mf_actuary_decisions')} d
                ON p.factory_run_id = d.factory_run_id AND p.model_config_id = d.model_config_id
            GROUP BY p.factory_run_id, p.created_at
            ORDER BY p.factory_run_id DESC
        """)
        return runs
    except Exception as e:
        logger.warning("Failed to list factory runs: %s", e)
        return []


# ---------------------------------------------------------------------------
# 2. Leaderboard for a run
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/leaderboard")
async def get_leaderboard(run_id: str):
    """Get the full leaderboard for a factory run."""
    try:
        leaderboard = await execute_query(f"""
            SELECT
                lb.*,
                d.decision,
                d.reviewer,
                d.decided_at,
                d.conditions
            FROM {fqn('mf_leaderboard')} lb
            LEFT JOIN {fqn('mf_actuary_decisions')} d
                ON lb.factory_run_id = d.factory_run_id
                AND lb.model_config_id = d.model_config_id
            WHERE lb.factory_run_id = '{run_id}'
            ORDER BY lb.target_column, lb.rank
        """)
        return leaderboard
    except Exception as e:
        logger.warning("Failed to get leaderboard: %s", e)
        raise HTTPException(500, f"Failed to load leaderboard: {e}")


# ---------------------------------------------------------------------------
# 3. Model detail
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/models/{config_id}")
async def get_model_detail(run_id: str, config_id: str):
    """Get detailed info for a specific model."""
    # Leaderboard entry
    lb = await execute_query(f"""
        SELECT * FROM {fqn('mf_leaderboard')}
        WHERE factory_run_id = '{run_id}' AND model_config_id = '{config_id}'
    """)
    if not lb:
        raise HTTPException(404, f"Model {config_id} not found in run {run_id}")

    # Training plan entry
    plan = await execute_query(f"""
        SELECT * FROM {fqn('mf_training_plan')}
        WHERE factory_run_id = '{run_id}' AND model_config_id = '{config_id}'
    """)

    # Training log entry
    log = await execute_query(f"""
        SELECT * FROM {fqn('mf_training_log')}
        WHERE factory_run_id = '{run_id}' AND model_config_id = '{config_id}'
    """)

    # Decision if any
    decision = await execute_query(f"""
        SELECT * FROM {fqn('mf_actuary_decisions')}
        WHERE factory_run_id = '{run_id}' AND model_config_id = '{config_id}'
        ORDER BY decided_at DESC LIMIT 1
    """)

    return {
        "leaderboard": lb[0],
        "plan": plan[0] if plan else None,
        "training_log": log[0] if log else None,
        "decision": decision[0] if decision else None,
    }


# ---------------------------------------------------------------------------
# 4. Record decision
# ---------------------------------------------------------------------------

@router.post("/runs/{run_id}/models/{config_id}/decide")
async def decide_model(run_id: str, config_id: str, req: ModelDecisionRequest):
    """Record actuary approval/rejection for a model."""
    if req.decision not in ("APPROVED", "REJECTED", "DEFERRED"):
        raise HTTPException(400, "Decision must be APPROVED, REJECTED, or DEFERRED")

    await ensure_model_factory_tables()

    # Verify model exists
    lb = await execute_query(f"""
        SELECT mlflow_run_id, composite_score, regulatory_suitability_score
        FROM {fqn('mf_leaderboard')}
        WHERE factory_run_id = '{run_id}' AND model_config_id = '{config_id}'
    """)
    if not lb:
        raise HTTPException(404, f"Model {config_id} not found in run {run_id}")

    reviewer = get_current_user()
    now = datetime.now(timezone.utc).isoformat()
    decision_id = f"DEC-{now.replace(':', '').replace('-', '').replace('.', '')[:20]}"

    await execute_query(f"""
        INSERT INTO {fqn('mf_actuary_decisions')} VALUES (
            '{decision_id}',
            '{run_id}',
            '{config_id}',
            '{lb[0].get("mlflow_run_id", "")}',
            '{req.decision}',
            '{reviewer}',
            '{req.reviewer_notes.replace("'", "''")}',
            '{now}',
            {str(req.decision == 'APPROVED').lower()},
            '{req.conditions.replace("'", "''")}'
        )
    """)

    # Audit log
    import json, uuid
    event = {
        "event_id": str(uuid.uuid4()),
        "factory_run_id": run_id,
        "event_type": f"ACTUARY_{req.decision}",
        "event_timestamp": now,
        "actor": reviewer,
        "details_json": json.dumps({
            "decision_id": decision_id,
            "model_config_id": config_id,
            "decision": req.decision,
            "notes": req.reviewer_notes,
            "conditions": req.conditions,
        }),
        "mlflow_run_id": lb[0].get("mlflow_run_id"),
        "upt_table_version": None,
    }
    cols = ", ".join(event.keys())
    vals = ", ".join(
        "NULL" if v is None else f"'{str(v).replace(chr(39), chr(39)+chr(39))}'"
        for v in event.values()
    )
    await execute_query(f"INSERT INTO {fqn('mf_audit_log')} ({cols}) VALUES ({vals})")

    return {
        "decision_id": decision_id,
        "factory_run_id": run_id,
        "model_config_id": config_id,
        "decision": req.decision,
        "reviewer": reviewer,
        "message": f"Model {config_id} has been {req.decision.lower()}.",
    }


# ---------------------------------------------------------------------------
# 5. Audit trail
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/audit")
async def get_audit_trail(run_id: str):
    """Get the full audit trail for a factory run."""
    try:
        events = await execute_query(f"""
            SELECT * FROM {fqn('mf_audit_log')}
            WHERE factory_run_id = '{run_id}'
            ORDER BY event_timestamp
        """)
        return events
    except Exception as e:
        logger.warning("Failed to get audit trail: %s", e)
        return []


# ---------------------------------------------------------------------------
# 6. Feature profile
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}/features")
async def get_feature_profile(run_id: str):
    """Get the feature profile for a factory run."""
    try:
        features = await execute_query(f"""
            SELECT * FROM {fqn('mf_feature_profile')}
            WHERE factory_run_id = '{run_id}'
            ORDER BY feature_group, feature_name
        """)
        return features
    except Exception as e:
        logger.warning("Failed to get feature profile: %s", e)
        return []
