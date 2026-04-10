"""AI Model Selection Agent routes (OPTIONAL).

Provides an LLM-powered analysis of the feature table with model
training recommendations. Fully logged and auditable.

This is entirely optional — the demo works perfectly without it.
All AI interactions are presented as recommendations, never auto-executed.
"""

import json
import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import fqn, get_workspace_client, get_current_user
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

# Default endpoint — configurable via env var
DEFAULT_MODEL_ENDPOINT = "databricks-claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an expert actuarial pricing AI advisor for a P&C insurance company.
You are analysing a Unified Pricing Table (wide denormalized feature table) to recommend
which pricing models should be trained.

CONTEXT: Commercial property & casualty insurance. The table contains policy data, claims
history, market benchmarks, geospatial risk scores, credit bureau data, and derived features.

RESPONSE FORMAT: Return valid JSON with this exact structure:
{
  "recommendations": [
    {
      "model_name": "string - descriptive name",
      "model_type": "GLM_Poisson | GLM_Gamma | GBM_Classifier | GBM_Regressor",
      "target_variable": "column name",
      "purpose": "what this model predicts and why",
      "recommended_features": ["list of column names"],
      "feature_rationale": "why these features",
      "regulatory_notes": "regulatory considerations",
      "priority": "high | medium | low"
    }
  ],
  "data_quality_observations": ["list of observations"],
  "overall_strategy": "plain English modelling strategy"
}"""


@router.get("/status")
async def agent_status():
    """Check if the AI agent is available."""
    endpoint = os.getenv("AGENT_MODEL_ENDPOINT", DEFAULT_MODEL_ENDPOINT)
    try:
        w = get_workspace_client()
        ep = w.serving_endpoints.get(endpoint)
        ready = ep.state and ep.state.ready == "READY"
        return {
            "available": ready,
            "endpoint": endpoint,
            "message": "AI assistant is ready" if ready else "Endpoint not ready",
        }
    except Exception as e:
        return {
            "available": False,
            "endpoint": endpoint,
            "message": f"AI assistant unavailable: {str(e)[:100]}",
        }


@router.post("/analyze")
async def run_analysis():
    """Run the AI model selection analysis against the current UPT."""
    endpoint = os.getenv("AGENT_MODEL_ENDPOINT", DEFAULT_MODEL_ENDPOINT)
    upt_table = fqn("unified_pricing_table_live")

    # Step 1: Profile the UPT
    try:
        profile = await execute_query(f"""
            SELECT count(*) as row_count,
                   count(DISTINCT policy_id) as unique_policies
            FROM {upt_table}
        """)
        row_count = int(profile[0]["row_count"]) if profile else 0
    except Exception as e:
        raise HTTPException(500, f"Cannot profile UPT: {e}")

    # Get column stats
    cols_info = await execute_query(f"""
        SELECT column_name, data_type
        FROM information_schema.columns
        WHERE table_catalog = '{upt_table.split('.')[0]}'
          AND table_schema = '{upt_table.split('.')[1]}'
          AND table_name = '{upt_table.split('.')[2]}'
        ORDER BY ordinal_position
    """)

    numeric_cols = [c["column_name"] for c in cols_info
                    if c.get("data_type") in ("DOUBLE", "BIGINT", "INT", "LONG", "FLOAT", "DECIMAL")][:25]

    # Get basic stats for numeric columns
    stats_parts = ", ".join(
        f"ROUND(AVG(CAST({c} AS DOUBLE)), 2) AS `{c}_mean`, "
        f"ROUND(STDDEV(CAST({c} AS DOUBLE)), 2) AS `{c}_std`"
        for c in numeric_cols[:15]
    )
    if stats_parts:
        stats = await execute_query(f"SELECT {stats_parts} FROM {upt_table}")
        stats_dict = stats[0] if stats else {}
    else:
        stats_dict = {}

    # Build profile text
    profile_text = f"Table: {upt_table}\nRows: {row_count:,} | Columns: {len(cols_info)}\n\n"
    profile_text += "Key numeric columns (name | mean | std):\n"
    for c in numeric_cols[:15]:
        mean = stats_dict.get(f"{c}_mean", "?")
        std = stats_dict.get(f"{c}_std", "?")
        profile_text += f"  {c} | mean={mean} | std={std}\n"

    string_cols = [c["column_name"] for c in cols_info if c.get("data_type") == "STRING"][:8]
    for c in string_cols:
        profile_text += f"  {c} | string\n"

    user_prompt = f"""Analyse this pricing feature table and recommend models to train.

{profile_text}

Target variables:
- claim_count_5y: Claims count — FREQUENCY modelling
- total_incurred_5y: Total claims cost — SEVERITY modelling
- loss_ratio_5y: Loss ratio — alternative severity
- (quote history available separately for DEMAND modelling)

Requirements:
1. At least one frequency model (GLM preferred for regulatory)
2. At least one severity model
3. At least one demand/conversion model
4. Consider GBM uplift on GLM residuals
5. Max 25 features per model
6. Suitable for UK/European Solvency II regulatory submission"""

    # Step 2: Call the LLM
    llm_response_text = ""
    llm_success = False
    token_usage = {}

    try:
        w = get_workspace_client()
        response = w.serving_endpoints.query(
            name=endpoint,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=4000,
            temperature=0.1,
        )
        llm_response_text = response.choices[0].message.content
        llm_success = True
        if hasattr(response, "usage") and response.usage:
            token_usage = {
                "prompt_tokens": getattr(response.usage, "prompt_tokens", 0),
                "completion_tokens": getattr(response.usage, "completion_tokens", 0),
                "total_tokens": getattr(response.usage, "total_tokens", 0),
            }
    except Exception as e:
        llm_response_text = f"LLM call failed: {e}"

    # Step 3: Parse recommendations
    recommendations = None
    if llm_success:
        json_text = llm_response_text
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0]
        elif "```" in json_text:
            json_text = json_text.split("```")[1].split("```")[0]
        try:
            recommendations = json.loads(json_text.strip())
        except json.JSONDecodeError:
            pass

    # Step 4: Log to audit trail
    reviewer = get_current_user()
    await log_audit_event(
        event_type="agent_recommendation",
        entity_type="model",
        entity_id="agent_model_selector",
        user_id=reviewer,
        details={
            "model_endpoint": endpoint,
            "llm_success": llm_success,
            "token_usage": token_usage,
            "recommendations_count": len(recommendations.get("recommendations", [])) if recommendations else 0,
            "upt_rows": row_count,
            "upt_columns": len(cols_info),
        },
    )

    return {
        "success": llm_success,
        "endpoint": endpoint,
        "token_usage": token_usage,
        "recommendations": recommendations,
        "profile": {
            "table": upt_table,
            "row_count": row_count,
            "column_count": len(cols_info),
        },
        "transparency": {
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": user_prompt,
            "raw_response": llm_response_text,
        },
    }
