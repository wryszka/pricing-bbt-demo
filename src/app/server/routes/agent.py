"""AI Agent routes (OPTIONAL).

Three agents, all optional, all fully logged and auditable:
1. Model Selection — recommends which models to train
2. DQ Monitor — detects data quality anomalies beyond rule-based checks
3. Explainability — explains pricing shifts in plain English for actuaries
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
        ready = ep.state and "READY" in str(ep.state.ready)
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


# ---------------------------------------------------------------------------
# DQ Monitor Agent
# ---------------------------------------------------------------------------

@router.post("/dq-monitor")
async def run_dq_monitor():
    """Run the DQ monitoring agent against all external datasets."""
    endpoint = os.getenv("AGENT_MODEL_ENDPOINT", DEFAULT_MODEL_ENDPOINT)

    # Profile each dataset
    datasets_config = [
        ("market_pricing_benchmark", "raw_market_pricing_benchmark", "silver_market_pricing_benchmark",
         ["market_median_rate", "competitor_a_min_premium", "price_index_trend"]),
        ("geospatial_hazard_enrichment", "raw_geospatial_hazard_enrichment", "silver_geospatial_hazard_enrichment",
         ["flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"]),
        ("credit_bureau_summary", "raw_credit_bureau_summary", "silver_credit_bureau_summary",
         ["credit_score", "ccj_count", "years_trading", "director_changes"]),
    ]

    profile_text = ""
    for ds_name, raw_t, silver_t, cols in datasets_config:
        try:
            raw_q = await execute_query(f"SELECT count(*) as cnt FROM {fqn(raw_t)}")
            silver_q = await execute_query(f"SELECT count(*) as cnt FROM {fqn(silver_t)}")
            raw_cnt = int(raw_q[0]["cnt"]) if raw_q else 0
            silver_cnt = int(silver_q[0]["cnt"]) if silver_q else 0
            drop_rate = round((raw_cnt - silver_cnt) / raw_cnt * 100, 1) if raw_cnt else 0

            stats_parts = ", ".join(
                f"ROUND(AVG(CAST({c} AS DOUBLE)), 3) AS {c}_mean, "
                f"ROUND(STDDEV(CAST({c} AS DOUBLE)), 3) AS {c}_std, "
                f"ROUND(SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS {c}_null"
                for c in cols
            )
            stats = await execute_query(f"SELECT {stats_parts} FROM {fqn(raw_t)}")
            s = stats[0] if stats else {}

            profile_text += f"Dataset: {ds_name} (raw={raw_cnt}, silver={silver_cnt}, drop={drop_rate}%)\n"
            for c in cols:
                profile_text += f"  {c}: mean={s.get(f'{c}_mean','?')}, std={s.get(f'{c}_std','?')}, null={s.get(f'{c}_null','?')}%\n"
            profile_text += "\n"
        except Exception as e:
            profile_text += f"Dataset: {ds_name} — error profiling: {str(e)[:80]}\n\n"

    system_prompt = """You are a data quality monitoring agent for P&C insurance pricing.
Analyse data profiles and detect anomalies. Classify each as CRITICAL, WARNING, or INFO.
Respond with valid JSON:
{"findings": [{"dataset": "name", "column": "name", "severity": "CRITICAL|WARNING|INFO",
  "finding": "description", "evidence": "numbers", "suggested_action": "action",
  "pricing_impact": "how this affects pricing"}],
 "overall_assessment": "summary", "recommended_priority": "what to fix first"}"""

    user_prompt = f"Analyse these data quality profiles:\n\n{profile_text}"

    llm_response_text = ""
    llm_success = False
    try:
        w = get_workspace_client()
        resp = w.serving_endpoints.query(
            name=endpoint, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], max_tokens=3000, temperature=0.1,
        )
        llm_response_text = resp.choices[0].message.content
        llm_success = True
    except Exception as e:
        llm_response_text = f"LLM call failed: {e}"

    findings = None
    if llm_success:
        jt = llm_response_text
        if "```json" in jt: jt = jt.split("```json")[1].split("```")[0]
        elif "```" in jt: jt = jt.split("```")[1].split("```")[0]
        try:
            findings = json.loads(jt.strip())
        except json.JSONDecodeError:
            pass

    await log_audit_event(
        event_type="agent_recommendation", entity_type="dataset",
        entity_id="dq_monitor",
        details={"agent_type": "dq_monitor", "llm_success": llm_success,
                 "findings_count": len((findings or {}).get("findings", []))},
    )

    return {
        "success": llm_success, "endpoint": endpoint, "findings": findings,
        "transparency": {"system_prompt": system_prompt, "user_prompt": user_prompt,
                         "raw_response": llm_response_text},
    }


# ---------------------------------------------------------------------------
# Explainability Agent
# ---------------------------------------------------------------------------

class ExplainRequest(BaseModel):
    question: str = "Why did premiums change in the latest data update?"


@router.post("/explain")
async def run_explainability(req: ExplainRequest):
    """Explain pricing shifts in plain English for actuarial use."""
    endpoint = os.getenv("AGENT_MODEL_ENDPOINT", DEFAULT_MODEL_ENDPOINT)
    upt_table = fqn("unified_pricing_table_live")

    # Gather context
    try:
        portfolio = await execute_query(f"""
            SELECT count(*) as total_policies, round(sum(current_premium)) as total_gwp,
                   round(avg(current_premium)) as avg_premium,
                   round(avg(combined_risk_score), 2) as avg_risk
            FROM {upt_table}
        """)
    except Exception:
        portfolio = [{}]

    shadow_context = ""
    try:
        shadow = await execute_query(f"""
            SELECT count(*) as affected, round(sum(premium_delta)) as total_delta,
                   round(avg(premium_delta_pct), 1) as avg_pct,
                   sum(case when churn_risk = 'HIGH' then 1 else 0 end) as high_churn
            FROM {fqn('shadow_pricing_impact')}
        """)
        s = shadow[0] if shadow else {}
        shadow_context = f"Shadow pricing: {s.get('affected',0)} affected, delta=£{s.get('total_delta',0)}, avg={s.get('avg_pct',0)}%"

        by_ind = await execute_query(f"""
            SELECT industry_risk_tier, count(*) as policies, round(sum(premium_delta)) as delta
            FROM {fqn('shadow_pricing_impact')} GROUP BY industry_risk_tier
        """)
        for r in by_ind:
            shadow_context += f"\n  {r.get('industry_risk_tier','?')}: {r.get('policies',0)} policies, £{r.get('delta',0)}"
    except Exception:
        shadow_context = "No shadow pricing data available"

    p = portfolio[0] if portfolio else {}
    context = f"""Portfolio: {p.get('total_policies',0)} policies, £{p.get('total_gwp',0)} GWP, avg premium £{p.get('avg_premium',0)}
{shadow_context}"""

    system_prompt = """You are an actuarial explainability agent. Explain pricing changes in plain English
suitable for regulatory filings. Ground claims in data. Respond with valid JSON:
{"headline": "one sentence", "explanation": "2-3 paragraphs",
 "key_drivers": [{"factor": "name", "contribution": "amount", "detail": "explanation"}],
 "affected_segments": [{"segment": "name", "policies": N, "premium_impact": "£X"}],
 "regulatory_statement": "paragraph for regulator",
 "recommended_actions": ["action1", "action2"]}"""

    user_prompt = f"Question: {req.question}\n\nContext:\n{context}"

    llm_response_text = ""
    llm_success = False
    try:
        w = get_workspace_client()
        resp = w.serving_endpoints.query(
            name=endpoint, messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ], max_tokens=3000, temperature=0.1,
        )
        llm_response_text = resp.choices[0].message.content
        llm_success = True
    except Exception as e:
        llm_response_text = f"LLM call failed: {e}"

    explanation = None
    if llm_success:
        jt = llm_response_text
        if "```json" in jt: jt = jt.split("```json")[1].split("```")[0]
        elif "```" in jt: jt = jt.split("```")[1].split("```")[0]
        try:
            explanation = json.loads(jt.strip())
        except json.JSONDecodeError:
            pass

    await log_audit_event(
        event_type="agent_recommendation", entity_type="model",
        entity_id="explainability_agent",
        details={"agent_type": "explainability", "question": req.question,
                 "llm_success": llm_success,
                 "headline": (explanation or {}).get("headline", "")},
    )

    return {
        "success": llm_success, "endpoint": endpoint, "explanation": explanation,
        "transparency": {"system_prompt": system_prompt, "user_prompt": user_prompt,
                         "raw_response": llm_response_text},
    }
