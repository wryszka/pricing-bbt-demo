"""Governance routes.

Two groups:

1. Summary endpoint (preserved from earlier) — `/api/governance/summary` —
   powering the dashboard-style aggregated view.
2. Model Governance tab endpoints (new) — packs catalog, PDF viewing,
   agent chat against a pack, synthetic policy scoring story.

The Model Governance tab is the flagship post-promotion view. Agent chat
calls Databricks Foundation Model API (Claude Sonnet 4.6) directly —
Agent Framework endpoint can slot in later as a one-line swap.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import (
    fqn, get_catalog, get_current_user, get_schema,
    get_workspace_client, get_workspace_host,
)
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/governance", tags=["governance"])

FAMILIES = [
    {"key": "freq_glm",         "label": "Frequency (GLM)"},
    {"key": "sev_glm",          "label": "Severity (GLM)"},
    {"key": "demand_gbm",       "label": "Demand (GBM)"},
    {"key": "fraud_gbm",        "label": "Fraud (GBM)"},
    {"key": "freq_glm_motor",   "label": "Motor Frequency (GLM)"},
    {"key": "sev_glm_motor",    "label": "Motor Severity (GLM)"},
    {"key": "demand_gbm_motor", "label": "Motor Demand (GBM)"},
    {"key": "fraud_gbm_motor",  "label": "Motor Fraud (GBM)"},
]
# Real Databricks Agent Framework endpoint — deployed from governance_agent.py
# via the governance_agent_deploy bundle job. The agent has 3 tools it calls
# on-demand (query_pack_index, read_pack_artefact, query_audit_log) and
# returns a tool-use trace for the UI "Show full LLM interaction" panel.
AGENT_ENDPOINT = "pricing_governance_agent"
# Direct FM call as fallback if the agent endpoint is unavailable (e.g.
# during first deploy / cold start). Preserves the chat UX.
FM_ENDPOINT = "databricks-claude-sonnet-4-6"


# ---------------------------------------------------------------------------
# Summary (preserved)
# ---------------------------------------------------------------------------

@router.get("/summary")
async def governance_summary():
    """Aggregate governance data across all systems."""
    host = get_workspace_host()

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
                "dataset": ds, "raw_rows": raw_cnt, "silver_rows": silver_cnt,
                "dropped": raw_cnt - silver_cnt,
                "pass_rate": round(silver_cnt / raw_cnt * 100, 1) if raw_cnt else 0,
            })
    except Exception:
        pass

    lineage = []
    try:
        lineage = await execute_query(f"""
            SELECT version, timestamp, operation, userName
            FROM (DESCRIBE HISTORY {fqn('unified_pricing_table_live')} LIMIT 10)
            ORDER BY version DESC
        """)
    except Exception:
        pass

    return {
        "events_by_type": events_by_type,
        "recent_activity": recent,
        "data_quality": dq,
        "delta_lineage": lineage,
        "workspace_host": host,
    }


# ---------------------------------------------------------------------------
# Packs catalog
# ---------------------------------------------------------------------------

@router.get("/packs")
async def list_packs() -> dict:
    """Return every pack in the index, grouped by family."""
    try:
        rows = await execute_query(f"""
            SELECT pack_id, model_family, model_version, model_uc_name,
                   mlflow_run_id, story, simulated, primary_metric, primary_value,
                   pdf_path, size_bytes, generated_by, generated_at
            FROM {fqn('governance_packs_index')}
            ORDER BY generated_at DESC
        """)
    except Exception as e:
        logger.info("governance_packs_index not queryable yet: %s", e)
        return {"families": [{"key": f["key"], "label": f["label"], "packs": []} for f in FAMILIES]}

    by_family: dict[str, list] = {f["key"]: [] for f in FAMILIES}
    for r in rows:
        fam = r.get("model_family")
        if fam not in by_family:
            continue
        by_family[fam].append({
            "pack_id":       r["pack_id"],
            "model_family":  fam,
            "model_version": r["model_version"],
            "story":         r.get("story"),
            "simulated":     r.get("simulated"),
            "primary_metric":r.get("primary_metric"),
            "primary_value": r.get("primary_value"),
            "pdf_path":      r.get("pdf_path"),
            "size_bytes":    r.get("size_bytes"),
            "generated_by":  r.get("generated_by"),
            "generated_at":  str(r.get("generated_at", "")),
        })
    return {
        "families": [
            {"key": f["key"], "label": f["label"], "packs": by_family.get(f["key"], [])}
            for f in FAMILIES
        ]
    }


@router.get("/packs/by-date")
async def packs_on_date(date: str) -> dict:
    """Return the pack that was the most-recent-at-or-before `date` for each
    family. Used by the By-date entry point to show what was in force on a
    historical day."""
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except Exception:
        raise HTTPException(400, "date must be YYYY-MM-DD")

    try:
        rows = await execute_query(f"""
            SELECT model_family, pack_id, model_version, story, primary_metric,
                   primary_value, pdf_path, generated_by, generated_at
            FROM (
                SELECT *, row_number() OVER (PARTITION BY model_family
                                             ORDER BY generated_at DESC) AS rn
                FROM {fqn('governance_packs_index')}
                WHERE CAST(generated_at AS DATE) <= DATE('{date}')
            )
            WHERE rn = 1
            ORDER BY model_family
        """)
    except Exception as e:
        logger.warning("by-date query failed: %s", e)
        return {"date": date, "packs": []}
    return {
        "date": date,
        "packs": [{
            "model_family":  r["model_family"],
            "pack_id":       r["pack_id"],
            "model_version": r["model_version"],
            "story":         r.get("story"),
            "primary_metric":r.get("primary_metric"),
            "primary_value": r.get("primary_value"),
            "generated_by":  r.get("generated_by"),
            "generated_at":  str(r.get("generated_at", "")),
        } for r in rows],
    }


@router.get("/packs/{pack_id}")
async def pack_detail(pack_id: str) -> dict:
    rows = await execute_query(f"""
        SELECT pack_id, model_family, model_version, model_uc_name,
               mlflow_run_id, story, simulated, primary_metric, primary_value,
               pdf_path, size_bytes, generated_by, generated_at
        FROM {fqn('governance_packs_index')}
        WHERE pack_id = '{pack_id}'
        LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"pack {pack_id} not found")

    await log_audit_event(
        event_type="governance_pack_viewed",
        entity_type="model",
        entity_id=rows[0]["model_family"],
        entity_version=str(rows[0]["model_version"]),
        details={"pack_id": pack_id},
    )
    r = rows[0]
    return {
        "pack_id": r["pack_id"], "model_family": r["model_family"],
        "model_version": r["model_version"], "model_uc_name": r["model_uc_name"],
        "mlflow_run_id": r["mlflow_run_id"], "story": r.get("story"),
        "simulated": r.get("simulated"),
        "primary_metric": r.get("primary_metric"),
        "primary_value":  r.get("primary_value"),
        "pdf_path": r.get("pdf_path"),
        "size_bytes": r.get("size_bytes"),
        "generated_by": r.get("generated_by"),
        "generated_at": str(r.get("generated_at", "")),
        "pdf_url": f"/api/governance/packs/{pack_id}/pdf",
    }


@router.get("/packs/{pack_id}/pdf")
async def pack_pdf(pack_id: str):
    """Stream the PDF from the UC volume so the frontend can display it
    inline (iframe or <object>)."""
    rows = await execute_query(f"""
        SELECT pdf_path, model_family, model_version
        FROM {fqn('governance_packs_index')}
        WHERE pack_id = '{pack_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"pack {pack_id} not found")
    path = rows[0]["pdf_path"]
    try:
        resp = get_workspace_client().files.download(file_path=path)
        data = resp.contents.read() if hasattr(resp.contents, "read") else resp.contents
    except Exception as e:
        raise HTTPException(500, f"Could not download PDF: {e}")
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/pdf",
        headers={"Content-Disposition": f"inline; filename={pack_id}.pdf"},
    )


# ---------------------------------------------------------------------------
# PDF text cache — extract once, reuse for the chat
# ---------------------------------------------------------------------------

_pdf_text_cache: dict[str, str] = {}


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    """Pull plain text from the pack PDF for use as agent context."""
    try:
        from pypdf import PdfReader
    except Exception as e:
        logger.warning("pypdf not available: %s", e)
        return ""
    try:
        reader = PdfReader(io.BytesIO(pdf_bytes))
        pages = []
        for i, p in enumerate(reader.pages):
            try:
                t = p.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                pages.append(f"--- page {i+1} ---\n{t}")
        return "\n".join(pages)
    except Exception as e:
        logger.warning("PDF text extract failed: %s", e)
        return ""


async def _pack_text(pack_id: str) -> tuple[dict, str]:
    """Return (pack metadata row, extracted text). Caches the text per pack."""
    rows = await execute_query(f"""
        SELECT pack_id, model_family, model_version, pdf_path
        FROM {fqn('governance_packs_index')}
        WHERE pack_id = '{pack_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"pack {pack_id} not found")
    r = rows[0]
    if pack_id in _pdf_text_cache:
        return r, _pdf_text_cache[pack_id]
    try:
        resp = get_workspace_client().files.download(file_path=r["pdf_path"])
        data = resp.contents.read() if hasattr(resp.contents, "read") else resp.contents
    except Exception as e:
        logger.warning("Pack PDF download failed: %s", e)
        return r, ""
    text = _extract_pdf_text(data)
    _pdf_text_cache[pack_id] = text
    return r, text


@router.get("/packs/{pack_id}/text")
async def pack_text(pack_id: str) -> dict:
    r, text = await _pack_text(pack_id)
    return {
        "pack_id": pack_id,
        "model_family": r["model_family"],
        "model_version": r["model_version"],
        "text_length": len(text),
        "preview": text[:3000],
    }


# ---------------------------------------------------------------------------
# Agent chat — Foundation Model API (Claude Sonnet 4.6)
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    pack_id: str
    question: str
    policy_id: str | None = None   # only populated on by-policy flow


SYSTEM_PROMPT = """You are a model-governance assistant for Bricksurance SE's pricing committee.
You help compliance officers, senior actuaries, and regulators understand a specific model version
by answering questions strictly from that model's governance pack.

Rules you MUST follow:
 * Answer ONLY using information contained in the pack text provided in the user message.
 * Cite the pack section whenever you quote a fact (e.g., "see Section 4 — Model specification").
 * If the pack does not contain the information needed, reply exactly: "The pack does not document this — further investigation required." Do not guess.
 * Never speculate about fairness, bias, or model behaviour beyond what is documented.
 * Keep answers concise (4-8 sentences unless the user asks for more detail).
 * When drafting regulator/customer responses, phrase them carefully and stay grounded in the pack.
"""


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """Delegate the question to the governance Agent Framework endpoint.

    The agent uses tools to look up pack metadata, read sidecar artefacts, and
    query the audit log. If the agent endpoint is unavailable (cold start,
    first deploy) we fall back to the direct Foundation Model API call over
    the pack's PDF text so the chat panel still works.
    """
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    pack_row = (await execute_query(f"""
        SELECT pack_id, model_family, model_version, pdf_path
        FROM {fqn('governance_packs_index')}
        WHERE pack_id = '{req.pack_id}' LIMIT 1
    """) or [{}])[0]
    if not pack_row:
        raise HTTPException(404, f"pack {req.pack_id} not found")

    # Try the real agent endpoint first — wrapped via to_thread because the
    # function does sync HTTP + workspace SDK calls that would otherwise
    # block the event loop while the agent's tool-use loop runs.
    import asyncio
    agent_result = await asyncio.to_thread(_query_agent_endpoint, req.pack_id, req.question, req.policy_id)

    if agent_result.get("ok"):
        answer = agent_result["answer"]
        trace  = agent_result.get("trace", [])
        model  = agent_result.get("model", AGENT_ENDPOINT)
        usage  = agent_result.get("usage", {})
        sections = sorted(set(re.findall(r"[Ss]ection\s+(\d+)", answer or "")))

        # Audit: capture every tool call for governance continuity
        await log_audit_event(
            event_type="governance_pack_chat",
            entity_type="model",
            entity_id=pack_row["model_family"],
            entity_version=str(pack_row["model_version"]),
            details={
                "pack_id": req.pack_id,
                "question": req.question[:500],
                "answer_length": len(answer or ""),
                "policy_id": req.policy_id,
                "cited_sections": sections,
                "model": model,
                "endpoint": AGENT_ENDPOINT,
                "tool_trace": [{"tool": t.get("tool"),
                                 "args": t.get("arguments"),
                                 "result_summary": t.get("result_summary")}
                                for t in trace],
                "usage": usage,
            },
        )
        return {
            "pack_id":        req.pack_id,
            "model_family":   pack_row["model_family"],
            "model_version":  pack_row["model_version"],
            "question":       req.question,
            "answer":         answer,
            "cited_sections": sections,
            "tool_trace":     trace,
            "model":          model,
            "endpoint":       AGENT_ENDPOINT,
            "usage":          usage,
            "source":         "agent_framework",
        }

    # Fallback — FM API direct over PDF text
    logger.warning("Agent endpoint unavailable (%s) — falling back to FM API",
                   agent_result.get("error", "unknown"))
    _, text = await _pack_text(req.pack_id)
    truncated = (text or "")[:40000] or "(pack text could not be extracted)"
    user_content = (
        f"Pack:\n  family:  {pack_row['model_family']}\n"
        f"  version: {pack_row['model_version']}\n  pack_id: {pack_row['pack_id']}\n\n"
        f"Pack contents (plain text extracted from PDF):\n===\n{truncated}\n===\n\n"
        f"User question: {req.question}"
    )
    if req.policy_id:
        user_content += f"\n\nContext: this question concerns policy_id={req.policy_id}."

    try:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        fm_resp = get_workspace_client().serving_endpoints.query(
            name=FM_ENDPOINT,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=SYSTEM_PROMPT),
                ChatMessage(role=ChatMessageRole.USER,   content=user_content),
            ],
            max_tokens=800, temperature=0.2,
        )
    except Exception as e:
        logger.exception("FM API fallback also failed")
        return {
            "pack_id":  req.pack_id,
            "model":    FM_ENDPOINT,
            "answer":   f"Chat temporarily unavailable ({e}).",
            "error":    str(e)[:300],
            "source":   "unavailable",
        }

    answer = ""
    try:
        choices = getattr(fm_resp, "choices", None) or fm_resp.get("choices", [])
        if choices:
            m = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message", {})
            answer = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
    except Exception:
        answer = str(fm_resp)[:2000]

    sections = sorted(set(re.findall(r"[Ss]ection\s+(\d+)", answer or "")))
    usage = {}
    try:
        u = getattr(fm_resp, "usage", None) or (fm_resp.get("usage") if isinstance(fm_resp, dict) else None)
        if u:
            usage = {
                "prompt_tokens":     getattr(u, "prompt_tokens", None) or u.get("prompt_tokens"),
                "completion_tokens": getattr(u, "completion_tokens", None) or u.get("completion_tokens"),
                "total_tokens":      getattr(u, "total_tokens", None) or u.get("total_tokens"),
            }
    except Exception:
        pass

    await log_audit_event(
        event_type="governance_pack_chat",
        entity_type="model",
        entity_id=pack_row["model_family"],
        entity_version=str(pack_row["model_version"]),
        details={
            "pack_id": req.pack_id,
            "question": req.question[:500],
            "answer_length": len(answer or ""),
            "policy_id": req.policy_id,
            "cited_sections": sections,
            "model": FM_ENDPOINT,
            "endpoint": FM_ENDPOINT,
            "source": "fm_api_fallback",
            "fallback_reason": agent_result.get("error", "unknown")[:200],
            "usage": usage,
        },
    )
    return {
        "pack_id":        req.pack_id,
        "model_family":   pack_row["model_family"],
        "model_version":  pack_row["model_version"],
        "question":       req.question,
        "answer":         answer,
        "cited_sections": sections,
        "tool_trace":     [],
        "model":          FM_ENDPOINT,
        "endpoint":       FM_ENDPOINT,
        "usage":          usage,
        "source":         "fm_api_fallback",
        "fallback_reason": agent_result.get("error", "agent unavailable"),
    }


def _query_agent_endpoint(pack_id: str, question: str, policy_id: str | None) -> dict:
    """Call the Databricks Agent Framework serving endpoint. Returns a result
    dict with `ok` flag plus answer/trace/model/usage or error info."""
    import requests as _rq
    try:
        w = get_workspace_client()
        # Confirm the endpoint is ready before invoking
        try:
            ep = w.serving_endpoints.get(AGENT_ENDPOINT)
            state = ep.state.ready if ep.state and ep.state.ready else None
            if state and "READY" not in str(state):
                return {"ok": False, "error": f"endpoint not ready (state={state})"}
        except Exception as e:
            return {"ok": False, "error": f"endpoint lookup failed: {e}"}

        host  = w.config.host.rstrip("/")
        token = w.config._header_factory()
        # The deployed agent's input signature only declares the pack_id field
        # in custom_inputs — additional fields are rejected. We squeeze the
        # policy_id context into the question text when present.
        q = question if not policy_id else f"[policy_id={policy_id}] {question}"
        body = {
            "dataframe_records": [{
                "messages": [{"role": "user", "content": q}],
                "custom_inputs": {"pack_id": pack_id},
            }],
        }
        # Long timeout — the agent may make several tool calls (SQL + volume
        # reads) before returning. First invocation after cold start is the
        # slowest.
        resp = _rq.post(
            f"{host}/serving-endpoints/{AGENT_ENDPOINT}/invocations",
            headers={**token, "Content-Type": "application/json"},
            json=body, timeout=240,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("Agent endpoint call failed: %s", e)
        return {"ok": False, "error": str(e)[:300]}

    # MLflow serving wraps pyfunc output — usually under "predictions"
    preds = data.get("predictions") or data.get("outputs") or data
    if isinstance(preds, list):
        preds = preds[0] if preds else {}
    if not isinstance(preds, dict):
        return {"ok": False, "error": f"unexpected response shape: {type(preds).__name__}"}

    messages = preds.get("messages") or []
    answer = ""
    if messages:
        msg = messages[0] if isinstance(messages[0], dict) else {}
        answer = msg.get("content") or ""

    return {
        "ok":     True,
        "answer": answer,
        "trace":  preds.get("trace", []),
        "model":  preds.get("model", AGENT_ENDPOINT),
        "usage":  preds.get("usage", {}),
    }


# ---------------------------------------------------------------------------
# By-policy scoring story (synthetic, deterministic per policy_id)
# ---------------------------------------------------------------------------

def _seeded_float(seed: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    r = (h % 100000) / 100000.0
    return lo + r * (hi - lo)


@router.get("/policy/{policy_id}/scoring")
async def policy_scoring(policy_id: str) -> dict:
    """Return the scoring story for a policy.

    Prefers real predictions from `{fqn}.inference_logs` (written by the
    `inference_backfill` job, which scores every UPT policy with the current
    champions). Falls back to deterministic hash-seeded predictions when the
    table doesn't exist or the policy hasn't been scored yet."""
    policy_id = policy_id.strip().upper()
    rows = await execute_query(f"""
        SELECT policy_id, current_premium, sum_insured, annual_turnover,
               industry_risk_tier, construction_type, region, postcode_sector,
               flood_zone_rating, credit_score, claim_count_5y, total_incurred_5y,
               is_coastal, urban_score
        FROM {fqn('unified_pricing_table_live')}
        WHERE policy_id = '{policy_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"policy {policy_id} not found in Modelling Mart")
    row = rows[0]

    # ---- try real inference_logs first ----
    simulated = True
    inf_row: dict[str, Any] | None = None
    try:
        inf_rows = await execute_query(f"""
            SELECT policy_id, scored_at,
                   freq_pred, freq_version,
                   sev_pred,  sev_version,
                   demand_pred, demand_version,
                   fraud_pred,  fraud_version,
                   base_premium, fraud_loading, demand_adj, technical_premium
            FROM {fqn('inference_logs')}
            WHERE policy_id = '{policy_id}' LIMIT 1
        """)
        if inf_rows:
            inf_row   = inf_rows[0]
            simulated = False
    except Exception as e:
        logger.info("inference_logs not available for %s: %s", policy_id, e)

    if inf_row is not None:
        freq_pred         = float(inf_row["freq_pred"])  if inf_row["freq_pred"]  is not None else None
        sev_pred          = float(inf_row["sev_pred"])   if inf_row["sev_pred"]   is not None else None
        demand_p          = float(inf_row["demand_pred"])if inf_row["demand_pred"]is not None else None
        fraud_p           = float(inf_row["fraud_pred"]) if inf_row["fraud_pred"] is not None else None
        base_premium      = float(inf_row["base_premium"]      or 0)
        fraud_loading     = float(inf_row["fraud_loading"]     or 0)
        demand_adj        = float(inf_row["demand_adj"]        or 0)
        technical_premium = float(inf_row["technical_premium"] or 0)
        model_versions    = {
            "freq_glm":   inf_row.get("freq_version"),
            "sev_glm":    inf_row.get("sev_version"),
            "demand_gbm": inf_row.get("demand_version"),
            "fraud_gbm":  inf_row.get("fraud_version"),
        }
        scored_at         = str(inf_row.get("scored_at", ""))
    else:
        # Fallback: deterministic hash-seeded predictions so the endpoint still
        # returns a plausible story pre-backfill.
        freq_pred  = round(_seeded_float(f"{policy_id}:freq",  0.05, 0.45), 4)
        sev_pred   = round(_seeded_float(f"{policy_id}:sev",   2_500.0, 18_000.0), 0)
        demand_p   = round(_seeded_float(f"{policy_id}:demand", 0.20, 0.85), 3)
        fraud_p    = round(_seeded_float(f"{policy_id}:fraud",  0.01, 0.45), 3)
        base_premium      = round(freq_pred * sev_pred, 2)
        fraud_loading     = round(base_premium * (0.05 if fraud_p > 0.25 else 0.0), 2)
        demand_adj        = round(base_premium * (0.02 if demand_p < 0.4 else -0.02), 2)
        technical_premium = round(base_premium + fraud_loading + demand_adj, 2)
        model_versions    = {k: None for k in ("freq_glm", "sev_glm", "demand_gbm", "fraud_gbm")}
        scored_at         = ""

    # Find the current champion pack for each family (most recent)
    try:
        pack_rows = await execute_query(f"""
            SELECT model_family, pack_id, model_version, pdf_path, generated_at
            FROM (
              SELECT *, row_number() OVER (PARTITION BY model_family ORDER BY generated_at DESC) AS rn
              FROM {fqn('governance_packs_index')}
            )
            WHERE rn = 1
        """)
    except Exception:
        pack_rows = []
    packs_by_fam = {r["model_family"]: r for r in pack_rows}

    def _fam(key, label, pred, unit):
        p = packs_by_fam.get(key) or {}
        return {
            "family":            key,
            "label":             label,
            "prediction":        pred,
            "unit":              unit,
            "pack_id":           p.get("pack_id"),
            "model_version":     model_versions.get(key) or p.get("model_version"),
            "pack_generated_at": str(p.get("generated_at", "")),
        }

    await log_audit_event(
        event_type="governance_policy_lookup",
        entity_type="policy",
        entity_id=policy_id,
        details={"scoring_simulated": simulated, "source": "inference_logs" if not simulated else "synth"},
    )

    return {
        "policy_id": policy_id,
        "simulated": simulated,
        "policy":    row,
        "models": [
            _fam("freq_glm",   "Frequency (GLM)",      freq_pred,  "claims/yr"),
            _fam("sev_glm",    "Severity (GLM)",       sev_pred,   "GBP"),
            _fam("demand_gbm", "Demand (GBM)",         demand_p,   "conversion p"),
            _fam("fraud_gbm",  "Fraud (GBM)",          fraud_p,    "fraud p"),
        ],
        "price_build_up": [
            {"label": "Base technical premium (freq × severity)", "amount": base_premium},
            {"label": "Fraud-risk loading",                        "amount": fraud_loading},
            {"label": "Demand-elasticity adjustment",              "amount": demand_adj},
            {"label": "Technical premium",                         "amount": technical_premium, "emphasis": True},
        ],
        "quote_timestamp": datetime.now(timezone.utc).isoformat(),
        "scored_at":       scored_at,
        "note": ("Real inference from the inference_logs table — scored against the current champions."
                 if not simulated else
                 "Scoring story synthesised from the policy's features (inference_logs empty or missing)."),
    }


# ---------------------------------------------------------------------------
# Free-form governance agent chat — no pack context required
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str


@router.post("/ask")
async def ask_governance_agent(req: AskRequest) -> dict:
    """Free-form chat against the governance Agent Framework endpoint. No
    pack_id required — the agent can use its tools (query_pack_index,
    read_pack_artefact, query_audit_log) to answer any governance question
    across the whole workbench."""
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    from server.agent_client import invoke_agent
    result = await invoke_agent(
        endpoint_name=AGENT_ENDPOINT,   # pricing_governance_agent
        question=req.question,
        custom_inputs={"pack_id": ""},  # pack-agnostic — agent picks up via tools
        timeout=300,
    )
    await log_audit_event(
        event_type="governance_pack_chat",
        entity_type="governance",
        entity_id="open_query",
        details={
            "question":      req.question[:500],
            "endpoint":      AGENT_ENDPOINT,
            "agent_ok":      result.get("ok"),
            "answer_length": len(result.get("answer") or ""),
            "trace":         result.get("trace", []),
            "usage":         result.get("usage", {}),
            "error":         result.get("error"),
        },
    )
    return {
        "ok":       result.get("ok"),
        "question": req.question,
        "answer":   result.get("answer") or "",
        "trace":    result.get("trace", []),
        "model":    result.get("model"),
        "usage":    result.get("usage", {}),
        "endpoint": AGENT_ENDPOINT,
        "error":    result.get("error"),
        "cached":   bool(result.get("cached")),
    }


# ---------------------------------------------------------------------------
# Bias monitor — aggregate stats + agent-driven investigation
# ---------------------------------------------------------------------------

CHAT_AGENT_ENDPOINT = "pricing_chat_agent"
_PROTECTED_ATTRS = {"director_gender", "postcode_demographic", "ethnicity_proxy", "director_age_band"}
_FAMILY_METRIC = {
    "freq_glm":   "freq_pred",
    "sev_glm":    "sev_pred",
    "demand_gbm": "demand_pred",
    "fraud_gbm":  "fraud_pred",
}


@router.get("/bias-monitor")
async def bias_monitor(
    protected_attribute: str = "director_gender",
    family: str | None = None,
) -> dict:
    """Return per-cohort prediction + premium stats for a protected attribute.
    Backs the Governance-tab dashboard card. No agent invocation — direct SQL
    read of inference_logs joined to policy_demographics. Fast and cacheable."""
    if protected_attribute not in _PROTECTED_ATTRS:
        raise HTTPException(400, f"protected_attribute must be one of {sorted(_PROTECTED_ATTRS)}")
    metric_col = _FAMILY_METRIC.get(family) if family else None
    if family and not metric_col:
        raise HTTPException(400, f"unknown family: {family}")

    if metric_col:
        sql = f"""
            SELECT d.{protected_attribute} AS cohort,
                   count(*) AS n,
                   round(avg(i.{metric_col}), 6)          AS metric,
                   round(avg(i.technical_premium), 2)     AS avg_premium
            FROM {fqn('policy_demographics')} d
            JOIN {fqn('inference_logs')}      i USING (policy_id)
            GROUP BY d.{protected_attribute}
            ORDER BY d.{protected_attribute}
        """
    else:
        sql = f"""
            SELECT d.{protected_attribute} AS cohort,
                   count(*) AS n,
                   round(avg(i.freq_pred),   6) AS freq_pred,
                   round(avg(i.sev_pred),    2) AS sev_pred,
                   round(avg(i.demand_pred), 6) AS demand_pred,
                   round(avg(i.fraud_pred),  6) AS fraud_pred,
                   round(avg(i.technical_premium), 2) AS avg_premium
            FROM {fqn('policy_demographics')} d
            JOIN {fqn('inference_logs')}      i USING (policy_id)
            GROUP BY d.{protected_attribute}
            ORDER BY d.{protected_attribute}
        """
    try:
        rows = await execute_query(sql)
    except Exception as e:
        raise HTTPException(500, f"bias_monitor SQL failed: {e}")

    # SQL Statement Execution API returns every column as a string — coerce
    # known numeric fields so the UI can .toFixed() / .toLocaleString() them.
    def _num(v):
        if v is None: return None
        try:    return float(v)
        except (TypeError, ValueError): return None
    for r in rows:
        for k in ("n", "metric", "avg_premium", "freq_pred", "sev_pred",
                 "demand_pred", "fraud_pred"):
            if k in r:
                r[k] = _num(r[k])
        # n is an integer count — keep it as int in JSON
        if isinstance(r.get("n"), float):
            r["n"] = int(r["n"])

    premium_values = [r["avg_premium"] for r in rows if r.get("avg_premium") is not None]
    headline = None
    if premium_values and len(premium_values) >= 2:
        mx, mn = max(premium_values), min(premium_values)
        headline = {
            "max_premium":  mx,
            "min_premium":  mn,
            "gap_abs":      round(mx - mn, 2),
            "gap_pct":      round((mx / mn - 1) * 100, 1) if mn > 0 else None,
        }

    return {
        "protected_attribute": protected_attribute,
        "family":              family or "all",
        "cohorts":             rows,
        "headline":            headline,
        "scan_timestamp":      datetime.now(timezone.utc).isoformat(),
    }


class BiasInvestigateRequest(BaseModel):
    question: str
    protected_attribute: str = "director_gender"
    family: str | None = None


@router.post("/bias-investigate")
async def bias_investigate(req: BiasInvestigateRequest) -> dict:
    """Live bias investigation — dispatches to the `pricing_chat_agent`
    endpoint with persona=bias_investigator. The agent runs its tool-use
    loop (bias monitor + actual experience + proxy features + pack fairness
    sections) and returns a structured DETECTION / DIAGNOSIS / JUSTIFICATION
    / EVIDENCE / MITIGATION / CONCLUSION response."""
    if req.protected_attribute not in _PROTECTED_ATTRS:
        raise HTTPException(400, f"protected_attribute must be one of {sorted(_PROTECTED_ATTRS)}")

    from server.agent_client import invoke_agent
    custom = {
        "persona":             "bias_investigator",
        "mode":                "live",
        "protected_attribute": req.protected_attribute,
    }
    if req.family:
        custom["family"] = req.family

    result = await invoke_agent(
        endpoint_name=CHAT_AGENT_ENDPOINT,
        question=req.question,
        custom_inputs=custom,
        timeout=300,
    )

    await log_audit_event(
        event_type="bias_investigation",
        entity_type="model",
        entity_id=req.family or "portfolio",
        details={
            "question":            req.question[:400],
            "protected_attribute": req.protected_attribute,
            "family":              req.family,
            "mode":                "live",
            "endpoint":            CHAT_AGENT_ENDPOINT,
            "agent_ok":            result.get("ok"),
            "model":               result.get("model"),
            "trace":               result.get("trace", []),
            "usage":               result.get("usage", {}),
            "error":               result.get("error"),
        },
    )

    return {
        "ok":        result.get("ok"),
        "question":  req.question,
        "answer":    result.get("answer") or "",
        "trace":     result.get("trace", []),
        "model":     result.get("model"),
        "usage":     result.get("usage", {}),
        "endpoint":  CHAT_AGENT_ENDPOINT,
        "error":     result.get("error"),
        "cached":    bool(result.get("cached")),
    }


class BiasReviewCandidateRequest(BaseModel):
    family: str
    version: str
    protected_attribute: str = "director_gender"
    question: str | None = None


@router.post("/bias-review-candidate")
async def bias_review_candidate(req: BiasReviewCandidateRequest) -> dict:
    """PRE-PROMOTION bias review. Asks the agent (mode=pre_promotion) to
    audit a specific candidate version for disparity before the actuary
    promotes it. The agent uses its `score_candidate_for_bias` tool and
    compares to the current champion."""
    if req.protected_attribute not in _PROTECTED_ATTRS:
        raise HTTPException(400, f"protected_attribute must be one of {sorted(_PROTECTED_ATTRS)}")
    if req.family not in _FAMILY_METRIC:
        raise HTTPException(400, f"unknown family: {req.family}")

    from server.agent_client import invoke_agent

    default_q = (
        f"Review candidate {req.family} v{req.version} for {req.protected_attribute} bias. "
        f"Compare the candidate's cohort stats to the current champion. Recommend whether it is "
        f"safe to promote, given actual loss experience and the governance pack."
    )
    question = req.question or default_q

    result = await invoke_agent(
        endpoint_name=CHAT_AGENT_ENDPOINT,
        question=question,
        custom_inputs={
            "persona":             "bias_investigator",
            "mode":                "pre_promotion",
            "family":              req.family,
            "version":             req.version,
            "protected_attribute": req.protected_attribute,
        },
        timeout=300,
    )

    await log_audit_event(
        event_type="bias_pre_promotion_review",
        entity_type="model",
        entity_id=req.family,
        entity_version=str(req.version),
        details={
            "question":            question[:400],
            "protected_attribute": req.protected_attribute,
            "mode":                "pre_promotion",
            "endpoint":            CHAT_AGENT_ENDPOINT,
            "agent_ok":            result.get("ok"),
            "model":               result.get("model"),
            "trace":               result.get("trace", []),
            "usage":               result.get("usage", {}),
            "error":               result.get("error"),
        },
    )

    return {
        "ok":        result.get("ok"),
        "family":    req.family,
        "version":   req.version,
        "question":  question,
        "answer":    result.get("answer") or "",
        "trace":     result.get("trace", []),
        "model":     result.get("model"),
        "usage":     result.get("usage", {}),
        "endpoint":  CHAT_AGENT_ENDPOINT,
        "error":     result.get("error"),
        "cached":    bool(result.get("cached")),
    }


# ---------------------------------------------------------------------------
# Data summary — what governance collects, how it's used
#
# Pure static. Reframed: "we collect everything end-to-end; below are example
# surfaces you can interrogate today, but any other slicing is a config change
# away because the data is there." No SQL, no row counts — answers the
# question of WHAT and HOW, not how-many.
# ---------------------------------------------------------------------------

@router.get("/data-summary")
async def data_summary() -> dict:
    """Static catalogue of inputs + example surfaces. Each input lists the
    examples we currently surface and what other slicings are possible without
    touching the data layer."""
    inputs = [
        {
            "key": "audit_log",
            "table": fqn("audit_log"),
            "label": "Audit log",
            "purpose": "Immutable record of every governance event: pack generation, model promotion/rollback, bias investigation, MTA reprice, dataset approval, agent dispatch.",
            "grain": "one row per event",
            "fields": ["event_id", "event_type", "entity_type", "entity_id", "entity_version",
                       "user_id", "timestamp", "details (JSON)"],
            "examples_shown": [
                "Pack history on the Search → By date / By model tabs",
                "Promotion + rollback timeline on Deployment",
                "Per-policy decision trail on Search → By policy",
            ],
            "extensible_to": "Any custom event filter — by user, by entity_type, by date range — surfaces in the same tab pattern with a one-line SQL change.",
        },
        {
            "key": "inference_logs",
            "table": fqn("inference_logs"),
            "label": "Inference logs",
            "purpose": "Every quote scored by the live pricing endpoint. Predictions per family, the premium build-up, model versions, and the feature snapshot at decision time.",
            "grain": "one row per scored quote",
            "fields": ["policy_id", "scored_at", "freq_pred", "sev_pred", "demand_pred", "fraud_pred",
                       "base_premium", "fraud_loading", "demand_adj", "technical_premium",
                       "model versions per family", "is_mta"],
            "examples_shown": [
                "Premium adequacy by industry tier / region / construction type",
                "Bias monitor's cohort averages",
                "Quote replay on the Quote Review add-on",
            ],
            "extensible_to": "Any cohort dimension that lives on the policy or the inference itself — broker channel, sum-insured band, time-of-day, exposure age. Add the column to the cohort selector and it's live.",
        },
        {
            "key": "governance_packs_index",
            "table": fqn("governance_packs_index"),
            "label": "Governance packs",
            "purpose": "PDF + sidecar artefacts generated on every promotion. Pinned to a UC model version. Indexed for retrieval by family, date, or pack ID.",
            "grain": "one row per generated pack",
            "fields": ["pack_id", "model_family", "model_version", "model_uc_name", "mlflow_run_id",
                       "story", "primary_metric", "primary_value", "pdf_path", "generated_by", "generated_at"],
            "examples_shown": [
                "Search → By model — drill into any past version",
                "Search → By date — what was in force on a chosen day",
                "Search → By policy — pack of record per family",
            ],
            "extensible_to": "Filter or group on any column — story type, primary metric range, generator. The pack PDFs and sidecar JSON are content-addressable so you can build any new query against them.",
        },
        {
            "key": "policy_demographics",
            "table": fqn("policy_demographics"),
            "label": "Policy demographics",
            "purpose": "Protected attributes held alongside the book — never input to any model. Used to group predictions for fairness monitoring.",
            "grain": "one row per policy",
            "fields": ["policy_id", "director_gender", "postcode_demographic",
                       "ethnicity_proxy", "director_age_band"],
            "examples_shown": [
                "Bias monitor — gender, ethnicity proxy, age band, postcode demographic",
            ],
            "extensible_to": (
                "Any other protected attribute or proxy you ask for — disability flag, "
                "veteran status, religion proxy, household-income band, business-age band. "
                "Add the column once, the bias monitor picks it up automatically."
            ),
        },
        {
            "key": "unified_pricing_table_live",
            "table": fqn("unified_pricing_table_live"),
            "label": "Modelling Mart",
            "purpose": "The training + scoring feature mart. Every approved feed joined onto the active book — internal book, vendor enrichment, public reference data.",
            "grain": "one row per policy",
            "fields": ["~50 columns: rating factors, geospatial, credit, market benchmarks, public-data enrichment, claim-derived"],
            "examples_shown": [
                "Premium adequacy cohorts (industry tier, region, construction)",
                "By-policy scoring story",
                "Genie SQL surface (Modelling Mart)",
            ],
            "extensible_to": "Any feature in the catalogue is a candidate cohort or filter. Add an external feed (a new vendor, a new public dataset) and it joins on policy_id without changing the surfaces.",
        },
        {
            "key": "feature_catalog",
            "table": fqn("feature_catalog"),
            "label": "Feature catalog",
            "purpose": "Per-factor metadata — provenance, source columns, transformation, regulatory sensitivity, owner. Powers lineage and the agent's rating-factor lookups.",
            "grain": "one row per factor",
            "fields": ["feature_name", "feature_group", "data_type", "description",
                       "source_tables", "source_columns", "transformation", "owner",
                       "regulatory_sensitive", "pii"],
            "examples_shown": [
                "Modelling Mart → factor catalog tab",
                "Agent's tool answers (\"which features feed this model?\")",
            ],
            "extensible_to": "Any custom view of the catalogue — by group, by sensitivity, by source vendor. The agent can also be pointed at any cut you ask for.",
        },
        {
            "key": "pricing_engine_releases",
            "table": fqn("pricing_engine_releases"),
            "label": "Pricing engine releases",
            "purpose": "Monthly release-of-record snapshots — model versions + rating-engine config + approval. Backs the MTA flow and historical scoring.",
            "grain": "one row per release",
            "fields": ["release_id", "display_name", "effective_date", "status",
                       "<family>_glm_version", "rating_engine_version", "approved_by", "narrative"],
            "examples_shown": [
                "Pricing Engine → MTA: auto-pick release-of-record by inception date",
                "Pricing Engine → Quote on any historical release",
            ],
            "extensible_to": "Any release attribute — by approver, by metric delta, by status — surfaces in the picker without code changes.",
        },
    ]

    surfaces = [
        {
            "key": "monitor_bias",
            "label": "Monitor → Bias",
            "uses":  ["inference_logs", "policy_demographics", "governance_packs_index"],
            "summary": "Joins predictions to protected attributes, groups by cohort, flags spread. Agent investigation reads the pack's fairness section + actual loss experience.",
            "currently_showing": "Director gender · ethnicity proxy · age band · postcode demographic",
            "extensible_to":     "Any other attribute on policy_demographics — add the column, the picker shows it, the investigation works.",
        },
        {
            "key": "monitor_adequacy",
            "label": "Monitor → Premium adequacy",
            "uses":  ["inference_logs", "unified_pricing_table_live"],
            "summary": "Predicted premium vs actual incurred per cohort. Spread highlights mis-priced segments.",
            "currently_showing": "Industry risk tier · region · construction type",
            "extensible_to":     "Any column on the Modelling Mart — broker channel, sum-insured band, exposure tenure, peril mix.",
        },
        {
            "key": "search_bymodel",
            "label": "Search → By model / date",
            "uses":  ["governance_packs_index"],
            "summary": "Browse pack history. The date selector resolves the pack in force at-or-before a chosen date for each family.",
            "currently_showing": "Family + version timeline · governance pack PDF + sidecar viewer",
            "extensible_to":     "Cross-family comparison views, story-tag filters, primary-metric thresholds — all over the same index table.",
        },
        {
            "key": "search_bypolicy",
            "label": "Search → By policy",
            "uses":  ["unified_pricing_table_live", "inference_logs", "governance_packs_index"],
            "summary": "Look up a policy, see the scoring story (freq × sev × demand × fraud), link to the pack of record per family.",
            "currently_showing": "Policy ID lookup → price build-up + linked packs",
            "extensible_to":     "Bulk policy diff (\"how do these 100 policies score on candidate v54 vs champion v53?\"), search by feature value, by audit event.",
        },
        {
            "key": "agent",
            "label": "Agent (free-form Q&A)",
            "uses":  ["governance_packs_index", "audit_log", "feature_catalog"],
            "summary": "Pricing governance agent runs a tool-use loop — `query_pack_index`, `read_pack_artefact`, `query_audit_log` — to answer open questions.",
            "currently_showing": "Pack defence, audit trail Q&A, regulator-facing summaries",
            "extensible_to":     "Add a tool the agent can call (a new SQL view, an external API, a different LLM). The Mosaic AI Agent Framework re-deploys without code changes elsewhere.",
        },
    ]

    return {
        "inputs":   inputs,
        "surfaces": surfaces,
        "narrative": (
            "Everything below is collected, end-to-end, in Unity Catalog. "
            "What we *show* on the other tabs is a curated set of examples — "
            "any other slicing is a config change away."
        ),
    }


# ---------------------------------------------------------------------------
# Premium adequacy — predicted vs actual loss ratio per cohort
# ---------------------------------------------------------------------------

_ADEQUACY_DIMENSIONS = {"industry_risk_tier", "region", "construction_type"}


@router.get("/premium-adequacy")
async def premium_adequacy(cohort_dimension: str = "industry_risk_tier") -> dict:
    """Loss ratio per cohort: avg(annual loss) / avg(technical premium).
    `total_incurred_5y` is a 5-year incurred figure on the unified mart, so we
    annualise it. Underpriced cohorts (loss_ratio > 1.0) are the demo signal."""
    if cohort_dimension not in _ADEQUACY_DIMENSIONS:
        raise HTTPException(400, f"cohort_dimension must be one of {sorted(_ADEQUACY_DIMENSIONS)}")

    sql = f"""
        SELECT u.{cohort_dimension} AS cohort,
               count(*) AS n,
               round(avg(i.technical_premium),     2) AS avg_premium,
               round(avg(u.total_incurred_5y / 5.0), 2) AS avg_annual_loss,
               round(avg(u.total_incurred_5y / 5.0)
                     / nullif(avg(i.technical_premium), 0), 3) AS loss_ratio
        FROM {fqn('unified_pricing_table_live')} u
        JOIN {fqn('inference_logs')}             i USING (policy_id)
        WHERE u.{cohort_dimension} IS NOT NULL
        GROUP BY u.{cohort_dimension}
        ORDER BY u.{cohort_dimension}
    """
    try:
        rows = await execute_query(sql)
    except Exception as e:
        raise HTTPException(500, f"premium-adequacy SQL failed: {e}")

    def _num(v):
        if v is None: return None
        try:    return float(v)
        except (TypeError, ValueError): return None
    for r in rows:
        for k in ("n", "avg_premium", "avg_annual_loss", "loss_ratio"):
            if k in r:
                r[k] = _num(r[k])
        if isinstance(r.get("n"), float):
            r["n"] = int(r["n"])

    lr = [r["loss_ratio"] for r in rows if r.get("loss_ratio") is not None]
    headline = None
    if len(lr) >= 2:
        headline = {
            "max_loss_ratio": max(lr),
            "min_loss_ratio": min(lr),
            "spread_pp":      round((max(lr) - min(lr)) * 100, 1),
            "underpriced_cohorts": sum(1 for x in lr if x > 1.0),
        }

    return {
        "cohort_dimension": cohort_dimension,
        "cohorts":          rows,
        "headline":         headline,
        "scan_timestamp":   datetime.now(timezone.utc).isoformat(),
    }


class AdequacyInvestigateRequest(BaseModel):
    question: str
    cohort_dimension: str = "industry_risk_tier"


@router.post("/adequacy-investigate")
async def adequacy_investigate(req: AdequacyInvestigateRequest) -> dict:
    """Free-form investigation against the governance agent — same plumbing as
    /governance/ask, with the cohort dimension stuffed into the question for
    grounding. Lean cookie-cutter — no separate agent persona needed."""
    if not req.question.strip():
        raise HTTPException(400, "question is required")
    if req.cohort_dimension not in _ADEQUACY_DIMENSIONS:
        raise HTTPException(400, f"cohort_dimension must be one of {sorted(_ADEQUACY_DIMENSIONS)}")

    from server.agent_client import invoke_agent
    framed_q = (
        f"Premium adequacy investigation. Cohort dimension: {req.cohort_dimension}. "
        f"Question: {req.question}"
    )
    result = await invoke_agent(
        endpoint_name=AGENT_ENDPOINT,
        question=framed_q,
        custom_inputs={"pack_id": ""},
        timeout=300,
    )
    await log_audit_event(
        event_type="premium_adequacy_investigation",
        entity_type="portfolio",
        entity_id=req.cohort_dimension,
        details={
            "question":         req.question[:400],
            "cohort_dimension": req.cohort_dimension,
            "endpoint":         AGENT_ENDPOINT,
            "agent_ok":         result.get("ok"),
            "trace":            result.get("trace", []),
            "usage":            result.get("usage", {}),
            "error":            result.get("error"),
        },
    )
    return {
        "ok":               result.get("ok"),
        "question":         req.question,
        "cohort_dimension": req.cohort_dimension,
        "answer":           result.get("answer") or "",
        "trace":            result.get("trace", []),
        "model":            result.get("model"),
        "usage":            result.get("usage", {}),
        "endpoint":         AGENT_ENDPOINT,
        "error":            result.get("error"),
        "cached":           bool(result.get("cached")),
    }
