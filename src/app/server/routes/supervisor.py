"""Model Supervisor — single chat surface that consolidates every agent in
the workbench under one address.

Dispatches to:
 * `pricing_governance_agent`               — packs, audit log, regulator defence
 * `pricing_chat_agent` (bias_investigator) — fairness investigations
 * `pricing_chat_agent` (explain)           — ingestion / portfolio impact
 * `pricing_chat_agent` (factory)           — model-factory plan review
 * AI/BI Genie (Modelling Mart)             — natural-language SQL on the live mart
 * AI/BI Genie (Quote Stream)               — quote-stream analytics

Auto-routing classifies the question via a short Foundation-Model call (~1-2s);
or the caller pins a specific sub-agent. Every dispatch is audit-logged so the
end-to-end trail is the same as direct sub-agent use.
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.agent_client import invoke_agent
from server.audit import log_audit_event
from server.config import get_workspace_client, get_workspace_host

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/supervisor", tags=["supervisor"])

GOVERNANCE_AGENT_ENDPOINT = "pricing_governance_agent"
CHAT_AGENT_ENDPOINT       = "pricing_chat_agent"
FM_CLASSIFIER_ENDPOINT    = "databricks-claude-sonnet-4-6"


# Catalogue used both for routing and for the UI to render its picker.
SUB_AGENTS: list[dict[str, Any]] = [
    {
        "id":          "governance",
        "label":       "Governance Agent",
        "subtitle":    "packs · audit log · regulator defence",
        "endpoint":    GOVERNANCE_AGENT_ENDPOINT,
        "persona":     None,
        "tools":       ["query_pack_index", "read_pack_artefact", "query_audit_log"],
        "good_for":    [
            "What pack defends frequency v14?",
            "Show me audit events for fraud_gbm in March",
            "Draft an FCA Consumer Duty response",
        ],
    },
    {
        "id":          "bias",
        "label":       "Bias Investigator",
        "subtitle":    "fairness disparities",
        "endpoint":    CHAT_AGENT_ENDPOINT,
        "persona":     "bias_investigator",
        "tools":       ["bias_monitor", "actual_loss_experience", "proxy_features", "pack_fairness_section"],
        "good_for":    [
            "Investigate the gender gap on the latest production scoring",
            "Is the postcode disparity defensible?",
            "Why is the ethnicity-proxy gap there?",
        ],
    },
    {
        "id":          "explain",
        "label":       "Impact Explainer",
        "subtitle":    "data-ingestion impact",
        "endpoint":    CHAT_AGENT_ENDPOINT,
        "persona":     "explain",
        "tools":       ["portfolio_diff", "shadow_impact", "rating_factor_lookup"],
        "good_for":    [
            "Why did premiums move on the geospatial vendor refresh?",
            "Which postcode sectors drove the credit-bureau diff?",
            "Plain-English summary of the latest dataset approval",
        ],
    },
    {
        "id":          "factory",
        "label":       "Plan Reviewer",
        "subtitle":    "model-factory candidate review",
        "endpoint":    CHAT_AGENT_ENDPOINT,
        "persona":     "factory",
        "tools":       ["leaderboard", "shortlist", "portfolio_what_if"],
        "good_for":    [
            "Walk through the latest factory leaderboard",
            "Which 3 candidates should I shortlist?",
            "What's the portfolio impact if I promote variant B07?",
        ],
        "needs_run_id": True,
    },
    {
        "id":          "genie_mart",
        "label":       "Mart Genie",
        "subtitle":    "natural-language SQL on the Modelling Mart",
        "endpoint":    "ai_bi_genie",
        "space_env":   "GENIE_SPACE_ID",
        "tools":       ["text-to-SQL", "execute on warehouse", "render result"],
        "good_for":    [
            "Loss ratio by industry tier",
            "Top 10 postcode sectors by GWP",
            "How many quotes did we score yesterday?",
        ],
    },
    {
        "id":          "genie_quote",
        "label":       "Quote Genie",
        "subtitle":    "quote-stream analytics",
        "endpoint":    "ai_bi_genie",
        "space_env":   "GENIE_QUOTE_SPACE_ID",
        "tools":       ["text-to-SQL", "execute on warehouse", "render result"],
        "good_for":    [
            "Conversion rate by broker channel last week",
            "Which quotes were outliers vs market median?",
        ],
    },
    {
        "id":          "multi",
        "label":       "Multi-agent",
        "subtitle":    "fan one question out to governance + bias + explain in parallel",
        "endpoint":    "multi_agent",
        "tools":       ["governance · bias · explain (parallel fan-out)"],
        "good_for":    [
            "For freq_glm_motor v4: which pack defends it, is there a director_gender disparity in its predictions, and why did premiums move on the last data refresh?",
        ],
    },
]
_AGENT_BY_ID = {a["id"]: a for a in SUB_AGENTS}
# Sub-agents the `multi` virtual agent fans the question out to.
_MULTI_FANOUT_IDS = ("governance", "bias", "explain")


@router.get("/agents")
async def list_agents() -> dict:
    """Catalogue of sub-agents for the supervisor UI — fronts the architecture
    diagram and the picker chips."""
    return {
        "agents": [
            {
                "id":         a["id"],
                "label":      a["label"],
                "subtitle":   a["subtitle"],
                "endpoint":   a["endpoint"],
                "persona":    a.get("persona"),
                "tools":      a["tools"],
                "good_for":   a["good_for"],
                "kind":       "genie" if a["endpoint"] == "ai_bi_genie" else "agent",
                "needs_run_id": a.get("needs_run_id", False),
            }
            for a in SUB_AGENTS
        ],
    }


# ---------------------------------------------------------------------------
# Auto-classification — short FM call to pick the best sub-agent
# ---------------------------------------------------------------------------

_CLASSIFIER_SYSTEM = (
    "You are a router. Read the user's pricing-workbench question and respond "
    "with EXACTLY ONE of these labels — no extra text, no punctuation:\n"
    "  governance  — model packs, audit log, regulator defence (FCA/PRA), pack history\n"
    "  bias        — fairness, disparity, proxy discrimination, protected attributes\n"
    "  explain     — why did premiums change after a dataset refresh, ingestion impact\n"
    "  factory     — model factory plan, leaderboard, shortlist, candidate variants\n"
    "  genie_mart  — portfolio data: GWP, loss ratio, premium by region/industry/cohort\n"
    "  genie_quote — quote stream: conversion rate, outlier quotes, broker channel funnel\n"
    "If the question is ambiguous, prefer governance."
)


def _classify_sync(question: str) -> str:
    """Sync FM call to pick a sub-agent. Returns one of the SUB_AGENTS ids,
    or 'governance' as fallback."""
    try:
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        w = get_workspace_client()
        resp = w.serving_endpoints.query(
            name=FM_CLASSIFIER_ENDPOINT,
            messages=[
                ChatMessage(role=ChatMessageRole.SYSTEM, content=_CLASSIFIER_SYSTEM),
                ChatMessage(role=ChatMessageRole.USER,   content=question[:1000]),
            ],
            max_tokens=10, temperature=0.0,
        )
        choices = getattr(resp, "choices", None) or (resp.get("choices", []) if isinstance(resp, dict) else [])
        if not choices:
            return "governance"
        m = choices[0].message if hasattr(choices[0], "message") else choices[0].get("message", {})
        text = getattr(m, "content", None) or (m.get("content") if isinstance(m, dict) else "")
        token = (text or "").strip().lower().split()[0] if text else ""
        token = token.strip(".,;:'\"")
        return token if token in _AGENT_BY_ID else "governance"
    except Exception as e:
        logger.warning("supervisor classifier call failed (%s) — defaulting to governance", e)
        return "governance"


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question:    str
    # 'auto' = let the supervisor classify; otherwise must be a SUB_AGENTS id
    sub_agent:   Literal["auto", "governance", "bias", "explain", "factory",
                         "genie_mart", "genie_quote", "multi"] = "auto"
    # Optional context for sub-agents that need it
    pack_id:     str | None = None    # governance / bias_investigator
    run_id:      str | None = None    # factory persona
    family:      str | None = None    # bias_investigator narrowing


@router.post("/ask")
async def ask_supervisor(req: AskRequest) -> dict:
    """Single-chat dispatcher. Resolves the sub-agent (auto-classify or explicit),
    invokes it, and audit-logs the routing decision so the trail is intact."""
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    chosen = req.sub_agent
    classifier_used = False
    if chosen == "auto":
        chosen = await asyncio.to_thread(_classify_sync, req.question)
        classifier_used = True
    if chosen not in _AGENT_BY_ID:
        raise HTTPException(400, f"unknown sub_agent '{chosen}'")

    agent = _AGENT_BY_ID[chosen]
    result: dict[str, Any] = {}

    if chosen == "multi":
        # Fan the same question out to a fixed set of specialist sub-agents in
        # parallel. Each one returns its own structured answer; we stitch them
        # together as a single multi-section reply so the chat panel renders
        # one turn with three labelled blocks.
        fanout = [_AGENT_BY_ID[i] for i in _MULTI_FANOUT_IDS]

        async def _call_one(a: dict) -> dict:
            ci: dict[str, Any] = {}
            if a.get("persona"):
                ci["persona"] = a["persona"]
            if a["id"] == "governance":
                ci["pack_id"] = req.pack_id or ""
            if a["id"] == "bias":
                ci.setdefault("mode", "live")
                if req.family:
                    ci["family"] = req.family
            r = await invoke_agent(
                endpoint_name=a["endpoint"],
                question=req.question,
                custom_inputs=ci,
                timeout=300,
            )
            return {"agent": a, "result": r}

        outs = await asyncio.gather(*(_call_one(a) for a in fanout))
        sections: list[str] = []
        traces: list[Any] = []
        total_tokens = 0
        any_ok = False
        first_error: str | None = None
        for o in outs:
            a = o["agent"]; r = o["result"]
            sections.append(f"## {a['label']} — {a['subtitle']}\n\n" +
                            (r.get("answer") or f"_(no answer — {r.get('error') or 'unknown'})_"))
            traces.append({"sub_agent": a["id"], "trace": r.get("trace", [])})
            total_tokens += int((r.get("usage") or {}).get("total_tokens") or 0)
            if r.get("ok"):
                any_ok = True
            elif first_error is None:
                first_error = r.get("error")
        result = {
            "ok":       any_ok,
            "kind":     "multi",
            "answer":   "\n\n---\n\n".join(sections),
            "trace":    traces,
            "model":    "multi_agent",
            "usage":    {"total_tokens": total_tokens},
            "endpoint": "multi_agent",
            "error":    None if any_ok else first_error,
            "fanout":   [o["agent"]["id"] for o in outs],
        }
    elif agent["endpoint"] == "ai_bi_genie":
        # The supervisor doesn't proxy Genie conversations — Genie has its own
        # streaming UX. Hand the embed URL back so the chat panel can render
        # the room inline (with the user's question pre-filled), or fall back
        # to a clear error when the env var isn't wired.
        space_id = os.getenv(agent["space_env"], "")
        host     = get_workspace_host() or ""
        if space_id and host:
            # Pre-filling the question via ?query= lets the Genie iframe
            # show the answer on first paint without a manual second click.
            from urllib.parse import quote
            q_param  = f"?query={quote(req.question[:1000])}"
            embed    = f"{host}/embed/genie/rooms/{space_id}{q_param}"
            open_url = f"{host}/genie/rooms/{space_id}{q_param}"
            answer = (
                f"Genie is best driven inline — opening the **{agent['label']}** room "
                f"with your question pre-filled. Click *Open in Databricks* to take "
                f"the chat full-page if you want history + saved follow-ups."
            )
        else:
            embed = open_url = ""
            answer = f"Genie space env var {agent['space_env']} is not set."
        result = {
            "ok":         bool(space_id),
            "kind":       "genie",
            "space_id":   space_id,
            "embed_url":  embed,
            "open_url":   open_url,
            "answer":     answer,
            "trace":      [],
            "usage":      {},
            "model":      "ai_bi_genie",
            "endpoint":   agent["endpoint"],
            "error":      None if space_id else f"{agent['space_env']} not configured",
        }
    else:
        custom_inputs: dict[str, Any] = {}
        if agent.get("persona"):
            custom_inputs["persona"] = agent["persona"]
        # Governance agent expects pack_id; pass empty if not provided
        if chosen == "governance":
            custom_inputs["pack_id"] = req.pack_id or ""
        # Bias investigator uses optional family + protected_attribute hooks
        if chosen == "bias":
            custom_inputs.setdefault("mode", "live")
            if req.family:
                custom_inputs["family"] = req.family
        # Factory persona must have a run_id to land context
        if chosen == "factory":
            if not req.run_id:
                return {
                    "ok":         False,
                    "sub_agent":  chosen,
                    "kind":       "agent",
                    "answer":     ("The Plan Reviewer needs a factory run_id "
                                   "for context — open the Model Factory tab "
                                   "and chat from there for full leaderboard / "
                                   "shortlist context."),
                    "error":      "missing_run_id",
                    "trace":      [],
                    "usage":      {},
                    "model":      agent["endpoint"],
                    "endpoint":   agent["endpoint"],
                    "classifier_used": classifier_used,
                }
            custom_inputs["run_id"] = req.run_id

        agent_result = await invoke_agent(
            endpoint_name=agent["endpoint"],
            question=req.question,
            custom_inputs=custom_inputs,
            timeout=300,
        )
        result = {
            "ok":       agent_result.get("ok"),
            "kind":     "agent",
            "answer":   agent_result.get("answer") or "",
            "trace":    agent_result.get("trace", []),
            "model":    agent_result.get("model", agent["endpoint"]),
            "usage":    agent_result.get("usage", {}),
            "endpoint": agent["endpoint"],
            "error":    agent_result.get("error"),
        }

    await log_audit_event(
        event_type="supervisor_dispatch",
        entity_type="supervisor",
        entity_id=chosen,
        details={
            "sub_agent":        chosen,
            "classifier_used":  classifier_used,
            "endpoint":         agent["endpoint"],
            "persona":          agent.get("persona"),
            "question":         req.question[:400],
            "ok":               result.get("ok"),
            "model":            result.get("model"),
            "trace_len":        len(result.get("trace") or []),
            "tokens":           result.get("usage", {}).get("total_tokens"),
            "error":            result.get("error"),
        },
    )

    return {
        **result,
        "sub_agent":        chosen,
        "sub_agent_label":  agent["label"],
        "classifier_used":  classifier_used,
    }
