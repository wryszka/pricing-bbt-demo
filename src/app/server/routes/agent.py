"""`/agent/explain` route — delegates to the `pricing_chat_agent` Agent
Framework endpoint (persona=explain). The agent calls its own portfolio /
shadow-impact tools, so the app just forwards the actuary's question."""

import json
import logging

from fastapi import APIRouter
from pydantic import BaseModel

from server.agent_client import invoke_agent
from server.audit import log_audit_event

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/agent", tags=["agent"])

CHAT_AGENT_ENDPOINT = "pricing_chat_agent"


class ExplainRequest(BaseModel):
    question: str = "Why did premiums change in the latest data update?"


@router.post("/explain")
async def run_explainability(req: ExplainRequest):
    """Explain pricing shifts in plain English for actuarial use. Forwards
    to the Agent Framework endpoint with persona='explain' — the agent pulls
    portfolio stats and shadow-impact figures via its own tools and returns
    a JSON-structured explanation."""
    result = await invoke_agent(
        endpoint_name=CHAT_AGENT_ENDPOINT,
        question=req.question,
        custom_inputs={"persona": "explain"},
    )

    # Agent returns a JSON string inside `answer`; parse it (strip fences if any).
    explanation = None
    raw = result.get("answer") or ""
    if raw:
        jt = raw
        if "```json" in jt:
            jt = jt.split("```json")[1].split("```")[0]
        elif "```" in jt:
            jt = jt.split("```")[1].split("```")[0]
        try:
            explanation = json.loads(jt.strip())
        except json.JSONDecodeError:
            pass

    await log_audit_event(
        event_type="agent_recommendation",
        entity_type="model",
        entity_id="explainability_agent",
        details={
            "question":      req.question,
            "endpoint":      CHAT_AGENT_ENDPOINT,
            "agent_ok":      result.get("ok"),
            "headline":      (explanation or {}).get("headline", ""),
            "trace":         result.get("trace", []),
            "usage":         result.get("usage", {}),
            "error":         result.get("error"),
        },
    )
    return {
        "success":     result.get("ok"),
        "endpoint":    CHAT_AGENT_ENDPOINT,
        "explanation": explanation,
        "cached":      bool(result.get("cached")),
        "transparency": {
            "persona":      "explain",
            "raw_response": raw,
            "trace":        result.get("trace", []),
            "error":        result.get("error"),
        },
    }
