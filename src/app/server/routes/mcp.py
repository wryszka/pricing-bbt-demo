"""MCP server exposing the motor pricing engine as carrier *services*.

An outside agent — a bank's assistant, a comparison agent, Claude Desktop —
speaks MCP here and can discover, without knowing anything about insurance,
exactly what this carrier needs in order to price a risk. That discovery step
(`get_quote_requirements`) is the difference between being an aggregator feed
and being a service provider: we publish the questions, the accepted value
domains, and why each one matters — not a raw data dump.

Transport is JSON-RPC 2.0 over a single POST endpoint (MCP's streamable-HTTP
shape), which is what MCP clients expect and is also trivial to curl on stage:

  POST /api/mcp            JSON-RPC: initialize | tools/list | tools/call
  GET  /api/mcp/manifest   human/manifest view of the same tool set

Auth is whatever the Databricks App already enforces in front of the container;
this route adds no separate credential path.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from server.audit import log_audit_event
from server.mcp_engine import (
    log_tool_call, new_session_id, price_risk,
)
from server.mcp_tools import (
    BOOK_MEANS, COVER_OPTIONS, QUOTE_QUESTIONS, REQUIRED_FIELDS,
    build_feature_vector, next_questions, validate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/mcp", tags=["mcp"])

PROTOCOL_VERSION = "2025-06-18"
SERVER_INFO = {"name": "bricksurance-motor-distribution", "version": "1.0.0"}

# ---------------------------------------------------------------------------
# Tool schemas — this is the contract an unfamiliar agent reads.
# ---------------------------------------------------------------------------

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "name": "get_quote_requirements",
        "description": (
            "Discover what this carrier needs in order to price a motor risk. "
            "Returns every question, its type, accepted values, whether it is "
            "required, and why it affects the price. Call this FIRST — do not "
            "guess field names or values."
        ),
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "check_answers",
        "description": (
            "Validate the answers collected so far against the carrier's "
            "contract and get the next most useful questions to ask. Use this "
            "to run a short conversational journey instead of asking everything "
            "at once. Does not price and does not charge an engine call."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "answers": {"type": "object",
                            "description": "Field/value pairs gathered so far."},
            },
            "required": ["answers"],
        },
    },
    {
        "name": "price_motor_risk",
        "description": (
            "Price a motor risk against the carrier's live pricing engine and "
            "return the annual and monthly premium with the full rating "
            "breakdown. The premium is computed by the carrier's deployed "
            "models — never estimate or invent a premium yourself. Fields the "
            "customer cannot know (telematics and driving-behaviour history for "
            "a new customer) fall back to the book mean; the response reports "
            "which fields came from the customer and which did not."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "answers": {"type": "object",
                            "description": "Answers keyed by the field names from "
                                           "get_quote_requirements."},
                "session_id": {"type": "string",
                               "description": "Optional id tying calls into one journey."},
            },
            "required": ["answers"],
        },
    },
    {
        "name": "explain_price",
        "description": (
            "Explain, in plain language, what drove a premium — which factors "
            "pushed it up or down. Use after price_motor_risk when the customer "
            "asks why the price is what it is. An aggregator feed cannot answer "
            "this; the carrier can."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "breakdown": {"type": "object",
                              "description": "The `breakdown` object returned by price_motor_risk."},
                "question": {"type": "string",
                             "description": "What the customer actually asked."},
            },
            "required": ["breakdown"],
        },
    },
    {
        "name": "policy_terms",
        "description": (
            "Return the cover levels this carrier offers, what each includes, "
            "and the available excess options."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "cover": {"type": "string",
                          "description": "Optional — narrow to one cover level."},
            },
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

async def _tool_get_quote_requirements(_: dict, session_id: str, agent_id: str) -> dict:
    await log_tool_call(session_id=session_id, agent_id=agent_id, surface="mcp",
                        tool="get_quote_requirements", ok=True)
    return {
        "product": "Motor — private car (UK)",
        "required_fields": REQUIRED_FIELDS,
        "questions": QUOTE_QUESTIONS,
        "not_asked": {
            "fields": sorted(BOOK_MEANS.keys()),
            "reason": ("Telematics and driving-behaviour history. A new customer "
                       "has no driving record with this carrier, so these start "
                       "at the book mean and tighten once real telematics arrive."),
        },
        "next_step": ("Collect the required fields, optionally call check_answers "
                      "as you go, then call price_motor_risk."),
    }


async def _tool_check_answers(args: dict, session_id: str, agent_id: str) -> dict:
    answers = args.get("answers") or {}
    missing, errors = validate(answers)
    ready = not missing and not errors
    await log_tool_call(session_id=session_id, agent_id=agent_id, surface="mcp",
                        tool="check_answers", ok=True,
                        fields_supplied=len(answers),
                        detail=f"ready={ready} missing={len(missing)}")
    return {
        "ready_to_price": ready,
        "missing_required": missing,
        "errors": errors,
        "next_questions": next_questions(answers),
    }


async def _tool_price_motor_risk(args: dict, session_id: str, agent_id: str) -> dict:
    answers = args.get("answers") or {}
    missing, errors = validate(answers)
    if missing or errors:
        await log_tool_call(session_id=session_id, agent_id=agent_id, surface="mcp",
                            tool="price_motor_risk", ok=False,
                            fields_supplied=len(answers),
                            detail="rejected: incomplete or invalid")
        return {
            "ok": False,
            "reason": "incomplete_or_invalid",
            "missing_required": missing,
            "errors": errors,
            "next_questions": next_questions(answers),
        }

    features, provenance = build_feature_vector(answers)
    priced = await price_risk(features)

    await log_tool_call(
        session_id=session_id, agent_id=agent_id, surface="mcp",
        tool="price_motor_risk", ok=bool(priced.get("ok")),
        latency_ms=priced.get("latency_ms"),
        annual_premium=priced.get("annual_premium"),
        fields_supplied=len(provenance["customer_supplied"]),
        detail=priced.get("error"))

    await log_audit_event(
        event_type="agent_quote",
        entity_type="quote",
        entity_id=session_id,
        details={"surface": "mcp", "agent_id": agent_id,
                 "annual_premium": priced.get("annual_premium"),
                 "engine": priced.get("engine"),
                 "customer_supplied": provenance["customer_supplied"],
                 "book_mean_fallback": provenance["book_mean_fallback"]},
    )

    if not priced.get("ok"):
        return priced

    return {
        "ok": True,
        "session_id": session_id,
        "annual_premium": priced["annual_premium"],
        "monthly_premium": priced["monthly_premium"],
        "currency": "GBP",
        "breakdown": priced["breakdown"],
        "priced_by": {
            "engine": priced["engine"],
            "latency_ms": priced["latency_ms"],
            "note": "Premium computed by the carrier's deployed pricing models.",
        },
        "input_provenance": provenance,
        "cover_options": COVER_OPTIONS,
    }


def _plain_text(raw: str) -> str:
    """The explain agent replies with a JSON envelope (headline + explanation),
    sometimes inside code fences. Flatten it to prose so a chat surface can read
    it out directly; fall back to the raw text if it isn't the expected shape."""
    import json as _json
    import re as _re

    text = (raw or "").strip()
    if not text:
        return ""
    fenced = _re.search(r"```(?:json)?\s*(.*?)```", text, _re.S)
    if fenced:
        text = fenced.group(1).strip()
    elif text.startswith("```"):
        # Unterminated fence — the agent hit its token cap mid-response.
        text = _re.sub(r"^```(?:json)?\s*", "", text)

    obj: Any = None
    try:
        obj = _json.loads(text)
    except (ValueError, TypeError):
        # Truncated JSON is common when the agent is cut off. Recover the
        # prose fields directly rather than dumping raw JSON into a customer
        # conversation.
        salvaged = []
        for key in ("headline", "explanation"):
            m = _re.search(rf'"{key}"\s*:\s*"(.*?)(?<!\\)"', text, _re.S)
            if m:
                salvaged.append(m.group(1).replace('\\"', '"').replace("\\n", "\n").strip())
        if salvaged:
            return "\n\n".join(salvaged)
        return _re.sub(r"^\s*[\{\}\[\]]\s*$", "", text, flags=_re.M).strip()
    if not isinstance(obj, dict):
        return (raw or "").strip()

    parts = [str(obj[k]).strip() for k in ("headline", "explanation")
             if obj.get(k)]
    drivers = obj.get("drivers") or obj.get("factors")
    if isinstance(drivers, list) and drivers:
        bullets = []
        for d in drivers[:6]:
            if isinstance(d, dict):
                label = d.get("factor") or d.get("name") or ""
                effect = d.get("effect") or d.get("impact") or d.get("direction") or ""
                bullets.append(f"- {label}{f': {effect}' if effect else ''}".rstrip())
            else:
                bullets.append(f"- {d}")
        parts.append("\n".join(bullets))
    return "\n\n".join(parts) if parts else (raw or "").strip()


async def _tool_explain_price(args: dict, session_id: str, agent_id: str) -> dict:
    breakdown = args.get("breakdown") or {}
    question = args.get("question") or "Why is my premium this amount?"

    from server.agent_client import invoke_agent
    prompt = (
        f"A motor customer asks: {question}\n\n"
        f"Here is the rating breakdown from our pricing engine:\n{breakdown}\n\n"
        "Explain in plain language what drove this premium — name the factors "
        "that pushed it up and any that brought it down. Be specific and honest. "
        "Do not quote a different premium than the one in the breakdown."
    )
    try:
        result = await invoke_agent(endpoint_name="pricing_chat_agent",
                                    question=prompt,
                                    custom_inputs={"persona": "explain"})
        answer = _plain_text(result.get("answer") or "")
        ok = bool(answer)
    except Exception as e:
        logger.warning("explain_price agent call failed: %s", str(e)[:200])
        answer, ok = "", False

    await log_tool_call(session_id=session_id, agent_id=agent_id, surface="mcp",
                        tool="explain_price", ok=ok)
    if not ok:
        return {"ok": False,
                "error": "explanation service unavailable",
                "premium": breakdown.get("final_premium")}
    return {"ok": True, "explanation": answer,
            "premium": breakdown.get("final_premium")}


async def _tool_policy_terms(args: dict, session_id: str, agent_id: str) -> dict:
    cover = (args.get("cover") or "").strip().lower()
    opts = COVER_OPTIONS
    if cover:
        opts = [c for c in COVER_OPTIONS if c["cover"].lower() == cover] or COVER_OPTIONS
    await log_tool_call(session_id=session_id, agent_id=agent_id, surface="mcp",
                        tool="policy_terms", ok=True)
    return {"cover_options": opts,
            "instalments": "Monthly instalments available; annual payment avoids "
                           "the credit charge."}


TOOL_IMPLS = {
    "get_quote_requirements": _tool_get_quote_requirements,
    "check_answers":          _tool_check_answers,
    "price_motor_risk":       _tool_price_motor_risk,
    "explain_price":          _tool_explain_price,
    "policy_terms":           _tool_policy_terms,
}


# ---------------------------------------------------------------------------
# JSON-RPC transport
# ---------------------------------------------------------------------------

def _ok(rpc_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _err(rpc_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


@router.post("")
async def jsonrpc(request: Request) -> dict:
    """Single JSON-RPC entry point: initialize, tools/list, tools/call."""
    try:
        body = await request.json()
    except Exception:
        return _err(None, -32700, "Parse error: body is not valid JSON")

    rpc_id = body.get("id")
    method = body.get("method")
    params = body.get("params") or {}
    agent_id = request.headers.get("user-agent", "unknown-agent")[:120]

    if method == "initialize":
        return _ok(rpc_id, {
            "protocolVersion": PROTOCOL_VERSION,
            "serverInfo": SERVER_INFO,
            "capabilities": {"tools": {}},
            "instructions": (
                "Motor insurance pricing services. Call get_quote_requirements "
                "first to learn what this carrier needs, then price_motor_risk. "
                "Premiums come from the carrier's live pricing engine — never "
                "estimate one yourself."
            ),
        })

    if method in ("notifications/initialized", "notifications/cancelled"):
        return _ok(rpc_id, {})

    if method == "tools/list":
        return _ok(rpc_id, {"tools": TOOL_SCHEMAS})

    if method == "tools/call":
        name = params.get("name")
        args = params.get("arguments") or {}
        impl = TOOL_IMPLS.get(name)
        if impl is None:
            return _err(rpc_id, -32601, f"Unknown tool: {name}")

        session_id = str(args.get("session_id") or "").strip() or new_session_id()
        try:
            payload = await impl(args, session_id, agent_id)
        except Exception as e:
            logger.exception("mcp tool %s failed", name)
            return _err(rpc_id, -32603, f"Tool execution failed: {str(e)[:200]}")

        import json as _json
        return _ok(rpc_id, {
            "content": [{"type": "text", "text": _json.dumps(payload, default=str)}],
            "structuredContent": payload,
            "isError": payload.get("ok") is False,
        })

    return _err(rpc_id, -32601, f"Method not found: {method}")


class ManifestResponse(BaseModel):
    server: dict
    protocol_version: str
    tools: list[dict]


@router.get("/manifest")
async def manifest() -> ManifestResponse:
    """Plain view of the tool surface — handy for the UI and for showing an
    audience what an agent sees before it has asked a single question."""
    return ManifestResponse(server=SERVER_INFO, protocol_version=PROTOCOL_VERSION,
                            tools=TOOL_SCHEMAS)
