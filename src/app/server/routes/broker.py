"""Conversational quote journey — the direct/broker chatbot surface.

Claude holds the conversation; the carrier's pricing engine sets the price.
Claude is given the same tools the MCP server publishes and must call
`price_motor_risk` to obtain a premium — the system prompt forbids inventing
one, and the UI shows every tool call with its latency so an audience can see
the price came from the engine rather than from the model.

  POST /api/broker/chat     one turn of the conversation
  GET  /api/broker/tools    the tool surface Claude is given (for the UI panel)

State lives in the request: the client sends the running message list and the
answers gathered so far, so the container stays stateless and multi-user safe.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel

from server.config import get_workspace_client
from server.mcp_engine import log_tool_call, new_session_id, price_risk
from server.mcp_tools import (
    COVER_OPTIONS, QUESTION_INDEX, QUOTE_QUESTIONS, REQUIRED_FIELDS,
    build_feature_vector, next_questions, validate,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/broker", tags=["broker"])

FM_ENDPOINT = "databricks-claude-sonnet-4-6"
MAX_TOOL_HOPS = 6

SYSTEM_PROMPT = """You are the quote assistant for Bricksurance Motor, a UK private-car insurer.

Your job is to have a short, natural conversation that collects what the pricing engine needs, then give the customer their price.

Rules you must follow:
- NEVER state, estimate, guess or imply a premium that did not come back from the `price_motor_risk` tool. You have no ability to price a risk yourself. If you have not called the tool, you do not know the price.
- Call `get_quote_requirements` at the START of a new conversation so you know exactly which fields exist and what values are accepted.
- Only ask about fields in that list. Do NOT ask for make, model, registration or anything the engine does not use — for the vehicle you need its value, its age in years, and optionally fuel type. If the customer volunteers a make and model, thank them and translate it yourself into vehicle age and value; do not ask them to repeat it.
- Every time the customer gives you one or more answers, call `record_answers` with what you understood, converting to the field names and accepted values from the contract (e.g. "I've had my licence 20 years" → license_years_held: 20). Do this before replying.
- Ask at most TWO questions per message. Keep it conversational, not a form.
- Once you have every required field, call `price_motor_risk` immediately — do not ask for confirmation first.
- After a price comes back, give the annual and monthly figure and offer to explain what drove it. If the customer asks why, call `explain_price`.
- If the customer asks about cover levels or excess, call `policy_terms`.
- Be warm and brief. British English. Never invent facts about cover.
- If the engine returns an error, say plainly that the quote system is briefly unavailable — do not substitute a made-up figure.
"""

# Tool definitions in OpenAI/FMAPI function-calling shape.
CHAT_TOOLS: list[dict[str, Any]] = [
    {"type": "function", "function": {
        "name": "get_quote_requirements",
        "description": "List every question the pricing engine needs, with accepted values.",
        "parameters": {"type": "object", "properties": {}},
    }},
    {"type": "function", "function": {
        "name": "record_answers",
        "description": ("Record answers the customer has just given, using the "
                        "carrier's field names and accepted values. Call this on "
                        "every turn where the customer supplies information. "
                        "Returns what is still outstanding."),
        "parameters": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "description": ("Field/value pairs understood from the customer's "
                                    "last message, e.g. {\"driver_age\": 42, "
                                    "\"vehicle_value\": 18000}."),
                },
            },
            "required": ["answers"],
        },
    }},
    {"type": "function", "function": {
        "name": "price_motor_risk",
        "description": ("Price the risk with the carrier's live pricing engine. "
                        "Returns the real annual and monthly premium. This is the "
                        "ONLY way to obtain a premium."),
        "parameters": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "description": ("All answers gathered, keyed by field name, e.g. "
                                    "{\"driver_age\": 42, \"vehicle_value\": 18000}."),
                },
            },
            "required": ["answers"],
        },
    }},
    {"type": "function", "function": {
        "name": "explain_price",
        "description": "Explain what drove the premium that was just quoted.",
        "parameters": {
            "type": "object",
            "properties": {"question": {"type": "string"}},
        },
    }},
    {"type": "function", "function": {
        "name": "policy_terms",
        "description": "Cover levels, what each includes, and excess options.",
        "parameters": {
            "type": "object",
            "properties": {"cover": {"type": "string"}},
        },
    }},
]


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, Any]] = []
    answers: dict[str, Any] = {}
    session_id: str | None = None
    # Carried back by the client so a later turn can still explain a premium
    # priced on an earlier one — the container keeps no per-session state.
    breakdown: dict[str, Any] | None = None


def _fm_chat(messages: list[dict[str, Any]]) -> Any:
    """One FMAPI chat completion with tools. Uses the raw API so tool_calls come
    back unflattened (the SDK's typed ChatMessage drops them)."""
    w = get_workspace_client()
    return w.api_client.do(
        "POST",
        f"/serving-endpoints/{FM_ENDPOINT}/invocations",
        body={
            "messages": messages,
            "tools": CHAT_TOOLS,
            "max_tokens": 900,
            "temperature": 0.3,
        },
    )


def _assistant_message(resp: Any) -> dict[str, Any]:
    choices = (resp or {}).get("choices") or []
    if not choices:
        return {"role": "assistant", "content": ""}
    msg = choices[0].get("message") or {}
    return {
        "role": "assistant",
        "content": msg.get("content") or "",
        **({"tool_calls": msg["tool_calls"]} if msg.get("tool_calls") else {}),
    }


async def _run_tool(name: str, args: dict[str, Any], state: dict[str, Any],
                    session_id: str) -> dict[str, Any]:
    """Execute one tool for the chat surface and record it for the UI panel."""
    if name == "get_quote_requirements":
        await log_tool_call(session_id=session_id, agent_id="broker-chat",
                            surface="broker_chat", tool=name, ok=True)
        state["tool_log"].append({"tool": name, "ok": True})
        return {"required_fields": REQUIRED_FIELDS, "questions": QUOTE_QUESTIONS}

    if name == "policy_terms":
        await log_tool_call(session_id=session_id, agent_id="broker-chat",
                            surface="broker_chat", tool=name, ok=True)
        state["tool_log"].append({"tool": name, "ok": True})
        return {"cover_options": COVER_OPTIONS}

    if name == "record_answers":
        incoming = args.get("answers") or {}
        # Keep only fields the engine knows, so a stray "make"/"model" from the
        # model never pollutes the answer set.
        clean = {k: v for k, v in incoming.items()
                 if k in QUESTION_INDEX and v is not None and str(v) != ""}
        rejected = [k for k in incoming if k not in QUESTION_INDEX]
        state["answers"].update(clean)
        missing, errors = validate(state["answers"])
        state["tool_log"].append({"tool": name, "ok": True,
                                  "detail": f"+{len(clean)} field(s)"})
        await log_tool_call(session_id=session_id, agent_id="broker-chat",
                            surface="broker_chat", tool=name, ok=True,
                            fields_supplied=len(state["answers"]),
                            detail=f"recorded={len(clean)} rejected={len(rejected)}")
        return {
            "recorded": clean,
            "ignored_unknown_fields": rejected,
            "still_missing": missing,
            "errors": errors,
            "ready_to_price": not missing and not errors,
            "next_questions": next_questions(state["answers"]),
        }

    if name == "price_motor_risk":
        # Merge whatever Claude gathered into the running answer set so a
        # partial later turn cannot lose earlier answers.
        incoming = args.get("answers") or {}
        state["answers"].update({k: v for k, v in incoming.items()
                                 if v is not None and str(v) != ""})
        answers = state["answers"]

        missing, errors = validate(answers)
        if missing or errors:
            state["tool_log"].append({"tool": name, "ok": False,
                                      "detail": "incomplete"})
            await log_tool_call(session_id=session_id, agent_id="broker-chat",
                                surface="broker_chat", tool=name, ok=False,
                                fields_supplied=len(answers),
                                detail="rejected: incomplete")
            return {"ok": False, "reason": "incomplete_or_invalid",
                    "missing_required": missing, "errors": errors,
                    "next_questions": next_questions(answers)}

        features, provenance = build_feature_vector(answers)
        priced = await price_risk(features)
        state["quote"] = priced if priced.get("ok") else None
        if priced.get("ok"):
            state["breakdown"] = priced["breakdown"]
            # Keys are already the wire contract the UI reads — see
            # build_feature_vector. Pass straight through, no remapping.
            state["provenance"] = provenance
        state["tool_log"].append({
            "tool": name, "ok": bool(priced.get("ok")),
            "latency_ms": priced.get("latency_ms"),
            "engine": priced.get("engine"),
            "annual_premium": priced.get("annual_premium"),
        })
        await log_tool_call(session_id=session_id, agent_id="broker-chat",
                            surface="broker_chat", tool=name,
                            ok=bool(priced.get("ok")),
                            latency_ms=priced.get("latency_ms"),
                            annual_premium=priced.get("annual_premium"),
                            fields_supplied=len(provenance["customer_supplied"]),
                            detail=priced.get("error"))
        if not priced.get("ok"):
            return {"ok": False, "error": priced.get("error"),
                    "detail": priced.get("detail")}
        return {"ok": True,
                "annual_premium": priced["annual_premium"],
                "monthly_premium": priced["monthly_premium"],
                "currency": "GBP",
                "cover_options": COVER_OPTIONS}

    if name == "explain_price":
        breakdown = state.get("breakdown")
        if not breakdown:
            state["tool_log"].append({"tool": name, "ok": False,
                                      "detail": "no quote yet"})
            return {"ok": False, "error": "no quote has been priced yet"}
        from server.routes.mcp import _tool_explain_price
        out = await _tool_explain_price(
            {"breakdown": breakdown, "question": args.get("question") or ""},
            session_id, "broker-chat")
        state["tool_log"].append({"tool": name, "ok": bool(out.get("ok"))})
        return out

    return {"error": f"unknown tool {name}"}


@router.post("/chat")
async def chat(req: ChatRequest) -> dict:
    """One turn: run Claude, execute any tools it calls, return its reply."""
    session_id = (req.session_id or "").strip() or new_session_id()
    state: dict[str, Any] = {
        "answers": dict(req.answers or {}),
        "tool_log": [],
        "quote": None,
        "breakdown": dict(req.breakdown) if req.breakdown else None,
        "provenance": None,
    }

    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.extend(req.history or [])
    messages.append({"role": "user", "content": req.message})

    reply = ""
    try:
        for _ in range(MAX_TOOL_HOPS):
            resp = _fm_chat(messages)
            assistant = _assistant_message(resp)
            messages.append(assistant)

            calls = assistant.get("tool_calls") or []
            if not calls:
                reply = assistant.get("content") or ""
                break

            for call in calls:
                fn = (call.get("function") or {})
                name = fn.get("name") or ""
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                result = await _run_tool(name, args, state, session_id)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.get("id"),
                    "content": json.dumps(result, default=str),
                })
        else:
            reply = ("Sorry — I got a bit tangled there. Could you tell me again "
                     "what you'd like to insure?")
    except Exception as e:
        logger.exception("broker chat turn failed")
        return {
            "ok": False,
            "session_id": session_id,
            "reply": "Sorry — the quote assistant is briefly unavailable. "
                     "Please try again in a moment.",
            "error": str(e)[:300],
            "answers": state["answers"],
            "tool_log": state["tool_log"],
        }

    # Trim the system prompt back off before handing history to the client.
    history_out = [m for m in messages[1:]]

    return {
        "ok": True,
        "session_id": session_id,
        "reply": reply,
        "history": history_out,
        "answers": state["answers"],
        "tool_log": state["tool_log"],
        "quote": state["quote"],
        "breakdown": state["breakdown"],
        "provenance": state["provenance"],
        "progress": {
            "required": REQUIRED_FIELDS,
            "collected": [f for f in REQUIRED_FIELDS if state["answers"].get(f) not in (None, "")],
        },
    }


@router.get("/tools")
async def tools() -> dict:
    """The tool surface Claude is given — rendered in the UI side panel."""
    return {"model": FM_ENDPOINT,
            "tools": [t["function"] for t in CHAT_TOOLS]}
