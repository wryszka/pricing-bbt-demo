"""Thin proxy around the Databricks Genie Conversation REST API.

Lets the app render Genie conversations inline (own chat UI, SQL displayed in a
collapsible panel, data rendered as a native table) instead of opening a full-page
iframe. Uses the REST API directly (via the SDK's authenticated HTTP client) so
we are resilient to SDK version drift on the `genie` Python service.
"""

import asyncio
import logging
from typing import Any, Optional

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import get_workspace_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/genie", tags=["genie"])


def _auth_headers() -> dict:
    w = get_workspace_client()
    return w.config._header_factory()


def _host() -> str:
    return get_workspace_client().config.host.rstrip("/")


def _api_sync(method: str, path: str, *, json_body: dict | None = None) -> dict:
    url = f"{_host()}{path}"
    headers = {**_auth_headers(), "Content-Type": "application/json"}
    if method == "GET":
        r = requests.get(url, headers=headers, timeout=60)
    elif method == "POST":
        r = requests.post(url, headers=headers, json=json_body or {}, timeout=60)
    else:
        raise ValueError(f"Unsupported method: {method}")
    if not r.ok:
        detail = (r.text or "")[:500]
        logger.warning("Genie API %s %s → %s: %s", method, path, r.status_code, detail)
        raise HTTPException(r.status_code, f"Genie API error: {detail}")
    try:
        return r.json()
    except Exception:
        return {}


async def _api(method: str, path: str, *, json_body: dict | None = None) -> dict:
    """Async wrapper around the Genie REST call so polling routes don't block
    the event loop on the 60s sync HTTP timeout."""
    return await asyncio.to_thread(_api_sync, method, path, json_body=json_body)


# ---------------------------------------------------------------------------
# Shape normalisation — REST responses are consistent, so the work is small.
# ---------------------------------------------------------------------------

def _flatten_message(msg: dict) -> dict:
    """Normalise a GenieMessage JSON into a consistent shape for the UI."""
    if not msg:
        return {"message_id": None, "conversation_id": None, "status": None, "attachments": []}

    attachments: list[dict] = []
    for a in msg.get("attachments") or []:
        att: dict[str, Any] = {}
        if a.get("attachment_id"):
            att["attachment_id"] = a["attachment_id"]
        text_obj = a.get("text")
        if text_obj:
            # May be {"content": "..."} or a plain string
            att["text"] = text_obj.get("content") if isinstance(text_obj, dict) else str(text_obj)
        if a.get("query"):
            q = a["query"]
            att["query"] = {
                "description":  q.get("description"),
                "query":        q.get("query"),
                "title":        q.get("title"),
                "statement_id": q.get("statement_id"),
            }
        sug = a.get("suggested_questions")
        if sug:
            qs = sug.get("questions") if isinstance(sug, dict) else sug
            att["suggested_questions"] = [str(q) for q in (qs or [])]
        attachments.append(att)

    return {
        "message_id":      msg.get("message_id") or msg.get("id"),
        "conversation_id": msg.get("conversation_id"),
        "status":          msg.get("status"),
        "content":         msg.get("content"),
        "error":           msg.get("error"),
        "attachments":     attachments,
    }


def _attachment_id_with_query(msg: dict) -> Optional[str]:
    for a in msg.get("attachments") or []:
        if a.get("query") and a.get("attachment_id"):
            return a["attachment_id"]
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

class StartRequest(BaseModel):
    content: str


@router.post("/{space_id}/start")
async def start_conversation(space_id: str, req: StartRequest):
    """Open a conversation with the first user message. Returns the conversation
    id and the initial message (still IN_PROGRESS — the client polls the
    get-message endpoint until COMPLETED)."""
    try:
        body = await _api("POST", f"/api/2.0/genie/spaces/{space_id}/start-conversation",
                    json_body={"content": req.content})
        # REST returns: conversation_id, conversation, message, message_id
        msg = body.get("message") or {}
        if not msg.get("conversation_id"):
            msg["conversation_id"] = body.get("conversation_id")
        return {
            "conversation_id": body.get("conversation_id"),
            "message":         _flatten_message(msg),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("genie start_conversation failed")
        raise HTTPException(500, f"Genie start failed: {str(e)[:300]}")


class MessageRequest(BaseModel):
    content: str


@router.post("/{space_id}/conversations/{conversation_id}/message")
async def create_message(space_id: str, conversation_id: str, req: MessageRequest):
    try:
        body = await _api("POST",
                    f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages",
                    json_body={"content": req.content})
        return {"message": _flatten_message(body)}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("genie create_message failed")
        raise HTTPException(500, f"Genie message failed: {str(e)[:300]}")


@router.get("/{space_id}/conversations/{conversation_id}/messages/{message_id}")
async def get_message(space_id: str, conversation_id: str, message_id: str):
    """Poll for a message's current state."""
    try:
        body = await _api("GET",
                    f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}")
        return _flatten_message(body)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("genie get_message failed")
        raise HTTPException(500, f"Genie get_message failed: {str(e)[:300]}")


@router.get("/{space_id}/conversations/{conversation_id}/messages/{message_id}/query-result")
async def query_result(space_id: str, conversation_id: str, message_id: str):
    """Return the SQL + executed result for a message's SQL attachment.
    If the reply is text-only, returns {"has_result": false}."""
    try:
        msg = await _api("GET",
                   f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}/messages/{message_id}")
        attachment_id = _attachment_id_with_query(msg)
        if not attachment_id:
            return {"has_result": False}

        body = await _api("GET",
                    f"/api/2.0/genie/spaces/{space_id}/conversations/{conversation_id}"
                    f"/messages/{message_id}/attachments/{attachment_id}/query-result")

        # Extract columns + rows from statement_response.{manifest, result}
        columns: list[str] = []
        rows: list[list[Any]] = []
        statement = body.get("statement_response") or body
        manifest = (statement or {}).get("manifest") or {}
        schema = manifest.get("schema") or {}
        columns = [c.get("name") for c in (schema.get("columns") or [])]
        result_block = (statement or {}).get("result") or {}
        rows = [list(r) for r in (result_block.get("data_array") or [])]

        # Extract SQL from the message attachment
        sql_text = sql_title = None
        for a in msg.get("attachments") or []:
            if a.get("query"):
                sql_text  = a["query"].get("query")
                sql_title = a["query"].get("title")
                break

        return {
            "has_result": True,
            "sql":        sql_text,
            "title":      sql_title,
            "columns":    columns,
            "rows":       rows,
            "row_count":  len(rows),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("genie query_result failed")
        raise HTTPException(500, f"Genie query-result failed: {str(e)[:300]}")
