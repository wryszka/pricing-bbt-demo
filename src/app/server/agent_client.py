"""Shared client for invoking Databricks Agent Framework serving endpoints.

All app-side chat/agent routes go through this helper so every interaction
hits a real governed endpoint (not a direct Foundation-Model-API call) and
the tool trace + token usage comes back in a consistent shape for audit
logging.

`invoke_agent` is async — it runs the blocking HTTP/SDK work on a thread pool
so a slow agent call (typical: 10-30s of tool-use loop) does not stall the
FastAPI event loop. Always await it from route handlers.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

import requests

from server.config import get_workspace_client

logger = logging.getLogger(__name__)

# Transient errors from the underlying Foundation Model endpoint surface here
# as 400 BAD_REQUEST wrapping a 503/502/504 from databricks-claude-sonnet-4-6.
# A single short retry absorbs most of these flakes during a demo.
_TRANSIENT_FM_HINTS = ("503 Server Error", "502 Server Error", "504 Server Error", "Service Unavailable")


def _invoke_sync(endpoint_name: str,
                 question: str,
                 custom_inputs: dict[str, Any] | None,
                 history: list[dict] | None,
                 timeout: int) -> dict:
    """Sync invocation — kept private. Wrap with asyncio.to_thread."""
    try:
        w = get_workspace_client()
        try:
            ep = w.serving_endpoints.get(endpoint_name)
            state = ep.state.ready if ep.state and ep.state.ready else None
            if state and "READY" not in str(state):
                return {"ok": False, "error": f"endpoint not ready (state={state})"}
        except Exception as e:
            return {"ok": False, "error": f"endpoint lookup failed: {e}"}

        host  = w.config.host.rstrip("/")
        token = w.config._header_factory()

        messages = list(history) if history else []
        messages.append({"role": "user", "content": question})

        body: dict[str, Any] = {
            "dataframe_records": [{
                "messages":       messages,
                "custom_inputs":  custom_inputs or {},
            }],
        }

        last_error: Exception | None = None
        data = None
        for attempt in (1, 2):
            try:
                resp = requests.post(
                    f"{host}/serving-endpoints/{endpoint_name}/invocations",
                    headers={**token, "Content-Type": "application/json"},
                    json=body, timeout=timeout,
                )
                # On 4xx the body usually contains the upstream FM error message.
                # If that error looks like a transient FM 5xx, retry once.
                if resp.status_code >= 400:
                    msg = resp.text[:500]
                    if attempt == 1 and any(h in msg for h in _TRANSIENT_FM_HINTS):
                        logger.info("Transient FM error on %s — retrying once: %s",
                                    endpoint_name, msg[:160])
                        time.sleep(2.0)
                        last_error = requests.HTTPError(f"{resp.status_code}: {msg}")
                        continue
                resp.raise_for_status()
                data = resp.json()
                break
            except requests.HTTPError as e:
                last_error = e
                break  # non-transient, give up
            except Exception as e:
                last_error = e
                if attempt == 1:
                    time.sleep(1.0)
                    continue
                break
        if data is None:
            raise last_error or RuntimeError("agent invocation failed with no response")
    except Exception as e:
        logger.warning("Agent endpoint %s failed: %s", endpoint_name, e)
        return {"ok": False, "error": str(e)[:300]}

    preds = data.get("predictions") or data.get("outputs") or data
    if isinstance(preds, list):
        preds = preds[0] if preds else {}
    if not isinstance(preds, dict):
        return {"ok": False, "error": f"unexpected response shape: {type(preds).__name__}"}

    msgs = preds.get("messages") or []
    answer = ""
    if msgs:
        msg = msgs[0] if isinstance(msgs[0], dict) else {}
        answer = msg.get("content") or ""

    return {
        "ok":      True,
        "answer":  answer,
        "trace":   preds.get("trace", []),
        "model":   preds.get("model", endpoint_name),
        "usage":   preds.get("usage", {}),
        "persona": preds.get("persona"),
    }


async def invoke_agent(endpoint_name: str,
                       question: str,
                       custom_inputs: dict[str, Any] | None = None,
                       history: list[dict] | None = None,
                       timeout: int = 240) -> dict:
    """Call an Agent Framework serving endpoint. Returns a dict with:

        ok:      bool
        answer:  str (assistant text) — populated when ok
        trace:   list (tool-call trace) — populated when ok
        model:   str (underlying FM endpoint name)
        usage:   dict (prompt/completion/total tokens)
        persona: str | None (echoed back by the agent if set)
        error:   str — populated when not ok
        cached:  bool — true if returned from the AI response cache

    `history` is an optional list of prior messages ({role, content}); if
    omitted a single user-message is sent.

    Cache behaviour: when the global AI mode is `cached`, this call looks
    up a deterministic hash of (endpoint, question, custom_inputs) in the
    on-volume cache first and returns the stored response on hit. On miss
    it falls through to the live endpoint and writes the response back so
    a subsequent identical call lands fast and verbatim.
    """
    from server import ai_cache
    mode = ai_cache.get_mode()
    key  = ai_cache.cache_key(endpoint_name, question, custom_inputs)
    if mode == "cached":
        hit = ai_cache.get_cached(key)
        if hit is not None:
            hit = dict(hit)
            hit["cached"] = True
            return hit

    result = await asyncio.to_thread(
        _invoke_sync, endpoint_name, question, custom_inputs, history, timeout,
    )

    if mode == "cached" and result.get("ok"):
        # Don't cache history-driven conversations — only single-question
        # invocations are reliably reproducible.
        if not history:
            try:
                ai_cache.put_cached(key, result, endpoint_name, question, custom_inputs)
            except Exception as e:
                logger.warning("ai_cache.put_cached failed for %s: %s", endpoint_name, e)
    result["cached"] = False
    return result
