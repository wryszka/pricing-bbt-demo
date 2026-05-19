import asyncio
import logging
from typing import Any

import requests
from databricks.sdk.service.sql import Disposition, Format, StatementState

from server.config import get_workspace_client, get_warehouse_id

logger = logging.getLogger(__name__)


def _fetch_external_link(url: str) -> list[list[Any]]:
    """Fetch a pre-signed chunk URL and parse its JSON_ARRAY body."""
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    body = r.json()
    # The payload is the raw JSON array of rows.
    return body if isinstance(body, list) else []


def _execute_sync(sql: str, large: bool = False) -> list[dict[str, Any]]:
    client = get_workspace_client()
    warehouse_id = get_warehouse_id()
    logger.debug("SQL: %s", sql[:200])

    # The default INLINE disposition caps the response at 25 MiB. Queries
    # that scan large UPT joins (e.g. impact-analysis row-level shadow
    # pricing) blow that limit. `large=True` flips to EXTERNAL_LINKS so the
    # server returns pre-signed URLs we GET ourselves; format stays
    # JSON_ARRAY so we can parse without an Arrow dependency.
    extra: dict[str, Any] = {}
    if large:
        extra["disposition"] = Disposition.EXTERNAL_LINKS
        extra["format"]      = Format.JSON_ARRAY
    response = client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        **extra,
    )

    if response.status and response.status.state == StatementState.FAILED:
        error_msg = response.status.error.message if response.status.error else "Unknown"
        raise RuntimeError(f"SQL failed: {error_msg}")

    if not response.manifest or not response.manifest.schema or not response.manifest.schema.columns:
        return []

    columns = [col.name for col in response.manifest.schema.columns]
    rows: list[dict[str, Any]] = []

    # Inline data — first chunk is on response.result.data_array.
    if response.result and response.result.data_array:
        for row_data in response.result.data_array:
            rows.append(dict(zip(columns, row_data)))

    # External links — pre-signed URLs in lieu of inline data. Walk both
    # the first chunk's links (response.result.external_links) and any
    # further chunks listed in manifest.chunks. For each link, GET the
    # URL and JSON-decode the rows.
    seen_chunks: set[int] = set()
    if response.result and response.result.external_links:
        for link in response.result.external_links:
            url = getattr(link, "external_link", None)
            if not url:
                continue
            try:
                for row_data in _fetch_external_link(url):
                    rows.append(dict(zip(columns, row_data)))
                if link.chunk_index is not None:
                    seen_chunks.add(link.chunk_index)
            except Exception as e:
                logger.warning("external chunk fetch failed: %s", e)

        # Walk subsequent chunks via the SDK to get their pre-signed URLs.
        manifest = response.manifest
        if manifest and getattr(manifest, "chunks", None):
            for chunk_info in manifest.chunks:
                idx = chunk_info.chunk_index
                if idx is None or idx in seen_chunks:
                    continue
                next_chunk = client.statement_execution.get_statement_result_chunk_n(
                    statement_id=response.statement_id,
                    chunk_index=idx,
                )
                for link in (next_chunk.external_links or []):
                    url = getattr(link, "external_link", None)
                    if not url:
                        continue
                    try:
                        for row_data in _fetch_external_link(url):
                            rows.append(dict(zip(columns, row_data)))
                    except Exception as e:
                        logger.warning("external chunk %s fetch failed: %s", idx, e)
        return rows

    # Inline data, additional chunks — when the result spans more than one
    # chunk (the default 16 MiB inline cap), only chunk 0 lands on
    # response.result.data_array. Chunks 1..N are listed in manifest.chunks
    # and must be fetched explicitly. Without this loop, large result sets
    # silently truncate to the first chunk's row count (sometimes a single
    # row when wide rows compress to one chunk).
    manifest = response.manifest
    if manifest and getattr(manifest, "chunks", None):
        first_chunk = response.result.chunk_index if response.result else 0
        for chunk_info in manifest.chunks:
            idx = chunk_info.chunk_index
            if idx is None or idx == first_chunk:
                continue
            chunk = client.statement_execution.get_statement_result_chunk_n(
                statement_id=response.statement_id,
                chunk_index=idx,
            )
            if chunk.data_array:
                for row_data in chunk.data_array:
                    rows.append(dict(zip(columns, row_data)))

    return rows


async def execute_query(sql: str, large: bool = False) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_execute_sync, sql, large)
