import asyncio
import logging
from typing import Any

from databricks.sdk.service.sql import StatementState

from server.config import get_workspace_client, get_warehouse_id

logger = logging.getLogger(__name__)


def _execute_sync(sql: str) -> list[dict[str, Any]]:
    client = get_workspace_client()
    warehouse_id = get_warehouse_id()
    logger.debug("SQL: %s", sql[:200])

    # INLINE disposition only — Databricks Apps' egress is firewalled away
    # from the cloud-storage hosts that EXTERNAL_LINKS would point to, so
    # large results have to be aggregated server-side before they come back
    # rather than streamed through pre-signed URLs.
    response = client.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
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


async def execute_query(sql: str) -> list[dict[str, Any]]:
    return await asyncio.to_thread(_execute_sync, sql)
