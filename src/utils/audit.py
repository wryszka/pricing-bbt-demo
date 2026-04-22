# Databricks notebook source
"""Unified audit trail for the Pricing UPT demo.

Provides a single `audit_log` table that records every significant event —
dataset approvals, model decisions, manual uploads, deployments, etc.

Usage from notebooks:
    %run ../utils/audit
    log_event(spark, catalog, schema,
              event_type="dataset_approved", entity_type="dataset",
              entity_id="market_pricing_benchmark", entity_version="v2",
              user_id="analyst@acme.com",
              details={"raw_rows": 150, "silver_rows": 142},
              source="notebook")

Usage from FastAPI app (async):
    from server.audit import log_audit_event
    await log_audit_event("dataset_approved", "dataset", ...)
"""

import json
import uuid
from datetime import datetime, timezone


AUDIT_TABLE = "audit_log"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS {fqn}.audit_log (
    event_id        STRING      COMMENT 'UUID for the event',
    event_type      STRING      COMMENT 'dataset_approved, model_rejected, manual_upload, etc.',
    entity_type     STRING      COMMENT 'dataset, model, feature, endpoint',
    entity_id       STRING      COMMENT 'Identifier of the entity acted upon',
    entity_version  STRING      COMMENT 'Version or snapshot reference',
    user_id         STRING      COMMENT 'Who triggered the event',
    timestamp       TIMESTAMP   COMMENT 'When the event occurred (UTC)',
    details         STRING      COMMENT 'JSON blob with flexible metadata',
    source          STRING      COMMENT 'app, notebook, api'
)
COMMENT 'Unified audit trail for all Pricing Workbench events'
"""


def _escape(val: str) -> str:
    """Escape single quotes for SQL string literals."""
    return val.replace("'", "''")


def log_event_sql(
    catalog: str,
    schema: str,
    event_type: str,
    entity_type: str,
    entity_id: str,
    entity_version: str = "",
    user_id: str = "system",
    details: dict | str | None = None,
    source: str = "notebook",
) -> str:
    """Return an INSERT SQL statement for the audit_log table."""
    event_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    fqn_table = f"{catalog}.{schema}.{AUDIT_TABLE}"

    if details is None:
        details_str = "{}"
    elif isinstance(details, dict):
        details_str = json.dumps(details)
    else:
        details_str = str(details)

    return f"""
        INSERT INTO {fqn_table} VALUES (
            '{event_id}',
            '{_escape(event_type)}',
            '{_escape(entity_type)}',
            '{_escape(entity_id)}',
            '{_escape(entity_version)}',
            '{_escape(user_id)}',
            '{now}',
            '{_escape(details_str)}',
            '{_escape(source)}'
        )
    """


def log_event(
    spark,
    catalog: str,
    schema: str,
    event_type: str,
    entity_type: str,
    entity_id: str,
    entity_version: str = "",
    user_id: str = "system",
    details: dict | str | None = None,
    source: str = "notebook",
) -> None:
    """Log an audit event using a Spark session (for notebooks)."""
    sql = log_event_sql(
        catalog, schema, event_type, entity_type,
        entity_id, entity_version, user_id, details, source,
    )
    spark.sql(sql)


def create_table_sql(catalog: str, schema: str) -> str:
    """Return the CREATE TABLE statement for the audit_log table."""
    return CREATE_TABLE_SQL.format(fqn=f"{catalog}.{schema}")
