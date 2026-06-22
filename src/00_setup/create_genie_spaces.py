# Databricks notebook source
# MAGIC %md
# MAGIC # Post-deploy — create Genie spaces + wire ids into app_config
# MAGIC
# MAGIC On a fresh deploy the app's `GENIE_SPACE_ID` / `GENIE_QUOTE_SPACE_ID` env
# MAGIC vars are blank, so the Pricing AI / Modelling-Mart Genie panels stay hidden.
# MAGIC This notebook creates the two Genie spaces over the freshly-built tables and
# MAGIC writes their ids to a small `app_config` table that the app reads as a
# MAGIC fallback (see `/api/config` in src/app/app.py) — so the panels light up
# MAGIC without an app redeploy.
# MAGIC
# MAGIC Idempotent: if `app_config` already records a space id and that space still
# MAGIC exists, the space is left as-is. Fully fault-tolerant: a failure creating one
# MAGIC space does not block the other, and never breaks the app (blank => hidden).
# MAGIC
# MAGIC Run AFTER the data pipeline (the spaces reference built tables) and the
# MAGIC grant job (the app SP needs read access). `deploy.sh` runs it for you.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("app_name", "pricing-workbench")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
warehouse_id = dbutils.widgets.get("warehouse_id")
app_name = dbutils.widgets.get("app_name")
fqn = f"{catalog}.{schema}"

import json
import uuid
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
if not warehouse_id:
    raise RuntimeError("warehouse_id is required to create Genie spaces.")

# The app queries Genie as its own service principal, so the SP needs CAN_RUN on
# each space. Resolve it now (best-effort - if the app is not deployed yet we
# still create the spaces; the grant just gets skipped).
try:
    app_sp = w.apps.get(name=app_name).service_principal_client_id
except Exception:
    app_sp = None

def grant_sp_on_space(space_id: str) -> None:
    if not (space_id and app_sp):
        return
    try:
        w.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/genie/{space_id}",
            body={"access_control_list": [
                {"service_principal_name": app_sp, "permission_level": "CAN_RUN"}
            ]},
        )
        print(f"  granted app SP CAN_RUN on genie space {space_id}")
    except Exception as e:
        print(f"  FAILED to grant app SP on genie space {space_id}: {e}")

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
parent_path = f"/Workspace/Users/{user}/genie"
try:
    w.workspace.mkdirs(parent_path)
except Exception:
    pass

# COMMAND ----------

# MAGIC %md
# MAGIC ## app_config table (key/value the app reads as a fallback)

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.app_config (
        key   STRING COMMENT 'config key, e.g. genie_space_id',
        value STRING COMMENT 'config value'
    ) COMMENT 'Post-deploy-populated ids (Genie spaces, dashboard) the app reads when env vars are blank.'
""")

def set_config(key: str, value: str) -> None:
    safe = (value or "").replace("'", "''")
    spark.sql(f"DELETE FROM {fqn}.app_config WHERE key = '{key}'")
    spark.sql(f"INSERT INTO {fqn}.app_config VALUES ('{key}', '{safe}')")

def get_config(key: str) -> str | None:
    rows = spark.sql(f"SELECT value FROM {fqn}.app_config WHERE key = '{key}'").collect()
    return rows[0]["value"] if rows else None

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create the two Genie spaces (idempotent)

# COMMAND ----------

def space_exists(space_id: str) -> bool:
    if not space_id:
        return False
    try:
        w.api_client.do("GET", f"/api/2.0/genie/spaces/{space_id}")
        return True
    except Exception:
        return False

def make_serialized_space(tables: list[str], questions: list[str]) -> str:
    return json.dumps({
        "version": 2,
        "config": {
            # sample_question.id must be a lowercase 32-hex UUID (no hyphens).
            "sample_questions": [{"id": uuid.uuid4().hex, "question": [q]} for q in questions],
        },
        "data_sources": {
            "tables": [{"identifier": t} for t in sorted(tables)],
        },
    })

def ensure_space(config_key: str, title: str, description: str,
                 tables: list[str], questions: list[str]) -> str | None:
    existing = get_config(config_key)
    if space_exists(existing):
        print(f"{config_key}: space {existing} already exists - leaving as-is")
        grant_sp_on_space(existing)
        return existing
    payload = {
        "title": title,
        "description": description,
        "parent_path": parent_path,
        "warehouse_id": warehouse_id,
        "serialized_space": make_serialized_space(tables, questions),
    }
    try:
        resp = w.api_client.do("POST", "/api/2.0/genie/spaces", body=payload)
        space_id = resp.get("space_id") or resp.get("id")
        if not space_id:
            print(f"{config_key}: create returned no id: {resp}")
            return None
        set_config(config_key, space_id)
        print(f"{config_key}: created space {space_id}")
        grant_sp_on_space(space_id)
        return space_id
    except Exception as e:
        print(f"{config_key}: FAILED to create ({e}) - panel stays hidden, app unaffected")
        return None

# Modelling Mart — Pricing Q&A over the unified pricing table.
ensure_space(
    "genie_space_id",
    title="Modelling Mart - Pricing Q&A",
    description="Ask questions across the unified pricing table (policies, claims, "
                "enrichment, factors).",
    tables=[f"{fqn}.unified_pricing_table_live"],
    questions=[
        "What is the total gross written premium by industry risk tier?",
        "Which 10 postcode sectors generate the most premium?",
        "Show average 5-year claim count by construction type",
    ],
)

# Commercial Quote Review — over the quote stream.
ensure_space(
    "genie_quote_space_id",
    title="Commercial Quote Review",
    description="Investigate the commercial quote stream - requests, rating-engine "
                "calls and responses.",
    tables=[f"{fqn}.quotes"],
    questions=[
        "How many quotes converted to bound policies last month?",
        "What is the average quoted premium by industry?",
        "Show the distribution of quote outcomes",
    ],
)

# COMMAND ----------

print("Genie space bootstrap complete. Current app_config:")
final = {r["key"]: r["value"] for r in spark.sql(f"SELECT key, value FROM {fqn}.app_config ORDER BY key").collect()}
for k, v in final.items():
    print(f"  {k} = {v}")

# Surface the outcome to the job run output. Does NOT fail the deploy if a space
# is missing — the app simply hides that panel.
dbutils.notebook.exit(json.dumps({
    "genie_space_id": final.get("genie_space_id", ""),
    "genie_quote_space_id": final.get("genie_quote_space_id", ""),
    "created_both": bool(final.get("genie_space_id") and final.get("genie_quote_space_id")),
}))
