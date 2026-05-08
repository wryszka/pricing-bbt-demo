# Databricks notebook source
# MAGIC %md
# MAGIC # Live Pricing System — teardown
# MAGIC
# MAGIC Inverse of `01_provision.py`. Brings the live pricing footprint down to
# MAGIC zero cost when the demo isn't running. Order matters — endpoint first,
# MAGIC then online table, then online store. The `live_pricing_metrics` table
# MAGIC stays so the load-test chart history survives a teardown / re-provision
# MAGIC cycle.
# MAGIC
# MAGIC Idempotent — anything already gone is a no-op.

# COMMAND ----------

dbutils.widgets.text("catalog_name",      "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",       "pricing_upt")
dbutils.widgets.text("online_store_name", "pricing-upt-online-store-live")
dbutils.widgets.text("endpoint_name",     "pricing_scorer")

# COMMAND ----------

# MAGIC %pip install databricks-sdk databricks-feature-engineering --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json
catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
online_store  = dbutils.widgets.get("online_store_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
fqn           = f"{catalog}.{schema}"
upt_table     = f"{fqn}.unified_pricing_table_live"

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %run ../../utils/audit

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

removed = {}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Delete the serving endpoint

# COMMAND ----------

try:
    w.serving_endpoints.delete(endpoint_name)
    print(f"endpoint deleted: {endpoint_name}")
    removed["endpoint"] = endpoint_name
except Exception as e:
    msg = str(e).lower()
    if "does not exist" in msg or "not found" in msg or "404" in msg:
        print(f"endpoint already absent: {endpoint_name}")
    else:
        print(f"endpoint delete error (continuing): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Delete the online table (CONTINUOUS-published Lakebase table)

# COMMAND ----------

try:
    w.online_tables.delete(upt_table)
    print(f"online table deleted: {upt_table}")
    removed["online_table"] = upt_table
except Exception as e:
    msg = str(e).lower()
    if "does not exist" in msg or "not found" in msg or "404" in msg:
        print(f"online table already absent: {upt_table}")
    else:
        print(f"online table delete error (continuing): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Delete the Lakebase online store

# COMMAND ----------

try:
    w.feature_store.delete_online_store(online_store)
    print(f"online store deleted: {online_store}")
    removed["online_store"] = online_store
except Exception as e:
    msg = str(e).lower()
    if "does not exist" in msg or "not found" in msg or "404" in msg:
        print(f"online store already absent: {online_store}")
    else:
        print(f"online store delete error (continuing): {e}")

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type   = "live_pricing_stopped",
    entity_type  = "endpoint",
    entity_id    = endpoint_name,
    entity_version="",
    user_id      = user,
    details={"removed": removed},
    source       = "notebook",
)

dbutils.notebook.exit(json.dumps({"removed": removed}))
