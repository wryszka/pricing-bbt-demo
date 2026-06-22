# Databricks notebook source
# MAGIC %md
# MAGIC # Motor live serving — teardown (soft stop)
# MAGIC
# MAGIC Deletes the `motor_pricing_scorer` Model Serving endpoint to stop
# MAGIC compute spend. Lakebase store + published online table are left in
# MAGIC place so the next `motor_provision` run skips the ~3 min publish step
# MAGIC and just rebuilds the endpoint container.
# MAGIC
# MAGIC Idempotent: missing endpoint is logged and ignored.

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("endpoint_name", "motor_pricing_scorer")

catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")

import json
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

removed = {}
try:
    w.serving_endpoints.delete(endpoint_name)
    removed["endpoint"] = endpoint_name
    print(f"endpoint deleted: {endpoint_name}")
except Exception as e:
    msg = str(e).lower()
    if "does not exist" in msg or "not found" in msg or "404" in msg:
        print(f"endpoint already absent: {endpoint_name}")
    else:
        print(f"endpoint delete error (continuing): {e}")

dbutils.notebook.exit(json.dumps({"removed": removed}))
