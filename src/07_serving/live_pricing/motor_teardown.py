# Databricks notebook source
# MAGIC %md
# MAGIC # Motor live serving — teardown (soft stop)
# MAGIC
# MAGIC Deletes the `motor_pricing_scorer` Model Serving endpoint AND stops the
# MAGIC Lakebase instance backing the online store, so deactivate brings the
# MAGIC whole stack down (no idle compute spend). The published online table is
# MAGIC left in place so the next `motor_provision` run skips the ~3 min publish
# MAGIC step and just resumes Lakebase + rebuilds the endpoint container.
# MAGIC
# MAGIC The Lakebase stop runs HERE (job identity = creator, which can manage the
# MAGIC instance) rather than from the app — the app's service principal is not
# MAGIC authorized to stop/resume the instance (403), which is why deactivate
# MAGIC previously left Lakebase running.
# MAGIC
# MAGIC Idempotent: missing endpoint / already-stopped instance are logged and ignored.

# COMMAND ----------

dbutils.widgets.text("catalog_name",      "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",       "pricing_upt")
dbutils.widgets.text("endpoint_name",     "motor_pricing_scorer")
dbutils.widgets.text("online_store_name", "motor-pricing-online-store")

catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
online_store  = dbutils.widgets.get("online_store_name")

import json, time
import requests as _rq
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

# Stop the Lakebase instance (retry through transitional states; verify it
# reaches stopped so deactivate is reliable rather than fire-and-forget).
_host = w.config.host.rstrip("/")
_hdrs = lambda: {**w.config._header_factory(), "Content-Type": "application/json"}

def _instance():
    r = _rq.get(f"{_host}/api/2.0/database/instances/{online_store}", headers=_hdrs(), timeout=30)
    return r.json() if r.status_code == 200 else {}

lakebase_stopped = False
for i in range(8):
    inst = _instance()
    state = (inst.get("state") or "").upper()
    if not inst:
        print(f"lakebase instance {online_store} not found — skipping")
        break
    if inst.get("effective_stopped") or state == "STOPPED":
        lakebase_stopped = True
        print(f"lakebase stopped after ~{i*8}s")
        break
    if state in ("AVAILABLE", "STARTING", "UPDATING"):
        rr = _rq.patch(
            f"{_host}/api/2.0/database/instances/{online_store}?update_mask=stopped",
            headers=_hdrs(), data=json.dumps({"stopped": True}), timeout=30)
        print(f"  lakebase stop PATCH (state={state}) -> {rr.status_code}")
    time.sleep(8)
removed["lakebase_stopped"] = lakebase_stopped

dbutils.notebook.exit(json.dumps({"removed": removed}))
