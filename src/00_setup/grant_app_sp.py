# Databricks notebook source
# MAGIC %md
# MAGIC # Grant the app service principal what it needs
# MAGIC
# MAGIC The Databricks App runs as its own service principal. Beyond the
# MAGIC bundle-managed warehouse CAN_USE and job CAN_MANAGE_RUN, the app SP needs
# MAGIC UC read/execute on the schema and CAN_QUERY on the serving endpoints —
# MAGIC otherwise the app shows champions=null / endpoint=null (it can't resolve
# MAGIC model aliases or reach the scorer). These grants are NOT bundle-managed,
# MAGIC so this step asserts them. Idempotent; safe to re-run.
# MAGIC
# MAGIC Skips cleanly if `app_service_principal_id` is empty (first deploy, before
# MAGIC the app exists).

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_pricing_v2_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_workbench")
dbutils.widgets.text("app_service_principal_id", "")
dbutils.widgets.text("endpoint_names", "pricing_scorer,pricing_chat_agent,pricing_governance_agent,motor_pricing_scorer_direct")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
app_sp = dbutils.widgets.get("app_service_principal_id").strip()
endpoints = [e.strip() for e in dbutils.widgets.get("endpoint_names").split(",") if e.strip()]

if not app_sp:
    print("app_service_principal_id not set — skipping app-SP grants (first deploy).")
    dbutils.notebook.exit("skipped: no app SP")

# COMMAND ----------

# UC privileges — the app resolves aliases (EXECUTE) and queries tables (SELECT)
for stmt in [
    f"GRANT USE CATALOG ON CATALOG {catalog} TO `{app_sp}`",
    f"GRANT USE SCHEMA ON SCHEMA {catalog}.{schema} TO `{app_sp}`",
    f"GRANT SELECT ON SCHEMA {catalog}.{schema} TO `{app_sp}`",
    f"GRANT EXECUTE ON SCHEMA {catalog}.{schema} TO `{app_sp}`",
    # READ VOLUME so the app can stream governance-pack PDFs + saved payloads.
    f"GRANT READ VOLUME ON SCHEMA {catalog}.{schema} TO `{app_sp}`",
]:
    try:
        spark.sql(stmt)
        print(f"✓ {stmt}")
    except Exception as e:
        print(f"⚠ {stmt} — {str(e)[:100]}")

# COMMAND ----------

# Serving-endpoint CAN_QUERY — additive PATCH, leaves other grantees untouched.
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

for ep in endpoints:
    try:
        eid = w.serving_endpoints.get(ep).id
        w.api_client.do(
            "PATCH", f"/api/2.0/permissions/serving-endpoints/{eid}",
            body={"access_control_list": [
                {"service_principal_name": app_sp, "permission_level": "CAN_QUERY"}
            ]},
        )
        print(f"✓ {ep}: CAN_QUERY → {app_sp}")
    except Exception as e:
        print(f"⚠ {ep}: grant skipped (endpoint may not be deployed yet) — {str(e)[:100]}")

print("App-SP grants complete.")
