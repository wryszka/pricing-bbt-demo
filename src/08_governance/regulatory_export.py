# Databricks notebook source
# MAGIC %md
# MAGIC # Regulatory Export — Complete Model Governance Package
# MAGIC
# MAGIC Generates a comprehensive regulatory submission package for a pricing model.
# MAGIC Combines data lineage, model card, approval chain, serving config, and DQ
# MAGIC reports into a single auditable document.
# MAGIC
# MAGIC **This format can be adapted for Solvency II, Lloyd's minimum standards,
# MAGIC FCA requirements, or other regulatory frameworks as needed.**

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("model_name", "glm_frequency_model")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
model_name_short = dbutils.widgets.get("model_name")

fqn = f"{catalog}.{schema}"
full_model_name = f"{fqn}.{model_name_short}"

# COMMAND ----------

import json
from datetime import datetime, timezone

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Collect model metadata from MLflow

# COMMAND ----------

import mlflow
mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()

versions = client.search_model_versions(f"name='{full_model_name}'")
if not versions:
    print(f"No model found: {full_model_name}")
    dbutils.notebook.exit("MODEL_NOT_FOUND")

latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
run = client.get_run(latest.run_id)

model_card = {
    "model_name": full_model_name,
    "model_version": latest.version,
    "mlflow_run_id": latest.run_id,
    "status": str(latest.status),
    "creation_timestamp": latest.creation_timestamp,
    "parameters": dict(run.data.params),
    "metrics": {k: round(v, 6) for k, v in run.data.metrics.items()},
    "tags": dict(run.data.tags),
}

print(f"Model: {full_model_name} v{latest.version}")
print(f"Run ID: {latest.run_id}")
print(f"Parameters: {len(run.data.params)}")
print(f"Metrics: {len(run.data.metrics)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Data lineage — training data provenance

# COMMAND ----------

upt_table = f"{fqn}.unified_pricing_table_live"
upt_version = run.data.params.get("upt_delta_version", "?")

# Get UPT metadata
history = spark.sql(f"DESCRIBE HISTORY {upt_table} LIMIT 5").collect()
upt_stats = spark.sql(f"""
    SELECT count(*) as rows, count(DISTINCT policy_id) as policies
    FROM {upt_table}
""").collect()[0]

# Tags
tags = spark.sql(f"""
    SELECT tag_name, tag_value
    FROM {catalog}.information_schema.table_tags
    WHERE schema_name = '{schema}' AND table_name = 'unified_pricing_table_live'
""").collect()

data_lineage = {
    "training_table": upt_table,
    "training_delta_version": upt_version,
    "current_delta_version": history[0]["version"] if history else "?",
    "row_count": upt_stats.rows,
    "unique_policies": upt_stats.policies,
    "table_tags": {r.tag_name: r.tag_value for r in tags},
    "source_tables": [
        f"{fqn}.internal_commercial_policies",
        f"{fqn}.internal_claims_history",
        f"{fqn}.quotes",
        f"{fqn}.silver_market_pricing_benchmark",
        f"{fqn}.silver_geospatial_hazard_enrichment",
        f"{fqn}.silver_credit_bureau_summary",
    ],
    "delta_history": [
        {"version": h["version"], "timestamp": str(h["timestamp"]), "operation": h["operation"]}
        for h in history[:5]
    ],
}

print(f"Training data: {upt_table} (Delta v{upt_version})")
print(f"  Rows: {upt_stats.rows:,} | Policies: {upt_stats.policies:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Approval chain from audit trail

# COMMAND ----------

audit_events = spark.sql(f"""
    SELECT event_id, event_type, entity_type, entity_id, user_id,
           timestamp, details, source
    FROM {fqn}.audit_log
    WHERE entity_id LIKE '%{model_name_short}%'
       OR entity_type = 'model'
    ORDER BY timestamp
""").collect()

approval_chain = [
    {
        "event_id": e.event_id,
        "event_type": e.event_type,
        "entity": f"{e.entity_type}/{e.entity_id}",
        "user": e.user_id,
        "timestamp": str(e.timestamp),
        "source": e.source,
    }
    for e in audit_events
]

print(f"Approval chain: {len(approval_chain)} events")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Data quality expectations

# COMMAND ----------

# DQ pass rates from raw → silver pipeline
dq_report = []
for ds_name, raw_table, silver_table in [
    ("Market Pricing", "raw_market_pricing_benchmark", "silver_market_pricing_benchmark"),
    ("Geospatial Hazard", "raw_geospatial_hazard_enrichment", "silver_geospatial_hazard_enrichment"),
    ("Credit Bureau", "raw_credit_bureau_summary", "silver_credit_bureau_summary"),
]:
    raw_count = spark.sql(f"SELECT count(*) FROM {fqn}.{raw_table}").collect()[0][0]
    silver_count = spark.sql(f"SELECT count(*) FROM {fqn}.{silver_table}").collect()[0][0]
    pass_rate = round(silver_count / raw_count * 100, 1) if raw_count > 0 else 0
    dq_report.append({
        "dataset": ds_name,
        "raw_rows": raw_count,
        "silver_rows": silver_count,
        "rows_dropped": raw_count - silver_count,
        "pass_rate_pct": pass_rate,
    })

print("Data Quality:")
for r in dq_report:
    print(f"  {r['dataset']}: {r['pass_rate_pct']}% pass rate ({r['rows_dropped']} dropped)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Serving configuration

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

serving_config = {}
try:
    ep = w.serving_endpoints.get("pricing-frequency-endpoint")
    serving_config = {
        "endpoint_name": "pricing-frequency-endpoint",
        "state": str(ep.state.ready) if ep.state else "UNKNOWN",
        "entities": [
            {"name": e.name, "version": e.entity_version, "model": e.entity_name}
            for e in (ep.config.served_entities or [])
        ] if ep.config else [],
        "traffic": [
            {"model": r.served_model_name, "traffic_pct": r.traffic_percentage}
            for r in (ep.config.traffic_config.routes or [])
        ] if ep.config and ep.config.traffic_config else [],
    }
    print(f"Serving: {serving_config['endpoint_name']} ({serving_config['state']})")
except Exception as e:
    serving_config = {"endpoint_name": "pricing-frequency-endpoint", "state": "NOT_DEPLOYED", "note": str(e)[:100]}
    print(f"Serving endpoint not deployed yet: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Online store status

# COMMAND ----------

online_store = {}
try:
    store = w.feature_store.get_online_store("pricing-upt-online-store")
    online_store = {
        "name": store.name,
        "state": str(store.state).split(".")[-1],
        "capacity": store.capacity,
    }
except Exception:
    online_store = {"state": "NOT_CREATED"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Assemble and save regulatory package

# COMMAND ----------

regulatory_package = {
    "metadata": {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "generated_by": dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
        "format_version": "1.0",
        "note": "This format can be adapted for Solvency II, Lloyd's minimum standards, FCA requirements, or other regulatory frameworks.",
    },
    "model_card": model_card,
    "data_lineage": data_lineage,
    "data_quality": dq_report,
    "approval_chain": approval_chain,
    "serving_config": serving_config,
    "online_store": online_store,
}

# Save as JSON
json_str = json.dumps(regulatory_package, indent=2, default=str)
spark.createDataFrame([(full_model_name, latest.version, json_str)],
                      ["model_name", "model_version", "package_json"]) \
    .write.mode("overwrite").saveAsTable(f"{fqn}.regulatory_export_{model_name_short}")

print(f"✓ Regulatory package saved to {fqn}.regulatory_export_{model_name_short}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Generate PDF report

# COMMAND ----------

from fpdf import FPDF

pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=20)

# Title
pdf.add_page()
pdf.set_font("Helvetica", "B", 22)
pdf.cell(0, 12, "Regulatory Model Package", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 12)
pdf.cell(0, 8, f"{full_model_name} v{latest.version}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(5)
pdf.set_font("Helvetica", "I", 9)
pdf.cell(0, 5, f"Generated: {datetime.now(timezone.utc).strftime('%d %B %Y %H:%M UTC')}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, "Bricksurance SE — Pricing Workbench", align="C", new_x="LMARGIN", new_y="NEXT")

# 1. Model Identity
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "1. Model Identity", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
for k, v in [
    ("Model Name", full_model_name),
    ("Version", latest.version),
    ("MLflow Run ID", latest.run_id),
    ("Model Type", model_card["parameters"].get("model_type", "?")),
    ("Features", model_card["parameters"].get("features", "?")),
    ("Train Rows", model_card["parameters"].get("train_rows", "?")),
    ("Test Rows", model_card["parameters"].get("test_rows", "?")),
]:
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(45, 5, k)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, str(v), new_x="LMARGIN", new_y="NEXT")

# 2. Performance Metrics
pdf.ln(3)
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "2. Performance Metrics", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
for metric, value in sorted(model_card["metrics"].items()):
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(45, 5, metric)
    pdf.set_font("Helvetica", "", 9)
    pdf.cell(0, 5, str(value), new_x="LMARGIN", new_y="NEXT")

# 3. Data Lineage
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "3. Data Lineage", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.cell(0, 5, f"Training Table: {data_lineage['training_table']}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Delta Version at Training: v{data_lineage['training_delta_version']}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Current Delta Version: v{data_lineage['current_delta_version']}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"Rows: {data_lineage['row_count']:,} | Policies: {data_lineage['unique_policies']:,}", new_x="LMARGIN", new_y="NEXT")
pdf.ln(2)
pdf.set_font("Helvetica", "B", 9)
pdf.cell(0, 5, "Source Tables:", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 8)
for t in data_lineage["source_tables"]:
    pdf.cell(0, 4, f"  - {t}", new_x="LMARGIN", new_y="NEXT")

# 4. Data Quality
pdf.ln(3)
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "4. Data Quality", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
for r in dq_report:
    pdf.cell(0, 5, f"{r['dataset']}: {r['pass_rate_pct']}% pass rate ({r['raw_rows']} raw → {r['silver_rows']} silver, {r['rows_dropped']} dropped)", new_x="LMARGIN", new_y="NEXT")

# 5. Approval Chain
pdf.add_page()
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "5. Approval Chain", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 8)
if approval_chain:
    for e in approval_chain:
        pdf.cell(0, 4, f"{e['timestamp']} | {e['event_type']} | {e['user']} | {e['entity']} | {e['source']}", new_x="LMARGIN", new_y="NEXT")
else:
    pdf.cell(0, 5, "No approval events recorded for this model.", new_x="LMARGIN", new_y="NEXT")

# 6. Serving Config
pdf.ln(3)
pdf.set_font("Helvetica", "B", 14)
pdf.cell(0, 8, "6. Serving Configuration", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.cell(0, 5, f"Endpoint: {serving_config.get('endpoint_name', '?')}", new_x="LMARGIN", new_y="NEXT")
pdf.cell(0, 5, f"State: {serving_config.get('state', '?')}", new_x="LMARGIN", new_y="NEXT")

# Disclaimer
pdf.add_page()
pdf.ln(20)
pdf.set_font("Helvetica", "B", 12)
pdf.cell(0, 8, "Regulatory Compliance Note", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.multi_cell(0, 5,
    "This document provides a complete audit trail for the referenced pricing model. "
    "All data lineage is tracked automatically by Databricks Unity Catalog. All human "
    "decisions are recorded in the audit_log with user identity and timestamp. "
    "Delta Lake Time Travel enables reconstruction of any historical state.\n\n"
    "This format can be adapted for Solvency II (EIOPA), Lloyd's Minimum Standards, "
    "FCA requirements (PS21/3), APRA CPS 234, or other regulatory frameworks.\n\n"
    "DEMO DISCLAIMER: This is a synthetic demonstration. All data, models, and "
    "financial figures are fictional.", align="C")

pdf_bytes = pdf.output()
pdf_path = f"/tmp/regulatory_package_{model_name_short}.pdf"
with open(pdf_path, "wb") as f:
    f.write(pdf_bytes)

print(f"✓ PDF saved: {pdf_path} ({len(pdf_bytes):,} bytes)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Log to audit trail

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="manual_download",
    entity_type="model",
    entity_id=model_name_short,
    entity_version=latest.version,
    user_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    details={
        "report_type": "regulatory_export",
        "model_name": full_model_name,
        "model_version": latest.version,
        "sections": ["model_card", "data_lineage", "data_quality", "approval_chain", "serving_config"],
    },
    source="notebook",
)
print("✓ Export logged to audit_log")

# COMMAND ----------

print(f"""
Regulatory Export Complete
===========================
Model:     {full_model_name} v{latest.version}
JSON:      {fqn}.regulatory_export_{model_name_short}
PDF:       {pdf_path}

Sections:
  1. Model Identity (type, params, MLflow run)
  2. Performance Metrics ({len(model_card['metrics'])} metrics)
  3. Data Lineage (training table, Delta version, sources)
  4. Data Quality ({len(dq_report)} datasets, pass rates)
  5. Approval Chain ({len(approval_chain)} events)
  6. Serving Configuration

Adaptable for: Solvency II, Lloyd's, FCA, APRA, or other frameworks.
""")
