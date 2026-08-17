# Databricks notebook source
# MAGIC %md
# MAGIC # Ensure supporting tables exist
# MAGIC
# MAGIC The agent serving endpoints declare several tables as `resources` at
# MAGIC deploy time, so those tables must EXIST before the agents deploy:
# MAGIC governance_packs_index, governance_pack_sidecars, factory_runs,
# MAGIC factory_variants. The governance PDF job and the app lifespan normally
# MAGIC create them, but the orchestrator can't depend on those, so we create
# MAGIC them here (idempotent, empty). Real content is populated later by the
# MAGIC governance-pack job / the app.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_pricing_v2_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_workbench")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

DDL = {
    "governance_packs_index": f"""
        CREATE TABLE IF NOT EXISTS {fqn}.governance_packs_index (
            pack_id STRING, model_family STRING, model_version STRING,
            model_uc_name STRING, mlflow_run_id STRING, story STRING,
            simulated BOOLEAN, primary_metric STRING, primary_value DOUBLE,
            pdf_path STRING, size_bytes BIGINT, generated_by STRING,
            generated_at TIMESTAMP
        )""",
    "governance_pack_sidecars": f"""
        CREATE TABLE IF NOT EXISTS {fqn}.governance_pack_sidecars (
            pack_id STRING, filename STRING, content STRING,
            content_type STRING, written_at TIMESTAMP
        )""",
    "factory_runs": f"""
        CREATE TABLE IF NOT EXISTS {fqn}.factory_runs (
            run_id STRING, model_family STRING, plan_json STRING,
            narrative STRING, approved_by STRING, started_at TIMESTAMP,
            duration_seconds DOUBLE, status STRING, variant_count INT
        )""",
    "factory_variants": f"""
        CREATE TABLE IF NOT EXISTS {fqn}.factory_variants (
            run_id STRING, variant_id STRING, name STRING, category STRING,
            config_json STRING, metrics_json STRING, n_features INT,
            created_at TIMESTAMP
        )""",
}

for name, ddl in DDL.items():
    spark.sql(ddl)
    print(f"✓ ensured {name}")

# --- seed a governance-pack history if the index is empty --------------------
# The Governance page lists packs from this table; without a seed a fresh
# self-deploy shows an empty packs section. Seed a representative version
# timeline (idempotent — only when empty; the real "Generate pack" button adds
# live packs on top). Same rows demo_reset re-asserts.
try:
    have = spark.sql(f"SELECT count(*) c FROM {fqn}.governance_packs_index").collect()[0]["c"]
except Exception:
    have = 0
if have == 0:
    try:
        author = spark.sql("SELECT current_user()").collect()[0][0]
    except Exception:
        author = "actuarial_pricing_team"
    vol = f"/Volumes/{catalog}/{schema}/governance_packs"
    seed_rows = [
        ("GP-20260219094501-freq_glm-v25",   "freq_glm",   "25", "[seed-history] Feb baseline — initial production cut",       "gini", 0.1342, "freq_glm_v41.pdf",   "2026-02-19 09:45:01"),
        ("GP-20260219102230-sev_glm-v22",    "sev_glm",    "22", "[seed-history] Feb baseline — initial production cut",       "gini", 0.0188, "sev_glm_v36.pdf",    "2026-02-19 10:22:30"),
        ("GP-20260219104812-demand_gbm-v25", "demand_gbm", "25", "[seed-history] Feb baseline — initial production cut",       "auc",  0.5102, "demand_gbm_v39.pdf", "2026-02-19 10:48:12"),
        ("GP-20260219111545-fraud_gbm-v25",  "fraud_gbm",  "25", "[seed-history] Feb baseline — initial production cut",       "auc",  0.6841, "fraud_gbm_v41.pdf",  "2026-02-19 11:15:45"),
        ("GP-20260317143205-freq_glm-v33",   "freq_glm",   "33", "[seed-history] Mar refit — credit bureau feature added",     "gini", 0.1410, "freq_glm_v41.pdf",   "2026-03-17 14:32:05"),
        ("GP-20260317150022-sev_glm-v28",    "sev_glm",    "28", "[seed-history] Mar refit — IBNR floor adjustment",          "gini", 0.0214, "sev_glm_v36.pdf",    "2026-03-17 15:00:22"),
        ("GP-20260317152715-demand_gbm-v32", "demand_gbm", "32", "[seed-history] Mar refit — broker-channel features",        "auc",  0.5193, "demand_gbm_v39.pdf", "2026-03-17 15:27:15"),
        ("GP-20260317155930-fraud_gbm-v34",  "fraud_gbm",  "34", "[seed-history] Mar refit — SIU referral threshold tuning",  "auc",  0.7012, "fraud_gbm_v41.pdf",  "2026-03-17 15:59:30"),
    ]
    vals = ",\n".join(
        f"('{pid}','{fam}','{ver}','{fqn}.{fam}',NULL,'{story}',true,'{metric}',{val},"
        f"'{vol}/{pdf}',180000,'{author}',TIMESTAMP'{ts}')"
        for pid, fam, ver, story, metric, val, pdf, ts in seed_rows
    )
    spark.sql(f"""
        INSERT INTO {fqn}.governance_packs_index
          (pack_id, model_family, model_version, model_uc_name, mlflow_run_id,
           story, simulated, primary_metric, primary_value, pdf_path, size_bytes,
           generated_by, generated_at)
        VALUES {vals}
    """)
    print(f"✓ seeded {len(seed_rows)} governance-pack history rows")
else:
    print(f"ℹ governance_packs_index already has {have} rows — seed skipped")

print("Supporting tables ready.")
