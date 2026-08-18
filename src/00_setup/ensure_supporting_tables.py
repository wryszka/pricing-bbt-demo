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

# --- governance-pack history + a real sample PDF -----------------------------
# The Governance page lists packs from the index and opens the PDF at pdf_path.
# The real "Generate pack" job is heavy/flaky on serverless, so for the demo we
# ship a representative sample PDF into the volume and point the seeded history
# rows at it — so every pack OPENS (was 404 "no file exists"). Idempotent.
VOL = f"/Volumes/{catalog}/{schema}/governance_packs"
SAMPLE = "sample_governance_pack.pdf"
SAMPLE_PATH = f"{VOL}/{SAMPLE}"

spark.sql(f"CREATE VOLUME IF NOT EXISTS {fqn}.governance_packs")

# Build a representative pack PDF with NO libraries (raw PDF bytes) — fpdf2/
# matplotlib envs stalled on serverless, and open() can't write UC volumes.
# Upload via the Files API (the supported volume-write path).
SECTIONS = [
    "1. Executive summary", "2. Business context & intended use",
    "3. Data lineage & sources", "4. Model specification",
    "5. Performance evidence", "6. Feature behaviour",
    "7. Stability & version history", "8. Fairness & ethical considerations",
    "9. Risks & controls", "10. Regulatory coverage",
    "11. Audit trail", "12. Committee sign-off",
]

def _make_pdf(title, lines):
    def esc(s): return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    content = ["BT", "/F1 20 Tf", "54 720 Td", f"({esc(title)}) Tj", "/F1 11 Tf", "0 -28 Td"]
    for ln in lines:
        content += [f"({esc(ln)}) Tj", "0 -18 Td"]
    content.append("ET")
    stream = "\n".join(content).encode("latin-1")
    objs = [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R/Resources<</Font<</F1 5 0 R>>>>>>",
        b"<</Length %d>>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]
    pdf = b"%PDF-1.4\n"; offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(pdf)); pdf += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(pdf); pdf += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    for off in offs: pdf += b"%010d 00000 n \n" % off
    pdf += b"trailer\n<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF" % (len(objs) + 1, xref)
    return pdf

_lines = ["Bricksurance SE - representative sample governance pack "
          "(illustrative; synthetic data).", ""] + SECTIONS
data = _make_pdf("Model Governance Pack", _lines)
try:
    from databricks.sdk import WorkspaceClient
    import io as _io
    WorkspaceClient().files.upload(SAMPLE_PATH, _io.BytesIO(data), overwrite=True)
    print(f"✓ uploaded sample pack PDF ({len(data)} bytes) → {SAMPLE_PATH}")
except Exception as e:
    print(f"⚠ could not upload sample pack PDF (non-fatal): {str(e)[:160]}")

try:
    author = spark.sql("SELECT current_user()").collect()[0][0]
except Exception:
    author = "actuarial_pricing_team"

try:
    have = spark.sql(f"SELECT count(*) c FROM {fqn}.governance_packs_index").collect()[0]["c"]
except Exception:
    have = 0
if have == 0:
    seed_rows = [
        ("GP-20260219094501-freq_glm-v25",   "freq_glm",   "25", "[seed-history] Feb baseline — initial production cut",      "gini", 0.1342, "2026-02-19 09:45:01"),
        ("GP-20260219102230-sev_glm-v22",    "sev_glm",    "22", "[seed-history] Feb baseline — initial production cut",      "gini", 0.0188, "2026-02-19 10:22:30"),
        ("GP-20260219104812-demand_gbm-v25", "demand_gbm", "25", "[seed-history] Feb baseline — initial production cut",      "auc",  0.5102, "2026-02-19 10:48:12"),
        ("GP-20260219111545-fraud_gbm-v25",  "fraud_gbm",  "25", "[seed-history] Feb baseline — initial production cut",      "auc",  0.6841, "2026-02-19 11:15:45"),
        ("GP-20260317143205-freq_glm-v33",   "freq_glm",   "33", "[seed-history] Mar refit — credit bureau feature added",    "gini", 0.1410, "2026-03-17 14:32:05"),
        ("GP-20260317150022-sev_glm-v28",    "sev_glm",    "28", "[seed-history] Mar refit — IBNR floor adjustment",          "gini", 0.0214, "2026-03-17 15:00:22"),
        ("GP-20260317152715-demand_gbm-v32", "demand_gbm", "32", "[seed-history] Mar refit — broker-channel features",       "auc",  0.5193, "2026-03-17 15:27:15"),
        ("GP-20260317155930-fraud_gbm-v34",  "fraud_gbm",  "34", "[seed-history] Mar refit — SIU referral threshold tuning", "auc",  0.7012, "2026-03-17 15:59:30"),
    ]
    vals = ",\n".join(
        f"('{pid}','{fam}','{ver}','{fqn}.{fam}',NULL,'{story}',true,'{metric}',{val},"
        f"'{SAMPLE_PATH}',{len(data) if 'data' in dir() else 180000},'{author}',TIMESTAMP'{ts}')"
        for pid, fam, ver, story, metric, val, ts in seed_rows
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
    # fix existing seed-history rows to point at the sample PDF that exists
    spark.sql(f"""
        UPDATE {fqn}.governance_packs_index
        SET pdf_path = '{SAMPLE_PATH}'
        WHERE story LIKE '[seed-history]%'
    """)
    print(f"ℹ index has {have} rows — repointed seed-history pdf_path to the sample")

print("Supporting tables ready.")
