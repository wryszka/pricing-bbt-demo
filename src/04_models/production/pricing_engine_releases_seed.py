# Databricks notebook source
# MAGIC %md
# MAGIC # Pricing engine releases — monthly rate book versions
# MAGIC
# MAGIC Creates `{fqn}.pricing_engine_releases` — one row per monthly rate-
# MAGIC book release. A release bundles:
# MAGIC
# MAGIC * one version of each of the 4 production model families
# MAGIC * one version of the rating engine config
# MAGIC * an effective date + approval metadata + human narrative
# MAGIC
# MAGIC The Pricing Engine tab talks about **releases**, not raw model
# MAGIC versions — matches how real rate books ship through a committee.
# MAGIC Exactly one row can carry `status='champion'` at a time.
# MAGIC
# MAGIC Idempotent: CREATE OR REPLACE. Safe to re-run.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

import json
from datetime import date
from pyspark.sql.types import (
    StructType, StructField, StringType, DateType
)

# ---------------------------------------------------------------------------
# Release inventory
# ---------------------------------------------------------------------------
# Version numbers reference the CURRENT-GENERATION (new self-encoding wrapper)
# UC model versions registered after the last production retrain:
#    freq_glm   54..64 are the 11 backdated replays; 53 is the true champion
#    sev_glm    49..59 are the replays; 48 is the true champion
#    demand_gbm 52..62 are the replays; 51 is the true champion
#    fraud_gbm  54..64 are the replays; 53 is the true champion
# Each backdate tuple (index i) corresponds to the i-th monthly story in
# STORIES[family] (see backdate_versions.py). For our 5-release timeline we
# pick a coherent slice of stories — these versions already exist in UC.
# Rating engine config versions: v1.0 (archived), v1.1 (previous champion),
# v2.0 (current champion).

RELEASES = [
    {
        "release_id":          "dec_2025",
        "display_name":        "December 2025",
        "effective_date":      date(2025, 12, 1),
        "status":              "archived",
        "freq_glm_version":    "62",   # year_end story
        "sev_glm_version":     "57",
        "demand_gbm_version":  "60",
        "fraud_gbm_version":   "62",
        "rating_engine_version": "v1.1",
        "approved_by":         "pricing_committee@bricksurance.com",
        "narrative":           "Year-end cut. Calibration stable across all four families; no material data changes this month. Retained the March expense-loading review (19.5%).",
    },
    {
        "release_id":          "jan_2026",
        "display_name":        "January 2026",
        "effective_date":      date(2026, 1, 1),
        "status":              "archived",
        "freq_glm_version":    "63",   # postcode_refresh
        "sev_glm_version":     "58",   # new_peril_split discussion
        "demand_gbm_version":  "61",
        "fraud_gbm_version":   "63",
        "rating_engine_version": "v2.0",   # new rating-engine config took effect Jan 1
        "approved_by":         "pricing_committee@bricksurance.com",
        "narrative":           "Rating engine v2.0 activated (broker commission cut to 15%, fraud loading tightened to 6% at 0.20 trigger, min premium raised to £150). Frequency model also refreshed with ONSPD 2026 postcode enrichment — +3% lift on IMD decile signal.",
    },
    {
        "release_id":          "feb_2026",
        "display_name":        "February 2026",
        "effective_date":      date(2026, 2, 1),
        "status":              "archived",
        "freq_glm_version":    "64",   # calibration_drift
        "sev_glm_version":     "59",
        "demand_gbm_version":  "62",
        "fraud_gbm_version":   "64",
        "rating_engine_version": "v2.0",
        "approved_by":         "pricing_committee@bricksurance.com",
        "narrative":           "Calibration-drift flag raised internally on the frequency model — observed overprediction on low-turnover SMEs (actual 0.08 vs predicted 0.11). Logged for remediation in March. No rating engine changes.",
    },
    {
        "release_id":          "mar_2026",
        "display_name":        "March 2026",
        "effective_date":      date(2026, 3, 1),
        "status":              "previous_champion",
        "freq_glm_version":    "52",   # calibration_fix (NB: v52 is the simulation_date=2025-05 baseline replica; for the demo we treat this as the March calibration-fix release to keep the story tight)
        "sev_glm_version":     "47",
        "demand_gbm_version":  "50",
        "fraud_gbm_version":   "52",
        "rating_engine_version": "v2.0",
        "approved_by":         "pricing_committee@bricksurance.com",
        "narrative":           "Calibration-fix release. Frequency overprediction on low-turnover SMEs corrected with an isotonic-regression overlay. Severity model unchanged. Rating engine held at v2.0 pending Q2 review.",
    },
    {
        "release_id":          "apr_2026",
        "display_name":        "April 2026",
        "effective_date":      date(2026, 4, 1),
        "status":              "champion",
        "freq_glm_version":    "53",
        "sev_glm_version":     "48",
        "demand_gbm_version":  "51",
        "fraud_gbm_version":   "53",
        "rating_engine_version": "v2.0",
        "approved_by":         "pricing_committee@bricksurance.com",
        "narrative":           "April rate book. All four champion families refreshed from current Modelling Mart, pinned to the new self-encoding scoring wrappers. Rating engine v2.0 retained. Committee approved 1 April after bias-monitor scan cleared on both director_gender and postcode_demographic.",
    },
]

schema_struct = StructType([
    StructField("release_id",              StringType(),  False),
    StructField("display_name",            StringType(),  False),
    StructField("effective_date",          DateType(),    False),
    StructField("status",                  StringType(),  False),
    StructField("freq_glm_version",        StringType(),  False),
    StructField("sev_glm_version",         StringType(),  False),
    StructField("demand_gbm_version",      StringType(),  False),
    StructField("fraud_gbm_version",       StringType(),  False),
    StructField("rating_engine_version",   StringType(),  False),
    StructField("approved_by",             StringType(),  True),
    StructField("narrative",               StringType(),  True),
])

df = spark.createDataFrame(RELEASES, schema_struct)
df.write.mode("overwrite").option("overwriteSchema", "true") \
  .saveAsTable(f"{fqn}.pricing_engine_releases")

print(f"Seeded {fqn}.pricing_engine_releases with {df.count()} releases:")
spark.sql(f"""
    SELECT release_id, status, cast(effective_date as string) as effective_date,
           freq_glm_version AS freq, sev_glm_version AS sev,
           demand_gbm_version AS demand, fraud_gbm_version AS fraud,
           rating_engine_version AS rating
    FROM {fqn}.pricing_engine_releases
    ORDER BY effective_date
""").show(truncate=False)

# COMMAND ----------

# Audit — one event per seed release.
for r in RELEASES:
    det = json.dumps({
        "action":              "publish" if r["status"] == "champion" else "seed",
        "release_id":          r["release_id"],
        "freq_glm":            r["freq_glm_version"],
        "sev_glm":             r["sev_glm_version"],
        "demand_gbm":          r["demand_gbm_version"],
        "fraud_gbm":           r["fraud_gbm_version"],
        "rating_engine":       r["rating_engine_version"],
        "narrative":           (r["narrative"] or "")[:220],
    }).replace("'", "''")
    spark.sql(f"""
        INSERT INTO {fqn}.audit_log
          (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
        SELECT uuid(), 'pricing_engine_release_published', 'pricing_engine_release',
               '{r["release_id"]}', '{r["release_id"]}',
               '{r["approved_by"]}', cast('{r["effective_date"]} 09:00:00' as timestamp),
               '{det}', 'seed_notebook'
    """)
print(f"Audit: {len(RELEASES)} release events logged.")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "rows":     len(RELEASES),
    "champion": "apr_2026",
    "table":    f"{fqn}.pricing_engine_releases",
}))
