# Databricks notebook source
# MAGIC %md
# MAGIC # Demo reset
# MAGIC
# MAGIC One-shot reset to clean demo state. Run from the landing page reset
# MAGIC button. Idempotent — safe to re-run.
# MAGIC
# MAGIC What it does:
# MAGIC
# MAGIC | Item | Reset to |
# MAGIC |---|---|
# MAGIC | UC champion aliases (4 model families) | freq_glm v53 / sev_glm v48 / demand_gbm v51 / fraud_gbm v53 |
# MAGIC | `rating_engine_config` table | v2.0 champion, v1.1 previous_champion, v1.0 archived |
# MAGIC | `pricing_engine_releases` table | apr_2026 champion, mar_2026 previous_champion, others archived |
# MAGIC | `compare_results` table | truncate (transient demo output) |
# MAGIC | `historical_quote_scores` table | truncate (transient demo output) |
# MAGIC | `inference_logs` rows where `is_mta = true` | delete (transient MTA simulations) |
# MAGIC | Geospatial vendor refresh state | re-seed so the impact tab shows the 25k-policy story |
# MAGIC | Audit log | NOT cleared — append a `demo_reset` event so the reset itself is auditable |

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

import json
from datetime import datetime

CHAMPION_VERSIONS = {
    "freq_glm":   "53",
    "sev_glm":    "48",
    "demand_gbm": "51",
    "fraud_gbm":  "53",
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Reset UC champion aliases

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
alias_results = {}
for fam, ver in CHAMPION_VERSIONS.items():
    full = f"{fqn}.{fam}"
    try:
        w.registered_models.set_alias(full_name=full, alias="champion", version_num=int(ver))
        alias_results[fam] = f"champion → v{ver}"
    except Exception as e:
        alias_results[fam] = f"FAILED: {str(e)[:120]}"
print("Champion aliases:")
for k, v in alias_results.items(): print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Reset rating_engine_config statuses

# COMMAND ----------

spark.sql(f"""
    UPDATE {fqn}.rating_engine_config
    SET status = CASE version
        WHEN 'v2.0' THEN 'champion'
        WHEN 'v1.1' THEN 'previous_champion'
        ELSE 'archived'
    END
""")
print("rating_engine_config statuses reset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Reset pricing_engine_releases statuses

# COMMAND ----------

spark.sql(f"""
    UPDATE {fqn}.pricing_engine_releases
    SET status = CASE release_id
        WHEN 'apr_2026' THEN 'champion'
        WHEN 'mar_2026' THEN 'previous_champion'
        ELSE 'archived'
    END
""")
print("pricing_engine_releases statuses reset.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Truncate transient demo outputs

# COMMAND ----------

cleanup_counts = {}
# compare_results is intentionally NOT truncated — the Compare & Test tab
# auto-hydrates from the most recent cached run for the active family, so
# preserving prior runs gives the demo a populated landing state without
# needing a fresh job to fire mid-presentation.
for tbl in ("historical_quote_scores",):
    try:
        before = spark.table(f"{fqn}.{tbl}").count()
        spark.sql(f"TRUNCATE TABLE {fqn}.{tbl}")
        cleanup_counts[tbl] = f"{before} rows truncated"
    except Exception as e:
        cleanup_counts[tbl] = f"skipped: {str(e)[:120]}"

# Delete only MTA-simulated rows from inference_logs (keep the backfill)
try:
    n = spark.sql(f"SELECT COUNT(*) AS n FROM {fqn}.inference_logs WHERE is_mta = true").collect()[0]["n"]
    spark.sql(f"DELETE FROM {fqn}.inference_logs WHERE is_mta = true")
    cleanup_counts["inference_logs (is_mta=true)"] = f"{n} rows deleted"
except Exception as e:
    cleanup_counts["inference_logs (is_mta=true)"] = f"skipped: {str(e)[:120]}"

# Revert the geospatial dataset to pending-approval so the demo can walk
# through the HITL approval flow from a fresh state. The other ingestion
# datasets (market_pricing_benchmark, credit_bureau_summary) stay approved
# since they're the "already in production" baseline.
try:
    n = spark.sql(
        f"SELECT COUNT(*) AS n FROM {fqn}.dataset_approvals "
        f"WHERE dataset_name = 'geospatial_hazard_enrichment'"
    ).collect()[0]["n"]
    spark.sql(
        f"DELETE FROM {fqn}.dataset_approvals "
        f"WHERE dataset_name = 'geospatial_hazard_enrichment'"
    )
    cleanup_counts["dataset_approvals (geospatial)"] = f"{n} rows deleted — back to pending"
except Exception as e:
    cleanup_counts["dataset_approvals (geospatial)"] = f"skipped: {str(e)[:120]}"

print("Cleanup:")
for k, v in cleanup_counts.items(): print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Re-seed geospatial vendor refresh
# MAGIC
# MAGIC So the Data Ingestion → Geospatial → Impact tab shows the 25k-policy
# MAGIC pricing impact story.

# COMMAND ----------

# Inline the seed (same as src/00_setup/seed_geospatial_vendor_refresh.sql)
spark.sql(f"""
    CREATE OR REPLACE TABLE {fqn}.raw_geospatial_hazard_enrichment AS
    WITH shifted_silver AS (
        SELECT
            postcode_sector,
            CASE
                WHEN abs(hash(concat(postcode_sector, 'flood_v2'))) % 100 < 15
                    THEN LEAST(10, flood_zone_rating + (abs(hash(postcode_sector)) % 3 + 1))
                WHEN abs(hash(concat(postcode_sector, 'flood_v2'))) % 100 < 30
                    THEN GREATEST(1, flood_zone_rating - (abs(hash(postcode_sector)) % 2 + 1))
                ELSE flood_zone_rating
            END AS flood_zone_rating,
            proximity_to_fire_station_km,
            CASE
                WHEN abs(hash(concat(postcode_sector, 'crime_v2'))) % 100 < 20
                    THEN ROUND(crime_theft_index * 1.15, 1)
                WHEN abs(hash(concat(postcode_sector, 'crime_v2'))) % 100 < 35
                    THEN ROUND(crime_theft_index * 0.90, 1)
                ELSE crime_theft_index
            END AS crime_theft_index,
            CASE
                WHEN abs(hash(concat(postcode_sector, 'subs_v2'))) % 100 < 8
                    THEN ROUND(LEAST(10.0, subsidence_risk + 1.5), 1)
                ELSE subsidence_risk
            END AS subsidence_risk,
            current_timestamp()                              AS _ingested_at,
            'manual_upload:vendor_refresh_2026_q2.csv'       AS _source_file
        FROM {fqn}.silver_geospatial_hazard_enrichment
    )
    SELECT * FROM shifted_silver
""")
n = spark.table(f"{fqn}.raw_geospatial_hazard_enrichment").count()
print(f"Geospatial raw vendor refresh re-seeded: {n} rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Re-seed protected-attribute cohorts on policy_demographics
# MAGIC
# MAGIC Idempotent. Adds ethnicity_proxy + director_age_band if missing — the
# MAGIC bias monitor reads four attributes off this table.

# COMMAND ----------

spark.sql(f"""
    CREATE OR REPLACE TABLE {fqn}.policy_demographics AS
    SELECT
      policy_id, director_gender, postcode_demographic,
      CASE postcode_demographic
        WHEN 'Q1_majority_white' THEN
          CASE WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 90 THEN 'White'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 95 THEN 'Asian'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 98 THEN 'Black'
               ELSE 'Mixed/Other' END
        WHEN 'Q2' THEN
          CASE WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 75 THEN 'White'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 87 THEN 'Asian'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 95 THEN 'Black'
               ELSE 'Mixed/Other' END
        WHEN 'Q3' THEN
          CASE WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 60 THEN 'White'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 78 THEN 'Asian'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 92 THEN 'Black'
               ELSE 'Mixed/Other' END
        WHEN 'Q4' THEN
          CASE WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 45 THEN 'White'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 67 THEN 'Asian'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 89 THEN 'Black'
               ELSE 'Mixed/Other' END
        WHEN 'Q5_most_diverse' THEN
          CASE WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 25 THEN 'White'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 53 THEN 'Asian'
               WHEN abs(hash(concat(policy_id,'eth'))) % 100 < 85 THEN 'Black'
               ELSE 'Mixed/Other' END
        ELSE 'White'
      END AS ethnicity_proxy,
      CASE
        WHEN abs(hash(concat(policy_id,'age'))) % 100 < 18 THEN '25-34'
        WHEN abs(hash(concat(policy_id,'age'))) % 100 < 42 THEN '35-44'
        WHEN abs(hash(concat(policy_id,'age'))) % 100 < 68 THEN '45-54'
        WHEN abs(hash(concat(policy_id,'age'))) % 100 < 88 THEN '55-64'
        ELSE '65+'
      END AS director_age_band
    FROM {fqn}.policy_demographics
""")
print("policy_demographics rebuilt — ethnicity_proxy correlates with postcode_demographic.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Re-seed historical governance pack rows
# MAGIC
# MAGIC Feb / March entries so the Governance "by date" tab shows a meaningful
# MAGIC timeline. PDFs are reused — the demo point is the version timeline.

# COMMAND ----------

spark.sql(f"""
    DELETE FROM {fqn}.governance_packs_index
    WHERE story LIKE '[seed-history]%'
""")
seed_rows = [
    ('GP-20260219094501-freq_glm-v25',   'freq_glm',   '25',
     '[seed-history] Feb baseline — initial production cut',
     'gini', 0.1342, 'freq_glm_v41_20260423_162042.pdf', '2026-02-19 09:45:01'),
    ('GP-20260219102230-sev_glm-v22',    'sev_glm',    '22',
     '[seed-history] Feb baseline — initial production cut',
     'gini', 0.0188, 'sev_glm_v36_20260423_170833.pdf',  '2026-02-19 10:22:30'),
    ('GP-20260219104812-demand_gbm-v25', 'demand_gbm', '25',
     '[seed-history] Feb baseline — initial production cut',
     'auc',  0.5102, 'demand_gbm_v39_20260423_170833.pdf','2026-02-19 10:48:12'),
    ('GP-20260219111545-fraud_gbm-v25',  'fraud_gbm',  '25',
     '[seed-history] Feb baseline — initial production cut',
     'auc',  0.6841, 'fraud_gbm_v41_20260423_171945.pdf','2026-02-19 11:15:45'),
    ('GP-20260317143205-freq_glm-v33',   'freq_glm',   '33',
     '[seed-history] Mar refit — credit bureau feature added',
     'gini', 0.1410, 'freq_glm_v41_20260423_162042.pdf', '2026-03-17 14:32:05'),
    ('GP-20260317150022-sev_glm-v28',    'sev_glm',    '28',
     '[seed-history] Mar refit — IBNR floor adjustment',
     'gini', 0.0214, 'sev_glm_v36_20260423_170833.pdf',  '2026-03-17 15:00:22'),
    ('GP-20260317152715-demand_gbm-v32', 'demand_gbm', '32',
     '[seed-history] Mar refit — broker-channel features',
     'auc',  0.5193, 'demand_gbm_v39_20260423_170833.pdf','2026-03-17 15:27:15'),
    ('GP-20260317155930-fraud_gbm-v34',  'fraud_gbm',  '34',
     '[seed-history] Mar refit — SIU referral threshold tuning',
     'auc',  0.7012, 'fraud_gbm_v41_20260423_171945.pdf','2026-03-17 15:59:30'),
]
vol_prefix = f"/Volumes/{catalog}/{schema}/governance_packs"
values_sql = ",\n".join(
    f"('{pid}', '{fam}', '{ver}', '{fqn}.{fam}', NULL, "
    f"'{story}', false, '{metric}', {val}, "
    f"'{vol_prefix}/{pdf}', 180000, 'laurence.ryszka', "
    f"TIMESTAMP'{ts}')"
    for pid, fam, ver, story, metric, val, pdf, ts in seed_rows
)
spark.sql(f"""
    INSERT INTO {fqn}.governance_packs_index
      (pack_id, model_family, model_version, model_uc_name, mlflow_run_id,
       story, simulated, primary_metric, primary_value, pdf_path, size_bytes,
       generated_by, generated_at)
    VALUES {values_sql}
""")
print(f"Pack history re-seeded: {len(seed_rows)} historical rows.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Audit the reset

# COMMAND ----------

try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except Exception:
    user = "system"

det = json.dumps({
    "champion_aliases":  CHAMPION_VERSIONS,
    "rating_engine":     "v2.0 champion",
    "active_release":    "apr_2026",
    "tables_truncated":  list(cleanup_counts.keys()),
}).replace("'", "''")
spark.sql(f"""
    INSERT INTO {fqn}.audit_log
      (event_id, event_type, entity_type, entity_id, entity_version,
       user_id, timestamp, details, source)
    SELECT uuid(), 'demo_reset', 'workbench', 'all',
           '-', '{user}', current_timestamp(), '{det}', 'reset_notebook'
""")

# COMMAND ----------

warm_outcome = {"called": False}
try:
    from databricks.sdk import WorkspaceClient as _W
    import requests as _rq
    _w = _W()
    _host = _w.config.host.rstrip("/")
    _token_hdr = _w.config._header_factory()
    # Find the workbench app on this workspace — name is stable across targets.
    _app_url = None
    try:
        for _a in _w.apps.list():
            if (_a.name or "").startswith("pricing-workbench"):
                _app_url = (_a.url or "").rstrip("/")
                break
    except Exception:
        _app_url = None
    if _app_url:
        # Wipe stale cached AI responses (champions / packs just got rebuilt)
        # then re-warm so a recorded demo lands on instant, identical answers.
        try:
            _rq.delete(f"{_app_url}/api/admin/ai-cache",
                       headers=_token_hdr, timeout=30)
        except Exception as _e:
            print(f"⚠ ai-cache clear failed (non-fatal): {_e}")
        try:
            r = _rq.post(f"{_app_url}/api/admin/ai-cache/warm",
                         headers=_token_hdr, timeout=600)
            warm_outcome = {"called": True, "status": r.status_code,
                            "body": r.text[:300]}
            print(f"ai-cache warm: {r.status_code}")
        except Exception as _e:
            warm_outcome = {"called": True, "error": str(_e)[:200]}
            print(f"⚠ ai-cache warm failed (non-fatal): {_e}")
    else:
        print("ℹ pricing-workbench app not found in this workspace — skipping warm-up")
except Exception as _e:
    print(f"⚠ ai-cache step skipped (non-fatal): {_e}")

dbutils.notebook.exit(json.dumps({
    "champion_aliases": CHAMPION_VERSIONS,
    "cleanup":          cleanup_counts,
    "geospatial_rows":  n,
    "reset_at":         datetime.utcnow().isoformat() + "Z",
    "ai_cache_warm":    warm_outcome,
}))
