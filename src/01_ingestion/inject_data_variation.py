# Databricks notebook source
# MAGIC %md
# MAGIC # Inject vendor-data variation into raw tables
# MAGIC
# MAGIC Runs AFTER the silver build completes — mutates the `raw_*` tables to
# MAGIC simulate fresh vendor data that's drifted from the previously-approved
# MAGIC silver snapshot. This is what gives the Data Ingestion UI a real
# MAGIC diff + impact to display: silver represents the last-approved state,
# MAGIC raw represents the new (un-reviewed) update.
# MAGIC
# MAGIC Deterministic per-key — re-runs produce the same mutations so demo
# MAGIC behaviour is stable. Idempotent — running twice is the same as once
# MAGIC because mutations are computed from the SOURCE silver values.
# MAGIC
# MAGIC Mutations applied:
# MAGIC  * **raw_geospatial_hazard_enrichment** — ~30% of postcodes get a
# MAGIC    risk uplift (flood +1 zone, crime +20%, subsidence +1)
# MAGIC  * **raw_credit_bureau_summary** — ~10% of policies get a credit
# MAGIC    score drift (±50 points) and ±1 CCJ
# MAGIC  * **raw_market_pricing_benchmark** — ~30% of SIC+region keys get a
# MAGIC    market median rate uplift (+8%)
# MAGIC
# MAGIC Tweak the WHERE clauses below if you want to change which fraction
# MAGIC of rows is affected — the demo narrative looks crispest with 20-40%.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

import json

# COMMAND ----------

# MAGIC %md
# MAGIC ## Geospatial hazard — risk uplift on ~30% of postcodes
# MAGIC
# MAGIC Mutate from the CURRENT silver values so re-runs are stable (idempotent
# MAGIC at the SQL level — same hash-based key selection each time, same
# MAGIC silver values feed the formula, same output).

# COMMAND ----------

spark.sql(f"""
    MERGE INTO {fqn}.raw_geospatial_hazard_enrichment AS r
    USING (
        SELECT s.postcode_sector,
               s.flood_zone_rating       AS old_flood,
               s.crime_theft_index       AS old_crime,
               s.subsidence_risk         AS old_subsidence,
               -- Deterministic 30% sample by hash
               (abs(hash(s.postcode_sector)) % 10) < 3 AS mutate
        FROM {fqn}.silver_geospatial_hazard_enrichment s
    ) AS m
    ON r.postcode_sector = m.postcode_sector
    WHEN MATCHED AND m.mutate THEN UPDATE SET
        r.flood_zone_rating = CAST(LEAST(10, m.old_flood + 1) AS STRING),
        r.crime_theft_index = CAST(ROUND(m.old_crime * 1.20, 2) AS STRING),
        r.subsidence_risk   = CAST(LEAST(10, m.old_subsidence + 1) AS STRING)
""")
geo_changed = spark.sql(f"""
    SELECT COUNT(*) AS n
    FROM {fqn}.raw_geospatial_hazard_enrichment r
    JOIN {fqn}.silver_geospatial_hazard_enrichment s
      ON r.postcode_sector = s.postcode_sector
    WHERE CAST(r.flood_zone_rating AS STRING) != CAST(s.flood_zone_rating AS STRING)
       OR CAST(r.crime_theft_index AS STRING) != CAST(s.crime_theft_index AS STRING)
       OR CAST(r.subsidence_risk   AS STRING) != CAST(s.subsidence_risk   AS STRING)
""").collect()[0]["n"]
print(f"geospatial — {geo_changed} postcodes diverged from silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Credit bureau — score drift on ~10% of policies
# MAGIC
# MAGIC Half drift downward (deteriorating credit), half upward (improving).
# MAGIC ±1 CCJ to mimic court-judgment updates.

# COMMAND ----------

spark.sql(f"""
    MERGE INTO {fqn}.raw_credit_bureau_summary AS r
    USING (
        SELECT s.policy_id,
               s.credit_score                                  AS old_score,
               s.ccj_count                                     AS old_ccj,
               (abs(hash(s.policy_id)) % 100) < 10             AS mutate,
               (abs(hash(s.policy_id || 'd')) % 2) = 0         AS drift_down
        FROM {fqn}.silver_credit_bureau_summary s
    ) AS m
    ON r.policy_id = m.policy_id
    WHEN MATCHED AND m.mutate THEN UPDATE SET
        r.credit_score = CAST(
            GREATEST(300, LEAST(900,
                CAST(m.old_score AS INT) + CASE WHEN m.drift_down THEN -50 ELSE 50 END
            )) AS STRING
        ),
        r.ccj_count = CAST(
            GREATEST(0, CAST(m.old_ccj AS INT) + CASE WHEN m.drift_down THEN 1 ELSE -1 END)
            AS STRING
        )
""")
credit_changed = spark.sql(f"""
    SELECT COUNT(*) AS n
    FROM {fqn}.raw_credit_bureau_summary r
    JOIN {fqn}.silver_credit_bureau_summary s
      ON r.policy_id = s.policy_id
    WHERE CAST(r.credit_score AS STRING) != CAST(s.credit_score AS STRING)
       OR CAST(r.ccj_count    AS STRING) != CAST(s.ccj_count    AS STRING)
""").collect()[0]["n"]
print(f"credit bureau — {credit_changed} policies diverged from silver")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Market pricing benchmark — rate uplift on ~30% of SIC+region keys

# COMMAND ----------

spark.sql(f"""
    MERGE INTO {fqn}.raw_market_pricing_benchmark AS r
    USING (
        SELECT s.match_key_sic_region,
               s.market_median_rate                                       AS old_rate,
               s.competitor_a_min_premium                                 AS old_comp,
               (abs(hash(s.match_key_sic_region)) % 10) < 3               AS mutate
        FROM {fqn}.silver_market_pricing_benchmark s
    ) AS m
    ON r.match_key_sic_region = m.match_key_sic_region
    WHEN MATCHED AND m.mutate THEN UPDATE SET
        r.market_median_rate       = CAST(ROUND(m.old_rate * 1.08, 2) AS STRING),
        r.competitor_a_min_premium = CAST(ROUND(m.old_comp * 1.05, 2) AS STRING)
""")
market_changed = spark.sql(f"""
    SELECT COUNT(*) AS n
    FROM {fqn}.raw_market_pricing_benchmark r
    JOIN {fqn}.silver_market_pricing_benchmark s
      ON r.match_key_sic_region = s.match_key_sic_region
    WHERE CAST(r.market_median_rate       AS STRING) != CAST(s.market_median_rate       AS STRING)
       OR CAST(r.competitor_a_min_premium AS STRING) != CAST(s.competitor_a_min_premium AS STRING)
""").collect()[0]["n"]
print(f"market pricing — {market_changed} SIC+region keys diverged from silver")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "geospatial_changed":    geo_changed,
    "credit_bureau_changed": credit_changed,
    "market_pricing_changed": market_changed,
}))
