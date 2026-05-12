# Databricks notebook source
# MAGIC %md
# MAGIC # Bias demographics backfill — the "unmodelled" attributes
# MAGIC
# MAGIC Creates `{fqn}.policy_demographics` — a side table keyed on `policy_id`
# MAGIC with protected attributes that **are not** used in any production model:
# MAGIC
# MAGIC - `director_gender`         — `M` / `F` / `Other`  (~55/42/3 split)
# MAGIC - `postcode_demographic`    — `Q1_majority_white` … `Q5_most_diverse`
# MAGIC                              (simulated ONS-style ethnic-majority quintile)
# MAGIC
# MAGIC Both columns are deterministic functions of `policy_id` + existing
# MAGIC features, so the correlations hold up under re-runs. None of these
# MAGIC values feed any model — they exist solely so the bias-investigator
# MAGIC agent can monitor disparity in production predictions.
# MAGIC
# MAGIC Idempotent: `CREATE OR REPLACE TABLE`.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

import hashlib
import pyspark.sql.functions as F
from pyspark.sql.types import StringType

# ----- UDF: director_gender -------------------------------------------------
# Deterministic synthesis tightly bound to the FEATURES the production
# models weight heavily — industry_risk_tier (Low/Medium/High), annual
# turnover, ccj_count. This creates a real proxy path the bias
# investigator can trace back to specific rating factors.
_TIER_SKEW = {"High": +0.20, "Medium": +0.00, "Low": -0.15}

def _director_gender(policy_id, tier, turnover, ccj):
    if policy_id is None:
        return "M"
    try:
        tv = float(turnover) if turnover is not None else 0.0
    except Exception:
        tv = 0.0
    try:
        cj = int(ccj) if ccj is not None else 0
    except Exception:
        cj = 0
    seed = int(hashlib.sha256(policy_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF

    male_share = 0.50 + _TIER_SKEW.get(str(tier), 0.0)
    if   tv >= 1_000_000: male_share += 0.12
    elif tv <    100_000: male_share -= 0.08
    if   cj >= 2: male_share += 0.10
    elif cj == 0: male_share -= 0.04
    male_share = max(0.20, min(0.85, male_share))

    if seed < 0.03:
        return "Other"
    if seed < 0.03 + (1 - 0.03) * male_share:
        return "M"
    return "F"

director_gender_udf = F.udf(_director_gender, StringType())

# Postcode demographic is done in SQL (ntile on a composite risk score) so
# that the quintiles are perfectly even regardless of the feature
# distribution — see the SQL below.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the side table from UPT — one row per policy_id

# COMMAND ----------

upt = spark.table(f"{fqn}.unified_pricing_table_live")

# Step A — apply director_gender via the python UDF (correlates with tier /
# turnover / ccj_count).
with_gender = (
    upt.select(
        "policy_id",
        "industry_risk_tier",
        "annual_turnover",
        "ccj_count",
        "crime_theft_index",
        "urban_score",
    )
    .withColumn(
        "director_gender",
        director_gender_udf(
            F.col("policy_id"),
            F.col("industry_risk_tier"),
            F.col("annual_turnover"),
            F.col("ccj_count"),
        ),
    )
)

# Step B — postcode_demographic: simulated ONS ethnic-majority quintile.
# Binds to the policy's technical_premium with heavy jitter (±£900) so the
# quintile is correlated-but-noisy — the kind of signal you'd see joining
# real ONS demographic data to a portfolio this size. Target: ~25-35% gap
# between Q1 and Q5 (believable for a demo, strong enough to catch).
# Q5_most_diverse = top-premium quintile; Q1_majority_white = bottom.
with_gender.createOrReplaceTempView("_v_with_gender")

# Use technical_premium from inference_logs if it exists; otherwise fall back
# to current_premium on the UPT. The bias signal is qualitatively the same —
# we only need a per-policy premium scalar to anchor the postcode quintile.
try:
    spark.table(f"{fqn}.inference_logs").limit(1).count()
    _premium_join = (
        f"LEFT JOIN {fqn}.inference_logs i ON i.policy_id = d.policy_id"
    )
    _premium_col = "coalesce(i.technical_premium, u.current_premium, 1000)"
except Exception:
    print(f"⚠ inference_logs missing — falling back to current_premium from UPT")
    _premium_join = ""
    _premium_col = "coalesce(u.current_premium, 1000)"

dem = spark.sql(f"""
    WITH with_pred AS (
      SELECT d.policy_id, d.director_gender,
             {_premium_col} AS tp,
             (abs(hash(d.policy_id)) % 1000000) / 1000000.0 AS jitter
      FROM _v_with_gender d
      JOIN {fqn}.unified_pricing_table_live u ON u.policy_id = d.policy_id
      {_premium_join}
    ),
    ranked AS (
      SELECT *, ntile(5) OVER (ORDER BY tp + (jitter - 0.5) * 3200 ASC) AS q
      FROM with_pred
    )
    SELECT policy_id,
           director_gender,
           CASE q
             WHEN 1 THEN 'Q1_majority_white'
             WHEN 2 THEN 'Q2'
             WHEN 3 THEN 'Q3'
             WHEN 4 THEN 'Q4'
             WHEN 5 THEN 'Q5_most_diverse'
           END AS postcode_demographic
    FROM ranked
""")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.policy_demographics (
        policy_id            STRING NOT NULL,
        director_gender      STRING,
        postcode_demographic STRING,
        ethnicity_proxy      STRING,
        director_age_band    STRING
    ) USING DELTA
""")

# Derive ethnicity_proxy from the postcode quintile (correlated but noisy
# enough to look real) and director_age_band from a deterministic hash so
# the demo is reproducible and the bias monitor can show 4 protected attrs.
dem.createOrReplaceTempView("_v_dem")
dem = spark.sql("""
    SELECT
      policy_id,
      director_gender,
      postcode_demographic,
      CASE postcode_demographic
        WHEN 'Q1_majority_white' THEN element_at(array('White British','White British','White British','White British','Asian','Mixed/Other'),                                  1 + cast(abs(hash(policy_id || 'eth')) % 6 as int))
        WHEN 'Q2'                THEN element_at(array('White British','White British','White British','Asian','Asian','Black','Mixed/Other'),                                  1 + cast(abs(hash(policy_id || 'eth')) % 7 as int))
        WHEN 'Q3'                THEN element_at(array('White British','White British','Asian','Asian','Black','Mixed/Other'),                                                   1 + cast(abs(hash(policy_id || 'eth')) % 6 as int))
        WHEN 'Q4'                THEN element_at(array('White British','Asian','Asian','Black','Black','Mixed/Other'),                                                           1 + cast(abs(hash(policy_id || 'eth')) % 6 as int))
        WHEN 'Q5_most_diverse'   THEN element_at(array('White British','Asian','Asian','Black','Black','Black','Mixed/Other','Mixed/Other'),                                     1 + cast(abs(hash(policy_id || 'eth')) % 8 as int))
        ELSE 'White British'
      END AS ethnicity_proxy,
      element_at(array('Under 30','30-39','40-49','50-59','60+'),
                 1 + cast(abs(hash(policy_id || 'age')) % 5 as int)) AS director_age_band
    FROM _v_dem
""")
dem.write.mode("overwrite").option("overwriteSchema", "true") \
   .saveAsTable(f"{fqn}.policy_demographics")

n   = spark.table(f"{fqn}.policy_demographics").count()
dg  = spark.sql(f"""
    SELECT director_gender, count(*) AS n,
           round(count(*) * 100.0 / sum(count(*)) OVER (), 1) AS pct
    FROM {fqn}.policy_demographics
    GROUP BY director_gender ORDER BY n DESC
""").toPandas()
pd_ = spark.sql(f"""
    SELECT postcode_demographic, count(*) AS n,
           round(count(*) * 100.0 / sum(count(*)) OVER (), 1) AS pct
    FROM {fqn}.policy_demographics
    GROUP BY postcode_demographic ORDER BY postcode_demographic
""").toPandas()

print(f"policy_demographics rows: {n:,}")
print("\ndirector_gender distribution:")
print(dg.to_string(index=False))
print("\npostcode_demographic distribution:")
print(pd_.to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick sanity check — does the bias show up in inference_logs?

# COMMAND ----------

try:
    gap_gender = spark.sql(f"""
        SELECT d.director_gender,
               count(*) AS policies,
               round(avg(i.technical_premium), 2) AS avg_tp,
               round(avg(i.freq_pred),   4) AS avg_freq,
               round(avg(i.sev_pred),    2) AS avg_sev,
               round(avg(i.fraud_pred),  4) AS avg_fraud,
               round(avg(i.demand_pred), 4) AS avg_demand
        FROM {fqn}.policy_demographics d
        JOIN {fqn}.inference_logs     i ON i.policy_id = d.policy_id
        GROUP BY d.director_gender
        ORDER BY avg_tp DESC
    """).toPandas()
    print("Premium by director_gender:")
    print(gap_gender.to_string(index=False))

    gap_post = spark.sql(f"""
        SELECT d.postcode_demographic,
               count(*) AS policies,
               round(avg(i.technical_premium), 2) AS avg_tp,
               round(avg(i.fraud_loading),     2) AS avg_fraud_loading
        FROM {fqn}.policy_demographics d
        JOIN {fqn}.inference_logs     i ON i.policy_id = d.policy_id
        GROUP BY d.postcode_demographic
        ORDER BY postcode_demographic
    """).toPandas()
    print("\nPremium by postcode_demographic:")
    print(gap_post.to_string(index=False))
except Exception as e:
    print(f"(bias check skipped — inference_logs may be empty: {e})")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "row_count":            int(n),
    "gender_distribution":  dg.astype(str).to_dict(orient="records"),
    "postcode_distribution": pd_.astype(str).to_dict(orient="records"),
}))
