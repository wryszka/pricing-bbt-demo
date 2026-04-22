# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Derived Factors
# MAGIC
# MAGIC Computes named, actuarially-meaningful factors from raw external + internal data.
# MAGIC Each factor has transparent, auditable weights — actuaries can read and reason
# MAGIC about the code. No black boxes.
# MAGIC
# MAGIC **Factors produced** (keyed on `postcode_sector`):
# MAGIC
# MAGIC 1. **`urban_score`** — weighted composite of population density, ONS urban
# MAGIC    classification, and NHS GP density. Higher = more urban.
# MAGIC
# MAGIC 2. **`neighbourhood_claim_frequency`** — credibility-weighted postcode-level
# MAGIC    claim frequency. Represents the *area's* claim propensity, smoothed toward the
# MAGIC    overall mean for postcodes with thin exposure. Not the individual
# MAGIC    policyholder's own history — that's already in the policy-level claim features.
# MAGIC
# MAGIC **Output table:** `derived_factors` — one row per postcode sector, with lineage
# MAGIC back to `silver_ons_reference`, `internal_commercial_policies`, and
# MAGIC `internal_claims_history`. Automatically joined into the UPT by `build_upt.py`.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, when, round as spark_round

ons     = spark.table(f"{fqn}.silver_ons_reference")
policies = spark.table(f"{fqn}.internal_commercial_policies")
claims   = spark.table(f"{fqn}.internal_claims_history")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Factor 1: Urban Score
# MAGIC
# MAGIC A weighted composite (0–1) of three signals, normalized per feature using min/max
# MAGIC across the observed postcodes:
# MAGIC
# MAGIC | Weight | Input                         | Rationale                                          |
# MAGIC |-------:|-------------------------------|----------------------------------------------------|
# MAGIC | 0.50   | population_density_per_km2    | Primary structural signal for urbanity             |
# MAGIC | 0.30   | urban_classification_score    | ONS rural-urban categorical (1–6)                  |
# MAGIC | 0.20   | gp_density_per_10k            | Health-infrastructure proxy — urban areas denser    |
# MAGIC
# MAGIC Weights chosen to mirror typical actuarial composite scoring — the primary
# MAGIC structural variable gets most of the weight, secondary and tertiary signals
# MAGIC decay. Every weight is visible here for audit.

# COMMAND ----------

# Normalize each input to 0-1 using min/max
stats = ons.agg(
    F.min("population_density_per_km2").alias("pop_min"),
    F.max("population_density_per_km2").alias("pop_max"),
    F.min("gp_density_per_10k").alias("gp_min"),
    F.max("gp_density_per_10k").alias("gp_max"),
).first()

pop_min, pop_max = float(stats["pop_min"]), float(stats["pop_max"])
gp_min,  gp_max  = float(stats["gp_min"]),  float(stats["gp_max"])

# Weights — visible, editable, auditable
W_POP_DENSITY      = 0.50
W_URBAN_CLASS      = 0.30
W_GP_DENSITY       = 0.20

urban_score_df = (ons
    .withColumn("_pop_norm",
        (col("population_density_per_km2") - lit(pop_min)) / lit(pop_max - pop_min))
    .withColumn("_urban_class_norm",
        (col("urban_classification_score") - lit(1)) / lit(5))
    .withColumn("_gp_norm",
        (col("gp_density_per_10k") - lit(gp_min)) / lit(gp_max - gp_min))
    .withColumn("urban_score",
        spark_round(
            lit(W_POP_DENSITY)  * col("_pop_norm") +
            lit(W_URBAN_CLASS)  * col("_urban_class_norm") +
            lit(W_GP_DENSITY)   * col("_gp_norm"),
            4))
    .select("postcode_sector", "urban_score")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Factor 2: Neighbourhood Claim Frequency (credibility-weighted)
# MAGIC
# MAGIC Raw postcode-level claim frequency is noisy — a postcode with 5 policies and
# MAGIC 2 claims looks like a 40% frequency area, but that's almost certainly noise.
# MAGIC
# MAGIC Credibility weighting (Bühlmann-style) pulls thin-exposure postcodes toward
# MAGIC the overall book mean, proportional to how much exposure they actually have:
# MAGIC
# MAGIC ```
# MAGIC Z = n / (n + K)
# MAGIC neighbourhood_claim_frequency = Z * postcode_frequency + (1 - Z) * overall_mean
# MAGIC ```
# MAGIC
# MAGIC Where `n` = number of policies in the postcode, `K` = credibility constant
# MAGIC (tuning parameter — we use 100 here, meaning a postcode needs 100 policies to
# MAGIC get ~50% weight on its own experience).

# COMMAND ----------

K_CREDIBILITY = 100  # actuarial tuning parameter — editable, documented

# Aggregate claims to policy level first
claims_per_policy = (claims
    .groupBy("policy_id")
    .agg(F.count("claim_id").alias("n_claims"))
)

# Join to policies and aggregate to postcode level
policy_claims = (policies
    .select("policy_id", "postcode_sector")
    .join(claims_per_policy, "policy_id", "left")
    .withColumn("n_claims", F.coalesce(col("n_claims"), lit(0)))
)

# Overall mean claim frequency across the book
overall_mean = policy_claims.agg(F.avg("n_claims")).first()[0]
print(f"Overall mean claim frequency (5y): {overall_mean:.4f}")

# Postcode-level aggregation with credibility
claim_freq_df = (policy_claims
    .groupBy("postcode_sector")
    .agg(
        F.count("policy_id").alias("n_policies"),
        F.avg("n_claims").alias("raw_postcode_frequency"),
    )
    .withColumn("credibility_z",
        col("n_policies") / (col("n_policies") + lit(K_CREDIBILITY)))
    .withColumn("neighbourhood_claim_frequency",
        spark_round(
            col("credibility_z") * col("raw_postcode_frequency") +
            (lit(1.0) - col("credibility_z")) * lit(overall_mean),
            4))
    .select(
        "postcode_sector",
        "n_policies",
        "raw_postcode_frequency",
        "credibility_z",
        "neighbourhood_claim_frequency",
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Combine factors and write `derived_factors` table

# COMMAND ----------

derived = (urban_score_df
    .join(claim_freq_df, "postcode_sector", "left")
    .select(
        "postcode_sector",
        "urban_score",
        "neighbourhood_claim_frequency",
        "raw_postcode_frequency",
        "n_policies",
        "credibility_z",
    )
    .withColumn("_derived_at", F.current_timestamp())
    .withColumn("_source_version", lit("derive_factors_v1"))
)

table_name = f"{fqn}.derived_factors"
derived.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

row_count = spark.table(table_name).count()
print(f"✓ {table_name} — {row_count:,} rows")

# COMMAND ----------

# Table + column comments for discoverability in Catalog Explorer
spark.sql(f"""
    ALTER TABLE {table_name}
    SET TBLPROPERTIES (
        'comment' = 'Derived pricing factors at postcode sector level. Computed by src/03_gold/derive_factors.py from silver_ons_reference + internal claims/policies. See notebook for weights and methodology.'
    )
""")

column_comments = {
    "postcode_sector":                "UK postcode sector (key)",
    "urban_score":                    "Weighted composite (0-1) of pop density (0.50) + ONS urban class (0.30) + GP density (0.20)",
    "neighbourhood_claim_frequency":  "Credibility-weighted postcode claim frequency (Buhlmann, K=100) — smoothed toward book mean for thin exposure",
    "raw_postcode_frequency":         "Raw postcode-level claim frequency before credibility smoothing",
    "n_policies":                     "Number of policies in the postcode — used as credibility exposure weight",
    "credibility_z":                  "Credibility weight applied to raw frequency (0=all book-mean, 1=all postcode-specific)",
}
for col_name, comment in column_comments.items():
    try:
        escaped = comment.replace("'", "\\'")
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} COMMENT '{escaped}'")
    except Exception:
        pass

# Tags
tags = {
    "factor_domain":      "location_and_claims",
    "factor_author":      "actuarial_pricing_team",
    "refresh_cadence":    "on_demand",
    "demo_environment":   "true",
}
tag_sql = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
spark.sql(f"ALTER TABLE {table_name} SET TAGS ({tag_sql})")
print(f"✓ Tags + comments applied to {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary — factor distributions

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        count(*)                                                 AS postcodes,
        round(avg(urban_score), 3)                               AS avg_urban_score,
        round(min(urban_score), 3)                               AS min_urban_score,
        round(max(urban_score), 3)                               AS max_urban_score,
        round(avg(neighbourhood_claim_frequency), 4)             AS avg_neigh_freq,
        round(min(neighbourhood_claim_frequency), 4)             AS min_neigh_freq,
        round(max(neighbourhood_claim_frequency), 4)             AS max_neigh_freq,
        sum(n_policies)                                          AS total_policies_covered
    FROM {table_name}
"""))
