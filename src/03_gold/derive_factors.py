# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Derived Factors
# MAGIC
# MAGIC Computes named, actuarially-meaningful factors from REAL UK public data.
# MAGIC Every weight is visible in this notebook — no black boxes.
# MAGIC
# MAGIC **Real data source:** `postcode_enrichment` (built by
# MAGIC `new_data_impact/00a_build_postcode_enrichment`) — the full ONS Postcode
# MAGIC Directory joined to the English Indices of Deprivation 2019 and the ONS
# MAGIC Rural-Urban Classification. ~1.5M English postcodes with real IMD / crime /
# MAGIC urban / coastal features.
# MAGIC
# MAGIC **Factors produced** (keyed on `postcode_sector` — matches the policy key
# MAGIC `internal_commercial_policies.postcode_sector`):
# MAGIC
# MAGIC 1. **`urban_score`** — real ONS urban-rural density + IMD composite.
# MAGIC 2. **`is_coastal`** — real coastal flag derived from ONS local authority codes.
# MAGIC 3. **`deprivation_composite`** — weighted blend of the IMD decile sub-domains.
# MAGIC 4. **`neighbourhood_claim_frequency`** — credibility-weighted postcode claim
# MAGIC    frequency (Bühlmann, K=100) — claims-driven, smoothed toward the book mean
# MAGIC    for thin exposure.
# MAGIC
# MAGIC **Join strategy:** postcode_enrichment is aggregated to the "postcode area"
# MAGIC level (the letters+first digits, e.g. `EC1`, `SW1`) so it matches the synthetic
# MAGIC `postcode_sector` keys on the policy book. Area-level means smooth out the real
# MAGIC postcode-to-postcode variance but preserve the between-area signal that the
# MAGIC pricing model cares about.
# MAGIC
# MAGIC **Output table:** `derived_factors` — one row per `postcode_sector`.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, when, round as spark_round

# Verify the real enrichment source exists — fail loudly if not
try:
    enrichment = spark.table(f"{fqn}.postcode_enrichment")
    n_enrichment = enrichment.count()
    print(f"✓ postcode_enrichment: {n_enrichment:,} real UK postcodes")
except Exception as e:
    raise RuntimeError(
        "postcode_enrichment table not found. "
        "Run `databricks bundle run build_postcode_enrichment` "
        "(or execute src/new_data_impact/00a_build_postcode_enrichment) first."
    ) from e

policies = spark.table(f"{fqn}.internal_commercial_policies")
claims   = spark.table(f"{fqn}.internal_claims_history")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Aggregate real enrichment to postcode-area level
# MAGIC
# MAGIC pricing-workbench's `postcode_sector` keys are strings like `EC1A`, `SW1A`, `M1A`.
# MAGIC Only some of these are real UK outcodes. Aggregating real data to the "area" level
# MAGIC (letters + first digits, e.g. `EC1`) gives us a robust join key that matches all
# MAGIC synthetic sectors.

# COMMAND ----------

# Postcode AREA: leading letters + first digit block
#   "EC1A 1AA" → "EC1"
#   "SW1A 1AA" → "SW1"
#   "M1 1AA"   → "M1"
AREA_RE = r"^([A-Z]{1,2}\d+)"

area_enrichment = (
    enrichment
    .filter(col("imd_decile").isNotNull())
    .withColumn("area", F.regexp_extract("postcode", AREA_RE, 1))
    .filter(col("area") != "")
    .groupBy("area")
    .agg(
        F.count("*").alias("postcodes_in_area"),
        F.avg("is_urban").alias("frac_urban"),
        F.max("is_coastal").cast("int").alias("is_coastal"),
        F.avg("imd_decile").alias("imd_decile"),
        F.avg("crime_decile").alias("crime_decile"),
        F.avg("income_decile").alias("income_decile"),
        F.avg("health_decile").alias("health_decile"),
        F.avg("living_env_decile").alias("living_env_decile"),
    )
)

print(f"✓ real enrichment aggregated to {area_enrichment.count()} postcode areas")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Factor: urban_score
# MAGIC
# MAGIC A weighted composite (0–1) of three real UK signals:
# MAGIC
# MAGIC | Weight | Input                              | Rationale                                              |
# MAGIC |-------:|------------------------------------|--------------------------------------------------------|
# MAGIC | 0.60   | `frac_urban` (ONS RUC 2011)        | Primary structural signal — fraction of area classified as urban |
# MAGIC | 0.40   | IMD living-environment decile      | Dense living environments correlate with urbanity       |
# MAGIC
# MAGIC IMD decile runs 1–10 where 1 = most deprived. We invert and normalise so that
# MAGIC dense/deprived environments (low decile) contribute positively to urban_score.

# COMMAND ----------

W_URBAN_FRAC    = 0.60
W_LIVING_ENV    = 0.40

# Build an outline of every policy sector (so every policy gets a row, even if the
# real area has no IMD coverage — we'll fill those with area-mean fallbacks later).
sectors = (
    policies.select("postcode_sector").distinct()
    .withColumn("area", F.regexp_extract("postcode_sector", AREA_RE, 1))
)

# Left-join area-level real enrichment onto sectors
sector_enrichment = (
    sectors
    .join(area_enrichment, "area", "left")
)

# Book-wide means (fallback for sectors whose area has no real data)
book_means = area_enrichment.agg(
    F.avg("frac_urban").alias("mean_frac_urban"),
    F.avg("living_env_decile").alias("mean_living_env"),
    F.avg("imd_decile").alias("mean_imd"),
    F.avg("crime_decile").alias("mean_crime"),
    F.avg("income_decile").alias("mean_income"),
    F.avg("health_decile").alias("mean_health"),
).first()

# Fill nulls with the book-wide average so every sector has usable values
sector_filled = (
    sector_enrichment
    .withColumn("frac_urban",        F.coalesce(col("frac_urban"),        lit(float(book_means["mean_frac_urban"]))))
    .withColumn("living_env_decile", F.coalesce(col("living_env_decile"), lit(float(book_means["mean_living_env"]))))
    .withColumn("imd_decile",        F.coalesce(col("imd_decile"),        lit(float(book_means["mean_imd"]))))
    .withColumn("crime_decile",      F.coalesce(col("crime_decile"),      lit(float(book_means["mean_crime"]))))
    .withColumn("income_decile",     F.coalesce(col("income_decile"),     lit(float(book_means["mean_income"]))))
    .withColumn("health_decile",     F.coalesce(col("health_decile"),     lit(float(book_means["mean_health"]))))
    .withColumn("is_coastal",        F.coalesce(col("is_coastal"),        lit(0)))
)

# urban_score: invert living_env_decile (lower = denser = more urban)
urban_scored = (
    sector_filled
    .withColumn(
        "urban_score",
        spark_round(
            lit(W_URBAN_FRAC)  * col("frac_urban") +
            lit(W_LIVING_ENV)  * ((lit(10.0) - col("living_env_decile")) / lit(9.0)),
            4,
        ),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Factor: deprivation_composite
# MAGIC
# MAGIC Equal-weighted average of the four IMD sub-domains (crime, income, health,
# MAGIC living environment), each normalised to [0, 1] with 1 = most deprived. Keeps
# MAGIC every sub-domain visible in the code for regulatory questions.

# COMMAND ----------

deprivation_scored = (
    urban_scored
    .withColumn(
        "deprivation_composite",
        spark_round(
            (
                (lit(10.0) - col("crime_decile")) / lit(9.0)
              + (lit(10.0) - col("income_decile")) / lit(9.0)
              + (lit(10.0) - col("health_decile")) / lit(9.0)
              + (lit(10.0) - col("living_env_decile")) / lit(9.0)
            ) / lit(4.0),
            4,
        ),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Factor: neighbourhood_claim_frequency (credibility-weighted)
# MAGIC
# MAGIC Bühlmann-style credibility. `Z = n / (n + K)`. Thin-exposure postcodes are
# MAGIC pulled toward the overall book mean.

# COMMAND ----------

K_CREDIBILITY = 100

claims_per_policy = (
    claims
    .groupBy("policy_id")
    .agg(F.count("claim_id").alias("n_claims"))
)
policy_claims = (
    policies.select("policy_id", "postcode_sector")
    .join(claims_per_policy, "policy_id", "left")
    .withColumn("n_claims", F.coalesce(col("n_claims"), lit(0)))
)

overall_mean_freq = policy_claims.agg(F.avg("n_claims")).first()[0]
print(f"overall mean 5y claim frequency: {overall_mean_freq:.4f}")

claim_freq_df = (
    policy_claims
    .groupBy("postcode_sector")
    .agg(
        F.count("policy_id").alias("n_policies"),
        F.avg("n_claims").alias("raw_postcode_frequency"),
    )
    .withColumn("credibility_z", col("n_policies") / (col("n_policies") + lit(K_CREDIBILITY)))
    .withColumn(
        "neighbourhood_claim_frequency",
        spark_round(
            col("credibility_z") * col("raw_postcode_frequency")
            + (lit(1.0) - col("credibility_z")) * lit(overall_mean_freq),
            4,
        ),
    )
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
# MAGIC ## 5. Combine factors and write `derived_factors`

# COMMAND ----------

derived = (
    deprivation_scored
    .select(
        "postcode_sector",
        "urban_score",
        "is_coastal",
        "deprivation_composite",
        "frac_urban",
        "imd_decile",
        "crime_decile",
        "income_decile",
        "health_decile",
        "living_env_decile",
    )
    .join(claim_freq_df, "postcode_sector", "left")
    .withColumn("_derived_at",     F.current_timestamp())
    .withColumn("_source_version", lit("derive_factors_v2_real_uk_data"))
)

table_name = f"{fqn}.derived_factors"
derived.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

row_count = spark.table(table_name).count()
print(f"✓ {table_name} — {row_count:,} rows")

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {table_name} SET TBLPROPERTIES (
        'comment' = 'Derived pricing factors at postcode sector level. Sourced from real UK public data (postcode_enrichment, built from ONSPD + IMD 2019 + ONS RUC) plus internal claims. Every weight and aggregation is visible in src/03_gold/derive_factors.py.'
    )
""")

column_comments = {
    "postcode_sector":                "UK postcode sector (join key onto policies)",
    "urban_score":                    "Weighted composite (0-1): 0.60*frac_urban + 0.40*inverted living_env_decile",
    "is_coastal":                     "1 if the postcode area falls in a coastal English local authority, 0 otherwise",
    "deprivation_composite":          "Equal-weighted average of inverted crime/income/health/living-env deciles (0-1, 1=most deprived)",
    "frac_urban":                     "Fraction of postcodes in the area classified urban by ONS RUC 2011",
    "imd_decile":                     "Area-average IMD overall decile (1=most deprived, 10=least)",
    "crime_decile":                   "Area-average IMD crime sub-decile",
    "income_decile":                  "Area-average IMD income sub-decile",
    "health_decile":                  "Area-average IMD health sub-decile",
    "living_env_decile":              "Area-average IMD living-environment sub-decile",
    "neighbourhood_claim_frequency":  "Credibility-weighted postcode claim frequency (Buhlmann, K=100)",
    "raw_postcode_frequency":         "Raw postcode-level claim frequency before smoothing",
    "n_policies":                     "Number of policies in the postcode — credibility exposure weight",
    "credibility_z":                  "Credibility weight (0=all book-mean, 1=all postcode-specific)",
}
for col_name, comment in column_comments.items():
    try:
        escaped = comment.replace("'", "\\'")
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} COMMENT '{escaped}'")
    except Exception:
        pass

tags = {
    "factor_domain":      "real_uk_public_data",
    "factor_author":      "actuarial_pricing_team",
    "refresh_cadence":    "on_demand",
    "demo_environment":   "true",
}
tag_sql = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
spark.sql(f"ALTER TABLE {table_name} SET TAGS ({tag_sql})")
print(f"✓ tags + comments applied to {table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        count(*)                                      AS sectors,
        round(avg(urban_score), 3)                    AS avg_urban_score,
        round(avg(deprivation_composite), 3)          AS avg_deprivation,
        sum(is_coastal)                               AS coastal_sectors,
        round(avg(neighbourhood_claim_frequency), 4)  AS avg_neigh_freq
    FROM {table_name}
"""))
