# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Feature Catalog
# MAGIC
# MAGIC Writes `feature_catalog` — one row per feature in the **training feature store**
# MAGIC (`unified_pricing_table_live`), with full provenance metadata:
# MAGIC
# MAGIC - **`feature_group`** — classification (rating_factor, enrichment, claim_derived, synthetic, derived, audit)
# MAGIC - **`source_tables`** / **`source_columns`** — upstream lineage
# MAGIC - **`transformation`** — plain-English description of the derivation
# MAGIC - **`owner`**, **`regulatory_sensitive`**, **`pii`** — governance flags
# MAGIC
# MAGIC This is the foundation for the Feature Catalog panel in the app and for future
# MAGIC bolt-ons around feature-level lineage and audit (e.g. "if the regulator bans
# MAGIC `crime_decile`, which models are affected?").
# MAGIC
# MAGIC **Output:** `feature_catalog` table in the demo schema.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

from datetime import datetime, timezone
import pyspark.sql.functions as F
from pyspark.sql.types import (
    ArrayType, BooleanType, StringType, StructField, StructType, TimestampType,
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Catalog definitions
# MAGIC
# MAGIC Keep these close to the `build_upt.py` logic — each feature shipped by the UPT
# MAGIC should be described here. Future bolt-ons query this table for lineage / audit.

# COMMAND ----------

# (feature_name, feature_group, data_type, description, source_tables, source_columns, transformation, owner, regulatory_sensitive, pii)
FEATURE_CATALOG = [
    # ---- Primary key + policy attributes ----
    ("policy_id",              "key",           "STRING",  "Primary key — one row per policy",                       ["internal_commercial_policies"], ["policy_id"], "identity", "actuarial_pricing_team", False, True),
    ("sic_code",               "rating_factor", "STRING",  "UK SIC 2007 industry code (4-digit)",                    ["internal_commercial_policies"], ["sic_code"], "identity", "actuarial_pricing_team", False, False),
    ("postcode_sector",        "rating_factor", "STRING",  "UK postcode sector — primary location key",              ["internal_commercial_policies"], ["postcode_sector"], "identity", "actuarial_pricing_team", False, False),
    ("region",                 "rating_factor", "STRING",  "Region derived from postcode prefix",                    ["internal_commercial_policies"], ["postcode_sector"], "postcode prefix → region lookup", "actuarial_pricing_team", False, False),
    ("annual_turnover",        "rating_factor", "BIGINT",  "Declared annual revenue (GBP)",                          ["internal_commercial_policies"], ["annual_turnover"], "identity", "actuarial_pricing_team", False, False),
    ("sum_insured",            "rating_factor", "BIGINT",  "Total sum insured (GBP)",                                ["internal_commercial_policies"], ["sum_insured"], "identity", "actuarial_pricing_team", False, False),
    ("construction_type",      "rating_factor", "STRING",  "ISO construction class",                                 ["internal_commercial_policies"], ["construction_type"], "identity", "actuarial_pricing_team", False, False),
    ("year_built",             "rating_factor", "INT",     "Year the primary premises was built",                    ["internal_commercial_policies"], ["year_built"], "identity", "actuarial_pricing_team", False, False),
    ("building_age_years",     "rating_factor", "INT",     "Age of the primary premises in years",                   ["internal_commercial_policies"], ["year_built"], "2026 - year_built", "actuarial_pricing_team", False, False),
    ("industry_risk_tier",     "rating_factor", "STRING",  "Industry risk classification derived from SIC code",     ["internal_commercial_policies"], ["sic_code"], "SIC → risk tier lookup", "actuarial_pricing_team", False, False),
    ("current_premium",        "rating_factor", "DOUBLE",  "Current annual premium on the policy (GBP)",             ["internal_commercial_policies"], ["current_premium"], "identity", "actuarial_pricing_team", False, False),

    # ---- Claim-derived features ----
    ("claim_count_5y",         "claim_derived", "BIGINT",  "Total number of claims in the last 5 years",             ["internal_claims_history"], ["policy_id"], "COUNT of claims", "actuarial_pricing_team", False, False),
    ("total_incurred_5y",      "claim_derived", "DOUBLE",  "Total incurred amount over 5 years (GBP)",               ["internal_claims_history"], ["incurred_amount"], "SUM of incurred_amount", "actuarial_pricing_team", False, False),
    ("total_paid_5y",          "claim_derived", "DOUBLE",  "Total paid amount over 5 years (GBP)",                   ["internal_claims_history"], ["paid_amount"], "SUM of paid_amount", "actuarial_pricing_team", False, False),
    ("loss_ratio_5y",          "derived",       "DOUBLE",  "5-year loss ratio (incurred / premium × 5)",             ["internal_claims_history", "internal_commercial_policies"], ["incurred_amount", "current_premium"], "total_incurred_5y / (current_premium * 5)", "actuarial_pricing_team", False, False),
    ("open_claims_count",      "claim_derived", "BIGINT",  "Open claims on the policy",                              ["internal_claims_history"], ["status"], "COUNT where status = Open", "actuarial_pricing_team", False, False),

    # ---- Peril breakdowns (derived from claims) ----
    ("fire_incurred",          "claim_derived", "DOUBLE",  "Incurred amount from fire claims (GBP)",                 ["internal_claims_history"], ["peril", "incurred_amount"], "SUM where peril = Fire", "actuarial_pricing_team", False, False),
    ("flood_incurred",         "claim_derived", "DOUBLE",  "Incurred amount from flood claims (GBP)",                ["internal_claims_history"], ["peril", "incurred_amount"], "SUM where peril = Flood", "actuarial_pricing_team", False, False),
    ("theft_incurred",         "claim_derived", "DOUBLE",  "Incurred amount from theft claims (GBP)",                ["internal_claims_history"], ["peril", "incurred_amount"], "SUM where peril = Theft", "actuarial_pricing_team", False, False),

    # ---- Quote-derived features ----
    ("quote_count",            "quote_derived", "BIGINT",  "Number of quotes on this policy",                        ["quotes"], ["policy_id"], "COUNT where policy_id = matched", "actuarial_pricing_team", False, False),
    ("avg_quoted_premium",     "quote_derived", "DOUBLE",  "Average quoted premium across quotes on this policy",    ["quotes"], ["gross_premium"], "AVG gross_premium", "actuarial_pricing_team", False, False),
    ("competitor_quote_count", "quote_derived", "BIGINT",  "Quotes where competitor was known to have quoted",       ["quotes"], ["competitor_quoted"], "COUNT where competitor_quoted = Y", "actuarial_pricing_team", False, False),

    # ---- Market enrichment ----
    ("market_median_rate",     "enrichment",    "DOUBLE",  "Market median premium per £1k SI (PCW)",                 ["silver_market_pricing_benchmark"], ["market_median_rate"], "external vendor benchmark", "pricing_analytics", False, False),
    ("competitor_a_min_premium", "enrichment",  "DOUBLE",  "Lowest observed competitor A premium",                   ["silver_market_pricing_benchmark"], ["competitor_a_min_premium"], "external vendor benchmark", "pricing_analytics", False, False),
    ("price_index_trend",      "enrichment",    "DOUBLE",  "Quarterly market price trend (%)",                       ["silver_market_pricing_benchmark"], ["price_index_trend"], "external vendor benchmark", "pricing_analytics", False, False),
    ("market_position_ratio",  "derived",       "DOUBLE",  "Our rate / market median",                               ["internal_commercial_policies", "silver_market_pricing_benchmark"], ["current_premium", "sum_insured", "market_median_rate"], "(current_premium / (sum_insured/1000)) / market_median_rate", "actuarial_pricing_team", False, False),

    # ---- Geospatial hazard enrichment (synthetic vendor) ----
    ("flood_zone_rating",      "enrichment",    "INT",     "Flood risk score (1=low, 10=high)",                      ["silver_geospatial_hazard_enrichment"], ["flood_zone_rating"], "vendor score", "pricing_analytics", False, False),
    ("proximity_to_fire_station_km", "enrichment", "DOUBLE", "Distance to nearest fire station (km)",                ["silver_geospatial_hazard_enrichment"], ["proximity_to_fire_station_km"], "vendor score", "pricing_analytics", False, False),
    ("crime_theft_index",      "enrichment",    "DOUBLE",  "Local area crime / theft index (synthetic vendor)",      ["silver_geospatial_hazard_enrichment"], ["crime_theft_index"], "vendor score", "pricing_analytics", True, False),
    ("subsidence_risk",        "enrichment",    "DOUBLE",  "Ground subsidence risk (0-10)",                          ["silver_geospatial_hazard_enrichment"], ["subsidence_risk"], "vendor score", "pricing_analytics", False, False),
    ("composite_location_risk","derived",       "DOUBLE",  "Weighted composite of flood + fire + crime + subsidence",["silver_geospatial_hazard_enrichment"], ["flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"], "0.3*flood + 0.2*fire_distance + 0.25*crime + 0.25*subsidence", "actuarial_pricing_team", False, False),
    ("location_risk_tier",     "derived",       "STRING",  "Location risk classification",                           ["silver_geospatial_hazard_enrichment"], ["flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"], "tier from composite_location_risk", "actuarial_pricing_team", False, False),

    # ---- Credit bureau enrichment (synthetic) ----
    ("credit_score",           "enrichment",    "INT",     "Company credit score (200-900)",                         ["silver_credit_bureau_summary"], ["credit_score"], "bureau feed", "credit_risk", True, True),
    ("ccj_count",              "enrichment",    "INT",     "County Court Judgements on file",                        ["silver_credit_bureau_summary"], ["ccj_count"], "bureau feed", "credit_risk", True, True),
    ("years_trading",          "enrichment",    "INT",     "Years the business has been trading",                    ["silver_credit_bureau_summary"], ["years_trading"], "bureau feed", "credit_risk", False, False),
    ("director_changes",       "enrichment",    "INT",     "Number of director changes in the last 5 years",         ["silver_credit_bureau_summary"], ["director_changes"], "bureau feed", "credit_risk", True, True),
    ("credit_risk_tier",       "derived",       "STRING",  "Credit classification: Prime/Standard/Sub-Standard/High",["silver_credit_bureau_summary"], ["credit_score", "ccj_count"], "tier from credit_score and ccj_count", "credit_risk", True, False),
    ("business_stability_score", "derived",     "INT",     "Composite business stability (0-100)",                   ["silver_credit_bureau_summary"], ["credit_score", "ccj_count", "years_trading", "director_changes"], "weighted composite", "credit_risk", True, False),

    # ---- Real UK enrichment (postcode_enrichment) ----
    ("urban_score",            "enrichment",    "DOUBLE",  "Real UK data: ONS RUC 2011 urban fraction + IMD living-env composite (0-1)", ["postcode_enrichment"], ["is_urban", "living_env_decile"], "0.60*frac_urban + 0.40*(10 - living_env_decile)/9", "actuarial_pricing_team", False, False),
    ("is_coastal",             "enrichment",    "INT",     "Real UK data: coastal postcode flag from ONS local authority codes", ["postcode_enrichment"], ["local_authority_code"], "lookup against coastal LA list", "actuarial_pricing_team", False, False),
    ("deprivation_composite",  "enrichment",    "DOUBLE",  "Real UK data: IMD 2019 crime+income+health+living-env composite (0-1, 1=most deprived)", ["postcode_enrichment"], ["crime_decile", "income_decile", "health_decile", "living_env_decile"], "mean of inverted IMD sub-deciles", "actuarial_pricing_team", True, False),
    ("imd_decile",             "enrichment",    "DOUBLE",  "Real UK data: IMD 2019 overall decile averaged to postcode area", ["postcode_enrichment"], ["imd_decile"], "area-level mean", "actuarial_pricing_team", True, False),
    ("crime_decile",           "enrichment",    "DOUBLE",  "Real UK data: IMD 2019 crime sub-decile averaged to postcode area", ["postcode_enrichment"], ["crime_decile"], "area-level mean", "actuarial_pricing_team", True, False),
    ("neighbourhood_claim_frequency", "derived","DOUBLE",  "Bühlmann credibility-weighted postcode claim frequency (K=100)", ["internal_claims_history", "internal_commercial_policies"], ["policy_id", "postcode_sector"], "Z*raw_freq + (1-Z)*book_mean", "actuarial_pricing_team", False, False),

    # ---- Key derived / audit ----
    ("combined_risk_score",    "derived",       "DOUBLE",  "Blended risk score across location, credit, industry, claims", ["silver_geospatial_hazard_enrichment", "silver_credit_bureau_summary", "internal_commercial_policies", "internal_claims_history"], ["composite_location_risk", "credit_score", "industry_risk_tier", "claim_count_5y"], "0.35*location + 0.30*credit + 0.20*industry + 0.15*claim_count", "actuarial_pricing_team", True, False),
    ("rate_per_1k_si",         "derived",       "DOUBLE",  "Current premium per £1,000 sum insured",                 ["internal_commercial_policies"], ["current_premium", "sum_insured"], "current_premium / (sum_insured/1000)", "actuarial_pricing_team", False, False),
    ("last_updated_by",        "audit",         "STRING",  "Who last updated the row",                               [], [], "system populated", "data_platform", False, False),
    ("approval_timestamp",     "audit",         "TIMESTAMP","Timestamp of the last approved merge",                  [], [], "current_timestamp()", "data_platform", False, False),
    ("upt_build_timestamp",    "audit",         "TIMESTAMP","Timestamp this UPT row was built",                      [], [], "current_timestamp()", "data_platform", False, False),
    ("source_version",         "audit",         "STRING",  "Source version stamp",                                   [], [], "string lit", "data_platform", False, False),
]

print(f"Catalog covers {len(FEATURE_CATALOG)} named features.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Write `feature_catalog`

# COMMAND ----------

now = datetime.now(timezone.utc).replace(tzinfo=None)
rows = [
    (name, group, dtype, desc, src_tbls, src_cols, trans, owner, reg, pii, now, now)
    for (name, group, dtype, desc, src_tbls, src_cols, trans, owner, reg, pii) in FEATURE_CATALOG
]

schema_ = StructType([
    StructField("feature_name",          StringType(),         False),
    StructField("feature_group",         StringType(),         False),
    StructField("data_type",             StringType()),
    StructField("description",           StringType()),
    StructField("source_tables",         ArrayType(StringType())),
    StructField("source_columns",        ArrayType(StringType())),
    StructField("transformation",        StringType()),
    StructField("owner",                 StringType()),
    StructField("regulatory_sensitive",  BooleanType()),
    StructField("pii",                   BooleanType()),
    StructField("added_at",              TimestampType()),
    StructField("last_modified",         TimestampType()),
])

df = spark.createDataFrame(rows, schema=schema_)

table_name = f"{fqn}.feature_catalog"
df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name)

print(f"✓ {table_name} — {df.count()} features")

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {table_name} SET TBLPROPERTIES (
        'comment' = 'Feature catalog for the training feature store (unified_pricing_table_live). One row per feature with source tables/columns, transformation, owner, regulatory flags. Foundation for feature lineage + audit bolt-ons.'
    )
""")
tags = {
    "pricing_domain":   "feature_governance",
    "refresh_cadence":    "manual",
    "demo_environment": "true",
}
tag_sql = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
spark.sql(f"ALTER TABLE {table_name} SET TAGS ({tag_sql})")

# COMMAND ----------

display(spark.sql(f"""
    SELECT feature_group, COUNT(*) AS n
    FROM {table_name}
    GROUP BY feature_group
    ORDER BY n DESC
"""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT feature_name, source_tables, owner, regulatory_sensitive
    FROM {table_name}
    WHERE regulatory_sensitive = true
    ORDER BY feature_name
"""))
