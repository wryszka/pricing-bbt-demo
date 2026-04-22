# Databricks notebook source
# MAGIC %md
# MAGIC # Gold: Build the Unified Pricing Table (UPT)
# MAGIC
# MAGIC Merges all internal and external silver-layer data into a single wide
# MAGIC denormalized table for pricing model training.
# MAGIC
# MAGIC **Sources:**
# MAGIC - Internal: `internal_commercial_policies`, `internal_claims_history`, `internal_quote_history`
# MAGIC - External (silver): `silver_market_pricing_benchmark`, `silver_geospatial_hazard_enrichment`, `silver_credit_bureau_summary`
# MAGIC
# MAGIC **Output:** `unified_pricing_table_live` — the Unified Pricing Table

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Load all source tables

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, round as spark_round, expr, log, greatest, least

# Internal tables (already in Databricks)
policies = spark.table(f"{fqn}.internal_commercial_policies")
claims = spark.table(f"{fqn}.internal_claims_history")
quotes = spark.table(f"{fqn}.internal_quote_history")

# External silver tables
market = spark.table(f"{fqn}.silver_market_pricing_benchmark")
geo = spark.table(f"{fqn}.silver_geospatial_hazard_enrichment")
bureau = spark.table(f"{fqn}.silver_credit_bureau_summary")

print(f"Policies: {policies.count()}, Claims: {claims.count()}, Quotes: {quotes.count()}")
print(f"Market: {market.count()}, Geo: {geo.count()}, Bureau: {bureau.count()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Aggregate claims to policy level

# COMMAND ----------

claims_agg = (claims
    .groupBy("policy_id")
    .agg(
        F.count("claim_id").alias("claim_count_5y"),
        F.sum("incurred_amount").alias("total_incurred_5y"),
        F.sum("paid_amount").alias("total_paid_5y"),
        F.sum("reserve").alias("total_reserve_5y"),
        F.countDistinct("peril").alias("distinct_perils"),
        F.max("loss_date").alias("last_claim_date"),
        F.sum(when(col("status") == "Open", 1).otherwise(0)).alias("open_claims_count"),
        # Peril-level breakdowns
        F.sum(when(col("peril") == "Fire", col("incurred_amount")).otherwise(0)).alias("fire_incurred"),
        F.sum(when(col("peril") == "Flood", col("incurred_amount")).otherwise(0)).alias("flood_incurred"),
        F.sum(when(col("peril") == "Theft", col("incurred_amount")).otherwise(0)).alias("theft_incurred"),
        F.sum(when(col("peril") == "Liability", col("incurred_amount")).otherwise(0)).alias("liability_incurred"),
        F.sum(when(col("peril") == "Storm", col("incurred_amount")).otherwise(0)).alias("storm_incurred"),
        F.sum(when(col("peril") == "Subsidence", col("incurred_amount")).otherwise(0)).alias("subsidence_incurred"),
        F.sum(when(col("peril") == "Escape of Water", col("incurred_amount")).otherwise(0)).alias("water_incurred"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Aggregate quotes to policy level

# COMMAND ----------

quotes_agg = (quotes
    .filter(col("converted") == "Y")
    .filter(col("policy_id").isNotNull())
    .groupBy("policy_id")
    .agg(
        F.count("quote_id").alias("quote_count"),
        F.avg("quoted_premium").alias("avg_quoted_premium"),
        F.min("quoted_premium").alias("min_quoted_premium"),
        F.max("quoted_premium").alias("max_quoted_premium"),
        F.sum(when(col("competitor_quoted") == "Y", 1).otherwise(0)).alias("competitor_quote_count"),
        F.max("quote_date").alias("last_quote_date"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Map postcodes to regions for market join

# COMMAND ----------

# Simple postcode-to-region mapping for the synthetic data
postcode_region_map = {
    "EC1": "London", "EC2": "London", "SW1": "London", "SE1": "London",
    "W1": "London", "N1": "London", "E1": "London",
    "M1": "North West", "M2": "North West",
    "B1": "Midlands", "B2": "Midlands",
    "LS1": "Yorkshire", "LS2": "Yorkshire",
    "L1": "North West", "L2": "North West",
    "CF1": "Wales",
    "EH1": "Scotland",
    "G1": "Scotland",
    "BS1": "South West",
    "NG1": "Midlands",
}

# Build mapping as DataFrame
region_rows = [(k, v) for k, v in postcode_region_map.items()]
region_df = spark.createDataFrame(region_rows, ["postcode_prefix", "region"])

# Extract postcode prefix (letters before digits) from postcode_sector
policies_with_region = (policies
    .withColumn("postcode_prefix", F.regexp_extract("postcode_sector", r"^([A-Z]+\d)", 1))
    .join(region_df, "postcode_prefix", "left")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: SIC code to risk tier mapping

# COMMAND ----------

sic_risk_map = {
    "1011": "Medium",   # Food processing
    "2562": "Medium",   # Machining
    "4110": "Medium",   # Construction
    "4520": "Medium",   # Vehicle maintenance
    "4711": "Low",      # Retail (non-specialised)
    "5610": "Medium",   # Restaurants
    "6201": "Low",      # Computer programming
    "6311": "Low",      # Data processing
    "6499": "Low",      # Financial services
    "6820": "Low",      # Real estate
    "7022": "Low",      # Management consultancy
    "7112": "Low",      # Engineering activities
    "8010": "High",     # Security
    "8622": "Medium",   # Medical practice
    "9311": "Medium",   # Sports facilities
}

sic_rows = [(k, v) for k, v in sic_risk_map.items()]
sic_df = spark.createDataFrame(sic_rows, ["sic_code", "industry_risk_tier"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Join everything into the Unified Pricing Table

# COMMAND ----------

# Build the market join key
policies_enriched = (policies_with_region
    .withColumn("market_join_key", F.concat(col("sic_code"), lit("_"), col("region")))
)

upt =(policies_enriched
    # Claims aggregation
    .join(claims_agg, "policy_id", "left")
    # Quote aggregation
    .join(quotes_agg, "policy_id", "left")
    # Market intelligence (join on SIC + region)
    .join(
        market.select("match_key_sic_region", "market_median_rate", "competitor_a_min_premium",
                       "price_index_trend", "competitor_ratio"),
        col("market_join_key") == col("match_key_sic_region"),
        "left"
    )
    # Geospatial hazard (join on postcode)
    .join(
        geo.select("postcode_sector", "flood_zone_rating", "proximity_to_fire_station_km",
                    "crime_theft_index", "subsidence_risk", "composite_location_risk", "location_risk_tier"),
        policies_enriched.postcode_sector == geo.postcode_sector,
        "left"
    )
    .drop(geo.postcode_sector)
    # Credit bureau (join on policy_id)
    .join(
        bureau.select("policy_id", "credit_score", "ccj_count", "years_trading",
                       "director_changes", "credit_risk_tier", "business_stability_score"),
        "policy_id",
        "left"
    )
    # SIC risk tier
    .join(sic_df, "sic_code", "left")
    # Drop intermediate columns
    .drop("postcode_prefix", "market_join_key", "match_key_sic_region")
)

# Derived factors (urban_score, neighbourhood_claim_frequency) — postcode-level
# Joined here so downstream models automatically pick them up. If the table
# doesn't exist yet, skip silently so the initial UPT build still works.
try:
    derived = spark.table(f"{fqn}.derived_factors")
    upt = upt.join(
        derived.select("postcode_sector", "urban_score", "neighbourhood_claim_frequency"),
        "postcode_sector",
        "left",
    )
    print(f"✓ Joined derived_factors ({derived.count()} postcodes)")
except Exception as e:
    print(f"Note: derived_factors not available, skipping — run 03_gold/derive_factors.py to enable. ({e})")

print(f"UPT columns: {len(upt.columns)}")
upt.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Generate synthetic/derived features
# MAGIC These simulate the 200+ bureau and geo proxy columns that would exist in a real UPT.

# COMMAND ----------

import hashlib

def deterministic_hash(seed_col, feature_name, min_val, max_val):
    """Generate a deterministic numeric feature from a seed column and feature name."""
    return (F.abs(F.hash(F.concat(seed_col, lit(feature_name)))) % (max_val - min_val) + min_val)

def deterministic_hash_double(seed_col, feature_name, min_val, max_val):
    """Generate a deterministic double feature."""
    return spark_round(
        (F.abs(F.hash(F.concat(seed_col, lit(feature_name)))) % ((max_val - min_val) * 100)) / 100.0 + min_val,
        2
    )

# Synthetic Bureau Features (credit/financial proxies)
bureau_features = {
    "credit_default_probability": (0.01, 0.35),
    "director_stability_score": (10, 100),
    "payment_history_score": (200, 900),
    "trade_credit_utilisation_pct": (0, 100),
    "debt_to_equity_ratio": (0.1, 5.0),
    "working_capital_ratio": (0.5, 3.5),
    "revenue_growth_3y_pct": (-30, 80),
    "employee_count_est": (1, 500),
    "industry_default_rate_pct": (0.5, 12.0),
    "supplier_concentration_score": (10, 100),
    "invoice_dispute_rate_pct": (0, 15),
    "bank_account_stability_months": (6, 240),
    "registered_charges_count": (0, 10),
    "profit_margin_est_pct": (-10, 40),
    "asset_tangibility_ratio": (0.1, 0.95),
    "interest_coverage_ratio": (0.5, 15.0),
    "accounts_filed_on_time": (0, 1),
    "company_age_months": (1, 600),
    "sector_bankruptcy_rate_pct": (0.5, 8.0),
    "management_experience_score": (10, 100),
}

# Synthetic Geo Features (location/environmental proxies)
geo_features = {
    "distance_to_coast_km": (0.1, 200.0),
    "local_unemployment_rate_pct": (2.0, 12.0),
    "traffic_density_index": (10, 100),
    "air_quality_index": (1, 10),
    "average_property_value_k": (80, 1200),
    "population_density_per_km2": (50, 15000),
    "commercial_density_score": (1, 100),
    "historic_flood_events_10y": (0, 15),
    "elevation_metres": (0, 500),
    "distance_to_hospital_km": (0.5, 40.0),
    "distance_to_motorway_km": (0.1, 50.0),
    "green_space_pct": (2, 60),
    "noise_pollution_index": (20, 90),
    "broadband_speed_mbps": (5, 1000),
    "listed_building_density": (0, 50),
    "average_wind_speed_mph": (5, 30),
    "annual_rainfall_mm": (400, 1500),
    "soil_clay_content_pct": (5, 70),
    "radon_risk_level": (1, 5),
    "tree_cover_pct": (1, 45),
}

# Apply synthetic features
for feat_name, (min_v, max_v) in bureau_features.items():
    if isinstance(min_v, float) or isinstance(max_v, float):
        upt =upt.withColumn(feat_name, deterministic_hash_double(col("policy_id"), feat_name, min_v, max_v))
    else:
        upt =upt.withColumn(feat_name, deterministic_hash(col("policy_id"), feat_name, min_v, max_v))

for feat_name, (min_v, max_v) in geo_features.items():
    if isinstance(min_v, float) or isinstance(max_v, float):
        upt =upt.withColumn(feat_name, deterministic_hash_double(col("postcode_sector"), feat_name, min_v, max_v))
    else:
        upt =upt.withColumn(feat_name, deterministic_hash(col("postcode_sector"), feat_name, min_v, max_v))

print(f"UPT columns after synthetic expansion: {len(upt.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 8: Add derived pricing features

# COMMAND ----------

upt = (upt
    # Loss ratio proxy
    .withColumn("loss_ratio_5y",
        spark_round(
            when(col("current_premium") > 0,
                 col("total_incurred_5y") / (col("current_premium") * 5))
            .otherwise(None), 3))
    # Premium per £1k sum insured
    .withColumn("rate_per_1k_si",
        spark_round(
            when(col("sum_insured") > 0,
                 col("current_premium") / (col("sum_insured") / 1000))
            .otherwise(None), 2))
    # Market position (our rate vs market median)
    .withColumn("market_position_ratio",
        spark_round(
            when(col("market_median_rate").isNotNull() & (col("market_median_rate") > 0),
                 col("rate_per_1k_si") / col("market_median_rate"))
            .otherwise(None), 3))
    # Building age
    .withColumn("building_age_years", lit(2026) - col("year_built"))
    # Combined risk score (weighted blend of location, credit, industry)
    .withColumn("combined_risk_score",
        spark_round(
            (F.coalesce(col("composite_location_risk"), lit(5.0)) * 0.35) +
            (F.coalesce(lit(10) - (col("credit_score") - 200) / 70.0, lit(5.0)) * 0.30) +
            (when(col("industry_risk_tier") == "High", 8.0)
             .when(col("industry_risk_tier") == "Medium", 5.0)
             .otherwise(2.5) * 0.20) +
            (when(col("claim_count_5y") > 3, 8.0)
             .when(col("claim_count_5y") > 1, 5.0)
             .when(col("claim_count_5y") > 0, 3.0)
             .otherwise(1.0) * 0.15),
            2))
    # Audit metadata
    .withColumn("last_updated_by", lit("system_upt_builder"))
    .withColumn("approval_timestamp", F.current_timestamp())
    .withColumn("source_version", lit("v1.0"))
    .withColumn("upt_build_timestamp", F.current_timestamp())
)

print(f"Final UPT columns: {len(upt.columns)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 9: Write the Unified Pricing Table

# COMMAND ----------

table_name = f"{fqn}.unified_pricing_table_live"
upt.write.mode("overwrite").saveAsTable(table_name)

row_count = spark.table(table_name).count()
col_count = len(spark.table(table_name).columns)
print(f"✓ {table_name} — {row_count:,} rows × {col_count} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 10: Register as UC Feature Table
# MAGIC Adding a PRIMARY KEY constraint makes this table automatically visible in
# MAGIC the Unity Catalog Features UI. Tags and column comments aid discovery.

# COMMAND ----------

# Primary key — makes UPT a feature table in UC
# NOT NULL is required before PK can be added (overwrite recreates nullable columns)
try:
    spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN policy_id SET NOT NULL")
except Exception:
    pass  # Already NOT NULL

try:
    spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT upt_pk PRIMARY KEY (policy_id)")
    print("✓ PRIMARY KEY constraint added (policy_id)")
except Exception as e:
    if "CONSTRAINT_ALREADY_EXISTS" in str(e) or "already exists" in str(e).lower():
        print("✓ PRIMARY KEY constraint already exists")
    else:
        print(f"⚠ Could not add PK constraint: {e}")

# COMMAND ----------

# Table-level tags — visible in Catalog Explorer and Features UI
tags = {
    "business_line": "commercial_property",
    "pricing_domain": "commercial_pricing",
    "table_owner": "actuarial_pricing_team",
    "refresh_cadence": "on_demand",
    "demo_environment": "true",
    "contains_pii": "false",
}
tag_sql = ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())
spark.sql(f"ALTER TABLE {table_name} SET TAGS ({tag_sql})")
print(f"✓ Table tags set: {list(tags.keys())}")

# COMMAND ----------

# Column-level comments — show up in Features UI for discoverability
column_comments = {
    # Primary key
    "policy_id": "Unique policy identifier — primary key",
    # Core policy fields
    "sic_code": "Standard Industrial Classification code (4-digit)",
    "postcode_sector": "UK postcode sector for the insured premises",
    "annual_turnover": "Declared gross revenue of the business (GBP)",
    "sum_insured": "Total sum insured under the policy (GBP)",
    "current_premium": "Current annual premium charged (GBP)",
    "construction_type": "ISO construction class of the primary premises",
    "building_age_years": "Age of the primary building in years",
    "renewal_date": "Next renewal date for the policy",
    # Claims features
    "claim_count_5y": "Total number of claims in the last 5 years",
    "total_incurred_5y": "Total incurred claim amount over 5 years (GBP)",
    "loss_ratio_5y": "5-year loss ratio (incurred / premium)",
    # Market features
    "market_median_rate": "Market median premium rate per £1k sum insured",
    "market_position_ratio": "Our rate vs market median (>1 = more expensive)",
    "price_index_trend": "Quarterly market price trend (%)",
    # Location risk
    "flood_zone_rating": "Flood risk score (1=low, 10=high)",
    "crime_theft_index": "Local area crime and theft index",
    "subsidence_risk": "Ground subsidence risk score (0-10)",
    "composite_location_risk": "Weighted composite location risk (flood+fire+crime+subsidence)",
    "location_risk_tier": "Location risk classification: High/Medium/Low",
    # Credit
    "credit_score": "Company credit score (200-900)",
    "credit_risk_tier": "Credit classification: Prime/Standard/Sub-Standard/High Risk",
    "business_stability_score": "Composite business stability score (0-100)",
    # Derived
    "combined_risk_score": "Blended risk score combining location, credit, industry, claims",
    "rate_per_1k_si": "Current premium rate per £1,000 sum insured",
    "industry_risk_tier": "Industry risk classification: High/Medium/Low",
    "urban_score": "Derived factor (0-1): weighted composite of population density, ONS urban class, GP density",
    "neighbourhood_claim_frequency": "Derived factor: credibility-weighted postcode-level claim frequency (Buhlmann K=100)",
    # Audit
    "last_updated_by": "User or system that last modified this row",
    "approval_timestamp": "Timestamp of the last approved data merge",
    "upt_build_timestamp": "Timestamp when this version of the UPT was built",
}

for col_name, comment in column_comments.items():
    try:
        escaped_comment = comment.replace("'", "\\'")
        spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN {col_name} COMMENT '{escaped_comment}'")
    except Exception:
        pass  # Column may not exist or comment may already be set

print(f"✓ Column comments set for {len(column_comments)} columns")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 11: Create initial monthly snapshot

# COMMAND ----------

from datetime import datetime

snapshot_suffix = datetime.now().strftime("%Y_%m")
snapshot_table = f"{fqn}.unified_pricing_table_{snapshot_suffix}"

spark.sql(f"CREATE TABLE IF NOT EXISTS {snapshot_table} DEEP CLONE {table_name}")
print(f"✓ Snapshot: {snapshot_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

display(spark.sql(f"""
    SELECT
        count(*) as total_rows,
        count(DISTINCT policy_id) as unique_policies,
        count(DISTINCT sic_code) as unique_sic_codes,
        count(DISTINCT postcode_sector) as unique_postcodes,
        avg(current_premium) as avg_premium,
        avg(loss_ratio_5y) as avg_loss_ratio,
        avg(combined_risk_score) as avg_risk_score
    FROM {table_name}
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Delta History
# MAGIC Shows the version chain for this table — used by Time Travel and audit.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {table_name} LIMIT 10"))
