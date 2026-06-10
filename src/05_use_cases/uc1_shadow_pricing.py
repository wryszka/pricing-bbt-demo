# Databricks notebook source
# MAGIC %md
# MAGIC # UC1: Shadow Pricing — Actuarial Impact Analysis
# MAGIC
# MAGIC **The "Shadow Run":** When a new vendor dataset version arrives, instead of
# MAGIC just checking for nulls, we automatically join it to our live policy book,
# MAGIC re-rate every active policy, and calculate the exact financial impact.
# MAGIC
# MAGIC **Demo flow:**
# MAGIC 1. Simulate an updated Geospatial Hazard dataset arriving (Flood Risk v2)
# MAGIC 2. Join the new data to active policies
# MAGIC 3. Re-rate using a proxy pricing formula (Base × Industry × Flood × Crime)
# MAGIC 4. Calculate premium deltas, churn risk, profitability impact
# MAGIC 5. Output the Impact Summary for actuarial review in the HITL App

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, round as spark_round, abs as spark_abs

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Simulate "Flood Risk v2" — updated geospatial data
# MAGIC In reality this would arrive via the ingestion pipeline. Here we simulate
# MAGIC an updated version where ~30% of postcodes have changed flood scores.

# COMMAND ----------

current_geo = spark.table(f"{fqn}.silver_geospatial_hazard_enrichment")

import random
random.seed(99)

# Create "v2" with shifted flood scores for ~30% of postcodes
# Some go up (increased risk), some go down (reduced risk)
geo_v2 = (current_geo
    .withColumn("rand_val", F.abs(F.hash(F.concat(col("postcode_sector"), lit("v2_shift")))) % 100)
    .withColumn("flood_zone_rating_v2",
        when(col("rand_val") < 15,
             # 15% get WORSE flood scores (+1 to +3)
             F.least(lit(10), col("flood_zone_rating") + (F.abs(F.hash(col("postcode_sector"))) % 3 + 1)))
        .when(col("rand_val") < 30,
             # 15% get BETTER flood scores (-1 to -2)
             F.greatest(lit(1), col("flood_zone_rating") - (F.abs(F.hash(col("postcode_sector"))) % 2 + 1)))
        .otherwise(col("flood_zone_rating"))  # 70% unchanged
    )
    .withColumn("flood_changed", col("flood_zone_rating_v2") != col("flood_zone_rating"))
    .withColumn("flood_direction",
        when(col("flood_zone_rating_v2") > col("flood_zone_rating"), "WORSENED")
        .when(col("flood_zone_rating_v2") < col("flood_zone_rating"), "IMPROVED")
        .otherwise("UNCHANGED"))
)

changed_count = geo_v2.filter(col("flood_changed")).count()
total_count = geo_v2.count()
print(f"Flood Risk v2: {changed_count}/{total_count} postcodes changed ({changed_count/total_count*100:.0f}%)")

display(
    geo_v2.filter(col("flood_changed"))
    .select("postcode_sector", "flood_zone_rating", "flood_zone_rating_v2", "flood_direction")
    .orderBy("postcode_sector")
    .limit(20)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Join to active policy book and re-rate
# MAGIC Using a proxy pricing formula that mimics a real rating engine:
# MAGIC `Technical_Price = Base_Rate × Industry_Factor × Flood_Factor × Crime_Factor × Construction_Factor`

# COMMAND ----------

policies = spark.table(f"{fqn}.internal_commercial_policies")
upt = spark.table(f"{fqn}.unified_pricing_table_live")

# Base rate per £1k sum insured (simplified)
BASE_RATE = 5.0  # £5 per £1k SI

# Industry factors
industry_factors = spark.createDataFrame([
    ("High", 1.8), ("Medium", 1.2), ("Low", 0.85),
], ["industry_risk_tier", "industry_factor"])

# Construction factors
construction_factors = spark.createDataFrame([
    ("Fire Resistive", 0.7), ("Non-Combustible", 0.85),
    ("Joisted Masonry", 1.0), ("Heavy Timber", 1.15), ("Frame", 1.4),
], ["construction_type", "construction_factor"])

# Build the shadow pricing comparison
shadow = (upt
    .join(geo_v2.select("postcode_sector", "flood_zone_rating_v2", "flood_changed", "flood_direction"),
          "postcode_sector", "left")
    .join(industry_factors, "industry_risk_tier", "left")
    .join(construction_factors, "construction_type", "left")
    # Current flood factor (0.7 for zone 1, up to 2.5 for zone 10)
    .withColumn("current_flood_factor",
        spark_round(0.7 + (F.coalesce(col("flood_zone_rating"), lit(5)) - 1) * 0.2, 2))
    # New flood factor
    .withColumn("new_flood_factor",
        spark_round(0.7 + (F.coalesce(col("flood_zone_rating_v2"), col("flood_zone_rating"), lit(5)) - 1) * 0.2, 2))
    # Crime factor (0.8 for low crime, up to 1.5 for high crime)
    .withColumn("crime_factor",
        spark_round(0.8 + F.coalesce(col("crime_theft_index"), lit(50)) / 100.0 * 0.7, 2))
    # Current technical price
    .withColumn("current_technical_price",
        spark_round(
            lit(BASE_RATE)
            * (col("sum_insured") / 1000)
            * F.coalesce(col("industry_factor"), lit(1.0))
            * col("current_flood_factor")
            * col("crime_factor")
            * F.coalesce(col("construction_factor"), lit(1.0)),
            0))
    # New technical price (with updated flood scores)
    .withColumn("new_technical_price",
        spark_round(
            lit(BASE_RATE)
            * (col("sum_insured") / 1000)
            * F.coalesce(col("industry_factor"), lit(1.0))
            * col("new_flood_factor")
            * col("crime_factor")
            * F.coalesce(col("construction_factor"), lit(1.0)),
            0))
    # Premium delta
    .withColumn("premium_delta", col("new_technical_price") - col("current_technical_price"))
    .withColumn("premium_delta_pct",
        spark_round(
            when(col("current_technical_price") > 0,
                 col("premium_delta") / col("current_technical_price") * 100)
            .otherwise(0), 1))
    # Churn risk flag: policies facing >£500 increase or >15% increase
    .withColumn("churn_risk",
        when((col("premium_delta") > 500) | (col("premium_delta_pct") > 15), "HIGH")
        .when((col("premium_delta") > 200) | (col("premium_delta_pct") > 5), "MEDIUM")
        .otherwise("LOW"))
    # Renewal proximity (months until renewal)
    .withColumn("months_to_renewal",
        F.months_between(col("renewal_date").cast("date"), F.current_date()))
)

print(f"Shadow pricing complete: {shadow.count()} policies re-rated")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Impact Summary — Portfolio Level

# COMMAND ----------

portfolio_impact = shadow.agg(
    F.count("*").alias("total_policies"),
    F.sum("current_premium").alias("total_current_gwp"),
    F.sum("current_technical_price").alias("total_current_technical"),
    F.sum("new_technical_price").alias("total_new_technical"),
    F.sum("premium_delta").alias("total_premium_change"),
    spark_round(F.avg("premium_delta_pct"), 1).alias("avg_change_pct"),
    F.sum(when(col("premium_delta") > 0, 1).otherwise(0)).alias("policies_increase"),
    F.sum(when(col("premium_delta") < 0, 1).otherwise(0)).alias("policies_decrease"),
    F.sum(when(col("premium_delta") == 0, 1).otherwise(0)).alias("policies_unchanged"),
    F.sum(when(col("churn_risk") == "HIGH", 1).otherwise(0)).alias("high_churn_risk"),
    F.sum(when(col("churn_risk") == "HIGH", col("current_premium")).otherwise(0)).alias("high_churn_gwp"),
    F.sum(when(col("churn_risk") == "MEDIUM", 1).otherwise(0)).alias("medium_churn_risk"),
).collect()[0]

print("=" * 60)
print("SHADOW PRICING IMPACT SUMMARY — Flood Risk v2")
print("=" * 60)
print(f"Total Policies Re-rated:    {portfolio_impact.total_policies:,}")
print(f"Current GWP:                £{portfolio_impact.total_current_gwp:,.0f}")
print(f"")
print(f"Technical Price Change:     £{portfolio_impact.total_premium_change:,.0f}")
print(f"Average Change:             {portfolio_impact.avg_change_pct}%")
print(f"")
print(f"Policies with INCREASE:     {portfolio_impact.policies_increase:,}")
print(f"Policies with DECREASE:     {portfolio_impact.policies_decrease:,}")
print(f"Policies UNCHANGED:         {portfolio_impact.policies_unchanged:,}")
print(f"")
print(f"HIGH Churn Risk:            {portfolio_impact.high_churn_risk:,} policies")
print(f"HIGH Churn GWP at Risk:     £{portfolio_impact.high_churn_gwp:,.0f}")
print(f"MEDIUM Churn Risk:          {portfolio_impact.medium_churn_risk:,} policies")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Write Impact Summary table (for App/Dashboard)

# COMMAND ----------

# Policy-level impact table
impact_detail = (shadow
    .select(
        "policy_id", "sic_code", "postcode_sector", "industry_risk_tier",
        "construction_type", "sum_insured", "current_premium",
        "current_technical_price", "new_technical_price",
        "premium_delta", "premium_delta_pct",
        "flood_zone_rating", col("flood_zone_rating_v2").alias("flood_zone_rating_new"),
        "flood_direction", "churn_risk",
        "months_to_renewal", "renewal_date",
    )
)

impact_detail.write.mode("overwrite").saveAsTable(f"{fqn}.shadow_pricing_impact")
print(f"✓ {fqn}.shadow_pricing_impact — {impact_detail.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Breakdown by Segment

# COMMAND ----------

# MAGIC %md
# MAGIC ### By Industry

# COMMAND ----------

display(
    shadow
    .groupBy("industry_risk_tier")
    .agg(
        F.count("*").alias("policies"),
        spark_round(F.sum("current_premium"), 0).alias("current_gwp"),
        spark_round(F.sum("premium_delta"), 0).alias("total_delta"),
        spark_round(F.avg("premium_delta_pct"), 1).alias("avg_change_pct"),
        F.sum(when(col("churn_risk") == "HIGH", 1).otherwise(0)).alias("high_churn"),
    )
    .orderBy(col("total_delta").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### By Region (Postcode Area)

# COMMAND ----------

display(
    shadow
    .withColumn("postcode_area", F.regexp_extract("postcode_sector", r"^([A-Z]+)", 1))
    .groupBy("postcode_area")
    .agg(
        F.count("*").alias("policies"),
        spark_round(F.sum("premium_delta"), 0).alias("total_delta"),
        spark_round(F.avg("premium_delta_pct"), 1).alias("avg_change_pct"),
        F.sum(when(col("flood_direction") == "WORSENED", 1).otherwise(0)).alias("flood_worsened"),
        F.sum(when(col("flood_direction") == "IMPROVED", 1).otherwise(0)).alias("flood_improved"),
        F.sum(when(col("churn_risk") == "HIGH", 1).otherwise(0)).alias("high_churn"),
    )
    .orderBy(col("total_delta").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Renewal Risk — Policies renewing in next 90 days with HIGH churn risk

# COMMAND ----------

display(
    shadow
    .filter(col("churn_risk") == "HIGH")
    .filter(col("months_to_renewal").between(0, 3))
    .select("policy_id", "postcode_sector", "industry_risk_tier",
            "current_premium", "premium_delta", "premium_delta_pct",
            "flood_zone_rating", "flood_zone_rating_v2", "renewal_date")
    .orderBy(col("premium_delta").desc())
    .limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Adverse Selection Analysis
# MAGIC If we DON'T update our prices, are we cheaper than market in high-risk areas?

# COMMAND ----------

display(
    shadow
    .filter(col("flood_direction") == "WORSENED")
    .filter(col("market_position_ratio").isNotNull())
    .select("policy_id", "postcode_sector",
            "flood_zone_rating", "flood_zone_rating_v2",
            "current_premium", "new_technical_price", "premium_delta",
            spark_round(col("market_position_ratio"), 2).alias("market_position"))
    .withColumn("adverse_selection_risk",
        when(col("market_position") < 0.9, "HIGH — We're cheaper than market in worsening zone")
        .when(col("market_position") < 1.1, "MEDIUM")
        .otherwise("LOW"))
    .orderBy("adverse_selection_risk", col("premium_delta").desc())
    .limit(30)
)
