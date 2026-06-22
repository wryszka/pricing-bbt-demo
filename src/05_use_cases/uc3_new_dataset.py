# Databricks notebook source
# MAGIC %md
# MAGIC # UC3: New Dataset Addition — Subsidence Risk Enhancement
# MAGIC
# MAGIC **The story:** A brand-new external dataset (Geospatial Subsidence Index)
# MAGIC arrives from a vendor. The system ingests it, maps it to existing policies,
# MAGIC expands the Unified Pricing Table with net-new features, and runs a shadow
# MAGIC simulation to show how the new factor creates micro-segments in our pricing.
# MAGIC
# MAGIC **Key insight:** By adding this data, we find safe risks we were overpricing
# MAGIC and hidden dangers we were underpricing. Time-to-market: minutes, not months.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, round as spark_round
import random

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Generate the new "Subsidence Risk Index" dataset
# MAGIC This simulates a vendor CSV arriving with detailed subsidence analysis —
# MAGIC more granular than the basic subsidence_risk already in the geo data.

# COMMAND ----------

# Get all postcodes from our policy book
postcodes = spark.table(f"{fqn}.unified_pricing_table_live").select("postcode_sector").distinct()

# Generate detailed subsidence data per postcode
subsidence_df = (postcodes
    .withColumn("subsidence_severity_score",
        spark_round(F.abs(F.hash(F.concat(col("postcode_sector"), lit("sub_sev")))) % 100 / 10.0, 1))
    .withColumn("clay_shrinkage_potential",
        when(F.abs(F.hash(F.concat(col("postcode_sector"), lit("clay")))) % 3 == 0, "High")
        .when(F.abs(F.hash(F.concat(col("postcode_sector"), lit("clay")))) % 3 == 1, "Medium")
        .otherwise("Low"))
    .withColumn("historic_subsidence_claims_per_1000",
        spark_round(F.abs(F.hash(F.concat(col("postcode_sector"), lit("hist_sub")))) % 50 / 10.0, 1))
    .withColumn("ground_movement_mm_per_year",
        spark_round(F.abs(F.hash(F.concat(col("postcode_sector"), lit("movement")))) % 30 / 10.0, 1))
    .withColumn("tree_proximity_risk",
        when(F.abs(F.hash(F.concat(col("postcode_sector"), lit("tree")))) % 4 == 0, "High")
        .when(F.abs(F.hash(F.concat(col("postcode_sector"), lit("tree")))) % 4 == 1, "Medium")
        .otherwise("Low"))
    .withColumn("water_table_depth_m",
        spark_round(F.abs(F.hash(F.concat(col("postcode_sector"), lit("water")))) % 200 / 10.0, 1))
    # Composite subsidence modifier for pricing
    .withColumn("subsidence_pricing_modifier",
        spark_round(
            0.8 + (col("subsidence_severity_score") / 10.0) * 0.6
            + when(col("clay_shrinkage_potential") == "High", 0.3)
              .when(col("clay_shrinkage_potential") == "Medium", 0.1)
              .otherwise(0.0)
            + col("historic_subsidence_claims_per_1000") / 50.0 * 0.2,
            3))
)

# Write to volume as "incoming vendor data" and to a staging table
subsidence_df.write.mode("overwrite").saveAsTable(f"{fqn}.staging_subsidence_risk_index")
print(f"✓ New dataset staged: {fqn}.staging_subsidence_risk_index — {subsidence_df.count()} postcodes")

display(subsidence_df.orderBy(col("subsidence_severity_score").desc()).limit(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Join new dataset to UPT and compute pricing impact
# MAGIC Compare: Old pricing (without subsidence) vs New pricing (with subsidence modifier)

# COMMAND ----------

upt = spark.table(f"{fqn}.unified_pricing_table_live")

# Join subsidence data
enriched = upt.join(
    subsidence_df.select("postcode_sector", "subsidence_severity_score", "clay_shrinkage_potential",
                          "historic_subsidence_claims_per_1000", "subsidence_pricing_modifier"),
    "postcode_sector",
    "left"
)

# Pricing impact: how does subsidence modifier change the technical price?
BASE_RATE = 5.0

impact = (enriched
    .withColumn("industry_factor",
        when(col("industry_risk_tier") == "High", 1.8)
        .when(col("industry_risk_tier") == "Medium", 1.2)
        .otherwise(0.85))
    .withColumn("flood_factor",
        spark_round(0.7 + (F.coalesce(col("flood_zone_rating"), lit(5)) - 1) * 0.2, 2))
    # Old price (no subsidence factor)
    .withColumn("price_without_subsidence",
        spark_round(lit(BASE_RATE) * (col("sum_insured") / 1000)
                    * col("industry_factor") * col("flood_factor"), 0))
    # New price (with subsidence factor)
    .withColumn("price_with_subsidence",
        spark_round(lit(BASE_RATE) * (col("sum_insured") / 1000)
                    * col("industry_factor") * col("flood_factor")
                    * F.coalesce(col("subsidence_pricing_modifier"), lit(1.0)), 0))
    # Delta
    .withColumn("subsidence_delta", col("price_with_subsidence") - col("price_without_subsidence"))
    .withColumn("subsidence_delta_pct",
        spark_round(
            when(col("price_without_subsidence") > 0,
                 col("subsidence_delta") / col("price_without_subsidence") * 100)
            .otherwise(0), 1))
    # Segment classification
    .withColumn("pricing_action",
        when(col("subsidence_delta_pct") > 10, "INCREASE — Hidden risk discovered")
        .when(col("subsidence_delta_pct") < -5, "DECREASE — Safe segment, can compete")
        .otherwise("MINOR CHANGE"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Portfolio Impact Summary

# COMMAND ----------

summary = impact.agg(
    F.count("*").alias("total_policies"),
    F.sum("current_premium").alias("total_gwp"),
    F.sum("subsidence_delta").alias("total_price_change"),
    spark_round(F.avg("subsidence_delta_pct"), 1).alias("avg_change_pct"),
    F.sum(when(col("pricing_action").contains("INCREASE"), 1).otherwise(0)).alias("need_increase"),
    F.sum(when(col("pricing_action").contains("INCREASE"), col("current_premium")).otherwise(0)).alias("increase_gwp"),
    F.sum(when(col("pricing_action").contains("DECREASE"), 1).otherwise(0)).alias("can_decrease"),
    F.sum(when(col("pricing_action").contains("DECREASE"), col("current_premium")).otherwise(0)).alias("decrease_gwp"),
    F.sum(when(col("pricing_action").contains("MINOR"), 1).otherwise(0)).alias("minor_change"),
).collect()[0]

print("=" * 60)
print("NEW DATASET IMPACT — Subsidence Risk Index")
print("=" * 60)
print(f"Total Policies:              {summary.total_policies:,}")
print(f"Total GWP:                   £{summary.total_gwp:,.0f}")
print(f"")
print(f"Net Price Change:            £{summary.total_price_change:,.0f}")
print(f"Average Change:              {summary.avg_change_pct}%")
print(f"")
print(f"INCREASE (hidden risk):      {summary.need_increase:,} policies (£{summary.increase_gwp:,.0f} GWP)")
print(f"DECREASE (competitive edge): {summary.can_decrease:,} policies (£{summary.decrease_gwp:,.0f} GWP)")
print(f"MINOR CHANGE:                {summary.minor_change:,} policies")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: "The Competitive Edge" — Safe segments we can price down

# COMMAND ----------

display(
    impact
    .filter(col("pricing_action").contains("DECREASE"))
    .groupBy("industry_risk_tier", "clay_shrinkage_potential")
    .agg(
        F.count("*").alias("policies"),
        spark_round(F.sum("current_premium"), 0).alias("gwp"),
        spark_round(F.avg("subsidence_delta_pct"), 1).alias("avg_reduction_pct"),
        spark_round(F.avg("subsidence_severity_score"), 1).alias("avg_subsidence_score"),
    )
    .orderBy(col("avg_reduction_pct"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: "Hidden Risk Discovery" — Exposed premium

# COMMAND ----------

display(
    impact
    .filter(col("pricing_action").contains("INCREASE"))
    .groupBy("industry_risk_tier", "clay_shrinkage_potential")
    .agg(
        F.count("*").alias("policies"),
        spark_round(F.sum("current_premium"), 0).alias("exposed_gwp"),
        spark_round(F.avg("subsidence_delta_pct"), 1).alias("avg_increase_pct"),
        spark_round(F.avg("subsidence_severity_score"), 1).alias("avg_subsidence_score"),
    )
    .orderBy(col("exposed_gwp").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 6: Most profitable regions with new factor

# COMMAND ----------

display(
    impact
    .withColumn("postcode_area", F.regexp_extract("postcode_sector", r"^([A-Z]+)", 1))
    .groupBy("postcode_area")
    .agg(
        F.count("*").alias("policies"),
        spark_round(F.sum("current_premium"), 0).alias("gwp"),
        spark_round(F.avg("subsidence_delta_pct"), 1).alias("avg_change_pct"),
        spark_round(F.avg("subsidence_severity_score"), 1).alias("avg_subsidence"),
        F.sum(when(col("pricing_action").contains("DECREASE"), 1).otherwise(0)).alias("can_compete"),
        F.sum(when(col("pricing_action").contains("INCREASE"), 1).otherwise(0)).alias("hidden_risk"),
    )
    .orderBy(col("can_compete").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 7: Write impact table for App/Dashboard

# COMMAND ----------

impact_output = (impact
    .select(
        "policy_id", "sic_code", "postcode_sector", "industry_risk_tier",
        "sum_insured", "current_premium",
        "price_without_subsidence", "price_with_subsidence",
        "subsidence_delta", "subsidence_delta_pct",
        "subsidence_severity_score", "clay_shrinkage_potential",
        "historic_subsidence_claims_per_1000", "subsidence_pricing_modifier",
        "pricing_action",
    )
)

impact_output.write.mode("overwrite").saveAsTable(f"{fqn}.new_dataset_subsidence_impact")
print(f"✓ {fqn}.new_dataset_subsidence_impact — {impact_output.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Messages for Demo
# MAGIC
# MAGIC 1. **The Competitive Edge (Better Price):** "By layering in this subsidence index,
# MAGIC    we identified safe sub-segments. We can drop rates for {can_decrease} policies,
# MAGIC    increasing our win-rate without sacrificing margin."
# MAGIC
# MAGIC 2. **Hidden Risk Discovery:** "The new dataset revealed £X of premium is exposed
# MAGIC    to subsidence risk our previous models completely missed."
# MAGIC
# MAGIC 3. **Time-to-Market:** "What used to take 6 months of IT integration now takes
# MAGIC    minutes. The data arrived, was evaluated for financial impact, and is ready
# MAGIC    for the rating engine today."
