# Databricks notebook source
# MAGIC %md
# MAGIC # UC5: Enriched Pricing — Combined Model Decision
# MAGIC
# MAGIC **The full pricing picture.** In real P&C insurance, the quoted premium
# MAGIC isn't just "frequency × severity". It layers in multiple model outputs:
# MAGIC
# MAGIC ```
# MAGIC Final Price = Technical Price × Demand Adjustment × Fraud Load × Retention Discount
# MAGIC ```
# MAGIC
# MAGIC | Component | Model | What it does |
# MAGIC |---|---|---|
# MAGIC | **Technical Price** | GLM Frequency × GLM Severity | Pure risk cost (the "burning cost") |
# MAGIC | **Demand Adjustment** | GBM Demand | Adjusts for price elasticity — how likely to convert |
# MAGIC | **Fraud Load** | GBM Fraud Propensity | Adds loading for high-fraud-risk segments |
# MAGIC | **Retention Discount** | GBM Retention | Applies discount for at-risk renewals to retain them |
# MAGIC
# MAGIC **Why this matters:** On Databricks, all 6 models use the same feature store,
# MAGIC same governance, same serving infrastructure. On legacy platforms, each model
# MAGIC is a separate system with its own data pipeline and integration overhead.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn_prefix = f"{catalog}.{schema}"

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, round as spark_round

upt = spark.table(f"{fqn_prefix}.unified_pricing_table_live")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Technical Price (Frequency × Severity)
# MAGIC
# MAGIC The actuarial base price. Using the proxy pricing formula (in production
# MAGIC this would come from the GLM frequency and severity models).

# COMMAND ----------

BASE_RATE = 5.0  # £5 per £1k SI

pricing = (upt
    .withColumn("industry_factor",
        when(col("industry_risk_tier") == "High", 1.8)
        .when(col("industry_risk_tier") == "Medium", 1.2)
        .otherwise(0.85))
    .withColumn("flood_factor",
        spark_round(0.7 + (F.coalesce(col("flood_zone_rating"), lit(5)) - 1) * 0.2, 2))
    .withColumn("crime_factor",
        spark_round(0.8 + F.coalesce(col("crime_theft_index"), lit(50)) / 100.0 * 0.7, 2))
    .withColumn("construction_factor",
        when(col("construction_type") == "Fire Resistive", 0.7)
        .when(col("construction_type") == "Non-Combustible", 0.85)
        .when(col("construction_type") == "Heavy Timber", 1.15)
        .when(col("construction_type") == "Frame", 1.4)
        .otherwise(1.0))
    .withColumn("technical_price",
        spark_round(
            lit(BASE_RATE) * (col("sum_insured") / 1000.0)
            * col("industry_factor") * col("flood_factor")
            * col("crime_factor") * col("construction_factor"),
        0))
)

avg_tech = pricing.agg(F.avg("technical_price")).collect()[0][0]
print(f"Step 1: Technical Price")
print(f"  Average: £{avg_tech:,.0f}")
print(f"  Based on: GLM Frequency × Severity (proxy formula)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Fraud Load Factor
# MAGIC
# MAGIC Policies in high-fraud-risk segments get a loading to cover the expected
# MAGIC additional cost of fraudulent claims. Typically 0-15% of technical price.

# COMMAND ----------

# Simulate fraud scores from the model's key features
# In production this comes from Model 5 (fraud propensity model)
pricing = (pricing
    .withColumn("fraud_score",
        spark_round(
            (F.least(F.greatest(F.coalesce(col("loss_ratio_5y"), lit(0)).cast("double"), lit(0)), lit(5)) / 5 * 0.3) +
            (F.least(F.greatest(F.coalesce(col("ccj_count"), lit(0)).cast("double"), lit(0)), lit(5)) / 5 * 0.3) +
            (F.least(F.greatest(F.coalesce(col("credit_default_probability"), lit(0)).cast("double"), lit(0)), lit(0.5)) / 0.5 * 0.4),
        3))
    # Fraud load: 0% for low risk, up to 15% for high risk
    .withColumn("fraud_load",
        when(col("fraud_score") > 0.6, 1.15)  # High: +15%
        .when(col("fraud_score") > 0.3, 1.05)  # Medium: +5%
        .otherwise(1.0))  # Low: no loading
    .withColumn("price_after_fraud",
        spark_round(col("technical_price") * col("fraud_load"), 0))
)

fraud_uplift = pricing.agg(
    F.avg("fraud_load").alias("avg_load"),
    F.sum(when(col("fraud_load") > 1.0, 1).otherwise(0)).alias("policies_loaded"),
).collect()[0]
print(f"Step 2: Fraud Load")
print(f"  Average load factor: {fraud_uplift.avg_load:.3f}")
print(f"  Policies with loading: {fraud_uplift.policies_loaded:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Retention Discount
# MAGIC
# MAGIC At-risk renewals get a discount to retain them. The discount is funded
# MAGIC by the margin on retained low-risk policies. Typically 0-10%.

# COMMAND ----------

pricing = (pricing
    .withColumn("churn_score",
        spark_round(
            (F.least(F.greatest(F.coalesce(col("market_position_ratio"), lit(1.0)).cast("double"), lit(0.5)), lit(2)) - 0.5) / 1.5 * 0.4 +
            (1.0 - F.least(F.greatest(F.coalesce(col("claim_count_5y"), lit(0)).cast("double"), lit(0)), lit(5)) / 5) * 0.3 +
            (F.least(F.greatest(F.coalesce(col("competitor_quote_count"), lit(0)).cast("double"), lit(0)), lit(5)) / 5) * 0.3,
        3))
    # Retention discount: 0% for sticky, up to 10% for at-risk
    .withColumn("retention_factor",
        when(col("churn_score") > 0.7, 0.90)   # High churn risk: -10%
        .when(col("churn_score") > 0.4, 0.95)   # Medium: -5%
        .otherwise(1.0))  # Low risk: no discount
    .withColumn("price_after_retention",
        spark_round(col("price_after_fraud") * col("retention_factor"), 0))
)

retention_discount = pricing.agg(
    F.avg("retention_factor").alias("avg_factor"),
    F.sum(when(col("retention_factor") < 1.0, 1).otherwise(0)).alias("policies_discounted"),
).collect()[0]
print(f"Step 3: Retention Discount")
print(f"  Average factor: {retention_discount.avg_factor:.3f}")
print(f"  Policies discounted: {retention_discount.policies_discounted:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Final Combined Price
# MAGIC
# MAGIC The pricing waterfall: Technical → Fraud Load → Retention Discount → Final.

# COMMAND ----------

pricing = pricing.withColumn("final_price", col("price_after_retention"))

summary = pricing.agg(
    F.count("*").alias("policies"),
    spark_round(F.avg("technical_price"), 0).alias("avg_technical"),
    spark_round(F.avg("price_after_fraud"), 0).alias("avg_after_fraud"),
    spark_round(F.avg("price_after_retention"), 0).alias("avg_after_retention"),
    spark_round(F.avg("final_price"), 0).alias("avg_final"),
    spark_round(F.sum("technical_price"), 0).alias("total_technical_gwp"),
    spark_round(F.sum("final_price"), 0).alias("total_final_gwp"),
).collect()[0]

print("=" * 60)
print("ENRICHED PRICING WATERFALL")
print("=" * 60)
print(f"Policies:              {summary.policies:,}")
print(f"")
print(f"Avg Technical Price:   £{summary.avg_technical:,}")
print(f"  + Fraud Load:        £{summary.avg_after_fraud:,}")
print(f"  - Retention Discount:£{summary.avg_after_retention:,}")
print(f"  = Final Price:       £{summary.avg_final:,}")
print(f"")
print(f"Total GWP (Technical): £{summary.total_technical_gwp:,.0f}")
print(f"Total GWP (Final):     £{summary.total_final_gwp:,.0f}")
print(f"Net Adjustment:        £{summary.total_final_gwp - summary.total_technical_gwp:,.0f}")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pricing Waterfall by Industry

# COMMAND ----------

display(
    pricing
    .groupBy("industry_risk_tier")
    .agg(
        F.count("*").alias("policies"),
        spark_round(F.avg("technical_price"), 0).alias("avg_technical"),
        spark_round(F.avg("fraud_load"), 3).alias("avg_fraud_load"),
        spark_round(F.avg("retention_factor"), 3).alias("avg_retention"),
        spark_round(F.avg("final_price"), 0).alias("avg_final"),
        spark_round(F.sum("final_price") - F.sum("technical_price"), 0).alias("net_adjustment"),
    )
    .orderBy("industry_risk_tier")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Policies — Full Pricing Breakdown

# COMMAND ----------

display(
    pricing
    .select(
        "policy_id", "industry_risk_tier", "construction_type",
        "sum_insured", "current_premium",
        "technical_price", "fraud_load", "fraud_score",
        "retention_factor", "churn_score", "final_price",
    )
    .withColumn("vs_current", spark_round(col("final_price") - col("current_premium"), 0))
    .orderBy(F.abs(col("vs_current")).desc())
    .limit(30)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Key Demo Messages
# MAGIC
# MAGIC 1. **One platform, 6 models:** Frequency, severity, demand, risk uplift,
# MAGIC    fraud, retention — all trained on the same Unified Pricing Table, all
# MAGIC    governed by Unity Catalog, all logged in MLflow.
# MAGIC
# MAGIC 2. **Composable pricing:** Each model output is a transparent factor in the
# MAGIC    final price. An actuary can inspect and override any component.
# MAGIC
# MAGIC 3. **No integration overhead:** On legacy platforms (Radar, Earnix, Emblem),
# MAGIC    adding a new model means a new data pipeline, new integration, new
# MAGIC    deployment process. On Databricks, it's one more notebook using the
# MAGIC    same feature store.
# MAGIC
# MAGIC 4. **Audit trail:** Every model, every score, every decision is tracked.
# MAGIC    A regulator can reconstruct the exact pricing for any historical quote.
