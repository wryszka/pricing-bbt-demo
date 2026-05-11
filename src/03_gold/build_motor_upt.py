# Databricks notebook source
# MAGIC %md
# MAGIC # Build Motor UPT (Unified Motor Table)
# MAGIC
# MAGIC Merges motor_policies + motor_telematics_aggregate + motor_claims_history
# MAGIC into `unified_motor_table_live` — the 1M-row feature table that backs
# MAGIC the live-serving motor scorer.
# MAGIC
# MAGIC Idempotent. Adds PRIMARY KEY on policy_id so the table is publishable
# MAGIC to Lakebase via the existing live-serving provision flow.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

import pyspark.sql.functions as F
from pyspark.sql.functions import col, lit, when, coalesce

# COMMAND ----------

policies   = spark.table(f"{fqn}.motor_policies")
telematics = spark.table(f"{fqn}.motor_telematics_aggregate")
claims     = spark.table(f"{fqn}.motor_claims_history")

# Aggregate claims to per-policy
claim_agg = (claims.groupBy("policy_id")
    .agg(
        F.count("claim_id").alias("claim_count_5y"),
        F.sum("incurred_amount").alias("total_incurred_5y"),
        F.sum("paid_amount").alias("total_paid_5y"),
        F.sum(when(col("status") == "Open", 1).otherwise(0)).alias("open_claims_count"),
        F.sum(when(col("fault_indicator") == "At fault", 1).otherwise(0)).alias("at_fault_count_5y"),
        F.countDistinct("peril").alias("distinct_perils"),
        F.max("loss_date").alias("last_claim_date"),
    ))

upt = (policies
    .join(telematics, "policy_id", "left")
    .join(claim_agg,  "policy_id", "left")
    # zero-fill claim aggregates for non-claimants
    .withColumn("claim_count_5y",     coalesce(col("claim_count_5y"),     lit(0)))
    .withColumn("total_incurred_5y",  coalesce(col("total_incurred_5y"),  lit(0)))
    .withColumn("total_paid_5y",      coalesce(col("total_paid_5y"),      lit(0)))
    .withColumn("open_claims_count",  coalesce(col("open_claims_count"),  lit(0)))
    .withColumn("at_fault_count_5y",  coalesce(col("at_fault_count_5y"),  lit(0)))
    .withColumn("distinct_perils",    coalesce(col("distinct_perils"),    lit(0)))
    # Derived risk signals
    .withColumn("vehicle_age", lit(2026) - col("vehicle_year"))
    .withColumn("loss_ratio_5y",
        when(col("current_premium") > 0,
             F.round(col("total_incurred_5y") / (col("current_premium") * 5), 3))
        .otherwise(lit(0.0)))
    .withColumn("telematics_recent_event_count",
        col("recent_speeding_events") + col("recent_curfew_breaches") + col("recent_harsh_braking_30d"))
    .withColumn("upt_build_timestamp", F.current_timestamp())
)

table_name = f"{fqn}.unified_motor_table_live"
upt.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(table_name)

# Primary key for Lakebase publish
try:
    spark.sql(f"ALTER TABLE {table_name} ALTER COLUMN policy_id SET NOT NULL")
except Exception:
    pass
try:
    spark.sql(f"ALTER TABLE {table_name} ADD CONSTRAINT motor_upt_pk PRIMARY KEY (policy_id)")
except Exception as e:
    if "already exists" not in str(e).lower(): raise

n_rows = spark.table(table_name).count()
n_cols = len(spark.table(table_name).columns)
print(f"✓ {table_name}: {n_rows:,} rows × {n_cols} columns")

# Verify John's row is intact
print("\nJohn's UPT row:")
print(spark.sql(f"""
    SELECT driver_age, vehicle_make, vehicle_model, current_premium,
           behaviour_score, recent_speeding_events, recent_curfew_breaches,
           claim_count_5y, total_incurred_5y
    FROM {table_name}
    WHERE policy_id = 'POL-MOTOR-00000001'
""").collect())

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "table": table_name,
    "rows":  n_rows,
    "cols":  n_cols,
}))
