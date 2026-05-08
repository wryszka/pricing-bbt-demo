# Databricks notebook source
# MAGIC %md
# MAGIC # Rating engine config — seed + version history
# MAGIC
# MAGIC Creates `{fqn}.rating_engine_config` — one row per versioned rating
# MAGIC engine configuration. Every field a real rating engine carries:
# MAGIC
# MAGIC | Column                   | Purpose                                                |
# MAGIC |---|---|
# MAGIC | `version`                | semver-ish string: `v1.0`, `v1.1`, `v2.0`              |
# MAGIC | `effective_date`         | when this version starts applying                      |
# MAGIC | `status`                 | `champion` / `previous_champion` / `archived` / `draft`|
# MAGIC | `expense_loading_pct`    | % added for expenses                                    |
# MAGIC | `commission_bp`          | basis points for broker commission                      |
# MAGIC | `fraud_loading_pct`      | premium uplift when fraud_pred > threshold              |
# MAGIC | `fraud_loading_threshold`| fraud_pred threshold that triggers the loading          |
# MAGIC | `demand_adj_pct`         | demand-elasticity adjustment (low-demand -> +adj, high -> -adj) |
# MAGIC | `demand_adj_threshold_lo`| demand_pred threshold below which we uplift             |
# MAGIC | `demand_adj_threshold_hi`| demand_pred threshold above which we discount           |
# MAGIC | `min_premium`            | floor                                                    |
# MAGIC | `max_premium`            | cap                                                     |
# MAGIC | `approved_by`            | who signed it off                                        |
# MAGIC | `narrative`              | one-paragraph rationale                                 |
# MAGIC
# MAGIC Only ONE row may have `status='champion'` at a time — enforced by the
# MAGIC update endpoint in the app (not at table level).
# MAGIC
# MAGIC Idempotent — CREATE OR REPLACE on the seed.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

from datetime import date
from pyspark.sql.types import (
    StructType, StructField, StringType, DoubleType, IntegerType, DateType
)

# Three real versions — tells a governance story: baseline, expense review,
# fraud-loading tightening.
versions = [
    {
        "version":                  "v1.0",
        "effective_date":           date(2024, 1, 1),
        "status":                   "archived",
        "expense_loading_pct":      22.0,
        "commission_bp":            1750,        # 17.5 %
        "fraud_loading_pct":        5.0,
        "fraud_loading_threshold":  0.25,
        "demand_adj_pct":           2.0,
        "demand_adj_threshold_lo":  0.40,
        "demand_adj_threshold_hi":  0.75,
        "min_premium":              120.0,
        "max_premium":              250_000.0,
        "approved_by":              "pricing_committee@bricksurance.com",
        "narrative":                "Launch configuration. Expense loading set to legacy book average (22%). Fraud loading 5% when predicted fraud probability > 25%. Demand elasticity band 40-75%.",
    },
    {
        "version":                  "v1.1",
        "effective_date":           date(2025, 4, 1),
        "status":                   "previous_champion",
        "expense_loading_pct":      19.5,
        "commission_bp":            1750,
        "fraud_loading_pct":        5.0,
        "fraud_loading_threshold":  0.25,
        "demand_adj_pct":           2.0,
        "demand_adj_threshold_lo":  0.40,
        "demand_adj_threshold_hi":  0.75,
        "min_premium":              120.0,
        "max_premium":              250_000.0,
        "approved_by":              "pricing_committee@bricksurance.com",
        "narrative":                "Expense loading review — benchmarked vs peer quarterly survey, reduced from 22% to 19.5% effective 1 April 2025. No other changes. Effect: approximately -2.5% across the entire book.",
    },
    {
        "version":                  "v2.0",
        "effective_date":           date(2026, 1, 1),
        "status":                   "champion",
        "expense_loading_pct":      19.5,
        "commission_bp":            1500,        # 15.0 % (post broker-deal renegotiation)
        "fraud_loading_pct":        6.0,         # tightened
        "fraud_loading_threshold":  0.20,        # lower trigger
        "demand_adj_pct":           2.5,         # slightly stronger elasticity band
        "demand_adj_threshold_lo":  0.40,
        "demand_adj_threshold_hi":  0.80,        # wider upper bound for discount
        "min_premium":              150.0,       # raised floor
        "max_premium":              250_000.0,
        "approved_by":              "pricing_committee@bricksurance.com",
        "narrative":                "Post-renegotiation broker deal dropped commission from 17.5% to 15%. Fraud loading tightened (6% at 0.20 trigger) following 2025 fraud-book review. Demand elasticity widened to capture more of the low-conversion tail. Min premium raised to £150 to cover acquisition cost on smallest SMEs.",
    },
]

schema_struct = StructType([
    StructField("version",                  StringType(),  False),
    StructField("effective_date",           DateType(),    False),
    StructField("status",                   StringType(),  False),
    StructField("expense_loading_pct",      DoubleType(),  False),
    StructField("commission_bp",            IntegerType(), False),
    StructField("fraud_loading_pct",        DoubleType(),  False),
    StructField("fraud_loading_threshold",  DoubleType(),  False),
    StructField("demand_adj_pct",           DoubleType(),  False),
    StructField("demand_adj_threshold_lo",  DoubleType(),  False),
    StructField("demand_adj_threshold_hi",  DoubleType(),  False),
    StructField("min_premium",              DoubleType(),  False),
    StructField("max_premium",              DoubleType(),  False),
    StructField("approved_by",              StringType(),  True),
    StructField("narrative",                StringType(),  True),
])

df = spark.createDataFrame(versions, schema_struct)
df.write.mode("overwrite").option("overwriteSchema", "true") \
  .saveAsTable(f"{fqn}.rating_engine_config")

print(f"Seeded {fqn}.rating_engine_config with {df.count()} versions:")
spark.sql(f"""
    SELECT version, status, cast(effective_date as string) as effective_date,
           expense_loading_pct, commission_bp, fraud_loading_pct
    FROM {fqn}.rating_engine_config
    ORDER BY effective_date
""").show(truncate=False)

# COMMAND ----------

# Seed audit events so the history shows up in the Governance tab right away.
import json
for v in versions:
    det = json.dumps({
        "action":              "seed" if v["status"] == "archived" else "publish",
        "version":             v["version"],
        "expense_loading_pct": v["expense_loading_pct"],
        "commission_bp":       v["commission_bp"],
        "fraud_loading_pct":   v["fraud_loading_pct"],
        "narrative":           v["narrative"][:220],
    }).replace("'", "''")
    spark.sql(f"""
        INSERT INTO {fqn}.audit_log
          (event_id, event_type, entity_type, entity_id, entity_version,
           user_id, timestamp, details, source)
        SELECT uuid(), 'rating_engine_config_change', 'rating_engine_config',
               'rating_engine', '{v["version"]}',
               '{v["approved_by"]}', cast('{v["effective_date"]} 09:00:00' as timestamp),
               '{det}', 'seed_notebook'
    """)

print("Seeded 3 audit events.")

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "rows":     3,
    "champion": "v2.0",
    "table":    f"{fqn}.rating_engine_config",
}))
