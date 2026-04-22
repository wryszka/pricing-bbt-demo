# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest: ONS Reference Data → Raw
# MAGIC Loads the free/public ONS + NHS reference CSV (postcode sector level) into a raw (bronze) table.
# MAGIC
# MAGIC Columns:
# MAGIC - `postcode_sector` — UK postcode sector
# MAGIC - `population_density_per_km2` — ONS mid-year estimates
# MAGIC - `urban_classification_score` — ONS rural-urban classification (1=rural .. 6=urban conurbation)
# MAGIC - `gp_density_per_10k` — NHS GP practices per 10k population
# MAGIC - `deprivation_decile` — IMD decile (1=most deprived .. 10=least deprived)

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("volume_name", "external_landing")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")

fqn = f"{catalog}.{schema}"
volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

# COMMAND ----------

import pyspark.sql.functions as F

df = (spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv(f"{volume_path}/ons_reference/")
)

df = df.withColumn("_ingested_at", F.current_timestamp()) \
       .withColumn("_source_file", F.col("_metadata.file_path"))

df.write.mode("overwrite").saveAsTable(f"{fqn}.raw_ons_reference")

print(f"✓ Ingested {df.count()} rows → {fqn}.raw_ons_reference")
