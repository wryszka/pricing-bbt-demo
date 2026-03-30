# Databricks notebook source
# MAGIC %md
# MAGIC # Ingest: Credit Bureau Summary → Raw
# MAGIC Loads the external credit bureau CSV from the landing volume into a raw (bronze) table.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_bbt")
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
    .csv(f"{volume_path}/credit_bureau_summary/")
)

df = df.withColumn("_ingested_at", F.current_timestamp()) \
       .withColumn("_source_file", F.input_file_name())

df.write.mode("overwrite").saveAsTable(f"{fqn}.raw_credit_bureau_summary")

print(f"✓ Ingested {df.count()} rows → {fqn}.raw_credit_bureau_summary")
