# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — Pricing BBT Demo
# MAGIC Creates the schema, volume, and "pre-existing" internal tables
# MAGIC (policies, claims, quotes) that Bricksurance already has on their platform.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_bbt")
dbutils.widgets.text("volume_name", "external_landing")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")

fqn = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create schema and volume

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {fqn}.{volume}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Internal table 1: Commercial Policies
# MAGIC These are policies already on the Databricks platform — the insurer's book of business.

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
import random

random.seed(42)

NUM_POLICIES = 50000
POSTCODES = [f"{area}{district}" for area in ["EC1","EC2","SW1","SE1","W1","N1","E1","M1","M2","B1","B2","LS1","LS2","L1","L2","CF1","EH1","G1","BS1","NG1"] for district in ["A","B","C","D","E"]]
SIC_CODES = ["1011","2562","4110","4520","4711","5610","6201","6311","6499","6820","7022","7112","8010","8622","9311"]
CONSTRUCTION = ["Non-Combustible","Joisted Masonry","Fire Resistive","Frame","Heavy Timber"]

policy_rows = []
for i in range(NUM_POLICIES):
    pid = f"POL-{100000 + i}"
    sic = random.choice(SIC_CODES)
    postcode = random.choice(POSTCODES)
    turnover = round(random.lognormvariate(13, 1.5))  # median ~440k
    construction = random.choice(CONSTRUCTION)
    year_built = random.randint(1920, 2024)
    sum_insured = round(turnover * random.uniform(1.5, 8.0))
    claims_5y = round(max(0, random.lognormvariate(8, 2.5)) if random.random() < 0.35 else 0)
    inception = f"202{random.randint(0,5)}-{random.randint(1,12):02d}-01"
    renewal = f"2026-{random.randint(1,12):02d}-01"
    premium = round(sum_insured * random.uniform(0.002, 0.015))

    policy_rows.append((pid, sic, postcode, turnover, construction, year_built,
                        sum_insured, claims_5y, inception, renewal, premium))

policy_schema = StructType([
    StructField("policy_id", StringType()),
    StructField("sic_code", StringType()),
    StructField("postcode_sector", StringType()),
    StructField("annual_turnover", LongType()),
    StructField("construction_type", StringType()),
    StructField("year_built", IntegerType()),
    StructField("sum_insured", LongType()),
    StructField("claims_history_5y", LongType()),
    StructField("inception_date", StringType()),
    StructField("renewal_date", StringType()),
    StructField("current_premium", LongType()),
])

df_policies = spark.createDataFrame(policy_rows, schema=policy_schema)
df_policies.write.mode("overwrite").saveAsTable(f"{fqn}.internal_commercial_policies")
print(f"✓ {fqn}.internal_commercial_policies — {df_policies.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Internal table 2: Claims History

# COMMAND ----------

# Generate claims linked to policies that have claims_history_5y > 0
policies_with_claims = df_policies.filter(F.col("claims_history_5y") > 0).select("policy_id", "sic_code", "postcode_sector", "claims_history_5y").collect()

PERILS = ["Fire","Flood","Theft","Liability","Storm","Subsidence","Escape of Water"]
STATUSES = ["Closed","Closed","Closed","Open","Open"]  # 60% closed

claim_rows = []
claim_id = 1
for row in policies_with_claims:
    # Each policy gets 1-5 claims
    n_claims = random.randint(1, 5)
    remaining = row.claims_history_5y
    for j in range(n_claims):
        cid = f"CLM-{claim_id:07d}"
        claim_id += 1
        peril = random.choice(PERILS)
        status = random.choice(STATUSES)
        loss_year = random.randint(2021, 2025)
        loss_month = random.randint(1, 12)
        loss_date = f"{loss_year}-{loss_month:02d}-{random.randint(1,28):02d}"

        if j < n_claims - 1:
            incurred = round(remaining * random.uniform(0.1, 0.5))
        else:
            incurred = remaining
        remaining = max(0, remaining - incurred)

        paid = round(incurred * random.uniform(0.5, 1.0)) if status == "Closed" else round(incurred * random.uniform(0.0, 0.4))
        reserve = incurred - paid

        claim_rows.append((cid, row.policy_id, peril, incurred, paid, reserve, loss_date, status))

claim_schema = StructType([
    StructField("claim_id", StringType()),
    StructField("policy_id", StringType()),
    StructField("peril", StringType()),
    StructField("incurred_amount", LongType()),
    StructField("paid_amount", LongType()),
    StructField("reserve", LongType()),
    StructField("loss_date", StringType()),
    StructField("status", StringType()),
])

df_claims = spark.createDataFrame(claim_rows, schema=claim_schema)
df_claims.write.mode("overwrite").saveAsTable(f"{fqn}.internal_claims_history")
print(f"✓ {fqn}.internal_claims_history — {df_claims.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Internal table 3: Quote History

# COMMAND ----------

quote_rows = []
for i in range(120000):
    qid = f"QTE-{200000 + i}"
    sic = random.choice(SIC_CODES)
    postcode = random.choice(POSTCODES)
    turnover = round(random.lognormvariate(13, 1.5))
    sum_insured = round(turnover * random.uniform(1.5, 8.0))
    quoted_premium = round(sum_insured * random.uniform(0.002, 0.015))
    competitor_quoted = random.choice(["Y", "N", "N"])  # 33% had competitor quote
    converted = "Y" if random.random() < 0.38 else "N"  # ~38% conversion
    policy_id = f"POL-{100000 + i}" if converted == "Y" and i < NUM_POLICIES else None
    quote_date = f"202{random.randint(0,5)}-{random.randint(1,12):02d}-{random.randint(1,28):02d}"

    quote_rows.append((qid, policy_id, sic, postcode, turnover, sum_insured,
                       quoted_premium, competitor_quoted, converted, quote_date))

quote_schema = StructType([
    StructField("quote_id", StringType()),
    StructField("policy_id", StringType()),
    StructField("sic_code", StringType()),
    StructField("postcode_sector", StringType()),
    StructField("annual_turnover", LongType()),
    StructField("sum_insured", LongType()),
    StructField("quoted_premium", LongType()),
    StructField("competitor_quoted", StringType()),
    StructField("converted", StringType()),
    StructField("quote_date", StringType()),
])

df_quotes = spark.createDataFrame(quote_rows, schema=quote_schema)
df_quotes.write.mode("overwrite").saveAsTable(f"{fqn}.internal_quote_history")
print(f"✓ {fqn}.internal_quote_history — {df_quotes.count()} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate external CSV files → Volume
# MAGIC These represent data that arrives from external vendors and needs ingestion.

# COMMAND ----------

import csv, os

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

# --- Market Pricing Benchmark ---
market_rows = []
for sic in SIC_CODES:
    for region in ["London","South East","North West","Midlands","Scotland","Wales","South West","East","Yorkshire","North East"]:
        key = f"{sic}_{region}"
        median_rate = round(random.uniform(1.5, 12.0), 2)
        comp_min = round(median_rate * random.uniform(0.6, 0.9), 2)
        trend = round(random.uniform(-8.0, 15.0), 1)
        # Intentional dirty data: some nulls, some bad values
        if random.random() < 0.03:
            median_rate = None  # null
        if random.random() < 0.02:
            trend = 999.9  # out of range
        market_rows.append((key, median_rate, comp_min, trend))

pdf_market = spark.createDataFrame(market_rows, ["match_key_sic_region", "market_median_rate", "competitor_a_min_premium", "price_index_trend"])
pdf_market.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/market_pricing_benchmark")
print(f"✓ market_pricing_benchmark — {len(market_rows)} rows")

# COMMAND ----------

# --- Geospatial Hazard Enrichment ---
geo_rows = []
for pc in POSTCODES:
    flood = random.randint(1, 10)
    fire_dist = round(random.uniform(0.5, 25.0), 1)
    crime = round(random.uniform(10, 95), 1)
    subsidence = round(random.uniform(0, 10), 1)
    # Dirty data: negative distances, nulls, out-of-range flood
    if random.random() < 0.04:
        fire_dist = round(random.uniform(-5, -0.1), 1)  # invalid negative
    if random.random() < 0.03:
        flood = random.randint(11, 15)  # out of range
    if random.random() < 0.03:
        crime = None
    geo_rows.append((pc, flood, fire_dist, crime, subsidence))

pdf_geo = spark.createDataFrame(geo_rows, ["postcode_sector", "flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"])
pdf_geo.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/geospatial_hazard_enrichment")
print(f"✓ geospatial_hazard_enrichment — {len(geo_rows)} rows")

# COMMAND ----------

# --- Credit Bureau Summary ---
# Keyed by a company_id derived from policy data
bureau_rows = []
company_ids_seen = set()
for i in range(NUM_POLICIES):
    cid = f"CMP-{300000 + i}"
    if cid in company_ids_seen:
        continue
    company_ids_seen.add(cid)
    credit_score = random.randint(200, 900)
    ccj_count = random.choice([0,0,0,0,0,0,1,1,2,3,5])
    years_trading = random.randint(0, 80)
    director_changes = random.randint(0, 8)
    # Dirty data: some impossible credit scores, nulls
    if random.random() < 0.02:
        credit_score = random.randint(950, 1100)  # impossible score
    if random.random() < 0.03:
        years_trading = None
    if random.random() < 0.02:
        ccj_count = -1  # invalid

    bureau_rows.append((cid, f"POL-{100000 + i}", credit_score, ccj_count, years_trading, director_changes))

pdf_bureau = spark.createDataFrame(bureau_rows, ["company_id", "policy_id", "credit_score", "ccj_count", "years_trading", "director_changes"])
pdf_bureau.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/credit_bureau_summary")
print(f"✓ credit_bureau_summary — {len(bureau_rows)} rows")

# COMMAND ----------

print(f"""
Setup complete.
  Schema:  {fqn}
  Volume:  {volume_path}

  Internal tables (pre-existing):
    - {fqn}.internal_commercial_policies  (50,000 rows)
    - {fqn}.internal_claims_history       (~50,000 rows)
    - {fqn}.internal_quote_history        (120,000 rows)

  External CSVs in volume (for ingestion):
    - market_pricing_benchmark/
    - geospatial_hazard_enrichment/
    - credit_bureau_summary/
""")
