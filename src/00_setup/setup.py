# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — Pricing UPT Demo
# MAGIC
# MAGIC Creates the schema, volume, internal tables (policies, claims, quotes),
# MAGIC and external vendor CSVs for the pricing accelerator.
# MAGIC
# MAGIC **SCALE_FACTOR** controls data volume:
# MAGIC - 1 (default): 50K policies — fast demo (~2 min setup)
# MAGIC - 10: 500K policies — realistic mid-market insurer
# MAGIC - 100: 5M policies — large commercial book
# MAGIC
# MAGIC All relationships (claim/policy ratios, quote conversion rates, fraud/churn
# MAGIC prevalence) are maintained at every scale.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("volume_name", "external_landing")
dbutils.widgets.text("scale_factor", "1")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")
SCALE = int(dbutils.widgets.get("scale_factor"))

fqn = f"{catalog}.{schema}"

# Base counts — multiply by SCALE_FACTOR
NUM_POLICIES = 50_000 * SCALE
NUM_QUOTES = 120_000 * SCALE
QUOTE_RATIO = NUM_QUOTES / NUM_POLICIES  # ~2.4 quotes per policy

print(f"Scale factor: {SCALE}x")
print(f"  Policies: {NUM_POLICIES:,}")
print(f"  Quotes:   {NUM_QUOTES:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create schema, volume, and audit log

# COMMAND ----------

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {fqn}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {fqn}.{volume}")

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.audit_log (
        event_id        STRING      COMMENT 'UUID for the event',
        event_type      STRING      COMMENT 'dataset_approved, model_rejected, manual_upload, etc.',
        entity_type     STRING      COMMENT 'dataset, model, feature, endpoint',
        entity_id       STRING      COMMENT 'Identifier of the entity acted upon',
        entity_version  STRING      COMMENT 'Version or snapshot reference',
        user_id         STRING      COMMENT 'Who triggered the event',
        timestamp       TIMESTAMP   COMMENT 'When the event occurred (UTC)',
        details         STRING      COMMENT 'JSON blob with flexible metadata',
        source          STRING      COMMENT 'app, notebook, api'
    )
    COMMENT 'Unified audit trail for all Pricing Workbench events'
""")
print(f"✓ {fqn}.audit_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference data: Postcodes, SIC codes, regions
# MAGIC UK-style postcodes with realistic geographic distribution.
# MAGIC Postcodes are weighted by commercial density (London > regions).

# COMMAND ----------

import pyspark.sql.functions as F
from pyspark.sql.types import *
import random
import math

random.seed(42)

# Postcode areas with approximate commercial density weights
# More policies in London/South East, fewer in Wales/Scotland (realistic)
POSTCODE_AREAS = {
    # London (highest density)
    "EC1": 8, "EC2": 7, "EC3": 6, "EC4": 5,
    "SW1": 6, "SE1": 5, "W1": 7, "N1": 4, "E1": 5, "WC1": 5, "WC2": 4,
    # South East
    "GU1": 3, "RG1": 3, "SL1": 2, "TN1": 2, "BN1": 2, "ME1": 2,
    # Midlands
    "B1": 4, "B2": 3, "CV1": 2, "NG1": 3, "DE1": 2, "LE1": 2, "WV1": 2,
    # North West
    "M1": 5, "M2": 4, "L1": 3, "L2": 2, "WA1": 2, "PR1": 2, "CH1": 2,
    # Yorkshire
    "LS1": 3, "LS2": 2, "BD1": 2, "S1": 3, "HU1": 2, "YO1": 2,
    # North East
    "NE1": 2, "SR1": 1, "TS1": 1, "DH1": 1,
    # Scotland
    "EH1": 3, "G1": 3, "G2": 2, "AB1": 1, "DD1": 1,
    # Wales
    "CF1": 2, "SA1": 1, "NP1": 1, "LL1": 1,
    # South West
    "BS1": 3, "EX1": 1, "BA1": 1, "GL1": 1, "PL1": 1,
    # East
    "CB1": 2, "IP1": 1, "NR1": 1, "CO1": 1,
}

# Expand postcodes with districts A-E
POSTCODES = []
POSTCODE_WEIGHTS = []
for area, weight in POSTCODE_AREAS.items():
    for district in ["A", "B", "C", "D", "E"]:
        POSTCODES.append(f"{area}{district}")
        POSTCODE_WEIGHTS.append(weight)

# Postcode to region mapping
POSTCODE_REGION = {}
REGION_MAP = {
    "London": ["EC1","EC2","EC3","EC4","SW1","SE1","W1","N1","E1","WC1","WC2"],
    "South East": ["GU1","RG1","SL1","TN1","BN1","ME1"],
    "Midlands": ["B1","B2","CV1","NG1","DE1","LE1","WV1"],
    "North West": ["M1","M2","L1","L2","WA1","PR1","CH1"],
    "Yorkshire": ["LS1","LS2","BD1","S1","HU1","YO1"],
    "North East": ["NE1","SR1","TS1","DH1"],
    "Scotland": ["EH1","G1","G2","AB1","DD1"],
    "Wales": ["CF1","SA1","NP1","LL1"],
    "South West": ["BS1","EX1","BA1","GL1","PL1"],
    "East": ["CB1","IP1","NR1","CO1"],
}
for region, areas in REGION_MAP.items():
    for a in areas:
        for d in ["A","B","C","D","E"]:
            POSTCODE_REGION[f"{a}{d}"] = region

REGIONS = list(REGION_MAP.keys())

# SIC codes with industry description and risk tier
SIC_CODES = [
    ("1011", "Food processing", "Medium"),
    ("2562", "Machining", "Medium"),
    ("4110", "Building construction", "High"),
    ("4520", "Vehicle maintenance", "Medium"),
    ("4711", "Retail (non-specialised)", "Low"),
    ("5610", "Restaurants & cafes", "Medium"),
    ("6201", "Computer programming", "Low"),
    ("6311", "Data processing", "Low"),
    ("6499", "Financial services", "Low"),
    ("6820", "Real estate", "Low"),
    ("7022", "Management consultancy", "Low"),
    ("7112", "Engineering activities", "Low"),
    ("8010", "Private security", "High"),
    ("8622", "Medical practice", "Medium"),
    ("9311", "Sports facilities", "Medium"),
]

SIC_CODE_LIST = [s[0] for s in SIC_CODES]
CONSTRUCTION = ["Non-Combustible", "Joisted Masonry", "Fire Resistive", "Frame", "Heavy Timber"]

print(f"Reference data: {len(POSTCODES)} postcodes, {len(SIC_CODES)} SIC codes, {len(REGIONS)} regions")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Internal table 1: Commercial Policies
# MAGIC
# MAGIC Realistic temporal patterns:
# MAGIC - Inception dates spread across 6 years (2020-2025)
# MAGIC - More inceptions in Jan/Apr/Jul/Oct (quarter starts)
# MAGIC - Renewal dates 12 months from inception

# COMMAND ----------

# Seasonal inception weights: more policies start at quarter boundaries
MONTH_WEIGHTS = [15, 8, 8, 12, 8, 8, 12, 8, 8, 12, 8, 8]  # Jan heavy, then quarterly

policy_rows = []
for i in range(NUM_POLICIES):
    pid = f"POL-{100000 + i}"
    sic_idx = random.randint(0, len(SIC_CODES) - 1)
    sic = SIC_CODES[sic_idx][0]
    postcode = random.choices(POSTCODES, weights=POSTCODE_WEIGHTS, k=1)[0]

    # Turnover: lognormal with industry-specific median
    base_turnover = 13.0 if SIC_CODES[sic_idx][2] == "Low" else 12.5
    turnover = round(random.lognormvariate(base_turnover, 1.5))
    construction = random.choice(CONSTRUCTION)
    year_built = random.randint(1920, 2024)
    sum_insured = round(turnover * random.uniform(1.5, 8.0))

    # Claims history: ~35% of policies have claims, skewed by risk tier
    claim_prob = 0.45 if SIC_CODES[sic_idx][2] == "High" else (0.35 if SIC_CODES[sic_idx][2] == "Medium" else 0.25)
    claims_5y = round(max(0, random.lognormvariate(8, 2.5)) if random.random() < claim_prob else 0)

    # Temporal: inception with seasonal pattern
    year = random.choice([2020, 2021, 2022, 2023, 2024, 2025])
    month = random.choices(range(1, 13), weights=MONTH_WEIGHTS, k=1)[0]
    inception = f"{year}-{month:02d}-01"
    renewal_month = month
    renewal = f"2026-{renewal_month:02d}-01"

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
print(f"✓ {fqn}.internal_commercial_policies — {df_policies.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Internal table 2: Claims History
# MAGIC
# MAGIC Seasonal claim patterns: more fire in winter, more flood in autumn,
# MAGIC more theft year-round but peaking in December.

# COMMAND ----------

policies_with_claims = df_policies.filter(F.col("claims_history_5y") > 0).select(
    "policy_id", "sic_code", "postcode_sector", "claims_history_5y"
).collect()

PERILS = ["Fire", "Flood", "Theft", "Liability", "Storm", "Subsidence", "Escape of Water"]
# Seasonal peril weights: [Jan..Dec]
PERIL_SEASON = {
    "Fire": [15,12,10,8,6,5,5,6,8,10,12,16],
    "Flood": [8,8,10,10,8,6,6,8,12,15,12,8],
    "Theft": [10,8,8,8,8,8,8,8,9,10,12,15],
    "Storm": [12,10,8,6,4,3,3,4,6,10,14,16],
}
STATUSES = ["Closed", "Closed", "Closed", "Open", "Open"]

claim_rows = []
claim_id = 1
for row in policies_with_claims:
    n_claims = random.randint(1, 5)
    remaining = row.claims_history_5y
    for j in range(n_claims):
        cid = f"CLM-{claim_id:07d}"
        claim_id += 1

        # Seasonal peril selection
        loss_year = random.randint(2021, 2025)
        loss_month = random.randint(1, 12)
        peril_weights = [PERIL_SEASON.get(p, [8]*12)[loss_month-1] for p in PERILS]
        peril = random.choices(PERILS, weights=peril_weights, k=1)[0]
        status = random.choice(STATUSES)
        loss_date = f"{loss_year}-{loss_month:02d}-{random.randint(1, 28):02d}"

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
print(f"✓ {fqn}.internal_claims_history — {df_claims.count():,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC (Quotes are produced by `setup_quote_stream.py` — single source of truth
# MAGIC for all quote data, both flat rows and captured JSON payloads. No separate
# MAGIC internal quote history is generated here.)

# COMMAND ----------

# MAGIC %md
# MAGIC ## External CSV files → Volume
# MAGIC 4 external vendor datasets, each with intentional dirty data for DQ demos.

# COMMAND ----------

volume_path = f"/Volumes/{catalog}/{schema}/{volume}"

# --- 1. Market Pricing Benchmark ---
# One row per SIC+region combination
market_rows = []
for sic, _, risk in SIC_CODES:
    for region in REGIONS:
        key = f"{sic}_{region}"
        base_rate = 3.0 if risk == "Low" else (6.0 if risk == "Medium" else 10.0)
        median_rate = round(base_rate * random.uniform(0.7, 1.5), 2)
        comp_min = round(median_rate * random.uniform(0.6, 0.9), 2)
        trend = round(random.uniform(-8.0, 15.0), 1)
        # Dirty data
        if random.random() < 0.03: median_rate = None
        if random.random() < 0.02: trend = 999.9
        market_rows.append((key, median_rate, comp_min, trend))

pdf_market = spark.createDataFrame(market_rows, ["match_key_sic_region", "market_median_rate", "competitor_a_min_premium", "price_index_trend"])
pdf_market.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/market_pricing_benchmark")
print(f"✓ market_pricing_benchmark — {len(market_rows)} rows")

# COMMAND ----------

# --- 2. Geospatial Hazard Enrichment ---
# Plausible geographic risk patterns: London flood risk lower (Thames Barrier),
# coastal areas higher flood, northern areas lower subsidence
geo_schema = StructType([
    StructField("postcode_sector", StringType()),
    StructField("flood_zone_rating", IntegerType()),
    StructField("proximity_to_fire_station_km", DoubleType()),
    StructField("crime_theft_index", DoubleType()),
    StructField("subsidence_risk", DoubleType()),
])

geo_rows = []
for pc in POSTCODES:
    region = POSTCODE_REGION.get(pc, "Unknown")
    flood_base = 3 if region in ("London", "East") else (5 if region in ("Yorkshire", "North East") else 4)
    flood = min(10, max(1, flood_base + random.randint(-2, 3)))
    fire_dist = round(random.uniform(0.5, 25.0), 1)
    crime_base = 60 if region == "London" else (45 if region in ("North West", "Midlands") else 35)
    crime = round(max(10.0, min(95.0, crime_base + random.gauss(0, 15))), 1)
    sub_base = 6 if region in ("London", "South East") else (3 if region in ("Scotland", "Wales") else 4)
    subsidence = round(max(0.0, min(10.0, sub_base + random.gauss(0, 2))), 1)
    # Dirty data
    if random.random() < 0.04: fire_dist = round(random.uniform(-5, -0.1), 1)
    if random.random() < 0.03: flood = random.randint(11, 15)
    if random.random() < 0.03: crime = None
    geo_rows.append((pc, flood, fire_dist, crime, subsidence))

pdf_geo = spark.createDataFrame(geo_rows, schema=geo_schema)
pdf_geo.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/geospatial_hazard_enrichment")
print(f"✓ geospatial_hazard_enrichment — {len(geo_rows)} rows")

# COMMAND ----------

# --- 3. Credit Bureau Summary ---
bureau_rows = []
for i in range(NUM_POLICIES):
    cid = f"CMP-{300000 + i}"
    credit_score = random.randint(200, 900)
    ccj_count = random.choice([0,0,0,0,0,0,1,1,2,3,5])
    years_trading = random.randint(0, 80)
    director_changes = random.randint(0, 8)
    if random.random() < 0.02: credit_score = random.randint(950, 1100)
    if random.random() < 0.03: years_trading = None
    if random.random() < 0.02: ccj_count = -1
    bureau_rows.append((cid, f"POL-{100000 + i}", credit_score, ccj_count, years_trading, director_changes))

pdf_bureau = spark.createDataFrame(bureau_rows, ["company_id", "policy_id", "credit_score", "ccj_count", "years_trading", "director_changes"])
pdf_bureau.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/credit_bureau_summary")
print(f"✓ credit_bureau_summary — {len(bureau_rows)} rows")

# COMMAND ----------

# --- 4. Economic Indicators by Region (NEW) ---
# Quarterly economic data used as pricing context
econ_rows = []
for region in REGIONS:
    for year in range(2021, 2026):
        for quarter in range(1, 5):
            gdp_growth = round(random.gauss(1.5, 1.2), 1)
            unemployment = round(max(2.0, random.gauss(4.5 if region != "London" else 3.8, 1.0)), 1)
            inflation = round(max(0.5, random.gauss(4.0, 1.5)), 1)
            construction_index = round(100 + random.gauss(0, 8), 1)
            econ_rows.append((region, year, quarter, gdp_growth, unemployment, inflation, construction_index))

pdf_econ = spark.createDataFrame(econ_rows,
    ["region", "year", "quarter", "gdp_growth_pct", "unemployment_rate_pct", "cpi_inflation_pct", "construction_cost_index"])
pdf_econ.coalesce(1).write.mode("overwrite").option("header", "true").csv(f"{volume_path}/economic_indicators")
print(f"✓ economic_indicators — {len(econ_rows)} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC The synthetic ONS reference dataset has been retired. Real UK public data now
# MAGIC comes from `new_data_impact/00a_build_postcode_enrichment` — run that notebook
# MAGIC (or the `build_postcode_enrichment` bundle job) once to produce the
# MAGIC `postcode_enrichment` table that `derive_factors.py` consumes.

# COMMAND ----------

print(f"""
Setup complete (scale_factor={SCALE}x).
  Schema:  {fqn}
  Volume:  {volume_path}

  Internal tables (pre-existing):
    - {fqn}.internal_commercial_policies  ({NUM_POLICIES:,} rows)
    - {fqn}.internal_claims_history       (~{len(claim_rows):,} rows)

  Run setup_quote_stream next to generate quotes + payload tables.

  External CSVs in volume (for ingestion):
    - market_pricing_benchmark/    ({len(market_rows)} rows)
    - geospatial_hazard_enrichment/ ({len(geo_rows)} rows)
    - credit_bureau_summary/       ({len(bureau_rows):,} rows)
    - economic_indicators/         ({len(econ_rows)} rows)

  Real UK enrichment comes from postcode_enrichment — build via the
  new_data_impact/00a notebook or `databricks bundle run build_postcode_enrichment`.

  Postcodes: {len(POSTCODES)} across {len(REGIONS)} regions
  SIC codes: {len(SIC_CODES)} industries
""")
