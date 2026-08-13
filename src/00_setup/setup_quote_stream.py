# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — Quotes (single source of truth)
# MAGIC
# MAGIC Generates the unified quote dataset used by the whole demo:
# MAGIC
# MAGIC ```
# MAGIC Sales channel  →  API  →  Rating Engine  →  API  →  Sales channel
# MAGIC ```
# MAGIC
# MAGIC **Four tables** (all live in the same schema, sort together in Catalog Explorer):
# MAGIC
# MAGIC | Table                            | Rows   | Purpose                                                 |
# MAGIC |----------------------------------|-------:|---------------------------------------------------------|
# MAGIC | `quotes`                         | ~120K  | Canonical flat quote table — one row per transaction     |
# MAGIC | `quote_payload_sales`            | ~1K    | Captured JSON from the sales channel (additive detail)  |
# MAGIC | `quote_payload_engine_request`   | ~1K    | Captured JSON sent to the rating engine                 |
# MAGIC | `quote_payload_engine_response`  | ~0.8K  | Captured JSON returned by the rating engine             |
# MAGIC
# MAGIC `quotes` replaces the previous `internal_quote_history` — everything downstream
# MAGIC (UPT aggregation, demand model, regulatory export, Quote Stream UI) reads from
# MAGIC this single table. The three payload tables are supplementary: they carry the
# MAGIC full JSON for a subset of transactions, keyed on the same `transaction_id`, so
# MAGIC operators can inspect and replay specific cases.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("scale_factor", "1")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
SCALE   = int(dbutils.widgets.get("scale_factor"))
fqn     = f"{catalog}.{schema}"

# Same scale as setup.py — covers all policies, UPT aggregation produces
# meaningful per-policy quote features.
N_QUOTES        = 120_000 * SCALE
N_WITH_PAYLOADS = 1_000           # Subset that gets full 3-JSON capture
DROPOUT_RATE    = 0.17

# Seeded anomalies — these always get full payloads so they're demoable.
OUTLIER_TRANSACTION_IDS = ["TX-BAKERY-48M-2026Q2", "TX-OUTLIER-RETAIL-02"]

# COMMAND ----------

import json
import math
import random
import uuid
from datetime import datetime, timedelta, timezone

from pyspark.sql import Row
from pyspark.sql.types import (
    BooleanType, DoubleType, IntegerType, LongType, StringType,
    StructField, StructType, TimestampType,
)

random.seed(42)

# COMMAND ----------

spark.sql(f"CREATE VOLUME IF NOT EXISTS {fqn}.saved_payloads")
print(f"✓ volume {fqn}.saved_payloads ready")

# Pre-load the policy IDs from internal_commercial_policies so BOUND quotes
# link to real policies. If the table doesn't exist yet, we fall back to a
# deterministic synthetic pool matching the scale used by setup.py.
try:
    policies_df = spark.table(f"{fqn}.internal_commercial_policies").select("policy_id")
    policy_ids = [r["policy_id"] for r in policies_df.collect()]
    print(f"✓ loaded {len(policy_ids):,} policy IDs from {fqn}.internal_commercial_policies")
except Exception as e:
    print(f"⚠ policies table not found ({e}) — falling back to synthetic IDs")
    policy_ids = [f"POL-{100000 + i}" for i in range(50_000 * SCALE)]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reference pools — commercial insurance flavour
# MAGIC Reuses the same postcodes and SIC codes the UPT uses so joins remain possible.

# COMMAND ----------

COMPANY_PREFIXES = ["Ashford", "Bright", "Castle", "Delta", "Eastern", "Foundry", "Grange",
                    "Harbour", "Iron", "Junction", "Kingfield", "Lakeside", "Meridian",
                    "Northstar", "Oakwell", "Pinewood", "Quarry", "Redcliffe", "Summit",
                    "Thornfield", "Uplands", "Vanguard", "Westbridge", "Yardley"]
COMPANY_SUFFIXES = ["Manufacturing Ltd", "Services Ltd", "Trading Ltd", "Holdings Ltd",
                    "Group plc", "Logistics Ltd", "Retail Ltd", "Construction Ltd",
                    "Engineering Ltd", "Hospitality Ltd"]
SIC_DESCRIPTIONS = {
    "1011": ("Food processing",        1.20),
    "2562": ("Machining",              1.30),
    "4110": ("Construction",           1.35),
    "4520": ("Vehicle maintenance",    1.15),
    "4711": ("Retail (general)",       0.95),
    "5610": ("Restaurants",            1.10),
    "6201": ("Computer programming",   0.80),
    "6311": ("Data processing",        0.85),
    "6499": ("Financial services",     0.90),
    "6820": ("Real estate",            0.95),
    "7022": ("Management consulting",  0.85),
    "7112": ("Engineering",            1.00),
    "8010": ("Security services",      1.40),
    "8622": ("Medical practice",       1.05),
    "9311": ("Sports facilities",      1.10),
}
SIC_CODES = list(SIC_DESCRIPTIONS.keys())

# Full POSTCODES match what setup.py builds — regenerated here to avoid a
# cross-notebook import. Keep in sync.
POSTCODE_REGION = {}
_region_prefixes = {
    "London":      ["EC1", "EC2", "SW1", "SE1", "W1", "N1", "E1"],
    "North West":  ["M1", "M2", "L1", "L2"],
    "Midlands":    ["B1", "B2", "NG1"],
    "Yorkshire":   ["LS1", "LS2"],
    "South West":  ["BS1"],
    "Scotland":    ["EH1", "G1"],
    "Wales":       ["CF1"],
}
POSTCODES = []
for region, prefixes in _region_prefixes.items():
    for pfx in prefixes:
        for d in "ABCDE":
            code = f"{pfx}{d}"
            POSTCODES.append(code)
            POSTCODE_REGION[code] = region

POSTCODE_LOADING = {}
for code, region in POSTCODE_REGION.items():
    if region == "London":
        POSTCODE_LOADING[code] = random.uniform(1.20, 1.45)
    elif region in ("North West", "Midlands"):
        POSTCODE_LOADING[code] = random.uniform(0.98, 1.10)
    else:
        POSTCODE_LOADING[code] = random.uniform(0.85, 1.05)

CONSTRUCTION_TYPES = ["Fire Resistive", "Non-Combustible", "Joisted Masonry", "Frame", "Heavy Timber"]
ROOF_TYPES         = ["Metal Deck", "Concrete", "Tiled", "Flat Membrane", "Slated"]
FLOOD_ZONES        = ["Low", "Medium", "High"]
PREVIOUS_INSURERS  = ["Aviva", "Zurich", "Allianz", "RSA", "AIG", "QBE", "Hiscox", "None"]
CHANNELS           = ["Direct", "Broker", "Aggregator", "Renewal"]
SALES_USERS        = ["sales.agent.01", "sales.agent.02", "sales.agent.03",
                      "self_service.portal", "broker.portal.uk"]
MODEL_VERSIONS     = ["pricing_v6.2", "pricing_v6.3", "pricing_v7.0_glm", "pricing_v7.1_glm"]
CURRENT_MODEL      = "pricing_v7.1_glm"

# COMMAND ----------

def random_postcode(area: str) -> str:
    return f"{area} {random.randint(1, 9)}{random.choice('ABDEFGHJLNPQRSTUWXYZ')}{random.choice('ABDEFGHJLNPQRSTUWXYZ')}"


def pricing_factors(sic_loading, pc_loading, construction, year_built,
                    buildings_si, contents_si, liability_si, flood_zone, claims_5y):
    base_building  = buildings_si * 0.0011
    base_contents  = contents_si  * 0.010
    base_liability = liability_si * 0.0004
    base = (base_building + base_contents + base_liability) * pc_loading * sic_loading

    loadings = {}
    if flood_zone == "High":
        loadings["flood_risk"] = round(base * 0.30, 2)
    elif flood_zone == "Medium":
        loadings["flood_risk"] = round(base * 0.10, 2)
    if year_built < 1930:
        loadings["subsidence_age"] = round(base * 0.07, 2)
    if construction in ("Frame", "Heavy Timber"):
        loadings["construction_risk"] = round(base * 0.08, 2)
    if claims_5y > 0:
        loadings["claims_history"] = round(base * 0.14 * claims_5y, 2)

    discounts = {}
    if claims_5y == 0:
        discounts["no_claims"] = round(base * 0.08, 2)
    if random.random() < 0.25:
        discounts["multi_cover"] = round(base * 0.06, 2)

    return {
        "base_building_premium":  round(base_building,  2),
        "base_contents_premium":  round(base_contents,  2),
        "base_liability_premium": round(base_liability, 2),
        "postcode_loading":       pc_loading,
        "sic_loading":            sic_loading,
        "base_premium":           round(base, 2),
        "loadings":               loadings,
        "discounts":              discounts,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate rows
# MAGIC
# MAGIC Pass 1: generate all ~120K flat quote rows (covers all policies for UPT).
# MAGIC Pass 2: for the first N_WITH_PAYLOADS quotes (plus the seeded outliers),
# MAGIC build full JSON payloads.

# COMMAND ----------

base_time = datetime(2026, 4, 1, 8, 0, tzinfo=timezone.utc)

# Pre-compute which quotes get payloads — set membership for O(1) lookup
payload_tx_ids: set = set(OUTLIER_TRANSACTION_IDS)
# The rest: first N_WITH_PAYLOADS - len(outliers) normal transactions
REGULAR_PAYLOAD_COUNT = max(0, N_WITH_PAYLOADS - len(OUTLIER_TRANSACTION_IDS))

flat_rows = []
sales_payload_rows = []
engine_req_rows    = []
engine_resp_rows   = []

n_policies = len(policy_ids)
for i in range(N_QUOTES):
    # Deterministic per-row RNG state (already seeded at module level)
    if i < len(OUTLIER_TRANSACTION_IDS):
        tx_id = OUTLIER_TRANSACTION_IDS[i]
        force_outlier, force_dropout = True, False
    else:
        tx_id = f"TX-{uuid.uuid4().hex[:8].upper()}"
        force_dropout = random.random() < DROPOUT_RATE
        force_outlier = False

    capture_payload = force_outlier or (
        i < N_WITH_PAYLOADS and not force_dropout  # prefer first N for payload capture
    ) or (i < N_WITH_PAYLOADS + len(OUTLIER_TRANSACTION_IDS))

    created_at = base_time - timedelta(minutes=random.randint(0, 365 * 24 * 60))

    company = f"{random.choice(COMPANY_PREFIXES)} {random.choice(COMPANY_SUFFIXES)}"
    sic = random.choice(SIC_CODES)
    sic_desc, sic_loading = SIC_DESCRIPTIONS[sic]
    postcode_sector = random.choice(POSTCODES)   # e.g. "EC1A" — matches UPT key
    region = POSTCODE_REGION[postcode_sector]
    pc_loading = POSTCODE_LOADING[postcode_sector]
    postcode = random_postcode(postcode_sector)  # e.g. "EC1A 3XY" — full postcode
    construction = random.choice(CONSTRUCTION_TYPES)
    year_built = random.randint(1905, 2023)
    flood_zone = random.choices(FLOOD_ZONES, weights=[0.70, 0.22, 0.08])[0]
    claims_5y  = random.choices([0, 1, 2, 3, 5], weights=[0.60, 0.22, 0.10, 0.05, 0.03])[0]
    buildings_si = random.choice([500_000, 1_000_000, 2_000_000, 5_000_000, 10_000_000, 25_000_000])
    contents_si  = random.choice([50_000, 100_000, 250_000, 500_000, 1_000_000])
    liability_si = random.choice([1_000_000, 2_000_000, 5_000_000, 10_000_000])
    sum_insured_total = buildings_si + contents_si     # legacy single SI field used by demand model
    annual_turnover = int(round(random.lognormvariate(13, 1.5)))
    channel = random.choice(CHANNELS)
    agent_user = random.choice(SALES_USERS)
    model_version = random.choice(MODEL_VERSIONS)
    floor_area_sqm = random.randint(150, 15_000)
    roof_type = random.choice(ROOF_TYPES)

    # Price the quote (unless journey was abandoned)
    gross_premium = None
    market_premium = None
    vs_market_rate = None
    quote_status  = "ABANDONED"
    if not force_dropout:
        factors = pricing_factors(sic_loading, pc_loading, construction, year_built,
                                  buildings_si, contents_si, liability_si, flood_zone, claims_5y)
        net = factors["base_premium"] + sum(factors["loadings"].values()) - sum(factors["discounts"].values())
        if force_outlier:
            net = 48_212_560.0
            factors["loadings"]["factor_override_anomaly"] = 48_000_000.0
        ipt = round(net * 0.12, 2)
        gross_premium = round(net + ipt, 2)

        # Market benchmark: what competitors roughly charge for this risk — the
        # technical price with competitive noise. Our offer relative to it drives
        # conversion (PRICE ELASTICITY), so the demand model learns a real,
        # downward-sloping demand curve and the optimiser has an interior optimum.
        market_premium = round(gross_premium * random.lognormvariate(0.0, 0.11), 2)
        vs_market_rate = round(gross_premium / market_premium, 4) if market_premium > 0 else 1.0
        # Logit: ~0.62 convert at market parity, falling as we price above market.
        # Larger accounts a touch more price-sensitive (they shop harder).
        _elasticity = 8.0 if gross_premium < 50_000 else 11.0
        _z = 0.5 - _elasticity * (vs_market_rate - 1.0)
        bind_prob = 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, _z))))
        quote_status = "BOUND" if random.random() < bind_prob else "QUOTED"

    converted = "Y" if quote_status == "BOUND" else "N"
    competitor_quoted = "Y" if random.random() < 0.35 else "N"

    # Link BOUND transactions to real policy_ids. Use index mod pool for even
    # distribution — gives roughly uniform quote coverage across policies when
    # the pool of bound quotes is aggregated.
    policy_id = None
    if quote_status == "BOUND" and n_policies > 0:
        policy_id = policy_ids[i % n_policies]

    has_payload = capture_payload

    flat_rows.append(Row(
        transaction_id=tx_id,
        policy_id=policy_id,
        created_at=created_at,
        channel=channel,
        agent_user=agent_user,
        company_name=company,
        sic_code=sic,
        sic_description=sic_desc,
        postcode=postcode,
        postcode_sector=postcode_sector,
        region=region,
        construction_type=construction,
        year_built=year_built,
        floor_area_sqm=floor_area_sqm,
        flood_zone=flood_zone,
        claims_last_5y=claims_5y,
        buildings_si=buildings_si,
        contents_si=contents_si,
        liability_si=liability_si,
        sum_insured=sum_insured_total,
        annual_turnover=annual_turnover,
        model_version=model_version,
        gross_premium=float(gross_premium) if gross_premium is not None else None,
        market_premium=float(market_premium) if market_premium is not None else None,
        vs_market_rate=float(vs_market_rate) if vs_market_rate is not None else None,
        quote_status=quote_status,
        converted=converted,
        competitor_quoted=competitor_quoted,
        is_outlier=force_outlier,
        has_payload=bool(has_payload),
    ))

    # Only build JSON for the subset with captured payloads
    if not has_payload:
        continue

    sales_payload = {
        "sales_transaction_id": tx_id,
        "timestamp": created_at.isoformat(),
        "agent_user": agent_user,
        "channel": channel,
        "business": {
            "company_name": company,
            "sic_code": sic,
            "sic_description": sic_desc,
            "company_number": f"GB{random.randint(1_000_000, 9_999_999)}",
            "years_trading": random.randint(1, 60),
        },
        "property": {
            "address_line_1": f"{random.randint(1, 250)} {random.choice(['Industrial', 'Business', 'Trade', 'Commerce'])} {random.choice(['Park', 'Estate', 'Way', 'Lane'])}",
            "postcode": postcode,
            "region": region,
            "construction_type": construction,
            "year_built": year_built,
            "roof_type": roof_type,
            "floor_area_sqm": floor_area_sqm,
            "flood_zone": flood_zone,
            "sprinklered": random.random() < 0.35,
            "alarmed": random.random() < 0.75,
        },
        "history": {
            "claims_last_5_years": claims_5y,
            "previous_insurer": random.choice(PREVIOUS_INSURERS),
            "bankruptcy_history": random.random() < 0.02,
        },
        "coverage_requested": {
            "buildings_sum_insured":  buildings_si,
            "contents_sum_insured":   contents_si,
            "public_liability_limit": liability_si,
            "voluntary_excess":       random.choice([1_000, 2_500, 5_000, 10_000]),
            "business_interruption":  random.random() < 0.50,
        },
    }
    sales_payload_rows.append(Row(
        transaction_id=tx_id, created_at=created_at, payload=json.dumps(sales_payload)))

    quote_ref = f"Q-{uuid.uuid4().hex[:10].upper()}"
    engine_request = {
        "quote_reference":      quote_ref,
        "sales_transaction_id": tx_id,
        "model_version":        model_version,
        "timestamp":            created_at.isoformat(),
        "factors": {
            "RATING_FACTOR_POSTCODE_AREA":  postcode_sector,
            "RATING_FACTOR_REGION":         region,
            "RATING_FACTOR_SIC":            sic,
            "RATING_FACTOR_CONSTRUCTION":   construction,
            "RATING_FACTOR_YEARS_BUILT":    datetime.now().year - year_built,
            "RATING_FACTOR_FLOOR_AREA":     floor_area_sqm,
            "RATING_FACTOR_FLOOD_ZONE":     flood_zone,
            "RATING_FACTOR_CLAIMS_5Y":      claims_5y,
            "RATING_FACTOR_SPRINKLERED":    sales_payload["property"]["sprinklered"],
            "RATING_FACTOR_ALARMED":        sales_payload["property"]["alarmed"],
            "RATING_FACTOR_BUILDINGS_SI":   buildings_si,
            "RATING_FACTOR_CONTENTS_SI":    contents_si,
            "RATING_FACTOR_LIABILITY_LIMIT": liability_si,
            "RATING_FACTOR_VOL_EXCESS":     sales_payload["coverage_requested"]["voluntary_excess"],
            "RATING_FACTOR_BUSINESS_INTERRUPTION": sales_payload["coverage_requested"]["business_interruption"],
            "RATING_FACTOR_CHANNEL":        channel,
        },
    }
    engine_req_rows.append(Row(
        transaction_id=tx_id, created_at=created_at, payload=json.dumps(engine_request)))

    # Only non-abandoned journeys get an engine response payload
    if quote_status != "ABANDONED":
        engine_response = {
            "quote_reference":      quote_ref,
            "sales_transaction_id": tx_id,
            "model_version":        model_version,
            "timestamp":            (created_at + timedelta(milliseconds=random.randint(150, 900))).isoformat(),
            "pricing": {
                "base_building_premium":  factors["base_building_premium"],
                "base_contents_premium":  factors["base_contents_premium"],
                "base_liability_premium": factors["base_liability_premium"],
                "postcode_loading":       factors["postcode_loading"],
                "sic_loading":            factors["sic_loading"],
                "base_premium":           factors["base_premium"],
                "loadings":               factors["loadings"],
                "discounts":              factors["discounts"],
                "net_premium":            round(net, 2),
                "ipt":                    ipt,
                "gross_premium":          gross_premium,
            },
            "decision": {
                "status":         "QUOTED",
                "decline_reason": None,
                "quote_expiry":   (created_at + timedelta(days=30)).date().isoformat(),
            },
        }
        engine_resp_rows.append(Row(
            transaction_id=tx_id,
            created_at=datetime.fromisoformat(engine_response["timestamp"]),
            payload=json.dumps(engine_response)))

print(f"""
Generated:
  quotes                         : {len(flat_rows):,} rows
  quote_payload_sales            : {len(sales_payload_rows):,} rows
  quote_payload_engine_request   : {len(engine_req_rows):,} rows
  quote_payload_engine_response  : {len(engine_resp_rows):,} rows
  seeded outliers                : {sum(1 for r in flat_rows if r['is_outlier'])}
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write tables

# COMMAND ----------

flat_schema = StructType([
    StructField("transaction_id",     StringType(), False),
    StructField("policy_id",          StringType()),
    StructField("created_at",         TimestampType(), False),
    StructField("channel",            StringType()),
    StructField("agent_user",         StringType()),
    StructField("company_name",       StringType()),
    StructField("sic_code",           StringType()),
    StructField("sic_description",    StringType()),
    StructField("postcode",           StringType()),
    StructField("postcode_sector",    StringType()),
    StructField("region",             StringType()),
    StructField("construction_type",  StringType()),
    StructField("year_built",         IntegerType()),
    StructField("floor_area_sqm",     IntegerType()),
    StructField("flood_zone",         StringType()),
    StructField("claims_last_5y",     IntegerType()),
    StructField("buildings_si",       LongType()),
    StructField("contents_si",        LongType()),
    StructField("liability_si",       LongType()),
    StructField("sum_insured",        LongType()),
    StructField("annual_turnover",    LongType()),
    StructField("model_version",      StringType()),
    StructField("gross_premium",      DoubleType()),
    StructField("market_premium",     DoubleType()),
    StructField("vs_market_rate",     DoubleType()),
    StructField("quote_status",       StringType()),
    StructField("converted",          StringType()),
    StructField("competitor_quoted",  StringType()),
    StructField("is_outlier",         BooleanType()),
    StructField("has_payload",        BooleanType()),
])
flat_df = spark.createDataFrame(flat_rows, schema=flat_schema)
(flat_df.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{fqn}.quotes"))
print(f"✓ {fqn}.quotes — {flat_df.count():,} rows")

# COMMAND ----------

def _write_payload(rows, table_name):
    schema_ = StructType([
        StructField("transaction_id", StringType(),    False),
        StructField("created_at",     TimestampType(), False),
        StructField("payload",        StringType(),    False),
    ])
    df = spark.createDataFrame(rows, schema=schema_) if rows else spark.createDataFrame([], schema_)
    (df.write.mode("overwrite").option("overwriteSchema", "true")
        .saveAsTable(f"{fqn}.{table_name}"))
    print(f"✓ {fqn}.{table_name} — {df.count():,} rows")


_write_payload(sales_payload_rows, "quote_payload_sales")
_write_payload(engine_req_rows,    "quote_payload_engine_request")
_write_payload(engine_resp_rows,   "quote_payload_engine_response")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Metadata — comments + tags for discoverability

# COMMAND ----------

spark.sql(f"""
    ALTER TABLE {fqn}.quotes SET TBLPROPERTIES (
        'comment' = 'Commercial quote stream — canonical flat table, one row per transaction. Every BOUND quote is linked to a real policy_id. Supersedes the old internal_quote_history. Used by build_upt, the demand model, and the Quote Stream UI. For the subset with has_payload=true, three companion quote_payload_* tables carry the full JSON captured from the sales channel → rating engine → sales channel flow.'
    )
""")

for pt in ("quote_payload_sales", "quote_payload_engine_request", "quote_payload_engine_response"):
    spark.sql(f"""
        ALTER TABLE {fqn}.{pt} SET TBLPROPERTIES (
            'comment' = 'Captured JSON payload for a subset of quotes. Keyed on transaction_id; join to quotes for context.'
        )
    """)

for t in ("quotes", "quote_payload_sales", "quote_payload_engine_request", "quote_payload_engine_response"):
    spark.sql(f"""
        ALTER TABLE {fqn}.{t} SET TAGS (
            'business_line' = 'commercial_property',
            'pricing_domain' = 'quote_stream',
            'refresh_cadence' = 'realtime',
            'demo_environment' = 'true'
        )
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Verify

# COMMAND ----------

display(spark.sql(f"""
    SELECT quote_status, COUNT(*) AS n,
           ROUND(AVG(gross_premium), 2) AS avg_gross,
           ROUND(MAX(gross_premium), 2) AS max_gross,
           COUNT_IF(has_payload) AS with_payload
    FROM {fqn}.quotes
    GROUP BY quote_status
"""))

# COMMAND ----------

display(spark.sql(f"""
    SELECT transaction_id, company_name, postcode, gross_premium, policy_id
    FROM {fqn}.quotes
    WHERE is_outlier = true
"""))
