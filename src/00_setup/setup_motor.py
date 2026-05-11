# Databricks notebook source
# MAGIC %md
# MAGIC # Setup — Motor Insurance dataset (Live Serving demo)
# MAGIC
# MAGIC Sits alongside the commercial book. Generates a 1M-policy motor book
# MAGIC with telematics features and claims history, designed to feed the
# MAGIC live-serving demo: "John, 19yo telematics driver, gets a quote in
# MAGIC milliseconds; a single black-box event flows through and changes his
# MAGIC price within minutes."
# MAGIC
# MAGIC Tables created:
# MAGIC   motor_policies                  — 1M rows, demographic + vehicle
# MAGIC   motor_telematics_aggregate      — 1M rows, per-policy behaviour snapshot
# MAGIC   motor_claims_history            — ~300k rows synthetic claims
# MAGIC
# MAGIC John lives at `POL-MOTOR-00000001`. He's deterministically seeded with
# MAGIC a clean record + behaviour_score 75 so the demo's first quote is
# MAGIC reproducible.
# MAGIC
# MAGIC Idempotent — runs in ~3-5 min on a small serverless warehouse.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("num_policies", "1000000")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
N       = int(dbutils.widgets.get("num_policies"))
fqn     = f"{catalog}.{schema}"

import pyspark.sql.functions as F
from pyspark.sql.types import StringType, IntegerType, DoubleType, DateType

print(f"Target: {fqn}  N policies: {N:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## motor_policies

# COMMAND ----------

# Reference distributions
VEHICLES = [
    # (make, model, group, base_value)
    ("Ford",       "Fiesta",    8,   8500),
    ("Vauxhall",   "Corsa",     7,   8000),
    ("VW",         "Polo",     10,  10000),
    ("Ford",       "Focus",    14,  14000),
    ("Toyota",     "Yaris",     9,   9500),
    ("Mini",       "Cooper",   18,  18000),
    ("BMW",        "1 Series", 24,  24000),
    ("BMW",        "3 Series", 30,  32000),
    ("Audi",       "A3",       22,  22000),
    ("Audi",       "A4",       28,  29000),
    ("Mercedes",   "A-Class",  23,  24000),
    ("Mercedes",   "C-Class",  31,  35000),
    ("Tesla",      "Model 3",  40,  42000),
    ("Range Rover","Evoque",   38,  45000),
    ("Honda",      "Civic",    13,  14500),
    ("Nissan",     "Qashqai",  16,  18500),
    ("Hyundai",    "i30",      11,  13000),
    ("Kia",        "Sportage", 15,  19000),
    ("Peugeot",    "208",       8,   9000),
    ("Renault",    "Clio",      9,   9800),
]
REGIONS = [
    ("London",      0.18),
    ("South East",  0.16),
    ("Midlands",    0.15),
    ("North West",  0.12),
    ("Yorkshire",   0.10),
    ("Scotland",    0.09),
    ("South West",  0.08),
    ("Wales",       0.05),
    ("East",        0.04),
    ("North East",  0.03),
]
POSTCODE_AREAS = ["EC1","EC2","SW1","N1","E1","M1","L1","B1","LS1","S1","NE1",
                  "EH1","G1","CF1","BS1","CB1","NR1","BN1","GU1","RG1"]

# 1M-row driver/vehicle generation via spark.range + deterministic hashing
df = spark.range(1, N + 1).withColumn("policy_id",
    F.format_string("POL-MOTOR-%08d", F.col("id")))

# Random uniforms (deterministic per id)
def U(name): return (F.abs(F.hash(F.col("policy_id"), F.lit(name))) % 10_000_000) / 10_000_000.0

# Driver age — skewed young (the synthetic book has lots of new drivers, mirrors a telematics-heavy book)
df = df.withColumn("driver_age",
    F.when(U("age") < 0.05, F.lit(17))    # learner-graduates
     .when(U("age") < 0.20, (F.rand() * 5 + 18).cast("int"))      # 18-22
     .when(U("age") < 0.45, (F.rand() * 10 + 23).cast("int"))     # 23-32
     .when(U("age") < 0.75, (F.rand() * 15 + 33).cast("int"))     # 33-47
     .when(U("age") < 0.92, (F.rand() * 13 + 48).cast("int"))     # 48-60
     .otherwise((F.rand() * 15 + 61).cast("int"))                  # 61-75
)
df = df.withColumn("license_years_held",
    F.greatest(F.lit(0), F.least(F.col("driver_age") - 17, (U("lic") * 30).cast("int")))
)
df = df.withColumn("no_claims_years",
    F.greatest(F.lit(0), F.least(F.col("license_years_held"),
        F.when(U("ncd") < 0.10, F.lit(0))                          # 10% have just claimed
         .when(U("ncd") < 0.30, (U("ncd2") * 3).cast("int"))      # 0-3 yrs
         .when(U("ncd") < 0.70, (U("ncd2") * 7 + 3).cast("int"))  # 3-10 yrs
         .otherwise(F.lit(15)))                                    # 10+ capped at 15
    )
)
df = df.withColumn("gender",
    F.when(U("gen") < 0.52, "M").otherwise("F"))
df = df.withColumn("marital_status",
    F.when(F.col("driver_age") < 25, "Single")
     .when(U("ms") < 0.55, "Married")
     .when(U("ms") < 0.85, "Single")
     .otherwise("Divorced"))
df = df.withColumn("occupation_class",
    F.when(U("occ") < 0.15, "Professional")
     .when(U("occ") < 0.45, "Office")
     .when(U("occ") < 0.65, "Skilled Manual")
     .when(U("occ") < 0.80, "Service")
     .when(U("occ") < 0.92, "Student")
     .otherwise("Self-Employed"))

# Geography
df = df.withColumn("postcode_area",
    F.element_at(F.array(*[F.lit(p) for p in POSTCODE_AREAS]),
                 ((F.abs(F.hash(F.col("policy_id"), F.lit("pc"))) % len(POSTCODE_AREAS)) + 1).cast("int")))
df = df.withColumn("region",
    F.when(F.col("postcode_area").isin("EC1","EC2","SW1","N1","E1"), "London")
     .when(F.col("postcode_area").isin("M1","L1"), "North West")
     .when(F.col("postcode_area").isin("B1"), "Midlands")
     .when(F.col("postcode_area").isin("LS1","S1"), "Yorkshire")
     .when(F.col("postcode_area").isin("NE1"), "North East")
     .when(F.col("postcode_area").isin("EH1","G1"), "Scotland")
     .when(F.col("postcode_area").isin("CF1"), "Wales")
     .when(F.col("postcode_area").isin("BS1"), "South West")
     .when(F.col("postcode_area").isin("CB1","NR1"), "East")
     .otherwise("South East"))

# Vehicle — pick deterministically; tweak value by age within ±25%
veh_arrays = {
    "make":   F.array(*[F.lit(v[0]) for v in VEHICLES]),
    "model":  F.array(*[F.lit(v[1]) for v in VEHICLES]),
    "group":  F.array(*[F.lit(v[2]) for v in VEHICLES]),
    "value":  F.array(*[F.lit(v[3]) for v in VEHICLES]),
}
veh_idx = ((F.abs(F.hash(F.col("policy_id"), F.lit("veh"))) % len(VEHICLES)) + 1).cast("int")
df = df.withColumn("vehicle_make",  F.element_at(veh_arrays["make"],  veh_idx)) \
       .withColumn("vehicle_model", F.element_at(veh_arrays["model"], veh_idx)) \
       .withColumn("vehicle_group", F.element_at(veh_arrays["group"], veh_idx)) \
       .withColumn("vehicle_value", F.element_at(veh_arrays["value"], veh_idx) *
                                     (F.lit(0.75) + U("vehv") * F.lit(0.5)))
# Vehicle year between 2010 and 2025, biased toward 2015-2023
df = df.withColumn("vehicle_year",
    F.lit(2010) + ((U("vyr") * 16) + (U("vyr2") * 6)).cast("int") % F.lit(16))
df = df.withColumn("fuel_type",
    F.when(U("fuel") < 0.55, "Petrol")
     .when(U("fuel") < 0.85, "Diesel")
     .when(U("fuel") < 0.95, "Hybrid")
     .otherwise("Electric"))

# Usage
df = df.withColumn("annual_mileage",
    (F.lit(4000) + U("mil") * F.lit(20000)).cast("int"))
df = df.withColumn("parking_overnight",
    F.when(U("park") < 0.40, "Garage")
     .when(U("park") < 0.75, "Driveway")
     .otherwise("Street"))
df = df.withColumn("business_use", F.when(U("biz") < 0.12, "Y").otherwise("N"))

# Risk history (independent of claims_history table — proxies pre-policy record)
df = df.withColumn("prior_convictions",
    F.when(U("conv") < 0.92, F.lit(0))
     .when(U("conv") < 0.98, F.lit(1))
     .otherwise(F.lit(2)))
df = df.withColumn("prior_accidents_5y",
    F.when(U("pa") < 0.85, F.lit(0))
     .when(U("pa") < 0.95, F.lit(1))
     .otherwise(F.lit(2)))

# Premium — synthesized from features so it correlates with future model predictions
base = F.lit(400)
age_factor = F.when(F.col("driver_age") < 21, F.lit(2.5)) \
              .when(F.col("driver_age") < 25, F.lit(1.7)) \
              .when(F.col("driver_age") < 30, F.lit(1.2)) \
              .when(F.col("driver_age") < 60, F.lit(0.85)) \
              .when(F.col("driver_age") < 70, F.lit(1.0)) \
              .otherwise(F.lit(1.4))
ncd_factor = F.greatest(F.lit(0.55), F.lit(1.0) - F.col("no_claims_years") * F.lit(0.04))
veh_factor = F.lit(0.6) + (F.col("vehicle_group") / F.lit(40))
mileage_factor = F.lit(0.85) + (F.col("annual_mileage") / F.lit(60000))
prior_factor = F.lit(1.0) + (F.col("prior_convictions") + F.col("prior_accidents_5y")) * F.lit(0.20)

df = df.withColumn("current_premium",
    F.round(base * age_factor * ncd_factor * veh_factor * mileage_factor * prior_factor, 0))

# Inception / renewal dates — staggered over the last 12 months
df = df.withColumn("inception_date",
    F.expr("date_add(current_date(), -cast(rand() * 365 as int))"))
df = df.withColumn("renewal_date",
    F.expr("date_add(inception_date, 365)"))

# Drop the id helper, write
df = df.drop("id")
print(f"motor_policies: {df.count():,} rows")
df.write.mode("overwrite").option("overwriteSchema","true") \
  .saveAsTable(f"{fqn}.motor_policies")

# Primary key for feature store / online sync
spark.sql(f"ALTER TABLE {fqn}.motor_policies ALTER COLUMN policy_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {fqn}.motor_policies ADD CONSTRAINT motor_policies_pk PRIMARY KEY (policy_id)")
except Exception as e:
    if "already exists" not in str(e).lower(): raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## motor_telematics_aggregate
# MAGIC
# MAGIC Per-policy black-box snapshot. The demo's "simulate event" button
# MAGIC mutates John's row here; everything else stays static.

# COMMAND ----------

tdf = spark.range(1, N + 1).withColumn("policy_id",
    F.format_string("POL-MOTOR-%08d", F.col("id")))

def Ut(name): return (F.abs(F.hash(F.col("policy_id"), F.lit("t_" + name))) % 10_000_000) / 10_000_000.0

# Behaviour score skews high (most drivers are decent), with a tail of risky drivers
tdf = tdf.withColumn("behaviour_score",
    F.when(Ut("bs") < 0.10, (F.lit(30) + Ut("bs2") * F.lit(25)).cast("int"))    # 30-55  risky
     .when(Ut("bs") < 0.30, (F.lit(55) + Ut("bs2") * F.lit(15)).cast("int"))    # 55-70
     .when(Ut("bs") < 0.65, (F.lit(70) + Ut("bs2") * F.lit(15)).cast("int"))    # 70-85
     .otherwise((F.lit(85) + Ut("bs2") * F.lit(13)).cast("int")))               # 85-98

tdf = tdf.withColumn("avg_speed_mph",        F.round(F.lit(28) + Ut("sp") * F.lit(15), 1))
tdf = tdf.withColumn("hours_driven_30d",     F.round(F.lit(10) + Ut("hr") * F.lit(60), 1))
tdf = tdf.withColumn("night_driving_pct",
    F.when(F.col("behaviour_score") > 80, F.round(Ut("nd") * F.lit(15), 1))
     .otherwise(F.round(F.lit(10) + Ut("nd") * F.lit(35), 1)))

# Recent events skew with behaviour score (poor score → more events)
tdf = tdf.withColumn("recent_speeding_events",
    F.when(F.col("behaviour_score") < 50, (Ut("sp_e") * F.lit(6)).cast("int"))
     .when(F.col("behaviour_score") < 70, (Ut("sp_e") * F.lit(3)).cast("int"))
     .when(F.col("behaviour_score") < 85, (Ut("sp_e") * F.lit(1.5)).cast("int"))
     .otherwise(F.lit(0)))
tdf = tdf.withColumn("recent_curfew_breaches",
    F.when(F.col("behaviour_score") < 50, (Ut("cb") * F.lit(4)).cast("int"))
     .when(F.col("behaviour_score") < 70, (Ut("cb") * F.lit(2)).cast("int"))
     .otherwise(F.lit(0)))
tdf = tdf.withColumn("recent_harsh_braking_30d",
    F.when(F.col("behaviour_score") < 50, (Ut("hb") * F.lit(15)).cast("int"))
     .when(F.col("behaviour_score") < 70, (Ut("hb") * F.lit(8)).cast("int"))
     .otherwise((Ut("hb") * F.lit(3)).cast("int")))

tdf = tdf.withColumn("total_miles_recorded", (Ut("tm") * F.lit(25000) + F.lit(2000)).cast("int"))
tdf = tdf.withColumn("days_since_install",   (Ut("ds") * F.lit(720) + F.lit(30)).cast("int"))
tdf = tdf.drop("id")

print(f"motor_telematics_aggregate: {tdf.count():,} rows")
tdf.write.mode("overwrite").option("overwriteSchema","true") \
   .saveAsTable(f"{fqn}.motor_telematics_aggregate")
spark.sql(f"ALTER TABLE {fqn}.motor_telematics_aggregate ALTER COLUMN policy_id SET NOT NULL")
try:
    spark.sql(f"ALTER TABLE {fqn}.motor_telematics_aggregate ADD CONSTRAINT motor_telematics_pk PRIMARY KEY (policy_id)")
except Exception as e:
    if "already exists" not in str(e).lower(): raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## motor_claims_history
# MAGIC
# MAGIC ~30% of policies have claims, with 1-3 claims each → ~400k rows.

# COMMAND ----------

# Sample policy_ids that have claims (deterministic by hash)
claimants = spark.table(f"{fqn}.motor_policies") \
    .select("policy_id", "driver_age") \
    .withColumn("_h", (F.abs(F.hash(F.col("policy_id"), F.lit("clm"))) % 100)) \
    .filter(F.col("_h") < 30) \
    .withColumn("n_claims",
        F.when(F.col("_h") < 5, F.lit(3))
         .when(F.col("_h") < 15, F.lit(2))
         .otherwise(F.lit(1)))

# Explode 1 row per claim
claims = claimants.select("policy_id", F.explode(F.expr("sequence(1, n_claims)")).alias("claim_idx"))

# Synthetic claim attributes
def Uc(name): return (F.abs(F.hash(F.col("policy_id"), F.col("claim_idx").cast("string"), F.lit(name))) % 10_000_000) / 10_000_000.0

claims = claims.withColumn("claim_id",
    F.concat(F.lit("CLM-MOTOR-"), F.col("policy_id"), F.lit("-"), F.col("claim_idx")))
claims = claims.withColumn("peril",
    F.when(Uc("pl") < 0.45, "Collision")
     .when(Uc("pl") < 0.65, "Theft")
     .when(Uc("pl") < 0.80, "Vandalism")
     .when(Uc("pl") < 0.92, "Third Party")
     .otherwise("Fire"))
claims = claims.withColumn("fault_indicator",
    F.when(F.col("peril").isin("Theft","Vandalism","Fire"), F.lit("Not at fault"))
     .when(Uc("flt") < 0.55, "At fault")
     .otherwise("Not at fault"))
claims = claims.withColumn("incurred_amount",
    F.round(F.lit(500) + Uc("inc") * F.lit(15000), 0))
claims = claims.withColumn("paid_amount",
    F.round(F.col("incurred_amount") * (F.lit(0.5) + Uc("pd") * F.lit(0.45)), 0))
claims = claims.withColumn("reserve",
    F.greatest(F.lit(0), F.col("incurred_amount") - F.col("paid_amount")))
claims = claims.withColumn("status",
    F.when(Uc("st") < 0.75, "Closed").otherwise("Open"))
claims = claims.withColumn("loss_date",
    F.expr("date_add(current_date(), -cast(rand() * 1825 as int))"))  # within 5y
claims = claims.drop("claim_idx")

print(f"motor_claims_history: {claims.count():,} rows")
claims.write.mode("overwrite").option("overwriteSchema","true") \
      .saveAsTable(f"{fqn}.motor_claims_history")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pre-seed John (POL-MOTOR-00000001)
# MAGIC
# MAGIC Override the synthetic values so the demo's first quote is reproducible:
# MAGIC John is 19, 1 year licence held, 0 NCD, drives a £8k Corsa, clean record,
# MAGIC behaviour_score 75, no recent telematics events. Premium around £1,800.

# COMMAND ----------

spark.sql(f"""
    UPDATE {fqn}.motor_policies SET
        driver_age = 19,
        license_years_held = 1,
        no_claims_years = 0,
        gender = 'M',
        marital_status = 'Single',
        occupation_class = 'Student',
        postcode_area = 'M1',
        region = 'North West',
        vehicle_make = 'Vauxhall',
        vehicle_model = 'Corsa',
        vehicle_group = 7,
        vehicle_value = 8200.0,
        vehicle_year = 2018,
        fuel_type = 'Petrol',
        annual_mileage = 8500,
        parking_overnight = 'Driveway',
        business_use = 'N',
        prior_convictions = 0,
        prior_accidents_5y = 0,
        current_premium = 1847
    WHERE policy_id = 'POL-MOTOR-00000001'
""")

spark.sql(f"""
    UPDATE {fqn}.motor_telematics_aggregate SET
        behaviour_score = 75,
        avg_speed_mph = 31.2,
        hours_driven_30d = 28.5,
        night_driving_pct = 12.0,
        recent_speeding_events = 0,
        recent_curfew_breaches = 0,
        recent_harsh_braking_30d = 2,
        total_miles_recorded = 9500,
        days_since_install = 365
    WHERE policy_id = 'POL-MOTOR-00000001'
""")

# Make sure John has no inflight claims
spark.sql(f"DELETE FROM {fqn}.motor_claims_history WHERE policy_id = 'POL-MOTOR-00000001'")

print("John seeded.")
print(spark.sql(f"""
    SELECT p.driver_age, p.vehicle_make, p.vehicle_model, p.current_premium,
           t.behaviour_score, t.recent_speeding_events, t.recent_curfew_breaches
    FROM {fqn}.motor_policies p
    JOIN {fqn}.motor_telematics_aggregate t USING (policy_id)
    WHERE p.policy_id = 'POL-MOTOR-00000001'
""").collect())

# COMMAND ----------

import json
dbutils.notebook.exit(json.dumps({
    "motor_policies": N,
    "motor_telematics_aggregate": N,
    "motor_claims_history": "~30% of policies",
    "john_policy_id": "POL-MOTOR-00000001",
}))
