# Databricks notebook source
# MAGIC %md
# MAGIC # Live Pricing System — realtime feature refresh
# MAGIC
# MAGIC Files a synthetic claim and recomputes the policy's claim-derived
# MAGIC features in `unified_pricing_table_live`. Continuous online sync
# MAGIC propagates the change into Lakebase within 5-15s, after which a
# MAGIC re-quote returns a higher premium (fraud_pred uses claim history).
# MAGIC
# MAGIC Backed by the FastAPI `/api/live-pricing/claim` endpoint — the
# MAGIC endpoint inserts the claim row and merges UPT inline (per project
# MAGIC decision: snappier demo). This notebook exists to (a) provide a
# MAGIC standalone way to test the claim flow from the workspace, and (b)
# MAGIC act as an async backstop the API can fire-and-forget for any extra
# MAGIC bookkeeping (e.g. recomputing derived risk scores).

# COMMAND ----------

dbutils.widgets.text("catalog_name",      "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",       "pricing_upt")
dbutils.widgets.text("policy_id",         "")
dbutils.widgets.text("claim_amount",      "75000")
dbutils.widgets.text("claim_type",        "ACCIDENTAL_DAMAGE")
dbutils.widgets.text("online_store_name", "pricing-upt-online-store-live")

# COMMAND ----------

import json, time, uuid
from datetime import date, datetime, timezone

catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
policy_id     = dbutils.widgets.get("policy_id").strip().upper()
claim_amount  = float(dbutils.widgets.get("claim_amount"))
claim_type    = dbutils.widgets.get("claim_type")
online_store  = dbutils.widgets.get("online_store_name")
fqn           = f"{catalog}.{schema}"

if not policy_id:
    raise ValueError("policy_id is required")

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %run ../../utils/audit

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verify policy exists

# COMMAND ----------

policy_row = spark.sql(f"""
    SELECT policy_id, current_premium,
           coalesce(claim_count_5y, 0)   AS claim_count_5y,
           coalesce(total_incurred_5y, 0) AS total_incurred_5y,
           coalesce(total_paid_5y, 0)     AS total_paid_5y,
           coalesce(open_claims_count, 0) AS open_claims_count
    FROM {fqn}.unified_pricing_table_live
    WHERE policy_id = '{policy_id}'
    LIMIT 1
""").collect()
if not policy_row:
    raise ValueError(f"policy {policy_id} not found in unified_pricing_table_live")
before = policy_row[0].asDict()
print(f"before: {before}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Append claim row to internal_claims_history
# MAGIC
# MAGIC Claim peril mapping mirrors the production claims handler — the
# MAGIC fraud model uses `distinct_perils` from this table.

# COMMAND ----------

claim_id  = f"CLM-{datetime.now(timezone.utc).strftime('%Y%m%d')}-{uuid.uuid4().hex[:8].upper()}"
loss_date = date.today().isoformat()
peril     = {
    "ACCIDENTAL_DAMAGE": "Other",
    "FIRE":              "Fire",
    "FLOOD":             "Flood",
    "THEFT":             "Theft",
    "STORM":             "Storm",
    "SUBSIDENCE":        "Subsidence",
    "WATER":             "Escape of Water",
}.get(claim_type, "Other")

t0 = time.perf_counter()
spark.sql(f"""
    INSERT INTO {fqn}.internal_claims_history (
        claim_id, policy_id, peril, incurred_amount, paid_amount, reserve,
        loss_date, status
    ) VALUES (
        '{claim_id}', '{policy_id}', '{peril}',
        {int(claim_amount)}, {int(claim_amount * 0.5)}, {int(claim_amount * 0.5)},
        '{loss_date}', 'Open'
    )
""")
claim_write_ms = (time.perf_counter() - t0) * 1000
print(f"claim {claim_id} inserted in {claim_write_ms:.0f}ms")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. MERGE updated claim aggregates into UPT
# MAGIC
# MAGIC Re-aggregates from the (now updated) claims history rather than
# MAGIC trusting incremental arithmetic — single source of truth.

# COMMAND ----------

t0 = time.perf_counter()
spark.sql(f"""
    MERGE INTO {fqn}.unified_pricing_table_live target
    USING (
        SELECT
            policy_id,
            COUNT(*)                                          AS claim_count_5y,
            SUM(incurred_amount)                              AS total_incurred_5y,
            SUM(paid_amount)                                  AS total_paid_5y,
            SUM(CASE WHEN status='Open' THEN 1 ELSE 0 END)    AS open_claims_count,
            COUNT(DISTINCT peril)                             AS distinct_perils
        FROM {fqn}.internal_claims_history
        WHERE policy_id = '{policy_id}'
        GROUP BY policy_id
    ) src
    ON target.policy_id = src.policy_id
    WHEN MATCHED THEN UPDATE SET
        target.claim_count_5y     = src.claim_count_5y,
        target.total_incurred_5y  = src.total_incurred_5y,
        target.total_paid_5y      = src.total_paid_5y,
        target.open_claims_count  = src.open_claims_count,
        target.distinct_perils    = src.distinct_perils,
        target.loss_ratio_5y      = ROUND(src.total_incurred_5y /
                                          (target.current_premium * 5), 3)
""")
upt_merge_ms = (time.perf_counter() - t0) * 1000
print(f"UPT merge in {upt_merge_ms:.0f}ms")

# COMMAND ----------

after = spark.sql(f"""
    SELECT claim_count_5y, total_incurred_5y, loss_ratio_5y, open_claims_count
    FROM {fqn}.unified_pricing_table_live WHERE policy_id = '{policy_id}'
""").collect()[0].asDict()
print(f"after: {after}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Audit the claim filing
# MAGIC
# MAGIC Continuous online sync pushes the merged UPT row to Lakebase — no
# MAGIC explicit `publish_table` required.

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type    = "live_pricing_claim_filed",
    entity_type   = "policy",
    entity_id     = policy_id,
    entity_version= "",
    user_id       = user,
    details={
        "claim_id":         claim_id,
        "claim_type":       claim_type,
        "peril":            peril,
        "claim_amount":     claim_amount,
        "before":           {k: float(v) if v is not None else None for k, v in before.items() if k != "policy_id"},
        "after":            {k: float(v) if v is not None else None for k, v in after.items()},
        "claim_write_ms":   round(claim_write_ms, 1),
        "upt_merge_ms":     round(upt_merge_ms, 1),
        "online_store":     online_store,
        "publish_mode":     "CONTINUOUS",
    },
    source="notebook",
)

dbutils.notebook.exit(json.dumps({
    "claim_id":       claim_id,
    "policy_id":      policy_id,
    "claim_amount":   claim_amount,
    "claim_write_ms": round(claim_write_ms, 1),
    "upt_merge_ms":   round(upt_merge_ms, 1),
    "total_ms":       round(claim_write_ms + upt_merge_ms, 1),
    "before":         {k: float(v) if v is not None else None for k, v in before.items() if k != "policy_id"},
    "after":          {k: float(v) if v is not None else None for k, v in after.items()},
}))
