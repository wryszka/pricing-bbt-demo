# Databricks notebook source
# MAGIC %md
# MAGIC # Autonomous Data Quality Agent
# MAGIC
# MAGIC **Optional.** Monitors incoming data for anomalies using an LLM to reason
# MAGIC about distribution shifts, null spikes, and schema changes — then suggests
# MAGIC remediation actions.
# MAGIC
# MAGIC Unlike rule-based DQ checks (which catch known issues), this agent can
# MAGIC identify unexpected patterns: "turnover values dropped 40% — is this a
# MAGIC data feed error or a real economic downturn?"
# MAGIC
# MAGIC All findings are logged to audit_log with full LLM transparency.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("model_endpoint", "databricks-claude-sonnet-4-6")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
model_endpoint = dbutils.widgets.get("model_endpoint")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

import json
from datetime import datetime, timezone
from databricks.sdk import WorkspaceClient
import pyspark.sql.functions as F
from pyspark.sql.functions import col

w = WorkspaceClient()
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Profile all external datasets — raw vs silver comparison

# COMMAND ----------

datasets = {
    "market_pricing_benchmark": {
        "raw": f"{fqn}.raw_market_pricing_benchmark",
        "silver": f"{fqn}.silver_market_pricing_benchmark",
        "key": "match_key_sic_region",
        "numeric_cols": ["market_median_rate", "competitor_a_min_premium", "price_index_trend"],
    },
    "geospatial_hazard_enrichment": {
        "raw": f"{fqn}.raw_geospatial_hazard_enrichment",
        "silver": f"{fqn}.silver_geospatial_hazard_enrichment",
        "key": "postcode_sector",
        "numeric_cols": ["flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"],
    },
    "credit_bureau_summary": {
        "raw": f"{fqn}.raw_credit_bureau_summary",
        "silver": f"{fqn}.silver_credit_bureau_summary",
        "key": "policy_id",
        "numeric_cols": ["credit_score", "ccj_count", "years_trading", "director_changes"],
    },
}

profiles = {}
for ds_name, ds in datasets.items():
    raw_df = spark.table(ds["raw"])
    silver_df = spark.table(ds["silver"])

    raw_count = raw_df.count()
    silver_count = silver_df.count()
    drop_rate = round((raw_count - silver_count) / raw_count * 100, 1) if raw_count > 0 else 0

    # Null rates and stats for numeric columns
    col_stats = {}
    for c in ds["numeric_cols"]:
        raw_stats = raw_df.agg(
            F.avg(col(c).cast("double")).alias("mean"),
            F.stddev(col(c).cast("double")).alias("std"),
            F.min(col(c).cast("double")).alias("min"),
            F.max(col(c).cast("double")).alias("max"),
            (F.sum(F.when(col(c).isNull(), 1).otherwise(0)) / F.count("*") * 100).alias("null_pct"),
        ).collect()[0]

        col_stats[c] = {
            "mean": round(float(raw_stats["mean"] or 0), 3),
            "std": round(float(raw_stats["std"] or 0), 3),
            "min": round(float(raw_stats["min"] or 0), 3),
            "max": round(float(raw_stats["max"] or 0), 3),
            "null_pct": round(float(raw_stats["null_pct"] or 0), 1),
        }

    profiles[ds_name] = {
        "raw_count": raw_count,
        "silver_count": silver_count,
        "drop_rate": drop_rate,
        "columns": col_stats,
    }
    print(f"{ds_name}: raw={raw_count}, silver={silver_count}, drop={drop_rate}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Build context and call LLM for anomaly detection

# COMMAND ----------

profile_text = "DATA QUALITY PROFILE — External Datasets\n\n"
for ds_name, p in profiles.items():
    profile_text += f"Dataset: {ds_name}\n"
    profile_text += f"  Raw rows: {p['raw_count']}, Silver rows: {p['silver_count']}, DQ drop rate: {p['drop_rate']}%\n"
    for c, s in p["columns"].items():
        profile_text += f"  {c}: mean={s['mean']}, std={s['std']}, min={s['min']}, max={s['max']}, null={s['null_pct']}%\n"
    profile_text += "\n"

system_prompt = """You are an autonomous data quality monitoring agent for a P&C insurance pricing system.
You analyse data profiles and detect anomalies that could affect pricing accuracy.

For each issue found, classify as:
- CRITICAL: Data feed is broken or severely corrupted (e.g. >20% nulls, impossible values)
- WARNING: Significant shift that needs investigation (e.g. mean changed >15%, new patterns)
- INFO: Minor observation worth noting (e.g. slight distribution shift)

Respond with valid JSON:
{
  "findings": [
    {
      "dataset": "dataset name",
      "column": "column name or 'general'",
      "severity": "CRITICAL | WARNING | INFO",
      "finding": "plain English description of the issue",
      "evidence": "the specific numbers that support this finding",
      "suggested_action": "what the data team should do",
      "pricing_impact": "how this could affect pricing if not addressed"
    }
  ],
  "overall_assessment": "1-2 sentence summary of data health",
  "recommended_priority": "which dataset needs attention first and why"
}"""

user_prompt = f"""Analyse these data quality profiles and identify any anomalies, risks, or issues.

{profile_text}

Context: This is commercial property & casualty insurance data. The datasets feed into a
Unified Pricing Table used for model training and live pricing. Any data quality issue
can directly affect premium calculations.

Known DQ rules already in place (DLT expectations):
- Market: median_rate > 0, price_trend between -50 and 50
- Geo: flood 1-10, fire_distance >= 0, crime not null, subsidence 0-10
- Credit: score 200-900, ccj >= 0, years_trading not null

Look for issues BEYOND what the rules catch — distribution shifts, unusual patterns,
cross-column inconsistencies, etc."""

print(f"Calling {model_endpoint}...")
try:
    response = w.serving_endpoints.query(
        name=model_endpoint,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=3000,
        temperature=0.1,
    )
    llm_response = response.choices[0].message.content
    llm_success = True
    print(f"Response received ({len(llm_response)} chars)")
except Exception as e:
    llm_response = f"LLM call failed: {e}"
    llm_success = False
    print(f"LLM call failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Parse and display findings

# COMMAND ----------

findings = None
if llm_success:
    json_text = llm_response
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0]
    try:
        findings = json.loads(json_text.strip())

        print("=" * 60)
        print("DATA QUALITY AGENT FINDINGS")
        print("=" * 60)
        print(f"\nOverall: {findings.get('overall_assessment', '')}")
        print(f"Priority: {findings.get('recommended_priority', '')}\n")

        for f in findings.get("findings", []):
            icon = "🔴" if f["severity"] == "CRITICAL" else ("🟡" if f["severity"] == "WARNING" else "🔵")
            print(f"{icon} [{f['severity']}] {f['dataset']} / {f['column']}")
            print(f"   {f['finding']}")
            print(f"   Evidence: {f['evidence']}")
            print(f"   Action: {f['suggested_action']}")
            print(f"   Impact: {f['pricing_impact']}")
            print()
    except json.JSONDecodeError as e:
        print(f"Failed to parse response: {e}")
        print(llm_response[:1000])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Log to audit trail

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="agent_recommendation",
    entity_type="dataset",
    entity_id="dq_monitor",
    entity_version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
    user_id=user,
    details={
        "agent_type": "dq_monitor",
        "model_endpoint": model_endpoint,
        "llm_success": llm_success,
        "findings_count": len(findings.get("findings", [])) if findings else 0,
        "critical_count": sum(1 for f in (findings or {}).get("findings", []) if f.get("severity") == "CRITICAL"),
        "warning_count": sum(1 for f in (findings or {}).get("findings", []) if f.get("severity") == "WARNING"),
        "overall_assessment": (findings or {}).get("overall_assessment", ""),
        "datasets_profiled": list(profiles.keys()),
    },
    source="notebook",
)
print("✓ DQ agent findings logged to audit_log")
