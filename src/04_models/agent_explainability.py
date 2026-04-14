# Databricks notebook source
# MAGIC %md
# MAGIC # Actuarial Explainability Agent
# MAGIC
# MAGIC **Optional.** Takes a pricing shift and explains WHY it happened in plain
# MAGIC English that an actuary can put directly into a regulatory submission.
# MAGIC
# MAGIC Example: "Premiums in North West increased by 8% — the agent identifies
# MAGIC that this is driven by a 3-point increase in flood risk scores for 15
# MAGIC postcodes, affecting 2,400 policies with £12M GWP."
# MAGIC
# MAGIC Uses the shadow pricing impact data and UPT to trace the causal chain.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("model_endpoint", "databricks-claude-sonnet-4-6")
dbutils.widgets.text("question", "Why did premiums change in the latest data update?")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
model_endpoint = dbutils.widgets.get("model_endpoint")
question = dbutils.widgets.get("question")
fqn = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

import json
from datetime import datetime, timezone
from databricks.sdk import WorkspaceClient
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, lit, round as spark_round

w = WorkspaceClient()
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Gather context from shadow pricing and UPT

# COMMAND ----------

# Portfolio summary
portfolio = spark.table(f"{fqn}.unified_pricing_table_live")
portfolio_summary = portfolio.agg(
    F.count("*").alias("total_policies"),
    F.sum("current_premium").alias("total_gwp"),
    F.avg("current_premium").alias("avg_premium"),
    F.avg("combined_risk_score").alias("avg_risk"),
).collect()[0]

# Shadow pricing impact (if available)
shadow_context = ""
try:
    shadow = spark.table(f"{fqn}.shadow_pricing_impact")
    shadow_summary = shadow.agg(
        F.count("*").alias("affected"),
        F.sum("premium_delta").alias("total_delta"),
        F.avg("premium_delta_pct").alias("avg_delta_pct"),
        F.sum(when(col("churn_risk") == "HIGH", 1).otherwise(0)).alias("high_churn"),
    ).collect()[0]

    by_industry = shadow.groupBy("industry_risk_tier").agg(
        F.count("*").alias("policies"),
        F.sum("premium_delta").alias("delta"),
        F.avg("premium_delta_pct").alias("avg_pct"),
    ).collect()

    by_region = (shadow
        .withColumn("region", F.regexp_extract("postcode_sector", r"^([A-Z]+)", 1))
        .groupBy("region").agg(
            F.count("*").alias("policies"),
            F.sum("premium_delta").alias("delta"),
        ).orderBy(F.abs(col("delta")).desc()).limit(10).collect())

    shadow_context = f"""
SHADOW PRICING IMPACT (latest simulation):
  Affected policies: {shadow_summary.affected:,}
  Total premium delta: £{shadow_summary.total_delta:,.0f}
  Average change: {shadow_summary.avg_delta_pct:.1f}%
  High churn risk: {shadow_summary.high_churn:,}

  By Industry:
"""
    for r in by_industry:
        shadow_context += f"    {r.industry_risk_tier}: {r.policies} policies, delta=£{r.delta:,.0f}, avg={r.avg_pct:.1f}%\n"
    shadow_context += "\n  By Region (top 10 by impact):\n"
    for r in by_region:
        shadow_context += f"    {r.region}: {r.policies} policies, delta=£{r.delta:,.0f}\n"

except Exception as e:
    shadow_context = f"Shadow pricing data not available: {e}\n"

# Data changes (raw vs silver)
data_changes = ""
try:
    for table_name, key in [("geospatial_hazard_enrichment", "postcode_sector"), ("market_pricing_benchmark", "match_key_sic_region")]:
        raw_count = spark.table(f"{fqn}.raw_{table_name}").count()
        silver_count = spark.table(f"{fqn}.silver_{table_name}").count()
        data_changes += f"  {table_name}: raw={raw_count}, silver={silver_count}, dropped={raw_count-silver_count}\n"
except Exception:
    pass

context = f"""PORTFOLIO:
  Total policies: {portfolio_summary.total_policies:,}
  Total GWP: £{portfolio_summary.total_gwp:,.0f}
  Average premium: £{portfolio_summary.avg_premium:,.0f}
  Average risk score: {portfolio_summary.avg_risk:.2f}

{shadow_context}

DATA PIPELINE STATUS:
{data_changes}"""

print(context)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Call LLM for explanation

# COMMAND ----------

system_prompt = """You are an actuarial explainability agent for a P&C insurance company.
When asked about pricing changes, you trace the causal chain from data changes through
to premium impact and explain it in plain English suitable for:
1. An actuary writing a regulatory filing
2. A head of pricing presenting to the board
3. An underwriter making portfolio decisions

Always ground your explanation in specific numbers from the data provided.
Structure your response as valid JSON:
{
  "headline": "one-sentence summary of the key finding",
  "explanation": "2-3 paragraph detailed explanation tracing the causal chain",
  "key_drivers": [
    {"factor": "name", "contribution": "percentage or magnitude", "detail": "explanation"}
  ],
  "affected_segments": [
    {"segment": "name", "policies": N, "premium_impact": "£X or X%"}
  ],
  "regulatory_statement": "one paragraph suitable for a regulatory submission",
  "recommended_actions": ["action 1", "action 2"]
}"""

user_prompt = f"""Question: {question}

Here is the current state of the pricing portfolio and recent changes:

{context}

Please explain the pricing dynamics, identify the root causes, and provide
actionable insights. Ground every claim in specific numbers from the data."""

print(f"Question: {question}")
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
except Exception as e:
    llm_response = f"LLM call failed: {e}"
    llm_success = False

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Display explanation

# COMMAND ----------

explanation = None
if llm_success:
    json_text = llm_response
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0]
    try:
        explanation = json.loads(json_text.strip())

        print("=" * 60)
        print("ACTUARIAL EXPLAINABILITY REPORT")
        print("=" * 60)
        print(f"\n{explanation.get('headline', '')}\n")
        print(explanation.get("explanation", ""))

        print("\nKey Drivers:")
        for d in explanation.get("key_drivers", []):
            print(f"  • {d['factor']} ({d['contribution']}): {d['detail']}")

        print("\nAffected Segments:")
        for s in explanation.get("affected_segments", []):
            print(f"  • {s['segment']}: {s['policies']} policies, {s['premium_impact']}")

        print(f"\nRegulatory Statement:\n  {explanation.get('regulatory_statement', '')}")

        print("\nRecommended Actions:")
        for a in explanation.get("recommended_actions", []):
            print(f"  → {a}")
    except json.JSONDecodeError:
        print(f"Response (not JSON):\n{llm_response[:2000]}")
else:
    print(llm_response)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Log to audit trail

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="agent_recommendation",
    entity_type="model",
    entity_id="explainability_agent",
    entity_version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
    user_id=user,
    details={
        "agent_type": "explainability",
        "model_endpoint": model_endpoint,
        "question": question,
        "llm_success": llm_success,
        "headline": (explanation or {}).get("headline", ""),
        "drivers_count": len((explanation or {}).get("key_drivers", [])),
        "segments_count": len((explanation or {}).get("affected_segments", [])),
    },
    source="notebook",
)
print("✓ Explainability report logged to audit_log")
