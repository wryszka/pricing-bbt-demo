# Databricks notebook source
# MAGIC %md
# MAGIC # AI Model Selection Agent (Optional)
# MAGIC
# MAGIC **This is optional.** The demo works perfectly without this agent.
# MAGIC It exists to demonstrate the art of the possible with AI-assisted
# MAGIC model governance in a regulated pricing context.
# MAGIC
# MAGIC The agent inspects the Unified Pricing Table and recommends which
# MAGIC models to train, which features to use, and explains its reasoning.
# MAGIC All LLM interactions are fully logged and auditable.

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

w = WorkspaceClient()
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Profile the Unified Pricing Table

# COMMAND ----------

upt = spark.table(f"{fqn}.unified_pricing_table_live")
row_count = upt.count()
col_count = len(upt.columns)

# Get column types and basic stats
schema_info = []
for field in upt.schema.fields:
    schema_info.append({
        "name": field.name,
        "type": str(field.dataType).replace("Type()", "").lower(),
    })

# Compute summary stats for numeric columns
import pyspark.sql.functions as F

numeric_cols = [f["name"] for f in schema_info if f["type"] in ("double", "long", "int", "float")][:30]
stats_aggs = []
for c in numeric_cols:
    stats_aggs.extend([
        F.round(F.avg(F.col(c).cast("double")), 3).alias(f"{c}_mean"),
        F.round(F.stddev(F.col(c).cast("double")), 3).alias(f"{c}_std"),
        F.round(F.count(F.when(F.col(c).isNull(), 1)) / F.count("*") * 100, 1).alias(f"{c}_null_pct"),
    ])

stats = upt.agg(*stats_aggs).collect()[0]

profile_text = f"Table: {fqn}.unified_pricing_table_live\n"
profile_text += f"Rows: {row_count:,} | Columns: {col_count}\n\n"
profile_text += "Column profiles (name | type | mean | std | null%):\n"
for c in numeric_cols:
    mean = stats[f"{c}_mean"]
    std = stats[f"{c}_std"]
    null_pct = stats[f"{c}_null_pct"]
    profile_text += f"  {c} | numeric | mean={mean} | std={std} | null={null_pct}%\n"

# Add categorical columns
cat_cols = [f["name"] for f in schema_info if f["type"] == "string"][:10]
for c in cat_cols:
    ndistinct = upt.select(c).distinct().count()
    profile_text += f"  {c} | string | distinct={ndistinct}\n"

print(profile_text)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Build the prompt and call the LLM

# COMMAND ----------

system_prompt = """You are an expert actuarial pricing AI advisor for a P&C insurance company.
You are analysing a Unified Pricing Table (wide denormalized feature table) to recommend
which pricing models should be trained.

CONTEXT: Commercial property & casualty insurance. The table contains policy data, claims
history, market benchmarks, geospatial risk scores, credit bureau data, and synthetic features.

RESPONSE FORMAT: Return valid JSON with this exact structure:
{
  "recommendations": [
    {
      "model_name": "string - descriptive name",
      "model_type": "GLM_Poisson | GLM_Gamma | GBM_Classifier | GBM_Regressor",
      "target_variable": "column name from the table",
      "purpose": "what this model predicts and why it matters for pricing",
      "recommended_features": ["list of column names to use as features"],
      "feature_rationale": "why these features were selected",
      "regulatory_notes": "considerations for regulatory approval",
      "priority": "high | medium | low"
    }
  ],
  "data_quality_observations": ["list of observations about the data"],
  "overall_strategy": "plain English explanation of the recommended modelling strategy"
}"""

user_prompt = f"""Analyse this pricing feature table and recommend which models to train.

{profile_text}

Available target variables for pricing:
- claim_count_5y (integer): Number of claims in 5 years — use for FREQUENCY modelling
- total_incurred_5y (numeric): Total claim cost — use for SEVERITY modelling
- loss_ratio_5y (numeric): Loss ratio — alternative severity target
- converted (in quote table): Conversion flag — use for DEMAND modelling

Requirements:
1. Recommend at least one frequency model (GLM preferred for regulatory)
2. Recommend at least one severity model
3. Recommend at least one demand/conversion model
4. Consider whether a GBM uplift model on GLM residuals would add value
5. For each model, select appropriate features from the table (max 25)
6. Flag any data quality concerns

The models must be suitable for UK/European regulatory submission (Solvency II context)."""

print(f"Calling {model_endpoint}...")

try:
    response = w.serving_endpoints.query(
        name=model_endpoint,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=4000,
        temperature=0.1,
    )
    llm_response = response.choices[0].message.content
    llm_success = True
    print(f"LLM response received ({len(llm_response)} chars)")
except Exception as e:
    llm_response = f"LLM call failed: {e}"
    llm_success = False
    print(f"LLM call failed: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Parse and display recommendations

# COMMAND ----------

if llm_success:
    # Extract JSON from response (handle markdown code blocks)
    json_text = llm_response
    if "```json" in json_text:
        json_text = json_text.split("```json")[1].split("```")[0]
    elif "```" in json_text:
        json_text = json_text.split("```")[1].split("```")[0]

    try:
        recommendations = json.loads(json_text.strip())
        print("=" * 60)
        print("AI MODEL SELECTION RECOMMENDATIONS")
        print("=" * 60)
        print(f"\nOverall Strategy: {recommendations.get('overall_strategy', '')}\n")

        for i, rec in enumerate(recommendations.get("recommendations", []), 1):
            print(f"\n{'─' * 50}")
            print(f"Model {i}: {rec['model_name']}")
            print(f"  Type:     {rec['model_type']}")
            print(f"  Target:   {rec['target_variable']}")
            print(f"  Purpose:  {rec['purpose']}")
            print(f"  Features: {len(rec['recommended_features'])} selected")
            print(f"  Priority: {rec['priority']}")
            print(f"  Regulatory: {rec.get('regulatory_notes', '')}")

        if recommendations.get("data_quality_observations"):
            print(f"\n{'─' * 50}")
            print("Data Quality Observations:")
            for obs in recommendations["data_quality_observations"]:
                print(f"  - {obs}")

    except json.JSONDecodeError as e:
        print(f"Failed to parse LLM JSON response: {e}")
        print(f"Raw response:\n{llm_response[:2000]}")
        recommendations = None
else:
    recommendations = None
    print("Skipping recommendations — LLM call was not successful")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Log to audit trail
# MAGIC Full LLM interaction is recorded for regulatory transparency.

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="agent_recommendation",
    entity_type="model",
    entity_id="agent_model_selector",
    entity_version=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
    user_id=user,
    details={
        "model_endpoint": model_endpoint,
        "llm_success": llm_success,
        "prompt_system": system_prompt[:500],
        "prompt_user_length": len(user_prompt),
        "response_length": len(llm_response),
        "response_preview": llm_response[:1000],
        "recommendations_count": len(recommendations.get("recommendations", [])) if recommendations else 0,
        "upt_rows": row_count,
        "upt_columns": col_count,
    },
    source="notebook",
)
print("✓ Agent interaction logged to audit_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM Governance & Transparency
# MAGIC
# MAGIC **What Databricks logs for every LLM call (AI Gateway):**
# MAGIC - Request timestamp, user identity, endpoint name
# MAGIC - Full input prompt (system + user messages)
# MAGIC - Full output response
# MAGIC - Token usage (input/output)
# MAGIC - Latency
# MAGIC
# MAGIC **Guardrails available:**
# MAGIC - Input filters: PII detection, prompt injection detection
# MAGIC - Output filters: content safety, hallucination detection
# MAGIC - Rate limiting per user/endpoint
# MAGIC - Cost controls (token budgets)
# MAGIC
# MAGIC **What a regulatory auditor sees:**
# MAGIC - The AI made a recommendation (logged in `audit_log`)
# MAGIC - A human actuary reviewed and approved/rejected the recommendation
# MAGIC - The full prompt and response are available for inspection
# MAGIC - The AI never executed anything — it only suggested
# MAGIC
# MAGIC This is a key differentiator vs black-box SaaS tools: **everything the AI
# MAGIC does is logged, auditable, and under human control.**

# COMMAND ----------

# MAGIC %md
# MAGIC ## Full LLM Response (for transparency)

# COMMAND ----------

print(llm_response)
