# Databricks notebook source
# MAGIC %md
# MAGIC # Apply uniform UC tags
# MAGIC
# MAGIC Tags every table in the schema and every registered model with a uniform
# MAGIC set so the whole demo is filterable in a shared platform ("show me
# MAGIC everything tagged `project=pricing_workbench`") — it stands out among a sea
# MAGIC of other assets. Idempotent; re-runnable.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_pricing_v2_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_workbench")
dbutils.widgets.text("tier", "core")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
tier = dbutils.widgets.get("tier")
fqn = f"{catalog}.{schema}"

try:
    owner = spark.sql("SELECT current_user()").collect()[0][0]
except Exception:
    owner = "actuarial_pricing_team"

# NOTE: keys are pw_-prefixed on purpose. This metastore enforces a governed
# tag policy on standard keys like `project` (fixed allowed-value vocabulary),
# which rejects arbitrary values. pw_-prefixed keys are free-form, so we own the
# values AND the whole demo is still filterable ("pw_project = pricing_workbench").
BASE_TAGS = {
    "pw_project": "pricing_workbench",
    "pw_environment": "demo",
    "pw_managed_by": "dab",
    "pw_tier": tier,
    "pw_owner": owner,
}

# Tables whose columns carry PII-shaped (synthetic) attributes.
PII_TABLES = {"policy_demographics", "internal_commercial_policies",
              "silver_credit_bureau_summary", "quotes"}

def _tag_clause(extra=None):
    tags = {**BASE_TAGS, **(extra or {})}
    return ", ".join(f"'{k}' = '{v}'" for k, v in tags.items())

# COMMAND ----------

# --- schema itself ---
try:
    spark.sql(f"ALTER SCHEMA {fqn} SET TAGS ({_tag_clause()})")
    print(f"✓ tagged schema {fqn}")
except Exception as e:
    print(f"⚠ schema tag failed: {str(e)[:120]}")

# --- every table / view ---
tables = [r["table_name"] for r in spark.sql(f"""
    SELECT table_name FROM {catalog}.information_schema.tables
    WHERE table_schema = '{schema}'
""").collect()]

ok = 0
for t in tables:
    extra = {"pw_contains_pii": "true"} if t in PII_TABLES else {"pw_contains_pii": "false"}
    try:
        spark.sql(f"ALTER TABLE {fqn}.{t} SET TAGS ({_tag_clause(extra)})")
        ok += 1
    except Exception as e:
        print(f"⚠ {t}: {str(e)[:100]}")
print(f"✓ tagged {ok}/{len(tables)} tables")

# COMMAND ----------

# --- registered models (best-effort — schema/syntax varies by workspace) ---
try:
    models = [r["model_name"] for r in spark.sql(f"""
        SELECT model_name FROM {catalog}.information_schema.models
        WHERE schema_name = '{schema}'
    """).collect()]
    mok = 0
    for m in models:
        try:
            spark.sql(f"ALTER MODEL {fqn}.{m} SET TAGS ({_tag_clause()})")
            mok += 1
        except Exception as e:
            print(f"⚠ model {m}: {str(e)[:100]}")
    print(f"✓ tagged {mok}/{len(models)} registered models")
except Exception as e:
    print(f"ℹ model tagging skipped (not supported here): {str(e)[:120]}")

print(f"\nUniform tags applied: {BASE_TAGS}")
