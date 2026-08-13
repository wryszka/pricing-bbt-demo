# Databricks notebook source
# MAGIC %md
# MAGIC # Set champion aliases
# MAGIC
# MAGIC `production_training` registers model versions but does NOT set the
# MAGIC `champion` alias. The unified scorer endpoint and the app both resolve
# MAGIC `@champion`, so this step asserts it: keep an existing champion if one is
# MAGIC set, otherwise pin the latest registered version. Idempotent.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_pricing_v2_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_workbench")
# Comma-separated model families to alias. Commercial champions by default;
# pass the *_motor families for the motor/agentic core.
dbutils.widgets.text("families", "freq_glm,sev_glm,demand_gbm,fraud_gbm")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
fqn = f"{catalog}.{schema}"

FAMILIES = [f.strip() for f in dbutils.widgets.get("families").split(",") if f.strip()]

# COMMAND ----------

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()


def _latest_version(full_name):
    try:
        vers = [int(v.version) for v in w.model_versions.list(full_name=full_name)]
        return max(vers) if vers else None
    except Exception:
        return None


results = {}
for fam in FAMILIES:
    full = f"{fqn}.{fam}"
    try:
        existing = w.registered_models.get_alias(full_name=full, alias="champion")
        results[fam] = f"champion already set → v{existing.version_num} (unchanged)"
        continue
    except Exception:
        pass
    ver = _latest_version(full)
    if ver is None:
        results[fam] = "SKIPPED: model not registered on this workspace"
        continue
    try:
        w.registered_models.set_alias(full_name=full, alias="champion", version_num=ver)
        results[fam] = f"champion → v{ver} (latest)"
    except Exception as e:
        results[fam] = f"FAILED: {str(e)[:120]}"

print("Champion aliases:")
for k, v in results.items():
    print(f"  {k}: {v}")

if any(v.startswith("FAILED") for v in results.values()):
    raise RuntimeError(f"One or more champion aliases failed: {results}")
