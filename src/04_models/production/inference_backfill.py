# Databricks notebook source
# MAGIC %md
# MAGIC # Inference-log backfill — score every UPT policy with the 4 champions
# MAGIC
# MAGIC Resolves the current `@champion` alias for each production family,
# MAGIC loads each champion's raw flavor model (sklearn wrapper for GLMs,
# MAGIC LightGBM booster for GBMs) and scores every policy in the Modelling
# MAGIC Mart. Writes one row per `policy_id` to `{fqn}.inference_logs` with
# MAGIC per-family predictions, champion versions, price breakdown and a
# MAGIC feature snapshot — this is the table the app's `/policy/{id}/scoring`
# MAGIC endpoint reads for real (vs deterministic-hash) scoring stories.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %pip install mlflow statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

import json, os, tempfile
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

FAMILIES = ["freq_glm", "sev_glm", "demand_gbm", "fraud_gbm"]
ALIAS    = "champion"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve champion versions + load raw flavor models

# COMMAND ----------

def _load_raw_champion(family: str):
    """Resolve the @champion alias for a family and load the deepest raw-flavor
    MLmodel from its artifact tree. Returns (version, model)."""
    uc_name = f"{fqn}.{family}"
    try:
        mv = client.get_model_version_by_alias(uc_name, ALIAS)
    except Exception:
        # No alias yet — fall back to the highest version number.
        vs = list(client.search_model_versions(f"name='{uc_name}'"))
        if not vs:
            raise RuntimeError(f"{family}: no registered versions")
        mv = max(vs, key=lambda x: int(x.version))

    tmpdir   = tempfile.mkdtemp(prefix=f"{family}_")
    model_uri = f"models:/{uc_name}/{mv.version}"
    model_root = download_artifacts(artifact_uri=model_uri, dst_path=tmpdir)
    mlmodel_dirs = [root for root, _, files in os.walk(model_root) if "MLmodel" in files]
    if not mlmodel_dirs:
        raise RuntimeError(f"{family}: no MLmodel under {model_root}")
    deepest = max(mlmodel_dirs, key=lambda p: p.count(os.sep))
    if family.endswith("_glm"):
        model = mlflow.sklearn.load_model(deepest)
    else:
        model = mlflow.lightgbm.load_model(deepest)
    print(f"  {family}: champion v{mv.version}  run={mv.run_id[:10]}  raw={os.path.relpath(deepest, model_root) or '<root>'}")
    return str(mv.version), model

champions = {}
for fam in FAMILIES:
    try:
        champions[fam] = _load_raw_champion(fam)
    except Exception as e:
        print(f"  {fam}: FAILED to load champion — {e}")

if len(champions) < 4:
    raise RuntimeError(f"Need all 4 champions loaded — got {list(champions.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull UPT + quotes, score in bulk, join

# COMMAND ----------

# Modelling Mart — all policies
upt_sdf  = spark.table(f"{fqn}.unified_pricing_table_live")
upt_pdf  = upt_sdf.toPandas()
print(f"UPT rows: {len(upt_pdf):,}")

# For each policy, we use the *most-recently converted* quote for demand
# scoring (i.e. the quote that actually became this policy). If none exists,
# demand_pred is NULL.
quotes_sdf = spark.table(f"{fqn}.quotes")
quotes_pdf = (
    quotes_sdf.filter("policy_id IS NOT NULL")
              .toPandas()
              .sort_values(["policy_id"])
              .drop_duplicates("policy_id", keep="last")
)
print(f"Quotes w/ policy link: {len(quotes_pdf):,}")

# COMMAND ----------

def _prep_object_to_str(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
    return out

# --- freq_glm ---
fv, fmodel = champions["freq_glm"]
freq_input = _prep_object_to_str(upt_pdf)
upt_pdf["_freq_pred"] = np.asarray(fmodel.predict(freq_input), dtype=float).ravel()

# --- sev_glm ---
sv, smodel = champions["sev_glm"]
sev_input = _prep_object_to_str(upt_pdf)
upt_pdf["_sev_pred"] = np.asarray(smodel.predict(sev_input), dtype=float).ravel()

# --- fraud_gbm ---
frv, frmodel = champions["fraud_gbm"]
fraud_input  = _prep_object_to_str(upt_pdf)
fraud_feats  = list(frmodel.feature_name())
for c in fraud_feats:
    if c not in fraud_input.columns:
        fraud_input[c] = 0
fraud_input = fraud_input[fraud_feats]
for c in fraud_input.columns:
    if fraud_input[c].dtype == "object" or str(fraud_input[c].dtype) == "string":
        fraud_input[c] = fraud_input[c].astype("category")
upt_pdf["_fraud_pred"] = np.asarray(frmodel.predict(fraud_input), dtype=float).ravel()

# --- demand_gbm (scored on the quote → joined back to policy) ---
dv, dmodel = champions["demand_gbm"]
demand_input = _prep_object_to_str(quotes_pdf)
demand_feats = list(dmodel.feature_name())
for c in demand_feats:
    if c not in demand_input.columns:
        demand_input[c] = 0
demand_input = demand_input[demand_feats]
for c in demand_input.columns:
    if demand_input[c].dtype == "object" or str(demand_input[c].dtype) == "string":
        demand_input[c] = demand_input[c].astype("category")
quotes_pdf["_demand_pred"] = np.asarray(dmodel.predict(demand_input), dtype=float).ravel()

# Join demand back onto UPT by policy_id
demand_lookup = quotes_pdf.set_index("policy_id")["_demand_pred"].to_dict()
upt_pdf["_demand_pred"] = upt_pdf["policy_id"].map(demand_lookup)

scored_n = {
    "freq":   int(upt_pdf["_freq_pred"].notna().sum()),
    "sev":    int(upt_pdf["_sev_pred"].notna().sum()),
    "fraud":  int(upt_pdf["_fraud_pred"].notna().sum()),
    "demand": int(upt_pdf["_demand_pred"].notna().sum()),
}
print(f"Scored: {scored_n}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Price build-up + feature snapshot

# COMMAND ----------

# Mirror the formula the governance route uses for price build-up
upt_pdf["_base_premium"]     = (upt_pdf["_freq_pred"] * upt_pdf["_sev_pred"]).round(2)
upt_pdf["_fraud_loading"]    = np.where(upt_pdf["_fraud_pred"].fillna(0) > 0.25,
                                        upt_pdf["_base_premium"] * 0.05, 0.0).round(2)
_demand_filled               = upt_pdf["_demand_pred"].fillna(0.5)
upt_pdf["_demand_adj"]       = np.where(_demand_filled < 0.40,
                                        upt_pdf["_base_premium"] * 0.02,
                                        upt_pdf["_base_premium"] * -0.02).round(2)
upt_pdf["_technical_premium"] = (upt_pdf["_base_premium"]
                                 + upt_pdf["_fraud_loading"]
                                 + upt_pdf["_demand_adj"]).round(2)

SNAPSHOT_COLS = [
    "current_premium", "sum_insured", "annual_turnover",
    "industry_risk_tier", "construction_type", "region", "postcode_sector",
    "flood_zone_rating", "credit_score", "ccj_count", "years_trading",
    "claim_count_5y", "total_incurred_5y", "is_coastal", "urban_score",
]
def _snapshot(row):
    return json.dumps({c: (None if pd.isna(row.get(c)) else
                           (str(row[c]) if isinstance(row[c], str) else
                            (float(row[c]) if pd.api.types.is_number(row[c]) else str(row[c]))))
                       for c in SNAPSHOT_COLS if c in row.index},
                      default=str)
upt_pdf["_features_snapshot"] = upt_pdf.apply(_snapshot, axis=1)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Write to `{fqn}.inference_logs` (overwrite — single backfill row per policy)

# COMMAND ----------

out_pdf = upt_pdf[["policy_id",
                   "_freq_pred", "_sev_pred", "_demand_pred", "_fraud_pred",
                   "_base_premium", "_fraud_loading", "_demand_adj", "_technical_premium",
                   "_features_snapshot"]].copy()
out_pdf.columns = ["policy_id",
                   "freq_pred", "sev_pred", "demand_pred", "fraud_pred",
                   "base_premium", "fraud_loading", "demand_adj", "technical_premium",
                   "features_snapshot"]
out_pdf["freq_version"]   = champions["freq_glm"][0]
out_pdf["sev_version"]    = champions["sev_glm"][0]
out_pdf["demand_version"] = champions["demand_gbm"][0]
out_pdf["fraud_version"]  = champions["fraud_gbm"][0]
out_pdf["scored_at"]      = datetime.now(timezone.utc)

for c in ("freq_pred", "sev_pred", "demand_pred", "fraud_pred",
          "base_premium", "fraud_loading", "demand_adj", "technical_premium"):
    out_pdf[c] = pd.to_numeric(out_pdf[c], errors="coerce").astype(float)

sdf = spark.createDataFrame(out_pdf)

# Create-or-replace the table (single backfill snapshot — we're not
# time-versioning inference yet).
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.inference_logs (
        policy_id         STRING NOT NULL,
        scored_at         TIMESTAMP,
        freq_pred         DOUBLE, freq_version    STRING,
        sev_pred          DOUBLE, sev_version     STRING,
        demand_pred       DOUBLE, demand_version  STRING,
        fraud_pred        DOUBLE, fraud_version   STRING,
        base_premium      DOUBLE, fraud_loading   DOUBLE,
        demand_adj        DOUBLE, technical_premium DOUBLE,
        features_snapshot STRING
    ) USING DELTA
""")
sdf.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fqn}.inference_logs")

row_count = spark.table(f"{fqn}.inference_logs").count()
print(f"inference_logs rows: {row_count:,}")

# COMMAND ----------

# Audit — record the backfill
try:
    user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()
except Exception:
    user = "system"

det = json.dumps({
    "row_count":       row_count,
    "champion_versions": {k: v[0] for k, v in champions.items()},
    "scored_counts":   scored_n,
}).replace("'", "''")
spark.sql(f"""
    INSERT INTO {fqn}.audit_log
      (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
    SELECT uuid(), 'inference_backfill', 'table', 'inference_logs',
           '-', '{user}', current_timestamp(), '{det}', 'notebook'
""")

dbutils.notebook.exit(json.dumps({"row_count": row_count,
                                    "champion_versions": {k: v[0] for k, v in champions.items()}}))
