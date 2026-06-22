# Databricks notebook source
# MAGIC %md
# MAGIC # Historical quote score — batch-run a single quote on any release
# MAGIC
# MAGIC Given a `release_id` and a JSON feature dict, loads the 4 model
# MAGIC versions pinned to that release straight from Unity Catalog (raw
# MAGIC flavor, no serving endpoint), applies the rating-engine config that
# MAGIC was in force, and returns a full price build-up.
# MAGIC
# MAGIC This is the counterpart to the `pricing_scorer` serving endpoint —
# MAGIC that one only serves the CURRENT champion release. Historical
# MAGIC releases are scored here, on demand, via a short batch run.
# MAGIC
# MAGIC Result is both returned via `dbutils.notebook.exit()` (JSON) and
# MAGIC persisted to `{fqn}.historical_quote_scores` for audit.

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("release_id",    "apr_2026")
dbutils.widgets.text("features_json", "{}")
dbutils.widgets.text("run_label",     "adhoc")

# COMMAND ----------

# MAGIC %pip install mlflow statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog     = dbutils.widgets.get("catalog_name")
schema      = dbutils.widgets.get("schema_name")
release_id  = dbutils.widgets.get("release_id")
features_js = dbutils.widgets.get("features_json")
run_label   = dbutils.widgets.get("run_label")
fqn         = f"{catalog}.{schema}"

import json, os, tempfile
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.artifacts import download_artifacts
from datetime import datetime, timezone

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Look up the release — which versions + rating engine to use

# COMMAND ----------

release_rows = spark.sql(f"""
    SELECT * FROM {fqn}.pricing_engine_releases WHERE release_id = '{release_id}' LIMIT 1
""").toPandas()
if release_rows.empty:
    raise RuntimeError(f"Release '{release_id}' not found in {fqn}.pricing_engine_releases")
release = release_rows.iloc[0].to_dict()
print(f"Release: {release['display_name']}  (effective {release['effective_date']}, status {release['status']})")
print(f"  freq_glm   v{release['freq_glm_version']}")
print(f"  sev_glm    v{release['sev_glm_version']}")
print(f"  demand_gbm v{release['demand_gbm_version']}")
print(f"  fraud_gbm  v{release['fraud_gbm_version']}")
print(f"  rating engine {release['rating_engine_version']}")

rating_rows = spark.sql(f"""
    SELECT * FROM {fqn}.rating_engine_config WHERE version = '{release['rating_engine_version']}' LIMIT 1
""").toPandas()
if rating_rows.empty:
    raise RuntimeError(f"Rating engine '{release['rating_engine_version']}' not found")
rating = rating_rows.iloc[0].to_dict()

# Parse input features
try:
    features = json.loads(features_js)
except Exception as e:
    raise RuntimeError(f"features_json could not be parsed: {e}")
print(f"\nFeatures ({len(features)} keys):", {k: str(v)[:40] for k, v in list(features.items())[:6]}, "…")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Download + load the 4 models from UC (raw flavor)

# COMMAND ----------

from mlflow.tracking import MlflowClient
_client = MlflowClient()

def _nudge_for(family: str, version: str) -> float:
    """Each backdated UC version is registered with the champion's bytes
    plus a `nudge_multiplier` MLflow param representing the story-driven
    drift relative to champion. Read it so historical scorings actually
    move across releases."""
    try:
        mv = _client.get_model_version(f"{fqn}.{family}", version)
        run = _client.get_run(mv.run_id)
        return float(run.data.params.get("nudge_multiplier") or 1.0)
    except Exception as e:
        print(f"  {family} v{version}: nudge lookup failed ({e}); using 1.0")
        return 1.0

def _load_raw(family: str, version: str):
    tmp  = tempfile.mkdtemp(prefix=f"{family}_v{version}_")
    uri  = f"models:/{fqn}.{family}/{version}"
    root = download_artifacts(artifact_uri=uri, dst_path=tmp)
    mlmodel_dirs = [r for r, _, fs in os.walk(root) if "MLmodel" in fs]
    if not mlmodel_dirs:
        raise RuntimeError(f"{family} v{version}: no MLmodel")
    deepest = max(mlmodel_dirs, key=lambda p: p.count(os.sep))
    if family.endswith("_glm"):
        return mlflow.sklearn.load_model(deepest)
    return mlflow.lightgbm.load_model(deepest)

freq_model   = _load_raw("freq_glm",   release["freq_glm_version"])
sev_model    = _load_raw("sev_glm",    release["sev_glm_version"])
demand_model = _load_raw("demand_gbm", release["demand_gbm_version"])
fraud_model  = _load_raw("fraud_gbm",  release["fraud_gbm_version"])

# Per-family nudges from the registered MLflow runs
nudges = {
    "freq_glm":   _nudge_for("freq_glm",   release["freq_glm_version"]),
    "sev_glm":    _nudge_for("sev_glm",    release["sev_glm_version"]),
    "demand_gbm": _nudge_for("demand_gbm", release["demand_gbm_version"]),
    "fraud_gbm":  _nudge_for("fraud_gbm",  release["fraud_gbm_version"]),
}
print(f"All 4 models loaded. Nudges: {nudges}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Score the single feature row

# COMMAND ----------

row = pd.DataFrame([features])
for c in row.columns:
    if row[c].dtype == "object":
        row[c] = row[c].astype(str).where(row[c].notna(), "(null)")

# Pad the GLM input so get_dummies in the wrapper sees every training
# category value; we keep only the original first-row prediction. (See
# pricing_scorer.py for the same pattern + rationale.)
GLM_CATS = {
    "industry_risk_tier": ["High", "Low", "Medium"],
    "construction_type":  ["Fire Resistive", "Frame", "Heavy Timber",
                           "Joisted Masonry", "Non-Combustible"],
}
template = row.iloc[0].to_dict()
pad_rows = []
for tier in GLM_CATS["industry_risk_tier"]:
    for ct in GLM_CATS["construction_type"]:
        r = dict(template); r["industry_risk_tier"] = tier; r["construction_type"] = ct
        pad_rows.append(r)
padded_glm = pd.concat([row, pd.DataFrame(pad_rows)], ignore_index=True)

# --- GLMs self-encode internally ---
freq_pred = float(np.asarray(freq_model.predict(padded_glm), dtype=float).ravel()[0])
sev_pred  = float(np.asarray(sev_model.predict(padded_glm),  dtype=float).ravel()[0])

# --- LightGBM: rebuild exact training schema with pandas_categorical ---
def _score_lgb(booster, df):
    feat_names  = list(booster.feature_name())
    pandas_cats = getattr(booster, "pandas_categorical", None)
    built = pd.DataFrame(index=df.index)
    for i, name in enumerate(feat_names):
        is_cat = bool(pandas_cats and i < len(pandas_cats) and pandas_cats[i] is not None)
        present = name in df.columns
        if is_cat:
            col = df[name] if present else pd.Series(["(null)"] * len(df), index=df.index)
            training_cats = [str(c) for c in pandas_cats[i]]
            built[name] = pd.Categorical(col.astype(str), categories=training_cats)
        else:
            if present:
                built[name] = pd.to_numeric(df[name], errors="coerce").fillna(0.0).astype(float)
            else:
                built[name] = pd.Series([0.0] * len(df), index=df.index, dtype=float)
    return float(np.asarray(booster.predict(built), dtype=float).ravel()[0])

demand_pred = _score_lgb(demand_model, row)
fraud_pred  = _score_lgb(fraud_model,  row)

# Apply each family's story-driven nudge
freq_pred   = freq_pred   * nudges["freq_glm"]
sev_pred    = sev_pred    * nudges["sev_glm"]
demand_pred = demand_pred * nudges["demand_gbm"]
fraud_pred  = fraud_pred  * nudges["fraud_gbm"]

print(f"Predictions — freq={freq_pred:.4f}  sev=£{sev_pred:.2f}  demand={demand_pred:.3f}  fraud={fraud_pred:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Apply the release's rating engine

# COMMAND ----------

def _num(v, d=0.0):
    try: return float(v) if v is not None else d
    except (TypeError, ValueError): return d

base = freq_pred * sev_pred
fraud_trigger = _num(rating.get("fraud_loading_threshold"), 0.25)
fraud_pct     = _num(rating.get("fraud_loading_pct"), 0.0)
fraud_loading = base * (fraud_pct / 100.0) if fraud_pred > fraud_trigger else 0.0

dlo = _num(rating.get("demand_adj_threshold_lo"), 0.40)
dhi = _num(rating.get("demand_adj_threshold_hi"), 0.75)
adj_pct = _num(rating.get("demand_adj_pct"), 0.0)
if   demand_pred < dlo: demand_adj = base * (adj_pct / 100.0)
elif demand_pred > dhi: demand_adj = -base * (adj_pct / 100.0)
else:                    demand_adj = 0.0

technical  = base + fraud_loading + demand_adj
expense    = technical * _num(rating.get("expense_loading_pct"), 0) / 100.0
with_exp   = technical + expense
commission = with_exp * _num(rating.get("commission_bp"), 0) / 10_000.0
gross      = with_exp + commission
gross      = max(_num(rating.get("min_premium"), 0),
                 min(_num(rating.get("max_premium"), 1e12), gross))

buildup = {
    "base_premium":      round(base, 2),
    "fraud_loading":     round(fraud_loading, 2),
    "demand_adj":        round(demand_adj, 2),
    "technical_premium": round(technical, 2),
    "expense_loading":   round(expense, 2),
    "commission":        round(commission, 2),
    "gross_premium":     round(gross, 2),
}
print(f"Gross: £{buildup['gross_premium']}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Persist + return

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.historical_quote_scores (
        score_id         STRING,
        scored_at        TIMESTAMP,
        release_id       STRING,
        release_display  STRING,
        freq_version     STRING,
        sev_version      STRING,
        demand_version   STRING,
        fraud_version    STRING,
        rating_engine    STRING,
        freq_pred        DOUBLE,
        sev_pred         DOUBLE,
        demand_pred      DOUBLE,
        fraud_pred       DOUBLE,
        base_premium     DOUBLE,
        fraud_loading    DOUBLE,
        demand_adj       DOUBLE,
        technical_premium DOUBLE,
        expense_loading  DOUBLE,
        commission       DOUBLE,
        gross_premium    DOUBLE,
        features_json    STRING,
        run_label        STRING
    ) USING DELTA
""")

score_id = f"HIS-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S%f')}"
spark.sql(f"""
    INSERT INTO {fqn}.historical_quote_scores VALUES (
        '{score_id}', current_timestamp(),
        '{release_id}', '{release['display_name'].replace("'", "''")}',
        '{release['freq_glm_version']}',  '{release['sev_glm_version']}',
        '{release['demand_gbm_version']}', '{release['fraud_gbm_version']}',
        '{release['rating_engine_version']}',
        {freq_pred}, {sev_pred}, {demand_pred}, {fraud_pred},
        {buildup['base_premium']}, {buildup['fraud_loading']}, {buildup['demand_adj']}, {buildup['technical_premium']},
        {buildup['expense_loading']}, {buildup['commission']}, {buildup['gross_premium']},
        '{features_js.replace("'", "''")}', '{run_label}'
    )
""")

result = {
    "score_id":       score_id,
    "release_id":     release_id,
    "display_name":   release["display_name"],
    "effective_date": str(release["effective_date"]),
    "model_versions": {
        "freq_glm":      release["freq_glm_version"],
        "sev_glm":       release["sev_glm_version"],
        "demand_gbm":    release["demand_gbm_version"],
        "fraud_gbm":     release["fraud_gbm_version"],
        "rating_engine": release["rating_engine_version"],
    },
    "predictions": {
        "freq_pred":   round(freq_pred, 6),
        "sev_pred":    round(sev_pred, 2),
        "demand_pred": round(demand_pred, 6),
        "fraud_pred":  round(fraud_pred, 6),
    },
    "price_buildup":  buildup,
    "source":         "historical_batch",
}

dbutils.notebook.exit(json.dumps(result))
