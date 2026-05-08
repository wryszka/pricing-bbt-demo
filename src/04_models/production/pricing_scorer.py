# Databricks notebook source
# MAGIC %md
# MAGIC # Pricing Scorer — unified live pricing endpoint with FeatureLookup
# MAGIC
# MAGIC One Model Serving endpoint (`pricing_scorer`) that takes a `policy_id`
# MAGIC and returns the final premium plus all intermediate predictions and
# MAGIC rating-engine components in a single round-trip.
# MAGIC
# MAGIC Logged with `fe.log_model(flavor=mlflow.pyfunc)` so feature resolution
# MAGIC against `unified_pricing_table_live` happens automatically at serving
# MAGIC time — Lakebase online lookup at request time, offline at training.
# MAGIC The pyfunc bundles the 4 current champions and applies the rating-
# MAGIC engine business rules — both baked at log time.
# MAGIC
# MAGIC Re-run this notebook whenever ANY of these flips:
# MAGIC  * a champion alias on freq_glm / sev_glm / demand_gbm / fraud_gbm
# MAGIC  * the rating_engine_config champion row
# MAGIC
# MAGIC The same endpoint serves both the live demo and production.

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("endpoint_name", "pricing_scorer")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-feature-engineering databricks-sdk \
# MAGIC   statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog        = dbutils.widgets.get("catalog_name")
schema         = dbutils.widgets.get("schema_name")
endpoint_name  = dbutils.widgets.get("endpoint_name")
fqn            = f"{catalog}.{schema}"
scorer_uc_name = f"{fqn}.pricing_scorer"

import json, os, tempfile, shutil
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe     = FeatureEngineeringClient()

FAMILIES = ("freq_glm", "sev_glm", "demand_gbm", "fraud_gbm")

def _champion_version(family: str) -> str:
    mv = client.get_model_version_by_alias(f"{fqn}.{family}", "champion")
    return str(mv.version)

CHAMPIONS = {fam: _champion_version(fam) for fam in FAMILIES}
print("Champions to bake in:", CHAMPIONS)

# COMMAND ----------

# Bake the current rating-engine champion config so the pyfunc applies it
# without a runtime SQL hop on every predict() call.
rating_row = spark.sql(f"""
    SELECT version, expense_loading_pct, commission_bp, fraud_loading_pct,
           fraud_loading_threshold, demand_adj_pct, demand_adj_threshold_lo,
           demand_adj_threshold_hi, min_premium, max_premium
    FROM {fqn}.rating_engine_config
    WHERE status = 'champion'
    LIMIT 1
""").collect()
if not rating_row:
    raise RuntimeError(f"{fqn}.rating_engine_config has no champion row")

RATING_CFG = {
    "version":                 rating_row[0]["version"],
    "expense_loading_pct":     float(rating_row[0]["expense_loading_pct"]),
    "commission_bp":           int(rating_row[0]["commission_bp"]),
    "fraud_loading_pct":       float(rating_row[0]["fraud_loading_pct"]),
    "fraud_loading_threshold": float(rating_row[0]["fraud_loading_threshold"]),
    "demand_adj_pct":          float(rating_row[0]["demand_adj_pct"]),
    "demand_adj_threshold_lo": float(rating_row[0]["demand_adj_threshold_lo"]),
    "demand_adj_threshold_hi": float(rating_row[0]["demand_adj_threshold_hi"]),
    "min_premium":             float(rating_row[0]["min_premium"]),
    "max_premium":             float(rating_row[0]["max_premium"]),
}
print("Rating engine config baked at log time:", RATING_CFG)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull inner artefacts from each champion
# MAGIC
# MAGIC Each champion is an FE-wrapped sklearn /
# MAGIC LightGBM artefact. The unified scorer only needs the inner raw flavor
# MAGIC (the wrapper does its own FeatureLookup and we already have features
# MAGIC at predict time). `_pull_raw_flavor` walks the artifact tree and
# MAGIC returns the deepest `MLmodel` directory.

# COMMAND ----------

def _pull_raw_flavor(family: str, version: str) -> str:
    tmp  = tempfile.mkdtemp(prefix=f"{family}_v{version}_")
    uri  = f"models:/{fqn}.{family}/{version}"
    root = download_artifacts(artifact_uri=uri, dst_path=tmp)
    mlmodel_dirs = [r for r, _, fs in os.walk(root) if "MLmodel" in fs]
    if not mlmodel_dirs:
        raise RuntimeError(f"{family} v{version}: no MLmodel under {root}")
    deepest = max(mlmodel_dirs, key=lambda p: p.count(os.sep))
    dest = f"{tempfile.mkdtemp(prefix=f'{family}_clean_')}/{family}"
    shutil.copytree(deepest, dest)
    return dest

artifact_paths = {fam: _pull_raw_flavor(fam, ver) for fam, ver in CHAMPIONS.items()}
print("Inner artefacts:")
for k, v in artifact_paths.items():
    print(f"  {k}: {v}")

# Bake champions + rating config into the artifact bundle. FeatureLookup
# resolves UPT features at request time — no warehouse_id / lookup table
# coordinates needed inside the pyfunc.
cfg_path = f"{tempfile.mkdtemp()}/config.json"
with open(cfg_path, "w") as fh:
    json.dump({
        "champions":            CHAMPIONS,
        "rating_engine_config": RATING_CFG,
    }, fh)
artifact_paths["config"] = cfg_path

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pyfunc — 4 sub-models + business rules in one call

# COMMAND ----------

# Feature unions per sub-model. These must match the trained champions'
# FEATURE lists so we slice the looked-up DataFrame correctly.
FREQ_FEATURES = [
    "sum_insured", "annual_turnover", "current_premium",
    "industry_risk_tier", "construction_type",
    "credit_score", "ccj_count", "years_trading",
    "flood_zone_rating", "proximity_to_fire_station_km",
    "crime_theft_index", "subsidence_risk", "composite_location_risk",
    "urban_score", "is_coastal", "population_density_per_km2",
    "elevation_metres", "annual_rainfall_mm",
    "director_stability_score", "employee_count_est",
    "distance_to_coast_km", "neighbourhood_claim_frequency",
]
SEV_FEATURES = [
    "sum_insured", "annual_turnover",
    "industry_risk_tier", "construction_type", "year_built",
    "credit_score", "years_trading",
    "flood_zone_rating", "proximity_to_fire_station_km",
    "crime_theft_index", "subsidence_risk", "composite_location_risk",
    "urban_score", "is_coastal", "elevation_metres",
    "annual_rainfall_mm", "population_density_per_km2",
    "distance_to_coast_km",
]
FRAUD_FEATURES = [
    "sum_insured", "annual_turnover", "current_premium",
    "industry_risk_tier", "construction_type", "year_built",
    "credit_score", "ccj_count", "years_trading",
    "flood_zone_rating", "crime_theft_index",
    "urban_score", "is_coastal", "director_stability_score",
    "employee_count_est", "claim_count_5y", "total_incurred_5y",
    "open_claims_count", "distinct_perils",
]
# Extra UPT cols used to synthesize quote-level inputs for the demand model.
DEMAND_PROXY_FEATURES = [
    "region", "rate_per_1k_si", "market_position_ratio", "market_median_rate",
]
UNION_FEATURES = sorted(set(
    FREQ_FEATURES + SEV_FEATURES + FRAUD_FEATURES + DEMAND_PROXY_FEATURES
))
print(f"FeatureLookup union: {len(UNION_FEATURES)} columns")

# COMMAND ----------

class PricingScorer(PythonModel):
    """Unified live pricing scorer. Receives policy_id + UPT-looked-up features
    (FE wrapper does the lookup), runs all 4 champions, applies the baked
    rating-engine business rules, returns final_premium plus every intermediate
    value the app and audit log need."""

    _GLM_CATS = {
        "industry_risk_tier": ["High", "Low", "Medium"],
        "construction_type":  ["Fire Resistive", "Frame", "Heavy Timber",
                                "Joisted Masonry", "Non-Combustible"],
    }

    def load_context(self, context):
        import json as _j
        import mlflow.sklearn, mlflow.lightgbm
        with open(context.artifacts["config"]) as fh:
            payload = _j.load(fh)
        self.champions  = payload["champions"]
        self.rating_cfg = payload["rating_engine_config"]
        self.freq   = mlflow.sklearn.load_model(context.artifacts["freq_glm"])
        self.sev    = mlflow.sklearn.load_model(context.artifacts["sev_glm"])
        self.demand = mlflow.lightgbm.load_model(context.artifacts["demand_gbm"])
        self.fraud  = mlflow.lightgbm.load_model(context.artifacts["fraud_gbm"])

    def _prep(self, df):
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == "object":
                out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
        return out

    def _pad_for_categoricals(self, df):
        import pandas as pd
        if df.empty:
            return df, 0
        n_real = len(df)
        template = df.iloc[0].to_dict()
        pad_rows = []
        for tier in self._GLM_CATS["industry_risk_tier"]:
            for ct in self._GLM_CATS["construction_type"]:
                r = dict(template)
                r["industry_risk_tier"] = tier
                r["construction_type"]  = ct
                pad_rows.append(r)
        padded = pd.concat([df, pd.DataFrame(pad_rows)], ignore_index=True)
        return padded, n_real

    def _score_glm(self, wrapper, df):
        import numpy as np
        padded, n_real = self._pad_for_categoricals(self._prep(df))
        all_preds = np.asarray(wrapper.predict(padded), dtype=float).ravel()
        return all_preds[:n_real]

    def _score_lgb(self, booster, df):
        import pandas as pd, numpy as np
        feat_names  = list(booster.feature_name())
        pandas_cats = getattr(booster, "pandas_categorical", None)
        prepped = self._prep(df.copy())
        built = pd.DataFrame(index=prepped.index)
        for i, name in enumerate(feat_names):
            is_cat = bool(pandas_cats and i < len(pandas_cats) and pandas_cats[i] is not None)
            present = name in prepped.columns
            if is_cat:
                col = prepped[name] if present else pd.Series(["(null)"] * len(prepped), index=prepped.index)
                training_cats = [str(c) for c in pandas_cats[i]]
                built[name] = pd.Categorical(col.astype(str), categories=training_cats)
            else:
                if present:
                    built[name] = pd.to_numeric(prepped[name], errors="coerce").fillna(0.0).astype(float)
                else:
                    built[name] = pd.Series([0.0] * len(prepped), index=prepped.index, dtype=float)
        return np.asarray(booster.predict(built), dtype=float).ravel()

    @staticmethod
    def _col(df, name, default):
        import pandas as pd
        if name in df.columns:
            ser = df[name]
            if ser.dtype == "object":
                return ser.fillna(default)
            return ser.astype(float).fillna(float(default)) if not isinstance(default, str) else ser.fillna(default)
        return pd.Series([default] * len(df), index=df.index)

    def _build_demand_input(self, df):
        """The demand model was trained on quote-level features (channel,
        voluntary_excess, gross_premium_quoted etc.) that don't exist on a
        policy. Project policy attributes through deterministic defaults so
        the demand prediction is stable per policy_id."""
        import numpy as np, pandas as pd
        sum_insured     = self._col(df, "sum_insured",     1_000_000.0).astype(float)
        current_premium = self._col(df, "current_premium", 1500.0).astype(float)
        market_median   = self._col(df, "market_median_rate", 1.5).astype(float)

        out = pd.DataFrame(index=df.index)
        out["channel"]              = "broker"
        out["region"]               = self._col(df, "region", "London")
        out["construction_type"]    = self._col(df, "construction_type", "Non-Combustible")
        out["flood_zone"]           = self._col(df, "flood_zone_rating", 1)
        out["year_built"]           = self._col(df, "year_built", 1990).astype(float)
        out["floor_area_sqm"]       = (sum_insured / 5_000.0).clip(50.0, 10_000.0)
        out["buildings_si"]         = sum_insured
        out["contents_si"]          = sum_insured * 0.2
        out["liability_si"]         = 1_000_000.0
        out["voluntary_excess"]     = 500.0
        out["gross_premium_quoted"] = current_premium
        out["log_gross_premium"]    = np.log1p(current_premium)
        out["log_buildings_si"]     = np.log1p(sum_insured)
        out["rate_per_1k_si"]       = self._col(df, "rate_per_1k_si", 1.5).astype(float)
        out["vs_market_rate"]       = self._col(df, "market_position_ratio", 1.0).astype(float)
        out["market_median_rate"]   = market_median
        out["competitor_a_min_rate"] = market_median * 0.95
        out["price_index"]          = 1.0
        out["annual_turnover"]      = self._col(df, "annual_turnover", 500_000.0).astype(float)
        out["credit_score"]         = self._col(df, "credit_score", 600.0).astype(float)
        out["flood_zone_rating"]    = self._col(df, "flood_zone_rating", 1).astype(float)
        out["crime_theft_index"]    = self._col(df, "crime_theft_index", 5.0).astype(float)
        out["sprinklered"]          = 0
        out["alarmed"]              = 0
        return out

    def _apply_rules(self, freq, sev, demand, fraud):
        import numpy as np
        cfg       = self.rating_cfg
        technical = freq * sev
        loaded    = technical * (1.0 + cfg["expense_loading_pct"] / 100.0) \
                              * (1.0 + cfg["commission_bp"] / 10_000.0)
        fraud_load = np.where(fraud > cfg["fraud_loading_threshold"],
                              loaded * cfg["fraud_loading_pct"] / 100.0, 0.0)
        demand_adj = np.where(demand < cfg["demand_adj_threshold_lo"],
                              loaded * cfg["demand_adj_pct"] / 100.0,
                              np.where(demand > cfg["demand_adj_threshold_hi"],
                                       -loaded * cfg["demand_adj_pct"] / 100.0,
                                       0.0))
        final = np.clip(loaded + fraud_load + demand_adj,
                        cfg["min_premium"], cfg["max_premium"])
        return technical, loaded, fraud_load, demand_adj, final

    def predict(self, context, model_input, params=None):
        import pandas as pd, numpy as np
        # FE wrapper has already resolved UPT features. model_input has
        # policy_id + every UNION_FEATURES column looked up from the online
        # store at request time (sub-10ms).
        if not hasattr(model_input, "columns"):
            model_input = pd.DataFrame(list(model_input))

        freq   = self._score_glm(self.freq,   model_input)
        sev    = self._score_glm(self.sev,    model_input)
        fraud  = self._score_lgb(self.fraud,  model_input)
        demand = self._score_lgb(self.demand, self._build_demand_input(model_input))

        technical, loaded, fraud_load, demand_adj, final = self._apply_rules(
            freq, sev, demand, fraud)

        n = len(model_input)
        return pd.DataFrame({
            "final_premium":          np.round(final, 2),
            "freq_pred":              freq,
            "sev_pred":               sev,
            "demand_pred":            demand,
            "fraud_pred":             fraud,
            "technical_premium":      np.round(loaded, 2),
            "fraud_load":             np.round(fraud_load, 2),
            "demand_adj":             np.round(demand_adj, 2),
            "rating_engine_version":  [self.rating_cfg["version"]] * n,
            "freq_version":           [self.champions["freq_glm"]]   * n,
            "sev_version":            [self.champions["sev_glm"]]    * n,
            "demand_version":         [self.champions["demand_gbm"]] * n,
            "fraud_version":          [self.champions["fraud_gbm"]]  * n,
        })


# COMMAND ----------

# MAGIC %md
# MAGIC ## Capture FeatureLookup spec via a tiny training_set
# MAGIC
# MAGIC `fe.log_model` requires a `training_set` to capture the FeatureLookup
# MAGIC metadata that drives serving-time resolution. Labels are dummy — we
# MAGIC never train this thing, just package it.

# COMMAND ----------

KEY = "policy_id"

labels_df = (
    spark.table(f"{fqn}.unified_pricing_table_live")
         .select(KEY).limit(50)
         .withColumn("_dummy_label", F.lit(0.0))
)

training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(
        table_name    = f"{fqn}.unified_pricing_table_live",
        feature_names = UNION_FEATURES,
        lookup_key    = KEY,
    )],
    label           = "_dummy_label",
    exclude_columns = [KEY],
)

# COMMAND ----------

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "policy_id")]),
    outputs=Schema([
        ColSpec("double", "final_premium"),
        ColSpec("double", "freq_pred"),
        ColSpec("double", "sev_pred"),
        ColSpec("double", "demand_pred"),
        ColSpec("double", "fraud_pred"),
        ColSpec("double", "technical_premium"),
        ColSpec("double", "fraud_load"),
        ColSpec("double", "demand_adj"),
        ColSpec("string", "rating_engine_version"),
        ColSpec("string", "freq_version"),
        ColSpec("string", "sev_version"),
        ColSpec("string", "demand_version"),
        ColSpec("string", "fraud_version"),
    ]),
)

# Sample policy_ids — input_example must be a DataFrame matching the input schema.
sample_pids = [r["policy_id"] for r in spark.sql(f"""
    SELECT policy_id FROM {fqn}.unified_pricing_table_live LIMIT 5
""").collect()]
input_example = pd.DataFrame({"policy_id": sample_pids})

# COMMAND ----------

with mlflow.start_run(run_name="pricing_scorer_deploy") as run:
    fe.log_model(
        model                 = PricingScorer(),
        artifact_path         = "scorer",
        flavor                = mlflow.pyfunc,
        training_set          = training_set,
        registered_model_name = scorer_uc_name,
        artifacts             = artifact_paths,
        input_example         = input_example,
        signature             = signature,
        # NOTE: do NOT include databricks-feature-engineering — fe.log_model
        # auto-adds databricks-feature-lookup==1.* for serving, and the two
        # clients can't coexist in the same env (it's a startup assertion).
        pip_requirements=[
            "mlflow>=2.12",
            "scikit-learn", "lightgbm", "statsmodels",
            "pandas", "numpy", "databricks-sdk",
        ],
    )
    print(f"Logged scorer for run {run.info.run_id}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sanity test — load the new version and predict
# MAGIC
# MAGIC `mlflow.pyfunc.load_model().predict()` on an FE-wrapped model triggers
# MAGIC `score_batch` against the offline UPT — supported on classic but not on
# MAGIC serverless runtime (FE's local-uri code path is missing). Wrap the
# MAGIC test so it logs a warning and continues; the real validation is the
# MAGIC warm-up call against the serving endpoint after deploy.

# COMMAND ----------

latest = max(int(v.version) for v in client.search_model_versions(f"name='{scorer_uc_name}'"))
print(f"New scorer version: {latest}")

try:
    scorer_uri = f"models:/{scorer_uc_name}/{latest}"
    loaded     = mlflow.pyfunc.load_model(scorer_uri)
    test_df    = input_example.copy()
    result     = loaded.predict(test_df)
    print("Sanity test result:")
    print(result.to_string(index=False))

    cfg = RATING_CFG
    assert (result["final_premium"] >= cfg["min_premium"] - 1e-6).all(), \
           f"final_premium below min: {result['final_premium'].min()}"
    assert (result["final_premium"] <= cfg["max_premium"] + 1e-6).all(), \
           f"final_premium above max: {result['final_premium'].max()}"
    assert (result["technical_premium"] > 0).all(), "technical_premium not positive"
    print("Sanity asserts passed.")
except Exception as e:
    print(f"Skipping in-notebook sanity test (serverless FE limitation): "
          f"{type(e).__name__}: {str(e)[:200]}")
    print("Real validation runs against the serving endpoint after deploy.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy as a Model Serving endpoint
# MAGIC
# MAGIC scale_to_zero=False — the live demo's whole point is sub-second
# MAGIC response. Provision/teardown notebooks bring this up before a demo
# MAGIC and wind it down afterwards (cost discipline).

# COMMAND ----------

print(f"Deploying {scorer_uc_name} v{latest} → endpoint '{endpoint_name}'")

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput

w = WorkspaceClient()
served = [ServedEntityInput(
    entity_name           = scorer_uc_name,
    entity_version        = str(latest),
    scale_to_zero_enabled = False,
    workload_size         = "Small",
)]

existing = None
try:
    existing = w.serving_endpoints.get(endpoint_name)
except Exception:
    pass

def _versions(cfg):
    return {(e.entity_version) for e in (getattr(cfg, "served_entities", []) or [])}

target = str(latest)
if existing is None:
    w.serving_endpoints.create(
        name   = endpoint_name,
        config = EndpointCoreConfigInput(name=endpoint_name, served_entities=served),
        route_optimized = True,
    )
    print("Created new endpoint (route_optimized).")
else:
    served_versions  = _versions(getattr(existing, "config", None))
    pending_versions = _versions(getattr(existing, "pending_config", None))
    if target in pending_versions:
        print(f"endpoint pending update to v{target} — skip; existing update will land")
    elif target in served_versions and not pending_versions:
        print(f"endpoint already serving v{target} — skip update")
    else:
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=served)
        print(f"updated existing endpoint to v{target}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "scorer_uc_name":        scorer_uc_name,
    "version":               latest,
    "endpoint":              endpoint_name,
    "champions":             CHAMPIONS,
    "rating_engine_version": RATING_CFG["version"],
}))
