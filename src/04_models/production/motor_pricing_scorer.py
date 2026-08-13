# Databricks notebook source
# MAGIC %md
# MAGIC # Motor Pricing Scorer — unified live serving endpoint
# MAGIC
# MAGIC One pyfunc that wraps the 3 motor champions (freq, sev, fraud) and a
# MAGIC motor-specific rating engine. Logged with `fe.log_model` so the
# MAGIC serving endpoint resolves features from Lakebase at request time
# MAGIC (sub-10ms). The endpoint takes `policy_id` and returns
# MAGIC `final_premium` plus every intermediate value.
# MAGIC
# MAGIC Rating engine (baked at log time):
# MAGIC   technical    = freq * sev
# MAGIC   loaded       = technical * (1 + expense_loading) * (1 + commission/10000)
# MAGIC   young_driver = +15% loading if driver_age < 25
# MAGIC   telematics   = +10% surcharge if any recent_speeding_events OR
# MAGIC                  recent_curfew_breaches in the last 30d
# MAGIC   fraud_load   = +8% if fraud_pred > 0.20
# MAGIC   final        = clip(loaded + loadings, [200, 50000])

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("endpoint_name", "motor_pricing_scorer")
# deploy_mode: "full" also deploys the route-optimized FeatureLookup endpoint
# (needs the Lakebase online store — the optional live-serving tier). "direct_only"
# deploys ONLY the scale-to-zero motor_pricing_scorer_direct endpoint, which is
# all the agentic buyer / MCP needs and requires no online store — the Core
# default on a workspace without the live tier.
dbutils.widgets.text("deploy_mode",   "full")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-feature-engineering databricks-sdk \
# MAGIC   statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog        = dbutils.widgets.get("catalog_name")
schema         = dbutils.widgets.get("schema_name")
endpoint_name  = dbutils.widgets.get("endpoint_name")
deploy_mode    = dbutils.widgets.get("deploy_mode").strip().lower()
fqn            = f"{catalog}.{schema}"
scorer_uc_name = f"{fqn}.motor_pricing_scorer"

import json, os, tempfile, shutil
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
import pyspark.sql.functions as F

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()
fe     = FeatureEngineeringClient()

FAMILIES = ("freq_glm_motor", "sev_glm_motor", "demand_gbm_motor", "fraud_gbm_motor")

def _champion_version(family: str) -> str:
    mv = client.get_model_version_by_alias(f"{fqn}.{family}", "champion")
    return str(mv.version)

CHAMPIONS = {fam: _champion_version(fam) for fam in FAMILIES}
print("Motor champions:", CHAMPIONS)

# COMMAND ----------

# Motor rating engine config — baked into the artifact.
RATING_CFG = {
    "version":                       "motor_v1.3",
    # freq_glm_motor is trained on `claim_count_5y` (a 5-year count). A premium
    # covers one year, so the scorer divides the GLM output by this before
    # multiplying by per-claim severity. If the frequency model is ever
    # retrained on a 1-year count, set this to 1.0.
    "freq_exposure_years":           5.0,
    "expense_loading_pct":           18.0,
    "commission_bp":                 1500,    # 15.0 %
    "young_driver_threshold":        25,
    "young_driver_loading_pct":      15.0,
    # Telematics surcharge SCALES with recent event count so every new
    # black-box event pushes the quote up monotonically (no reset needed).
    # load = loaded * min(events * per_event_pct, max_pct) / 100
    # where events = recent_speeding_events + recent_curfew_breaches.
    # Tuned so one event (≈2 units) ≈ 10% (matches the old binary surcharge)
    # and it climbs to a 40% cap. Legacy key kept for back-compat readers.
    "telematics_event_loading_pct":  10.0,
    "telematics_per_event_pct":      5.0,
    "telematics_max_load_pct":       40.0,
    "fraud_loading_pct":             8.0,
    "fraud_loading_threshold":       0.20,
    # Demand adjustment: low demand → small discount to win, high demand → small loading.
    "demand_adjust_pct":             4.0,
    "demand_low_threshold":          0.35,
    "demand_high_threshold":         0.70,
    "min_premium":                   200.0,
    "max_premium":                   50_000.0,
}

# COMMAND ----------

# Pull each champion's inner raw flavor — same pattern as commercial scorer.
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

cfg_path = f"{tempfile.mkdtemp()}/config.json"
with open(cfg_path, "w") as fh:
    json.dump({"champions": CHAMPIONS, "rating_engine_config": RATING_CFG}, fh)
artifact_paths["config"] = cfg_path

# COMMAND ----------

# Feature unions across the three motor sub-models.
FREQ_FEATURES = [
    "driver_age", "license_years_held", "no_claims_years",
    "gender", "marital_status", "occupation_class",
    "vehicle_group", "vehicle_value", "vehicle_age",
    "annual_mileage", "parking_overnight", "business_use", "fuel_type",
    "prior_convictions", "prior_accidents_5y",
    "behaviour_score", "avg_speed_mph", "hours_driven_30d",
    "night_driving_pct",
    "recent_speeding_events", "recent_curfew_breaches", "recent_harsh_braking_30d",
    "telematics_recent_event_count",
]
SEV_FEATURES = [
    "driver_age", "license_years_held",
    "vehicle_group", "vehicle_value", "vehicle_age", "fuel_type",
    "annual_mileage", "parking_overnight",
    "behaviour_score", "avg_speed_mph",
    "recent_speeding_events", "recent_harsh_braking_30d",
    "at_fault_count_5y", "distinct_perils",
]
FRAUD_FEATURES = [
    "driver_age", "license_years_held", "no_claims_years",
    "gender", "occupation_class",
    "vehicle_group", "vehicle_value", "vehicle_age", "annual_mileage",
    "prior_convictions", "prior_accidents_5y",
    "behaviour_score", "avg_speed_mph", "night_driving_pct",
    "recent_speeding_events", "recent_curfew_breaches", "recent_harsh_braking_30d",
    "telematics_recent_event_count",
    "claim_count_5y", "at_fault_count_5y", "open_claims_count", "distinct_perils",
]
DEMAND_FEATURES = [
    "driver_age", "license_years_held", "no_claims_years",
    "gender", "marital_status", "occupation_class",
    "vehicle_group", "vehicle_value", "vehicle_age", "fuel_type",
    "annual_mileage", "parking_overnight", "business_use",
    "current_premium", "behaviour_score",
    "claim_count_5y", "at_fault_count_5y",
]
UNION_FEATURES = sorted(set(FREQ_FEATURES + SEV_FEATURES + FRAUD_FEATURES + DEMAND_FEATURES))
print(f"FeatureLookup union: {len(UNION_FEATURES)} columns")

# COMMAND ----------

class MotorPricingScorer(PythonModel):
    """Live motor scorer. FE wrapper resolves features; pyfunc scores three
    sub-models, applies the motor rating engine, returns final premium and
    every intermediate."""

    # Categorical (string) UPT columns — used by _prep to route NULL-handling.
    _CATEGORICAL_FEATURES = {
        "gender", "marital_status", "occupation_class", "region",
        "postcode_area", "vehicle_make", "vehicle_model",
        "fuel_type", "parking_overnight", "business_use",
        "policy_id",
    }
    # GLM categoricals we pad rows for so get_dummies sees every value.
    _GLM_CATS = {
        "gender":           ["F", "M"],
        "marital_status":   ["Single", "Married", "Divorced"],
        "occupation_class": ["Professional", "Office", "Skilled Manual",
                              "Service", "Student", "Self-Employed"],
        "fuel_type":        ["Petrol", "Diesel", "Hybrid", "Electric"],
        "parking_overnight":["Garage", "Driveway", "Street"],
        "business_use":     ["N", "Y"],
    }

    def load_context(self, context):
        import json as _j
        import mlflow.sklearn, mlflow.lightgbm
        with open(context.artifacts["config"]) as fh:
            payload = _j.load(fh)
        self.champions  = payload["champions"]
        self.rating_cfg = payload["rating_engine_config"]
        self.freq   = mlflow.sklearn.load_model(context.artifacts["freq_glm_motor"])
        self.sev    = mlflow.sklearn.load_model(context.artifacts["sev_glm_motor"])
        self.demand = mlflow.lightgbm.load_model(context.artifacts["demand_gbm_motor"])
        self.fraud  = mlflow.lightgbm.load_model(context.artifacts["fraud_gbm_motor"])

    def _prep(self, df):
        import pandas as pd
        out = df.copy()
        for c in out.columns:
            s = out[c]
            kind = getattr(s.dtype, "kind", "")
            if c in self._CATEGORICAL_FEATURES:
                out[c] = s.astype(str).where(s.notna(), "(null)").astype(object)
            elif kind == "b":
                out[c] = s.fillna(False).astype(int).astype(float)
            else:
                out[c] = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float)
        return out

    def _pad_for_categoricals(self, df):
        import itertools, pandas as pd
        if df.empty:
            return df, 0
        n_real = len(df)
        template = df.iloc[0].to_dict()
        pad_rows = []
        # Cross-product of every cat value combo — fewer than commercial
        # because motor only needs each cat exercised, not every combo.
        for col, vals in self._GLM_CATS.items():
            for v in vals:
                r = dict(template)
                r[col] = v
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

    def _apply_rules(self, df, freq, sev, demand, fraud):
        """Motor rating engine. Returns
        (technical, loaded, young_driver_load, telematics_load, fraud_load,
         demand_adj, final)."""
        import numpy as np
        cfg = self.rating_cfg
        # The frequency GLM is trained on `claim_count_5y` — a FIVE-YEAR claim
        # count — but a premium covers ONE year, so it must be annualised
        # before it meets a per-claim severity. Without this the technical
        # premium is ~5x too high (book avg premium £523 vs ~£4.3k quoted).
        # `freq_exposure_years` keeps the divisor explicit and configurable:
        # retrain the GLM on a 1-year count and set it to 1.0.
        annual_freq = freq / float(cfg.get("freq_exposure_years", 1.0) or 1.0)
        technical = annual_freq * sev
        loaded    = technical * (1.0 + cfg["expense_loading_pct"] / 100.0) \
                              * (1.0 + cfg["commission_bp"] / 10_000.0)
        # Young driver loading
        age = df["driver_age"].astype(float).values if "driver_age" in df.columns else np.zeros(len(df))
        young_driver_load = np.where(age < cfg["young_driver_threshold"],
                                     loaded * cfg["young_driver_loading_pct"] / 100.0, 0.0)
        # Telematics-event surcharge — SCALES with recent event count so each
        # new black-box event raises the quote monotonically (no reset needed).
        sp = df["recent_speeding_events"].astype(float).values if "recent_speeding_events" in df.columns else np.zeros(len(df))
        cb = df["recent_curfew_breaches"].astype(float).values if "recent_curfew_breaches" in df.columns else np.zeros(len(df))
        events = sp + cb
        per_event = cfg.get("telematics_per_event_pct", 5.0)
        max_pct   = cfg.get("telematics_max_load_pct", 40.0)
        telematics_pct = np.minimum(events * per_event, max_pct)
        telematics_load = loaded * telematics_pct / 100.0
        # Fraud loading
        fraud_load = np.where(fraud > cfg["fraud_loading_threshold"],
                              loaded * cfg["fraud_loading_pct"] / 100.0, 0.0)
        # Demand adjustment: low predicted acceptance → small discount to win the
        # renewal; high predicted acceptance → small loading to capture margin.
        demand_adj = np.where(demand < cfg["demand_low_threshold"],
                              -loaded * cfg["demand_adjust_pct"] / 100.0,
                       np.where(demand > cfg["demand_high_threshold"],
                                 loaded * cfg["demand_adjust_pct"] / 100.0,
                                 0.0))
        final = np.clip(loaded + young_driver_load + telematics_load + fraud_load + demand_adj,
                        cfg["min_premium"], cfg["max_premium"])
        return technical, loaded, young_driver_load, telematics_load, fraud_load, demand_adj, final

    def predict(self, context, model_input, params=None):
        import pandas as pd, numpy as np
        if not hasattr(model_input, "columns"):
            model_input = pd.DataFrame(list(model_input))

        freq   = self._score_glm(self.freq,   model_input)
        sev    = self._score_glm(self.sev,    model_input)
        demand = self._score_lgb(self.demand, model_input)
        fraud  = self._score_lgb(self.fraud,  model_input)

        technical, loaded, young_load, telematics_load, fraud_load, demand_adj, final = \
            self._apply_rules(model_input, freq, sev, demand, fraud)

        # Surface the annualised frequency alongside the raw 5-year GLM output
        # so a reader can reconcile technical = annual_freq * sev by hand.
        _exposure = float(self.rating_cfg.get("freq_exposure_years", 1.0) or 1.0)
        annual_freq = freq / _exposure

        n = len(model_input)
        return pd.DataFrame({
            "final_premium":           np.round(final, 2),
            "freq_pred":               freq,          # raw GLM: claims per 5 years
            "annual_freq":             annual_freq,   # freq / freq_exposure_years
            "sev_pred":                sev,
            "demand_pred":             demand,
            "fraud_pred":              fraud,
            "technical_premium":       np.round(loaded, 2),
            "young_driver_load":       np.round(young_load, 2),
            "telematics_event_load":   np.round(telematics_load, 2),
            "fraud_load":              np.round(fraud_load, 2),
            "demand_adj":              np.round(demand_adj, 2),
            "rating_engine_version":   [self.rating_cfg["version"]] * n,
            "freq_version":            [self.champions["freq_glm_motor"]]   * n,
            "sev_version":             [self.champions["sev_glm_motor"]]    * n,
            "demand_version":          [self.champions["demand_gbm_motor"]] * n,
            "fraud_version":           [self.champions["fraud_gbm_motor"]]  * n,
        })

# COMMAND ----------

# Tiny training_set so fe.log_model captures the FeatureLookup spec.
KEY = "policy_id"
labels_df = (
    spark.table(f"{fqn}.unified_motor_table_live")
         .select(KEY).limit(50)
         .withColumn("_dummy_label", F.lit(0.0))
)
training_set = fe.create_training_set(
    df              = labels_df,
    feature_lookups = [FeatureLookup(
        table_name    = f"{fqn}.unified_motor_table_live",
        feature_names = UNION_FEATURES,
        lookup_key    = KEY,
    )],
    label           = "_dummy_label",
    exclude_columns = [KEY],
)

sample_pids = [r["policy_id"] for r in spark.sql(f"""
    SELECT policy_id FROM {fqn}.unified_motor_table_live LIMIT 3
""").collect()]
input_example = pd.DataFrame({"policy_id": sample_pids})

# COMMAND ----------

with mlflow.start_run(run_name="motor_pricing_scorer_deploy") as run:
    fe.log_model(
        model                 = MotorPricingScorer(),
        artifact_path         = "scorer",
        flavor                = mlflow.pyfunc,
        training_set          = training_set,
        registered_model_name = scorer_uc_name,
        artifacts             = artifact_paths,
        # NB: do NOT include databricks-feature-engineering — fe.log_model
        # auto-injects databricks-feature-lookup at serving time and they
        # can't coexist.
        pip_requirements=[
            "mlflow>=2.12",
            "scikit-learn", "lightgbm", "statsmodels",
            "pandas", "numpy", "databricks-sdk",
        ],
    )
    print(f"Logged motor scorer for run {run.info.run_id}")

# COMMAND ----------

latest = max(int(v.version) for v in client.search_model_versions(f"name='{scorer_uc_name}'"))
print(f"New scorer version: v{latest}")

# COMMAND ----------

# Deploy as a Model Serving endpoint. Race-safe reconcile.
from databricks.sdk import WorkspaceClient
import requests as _rq, json as _json

w = WorkspaceClient()
target = str(latest)
# Live demo endpoint: explicit 4-64 provisioned concurrency, scale_to_zero
# DISABLED so it stays warm while the system is on. Set via REST so the
# min/max concurrency fields apply regardless of databricks-sdk version.
_served_entity = {
    "entity_name": scorer_uc_name,
    "entity_version": target,
    "scale_to_zero_enabled": False,
    "min_provisioned_concurrency": 4,
    "max_provisioned_concurrency": 64,
    "workload_type": "CPU",
}
_host = w.config.host.rstrip("/")
_hdrs = {**w.config._header_factory(), "Content-Type": "application/json"}

import time as _time

if deploy_mode == "direct_only":
    print(f"deploy_mode=direct_only — skipping the route-optimized "
          f"'{endpoint_name}' endpoint (needs the Lakebase online store). "
          f"Only the scale-to-zero direct endpoint is deployed below.")

existing = None
try:
    existing = w.serving_endpoints.get(endpoint_name)
except Exception:
    pass

# Route optimization gives a direct data-plane path → lower, steadier latency
# at QPS (the millisecond-pricing story). It is CREATE-TIME ONLY and cannot be
# toggled on an existing endpoint, so if the endpoint exists but isn't
# route-optimized we delete and recreate it. The endpoint is then queried via
# its `endpoint_url` host (the app resolves this); OAuth is required (PATs are
# rejected) — the app SP already authenticates via OAuth.
_create_payload = {"name": endpoint_name, "route_optimized": True,
                   "config": {"served_entities": [_served_entity]}}
_is_ro = bool(getattr(existing, "route_optimized", False)) if existing else False

if deploy_mode == "direct_only":
    pass  # route-optimized endpoint intentionally not deployed in Core
elif existing is None:
    _r = _rq.post(f"{_host}/api/2.0/serving-endpoints", headers=_hdrs,
                  data=_json.dumps(_create_payload), timeout=60)
    print(f"Created route-optimized endpoint {endpoint_name} v{target} -> {_r.status_code}: {_r.text[:200]}")
elif not _is_ro:
    print(f"Endpoint {endpoint_name} exists but is NOT route-optimized — deleting to recreate…")
    _rq.delete(f"{_host}/api/2.0/serving-endpoints/{endpoint_name}", headers=_hdrs, timeout=60)
    for _ in range(60):
        try:
            w.serving_endpoints.get(endpoint_name); _time.sleep(5)
        except Exception:
            break
    _r = _rq.post(f"{_host}/api/2.0/serving-endpoints", headers=_hdrs,
                  data=_json.dumps(_create_payload), timeout=60)
    print(f"Recreated route-optimized endpoint {endpoint_name} v{target} -> {_r.status_code}: {_r.text[:200]}")
else:
    # Already route-optimized — reconcile the served version via PUT config
    # (the route_optimized flag persists across config updates).
    served_versions  = {e.entity_version for e in (existing.config.served_entities or [])} if existing.config else set()
    pending_versions = {e.entity_version for e in (existing.pending_config.served_entities or [])} if existing.pending_config else set()
    if target in pending_versions:
        print(f"Endpoint pending v{target} — skip")
    elif target in served_versions and not pending_versions:
        print(f"Endpoint already serving v{target} (route-optimized) — skip")
    else:
        _r = _rq.put(f"{_host}/api/2.0/serving-endpoints/{endpoint_name}/config", headers=_hdrs,
                     data=_json.dumps({"served_entities": [_served_entity]}), timeout=60)
        print(f"Updated route-optimized endpoint to v{target} -> {_r.status_code}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Direct scorer — interactive what-if quote form
# MAGIC
# MAGIC The same `MotorPricingScorer` class + artifacts, logged as a PLAIN
# MAGIC pyfunc (no FeatureLookup) so it accepts a full feature vector directly.
# MAGIC Backs a small, scale-to-zero `motor_pricing_scorer_direct` endpoint. The
# MAGIC quote form pushes the editable fields; the app composes them onto the
# MAGIC policy's other features (pulled from the feature table) and posts the
# MAGIC full vector here — so edits (mileage, value, age…) move the price live.
# MAGIC The FeatureLookup endpoint above (policy_id) is untouched and still
# MAGIC powers the live demo + load tester.

# COMMAND ----------

direct_uc_name = f"{fqn}.motor_pricing_scorer_direct"
direct_endpoint = "motor_pricing_scorer_direct"
_sample_features = (
    spark.table(f"{fqn}.unified_motor_table_live").select(*UNION_FEATURES).limit(1).toPandas()
)
with mlflow.start_run(run_name="motor_pricing_scorer_direct") as _drun:
    mlflow.pyfunc.log_model(
        artifact_path         = "scorer_direct",
        python_model          = MotorPricingScorer(),
        artifacts             = artifact_paths,
        registered_model_name = direct_uc_name,
        input_example         = _sample_features,
        pip_requirements=[
            "mlflow>=2.12",
            "scikit-learn", "lightgbm", "statsmodels",
            "pandas", "numpy", "databricks-sdk",
        ],
    )
    print(f"Logged direct scorer for run {_drun.info.run_id}")

direct_latest = max(int(v.version) for v in client.search_model_versions(f"name='{direct_uc_name}'"))
client.set_registered_model_alias(direct_uc_name, "champion", direct_latest)
print(f"Direct scorer v{direct_latest} aliased champion")

# Deploy/reconcile the small scale-to-zero direct endpoint (REST for sizing).
_direct_entity = {
    "entity_name": direct_uc_name,
    "entity_version": str(direct_latest),
    "scale_to_zero_enabled": True,
    "workload_size": "Small",
    "workload_type": "CPU",
}
try:
    w.serving_endpoints.get(direct_endpoint)
    _de_exists = True
except Exception:
    _de_exists = False
if _de_exists:
    _r = _rq.put(f"{_host}/api/2.0/serving-endpoints/{direct_endpoint}/config", headers=_hdrs,
                 data=_json.dumps({"served_entities": [_direct_entity]}), timeout=60)
    print(f"direct endpoint reconcile -> {_r.status_code}")
else:
    _r = _rq.post(f"{_host}/api/2.0/serving-endpoints", headers=_hdrs,
                  data=_json.dumps({"name": direct_endpoint, "config": {"served_entities": [_direct_entity]}}), timeout=60)
    print(f"direct endpoint create -> {_r.status_code}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "scorer_uc_name":        scorer_uc_name,
    "version":               latest,
    "endpoint":              endpoint_name,
    "direct_uc_name":        direct_uc_name,
    "direct_version":        direct_latest,
    "direct_endpoint":       direct_endpoint,
    "champions":             CHAMPIONS,
    "rating_engine_version": RATING_CFG["version"],
}))
