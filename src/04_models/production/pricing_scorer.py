# Databricks notebook source
# MAGIC %md
# MAGIC # Pricing Scorer — unified serving endpoint for the 4 champions
# MAGIC
# MAGIC One pyfunc that wraps `freq_glm`, `sev_glm`, `demand_gbm`, `fraud_gbm`
# MAGIC current champions. Deployed as a single Model Serving endpoint
# MAGIC (`pricing_scorer`) — the live pricing runtime for the Pricing Engine
# MAGIC tab.  Sub-second per-quote latency after warm start.
# MAGIC
# MAGIC Why unified:
# MAGIC  * single cold-start, one round-trip from the app
# MAGIC  * one endpoint to grant + monitor + audit
# MAGIC  * the app doesn't have to know about individual UC model versions
# MAGIC
# MAGIC Re-run this notebook whenever any champion alias flips — it bakes the
# MAGIC current versions in at log time, then deploys as a new endpoint
# MAGIC version.  Historical (non-champion) scoring stays on the Compare &
# MAGIC Test batch job.

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("endpoint_name", "pricing_scorer")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-agents databricks-sdk statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
fqn           = f"{catalog}.{schema}"
scorer_uc_name = f"{fqn}.pricing_scorer"

import json, os, tempfile
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec
from mlflow.tracking import MlflowClient
from mlflow.artifacts import download_artifacts

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

FAMILIES = ("freq_glm", "sev_glm", "demand_gbm", "fraud_gbm")

# Resolve current champion per family — baked into the logged artefact.
def _champion_version(family: str) -> str:
    mv = client.get_model_version_by_alias(f"{fqn}.{family}", "champion")
    return str(mv.version)

CHAMPIONS = {fam: _champion_version(fam) for fam in FAMILIES}
print("Champions to bake in:", CHAMPIONS)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download the raw flavor models into the log payload
# MAGIC
# MAGIC Each champion is downloaded from UC and the deepest MLmodel directory
# MAGIC (the raw sklearn wrapper or LightGBM booster — not the FE pyfunc) is
# MAGIC packaged as an artifact of this scorer. The scorer's `load_context`
# MAGIC loads them back at serving-endpoint startup.

# COMMAND ----------

def _pull_raw_flavor(family: str, version: str) -> str:
    """Download the UC model artifact tree, find the deepest MLmodel
    directory, copy just that subtree. Returns the local dir."""
    import shutil
    tmp  = tempfile.mkdtemp(prefix=f"{family}_v{version}_")
    uri  = f"models:/{fqn}.{family}/{version}"
    root = download_artifacts(artifact_uri=uri, dst_path=tmp)
    mlmodel_dirs = [r for r, _, fs in os.walk(root) if "MLmodel" in fs]
    if not mlmodel_dirs:
        raise RuntimeError(f"{family} v{version}: no MLmodel under {root}")
    deepest = max(mlmodel_dirs, key=lambda p: p.count(os.sep))
    # Copy to a clean path so log_model's artifacts dict is happy
    dest = f"{tempfile.mkdtemp(prefix=f'{family}_clean_')}/{family}"
    shutil.copytree(deepest, dest)
    return dest

artifact_paths = {
    fam: _pull_raw_flavor(fam, ver)
    for fam, ver in CHAMPIONS.items()
}
print("Artifact paths:")
for k, v in artifact_paths.items():
    print(f"  {k}: {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pyfunc definition — loads 4 models, scores a batch in one call

# COMMAND ----------

class PricingScorer(PythonModel):
    """Unified scorer across the 4 production champions. Reads champion
    versions from a config.json artefact so the logged pickle works
    regardless of whether the source class is inspectable."""

    def load_context(self, context):
        import json as _j
        import mlflow.sklearn, mlflow.lightgbm
        with open(context.artifacts["config"]) as fh:
            self.champions = _j.load(fh)
        self.freq   = mlflow.sklearn.load_model(context.artifacts["freq_glm"])
        self.sev    = mlflow.sklearn.load_model(context.artifacts["sev_glm"])
        self.demand = mlflow.lightgbm.load_model(context.artifacts["demand_gbm"])
        self.fraud  = mlflow.lightgbm.load_model(context.artifacts["fraud_gbm"])

    # GLM wrappers run get_dummies(drop_first=True) internally on object
    # dtype. On a single-row DataFrame, get_dummies sees only one value
    # per categorical and silently drops the category contribution.
    # Workaround: pad the input with synthetic rows that exercise every
    # training-time category value so get_dummies produces the full set
    # of indicator columns. Score the padded frame, then return only
    # the original-row predictions.
    _GLM_CATS = {
        "industry_risk_tier": ["High", "Low", "Medium"],
        "construction_type":  ["Fire Resistive", "Frame", "Heavy Timber",
                                "Joisted Masonry", "Non-Combustible"],
    }

    def _prep(self, df):
        out = df.copy()
        for c in out.columns:
            if out[c].dtype == "object":
                out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
        return out

    def _pad_for_categoricals(self, df):
        """Append rows that span every (industry_risk_tier × construction_type)
        combination so get_dummies sees every category. Returns the padded
        frame and the count of original rows."""
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
        """LightGBM needs each feature column in the exact dtype it had at
        training — categorical cols with training category lists, numeric
        cols as float. Use booster.pandas_categorical to tell them apart."""
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
                # Numeric feature — coerce to float, fill missing with 0.0
                if present:
                    built[name] = pd.to_numeric(prepped[name], errors="coerce").fillna(0.0).astype(float)
                else:
                    built[name] = pd.Series([0.0] * len(prepped), index=prepped.index, dtype=float)
        return np.asarray(booster.predict(built), dtype=float).ravel()

    def predict(self, context, model_input, params=None):
        import pandas as pd, json as _j

        # Accept DataFrame or list-of-dicts; also accept a `features` column
        # containing JSON-string rows (that's what the serving endpoint gets
        # from our FastAPI app).
        if not hasattr(model_input, "columns"):
            model_input = pd.DataFrame(list(model_input))
        if "features" in model_input.columns and len(model_input.columns) == 1:
            parsed = [
                _j.loads(r) if isinstance(r, str) else (r or {})
                for r in model_input["features"].tolist()
            ]
            model_input = pd.DataFrame(parsed)

        freq   = self._score_glm(self.freq,   model_input)
        sev    = self._score_glm(self.sev,    model_input)
        demand = self._score_lgb(self.demand, model_input)
        fraud  = self._score_lgb(self.fraud,  model_input)

        n = len(model_input)
        return pd.DataFrame({
            "freq_pred":       freq,
            "sev_pred":        sev,
            "demand_pred":     demand,
            "fraud_pred":      fraud,
            "freq_version":    [self.champions["freq_glm"]]   * n,
            "sev_version":     [self.champions["sev_glm"]]    * n,
            "demand_version":  [self.champions["demand_gbm"]] * n,
            "fraud_version":   [self.champions["fraud_gbm"]]  * n,
        })


# COMMAND ----------

# Write the champion config to a file — loaded back in load_context.
cfg_path = f"{tempfile.mkdtemp()}/config.json"
with open(cfg_path, "w") as fh:
    json.dump(CHAMPIONS, fh)

# Assemble artefact bundle (4 models + config)
artifact_paths["config"] = cfg_path

signature = ModelSignature(
    inputs=Schema([ColSpec("string", "features")]),
    outputs=Schema([
        ColSpec("double", "freq_pred"),
        ColSpec("double", "sev_pred"),
        ColSpec("double", "demand_pred"),
        ColSpec("double", "fraud_pred"),
        ColSpec("string", "freq_version"),
        ColSpec("string", "sev_version"),
        ColSpec("string", "demand_version"),
        ColSpec("string", "fraud_version"),
    ]),
)

input_example = {"features": json.dumps({
    "sum_insured": 2500000, "annual_turnover": 850000, "current_premium": 1200,
    "industry_risk_tier": "Medium", "construction_type": "Non-Combustible",
    "postcode_sector": "EC1A", "region": "London",
    "credit_score": 620, "ccj_count": 1, "years_trading": 12,
})}

# COMMAND ----------

from mlflow.models.resources import DatabricksServingEndpoint  # not used, no inner endpoint call

with mlflow.start_run(run_name="pricing_scorer_deploy"):
    mi = mlflow.pyfunc.log_model(
        artifact_path="scorer",
        python_model=PricingScorer(),
        artifacts=artifact_paths,
        input_example=input_example,
        signature=signature,
        registered_model_name=scorer_uc_name,
        pip_requirements=[
            "mlflow>=2.12", "scikit-learn", "lightgbm",
            "statsmodels", "pandas", "numpy", "databricks-sdk",
        ],
    )
    print("Logged:", mi.model_uri)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy as a Model Serving endpoint (scale to zero)

# COMMAND ----------

latest = max(int(v.version) for v in client.search_model_versions(f"name='{scorer_uc_name}'"))
print(f"Deploying {scorer_uc_name} v{latest} → endpoint '{endpoint_name}'")

try:
    from databricks import agents
    deployment = agents.deploy(
        model_name=scorer_uc_name,
        model_version=latest,
        scale_to_zero=True,
        tags={"project": "pricing_workbench", "purpose": "unified_pricing_scorer",
              **{f"baked_{k}": v for k, v in CHAMPIONS.items()}},
    )
    print("databricks-agents deploy kicked off:", deployment)
except Exception as e:
    print(f"databricks-agents.deploy failed, trying serving_endpoints: {e}")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    w = WorkspaceClient()
    served = [ServedEntityInput(
        entity_name=scorer_uc_name, entity_version=str(latest),
        scale_to_zero_enabled=True, workload_size="Small",
    )]
    cfg = EndpointCoreConfigInput(name=endpoint_name, served_entities=served)
    try:
        w.serving_endpoints.get(endpoint_name)
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=served)
        print("Updated existing endpoint.")
    except Exception:
        w.serving_endpoints.create(name=endpoint_name, config=cfg)
        print("Created new endpoint.")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "scorer_uc_name": scorer_uc_name,
    "version":        latest,
    "endpoint":       endpoint_name,
    "champions":      CHAMPIONS,
}))
