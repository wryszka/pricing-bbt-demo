# Databricks notebook source
# MAGIC %md
# MAGIC # Backdate historical versions of the production models
# MAGIC
# MAGIC For each of the 4 current champions (freq_glm, sev_glm, demand_gbm,
# MAGIC fraud_gbm), register 11 additional UC versions representing monthly
# MAGIC retraining over 2025-05 → 2026-03. Each uses the **same model bytes**
# MAGIC as the current champion but carries its own MLflow run with
# MAGIC:
# MAGIC - a `simulation_date` tag
# MAGIC - a `story` tag linking to a narrative in the governance pack
# MAGIC - plausibly-drifted metrics (the Gini nudges etc. come from `_shared`)
# MAGIC
# MAGIC Marked `simulated=true` so a regulator querying the pack history gets
# MAGIC the honest signal that these are replayed from synthetic data — the
# MAGIC current champion is the only one trained on real (current) data.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering lightgbm --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

import json, random
from datetime import datetime
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# Story catalogue — copied inline so this notebook doesn't import _shared (%pip
# restart makes relative imports painful). Keep in sync if stories change.
STORIES = {
    "freq_glm": [
        ("2025-05-01","baseline",              "Monthly retrain — stable book, no material data changes.", 1.00),
        ("2025-06-01","baseline",              "Routine refresh. Mild seasonal uplift on liability.",       1.01),
        ("2025-07-01","flood_spike",           "Summer flood event spiked claim frequency in flood zones 6-8. Retrain surfaces a +14% loading on those sectors.", 0.94),
        ("2025-08-01","flood_spike_correction","Post-flood correction. Model re-weights historical flood to avoid over-loading.", 0.97),
        ("2025-09-01","bureau_v3",             "Credit bureau feed v3 swap introduced a bias in credit_score mean — Gini down 6%, caught on review.", 0.94),
        ("2025-10-01","bureau_fix",            "Bureau v3 bias remediated; retrain with re-centered scores recovers lift.", 1.00),
        ("2025-11-01","baseline",              "Steady state. Monthly refresh.",                           1.01),
        ("2025-12-01","year_end",              "Year-end training cut-off. Calibration solid.",             1.02),
        ("2026-01-01","postcode_refresh",      "ONSPD 2026 postcode enrichment applied — IMD deciles refreshed, +3% lift.", 1.04),
        ("2026-02-01","calibration_drift",     "Observed overprediction on low-turnover SMEs (actual freq 0.08, predicted 0.11). Flagged for calibration fix.", 0.96),
        ("2026-03-01","calibration_fix",       "Calibration recalibrated with isotonic regression overlay — overprediction corrected.", 1.01),
    ],
    "sev_glm": [
        ("2025-05-01","baseline",              "Monthly retrain.", 1.00),
        ("2025-06-01","baseline",              "Routine refresh.", 1.01),
        ("2025-07-01","large_loss_outlier",    "Single £8M loss in Manchester distorts severity Gamma fit. Outlier flagged for robust-regression variant.", 0.92),
        ("2025-08-01","outlier_excluded",      "Large-loss outlier excluded from training set per large-loss policy; model stabilises.", 1.02),
        ("2025-09-01","baseline",              "Stable retrain. Minor drift.", 1.00),
        ("2025-10-01","claims_handling_change","New claims-handling guidelines reduce average settlement. Severity baseline shifts down ~7%.", 0.95),
        ("2025-11-01","baseline",              "Steady.", 1.00),
        ("2025-12-01","year_end",              "Year-end cut. Calibration fine.", 1.01),
        ("2026-01-01","new_peril_split",       "Storm + subsidence losses split into separate severity models next iteration. This version still combined.", 1.00),
        ("2026-02-01","baseline",              "Stable.", 1.00),
        ("2026-03-01","calibration_fix",       "Severity overshot on Heavy Timber construction — re-ran with construction interaction term.", 1.02),
    ],
    "demand_gbm": [
        ("2025-05-01","baseline",              "Monthly retrain on quote stream.", 1.00),
        ("2025-06-01","baseline",              "Routine refresh.", 1.01),
        ("2025-07-01","competitor_pricing",    "Competitor A dropped rates on retail — our conversion dipped 6% in that segment. Model rebaselines.", 0.96),
        ("2025-08-01","baseline",              "Stable after competitor noise subsides.", 1.00),
        ("2025-09-01","broker_channel_drift",  "New broker partnership skewed the quote mix — model under-fitted on direct channel. Retrained with channel stratification.", 0.97),
        ("2025-10-01","channel_fix",           "Channel stratification fix applied. Lift recovered.", 1.02),
        ("2025-11-01","baseline",              "Steady.", 1.00),
        ("2025-12-01","year_end",              "Year-end stable.", 1.01),
        ("2026-01-01","price_elasticity_shift","Post-budget price sensitivity increased across SME book — elasticity refit.", 1.02),
        ("2026-02-01","baseline",              "Stable.", 1.00),
        ("2026-03-01","baseline",              "Minor refresh.", 1.01),
    ],
    "fraud_gbm": [
        ("2025-05-01","baseline",              "Monthly retrain.", 1.00),
        ("2025-06-01","baseline",              "Routine refresh.", 1.01),
        ("2025-07-01","fraud_ring_detected",   "Detected organised fraud ring in London E postcodes (inflated theft claims). Retrain picked up the pattern — AUC +0.04.", 1.04),
        ("2025-08-01","ring_remediated",       "Fraud ring prosecuted; training drops flagged cases to avoid re-learning the same pattern.", 1.00),
        ("2025-09-01","label_drift",           "Claim-handlers adopted new fraud-label taxonomy — training labels shifted. Model flagged for revalidation.", 0.94),
        ("2025-10-01","label_fix",             "Label taxonomy reconciled against historical data; model retrained cleanly.", 1.01),
        ("2025-11-01","baseline",              "Steady.", 1.00),
        ("2025-12-01","year_end",              "Year-end calibration.", 1.01),
        ("2026-01-01","baseline",              "Routine.", 1.00),
        ("2026-02-01","high_cost_fp",          "High-cost false positives on legitimate commercial theft claims — precision dropped, senior underwriter reviews raised.", 0.96),
        ("2026-03-01","threshold_tuning",      "Decision threshold raised from 0.5 to 0.62 to cut FPR; AUC unchanged but precision recovered.", 1.00),
    ],
}

MODEL_CONFIGS = [
    {"name": "freq_glm",   "experiment": "production_freq",   "flavor": "sklearn",  "primary": "gini", "feature_table_suffix": "unified_pricing_table_live"},
    {"name": "sev_glm",    "experiment": "production_sev",    "flavor": "sklearn",  "primary": "gini", "feature_table_suffix": "unified_pricing_table_live"},
    {"name": "demand_gbm", "experiment": "production_demand", "flavor": "lightgbm", "primary": "auc",  "feature_table_suffix": "quotes"},
    {"name": "fraud_gbm",  "experiment": "production_fraud",  "flavor": "lightgbm", "primary": "auc",  "feature_table_suffix": "unified_pricing_table_live"},
]

# COMMAND ----------

def nudge_metrics(base: dict, nudge: float, seed: int) -> dict:
    """Apply the story's primary-metric nudge + ±3% secondary noise."""
    rng = random.Random(seed)
    out = {}
    for k, v in base.items():
        if v is None:
            continue
        if k in ("gini", "auc"):
            out[k] = round(float(v) * nudge, 4)
        elif k in ("rmse", "mae_gbp", "logloss", "deviance", "aic", "bic"):
            # Loss metrics move inversely with ranking nudge
            out[k] = round(float(v) / max(0.7, nudge), 4)
        else:
            out[k] = round(float(v) * (1 + rng.uniform(-0.03, 0.03)), 4)
    return out


def get_latest_version(client: MlflowClient, name: str):
    versions = client.search_model_versions(f"name='{name}'")
    if not versions:
        raise RuntimeError(f"No versions registered for {name}")
    return max(versions, key=lambda v: int(v.version))


def register_replay_version(client: MlflowClient, uc_name: str, source: str, run_id: str, tags: dict):
    """Register a new UC version of `uc_name` pointing at the champion's model
    artifact (`source`). We don't re-serialize the model — we just create a
    new UC version that references the same bytes, tied to a fresh run that
    holds the simulation-date tags + nudged metrics."""
    return client.create_model_version(
        name      = uc_name,
        source    = source,
        run_id    = run_id,
        tags      = {k: str(v) for k, v in tags.items()},
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Backdate loop

# COMMAND ----------

client = MlflowClient()
summary = []

for cfg in MODEL_CONFIGS:
    uc_name = f"{fqn}.{cfg['name']}"
    print(f"\n=== {cfg['name']} ===")

    # Load the current champion — assumed to be the most recent version
    try:
        champ = get_latest_version(client, uc_name)
    except Exception as e:
        print(f"  ERROR: {e}. Skipping.")
        continue
    print(f"  Champion version: v{champ.version}, run_id {champ.run_id}")

    # Pull champion metrics as the base for nudging
    try:
        run = client.get_run(champ.run_id)
        base_metrics = {k: v for k, v in run.data.metrics.items()}
    except Exception:
        base_metrics = {"gini": 0.22, "rmse": 0.5}
    print(f"  Base metrics: { {k: round(v, 4) for k, v in base_metrics.items()} }")

    champ_source = champ.source   # UC artifact URI of the champion version
    mlflow.set_experiment(f"/Workspace/Users/{user}/pricing_workbench_{cfg['experiment']}")

    import hashlib
    for i, (sim_date, story_tag, story_text, nudge) in enumerate(STORIES[cfg["name"]]):
        seed = int(hashlib.sha256((sim_date + cfg["name"]).encode()).hexdigest()[:8], 16)
        metrics = nudge_metrics(base_metrics, nudge, seed)
        with mlflow.start_run(run_name=f"{cfg['name']}_{sim_date[:7]}") as run:
            mlflow.set_tags({
                "simulation_date": sim_date,
                "story":           story_tag,
                "story_text":      story_text,
                "simulated":       "true",
                "feature_table":   f"{fqn}.{cfg['feature_table_suffix']}",
                "model_type":      "simulated_replay",
            })
            mlflow.log_params({"nudge_multiplier": nudge,
                               "replay_of_run":    champ.run_id,
                               "replay_of_version":champ.version})
            mlflow.log_metrics(metrics)
            register_replay_version(
                client    = client,
                uc_name   = uc_name,
                source    = champ_source,
                run_id    = run.info.run_id,
                tags      = {"simulation_date": sim_date, "story": story_tag,
                             "simulated": "true"},
            )

        # Audit
        try:
            det = json.dumps({"simulated": True, "simulation_date": sim_date,
                              "story": story_tag,
                              "mlflow_run_id": run.info.run_id,
                              "replay_of_run": champ.run_id, **metrics}).replace("'", "''")
            spark.sql(f"""
                INSERT INTO {fqn}.audit_log
                  (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
                SELECT uuid(), 'model_trained', 'model', '{cfg['name']}', '{sim_date}', '{user}',
                       CAST('{sim_date}T09:00:00.000+00:00' AS TIMESTAMP), '{det}', 'notebook'
            """)
        except Exception as e:
            print(f"  audit (sim_date={sim_date}): {e}")

        summary.append({"model": cfg["name"], "simulation_date": sim_date, "story": story_tag,
                        "primary_metric_name": cfg["primary"], "primary_metric_value": metrics.get(cfg["primary"])})
        print(f"  v{sim_date[:7]} [{story_tag:28}] {cfg['primary']}={metrics.get(cfg['primary'])}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

display(spark.createDataFrame([(r["model"], r["simulation_date"], r["story"],
                                r["primary_metric_name"], r["primary_metric_value"])
                               for r in summary],
                              ["model", "simulation_date", "story",
                               "primary_metric", "metric_value"])
        .orderBy("model", "simulation_date"))
