"""Model Development tab — notebook metadata, recent MLflow runs, audit-logged
open-notebook events, and the static library pin list.

All notebooks live in the repo under src/04_models/ and are synced to the
workspace during bundle deploy. Opening a notebook from the app:
  1) Logs an audit event (event_type = "notebook_opened").
  2) Returns the workspace URL the client navigates to.
"""
from __future__ import annotations
import logging
from typing import Any
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import get_workspace_client, get_workspace_host

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/development", tags=["development"])

# ---------------------------------------------------------------------------
# Notebook catalogue — the hand-curated set shown on the Model Development tab.
# `is_featured` controls whether the notebook gets a large card up top; the
# rest surface in the "Can you also do this?" model library tile grid.
#
# `path` is the workspace path where the bundle deploys this notebook (see
# databricks.yml → sync target). The workspace URL is
# {host}/?o=WORKSPACE_ID#folder/{path_without_extension}.
# ---------------------------------------------------------------------------

_BUNDLE_BASE = "/Workspace/Users/laurence.ryszka@databricks.com/.bundle/pricing-upt-demo/dev/files/src/04_models"

NOTEBOOKS: list[dict[str, Any]] = [
    # ---- featured: the headline cards ----
    # NOTE: paths are the deployed workspace paths WITHOUT the .py extension —
    # Databricks strips it when it imports a source file as a notebook. We
    # resolve each path to an object_id at open-time via workspace.get_status
    # so the URL is /#notebook/{id}, which opens directly in the editor.
    {
        "id":          "hello_world_freq",
        "title":       "Hello, Pricing Workbench — Minimal Freq GLM",
        "description": "The shortest end-to-end example: load the Mart, fit a Poisson GLM, log to MLflow, register in Unity Catalog. Start here if you've never trained a pricing model on Databricks.",
        "tags":        ["GLM", "beginner", "~2 min"],
        "is_featured": True,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/hello_world_freq",
    },
    {
        "id":          "model_01_glm_frequency",
        "title":       "Frequency GLM (Poisson, production-grade)",
        "description": "Full frequency model: rating-factor subset, relativity tables, p-values, MLflow + FeatureLookup for serving-time auto-binding. This is the reference GLM.",
        "tags":        ["GLM", "frequency", "production", "~4 min"],
        "is_featured": True,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/model_01_glm_frequency",
    },
    {
        "id":          "model_02_glm_severity",
        "title":       "Severity GLM (Gamma, log link)",
        "description": "Gamma GLM on observed claim severity. Relativity table, diagnostics, registered to UC. Pair this with the frequency model for a two-part pricing approach.",
        "tags":        ["GLM", "severity", "production", "~4 min"],
        "is_featured": True,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/model_02_glm_severity",
    },
    {
        "id":          "model_03_gbm_demand",
        "title":       "GBM Demand (LightGBM)",
        "description": "LightGBM demand / conversion model over quoted premiums. Runs over the quote stream, learns the price-elasticity surface, feeds the pricing optimisation step.",
        "tags":        ["GBM", "LightGBM", "demand", "~5 min"],
        "is_featured": True,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/model_03_gbm_demand",
    },
    # ---- model library: everything else, shown as compact tiles ----
    {
        "id":          "model_04_gbm_risk_uplift",
        "title":       "Risk Uplift GBM (residuals)",
        "description": "Second-stage GBM on GLM residuals — captures non-linear interactions the GLM misses without sacrificing the GLM's interpretability.",
        "tags":        ["GBM", "two-stage", "uplift"],
        "is_featured": False,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/model_04_gbm_risk_uplift",
    },
    {
        "id":          "model_05_fraud_propensity",
        "title":       "Fraud Propensity",
        "description": "Binary classifier flagging suspicious quotes. Used as a rating loading, not a bind/reject decision.",
        "tags":        ["classifier", "fraud"],
        "is_featured": False,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/model_05_fraud_propensity",
    },
    {
        "id":          "model_06_retention",
        "title":       "Retention / Churn GBM",
        "description": "Renewal-time churn model feeding the retention loading in the pricing waterfall.",
        "tags":        ["classifier", "retention"],
        "is_featured": False,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/model_06_retention",
    },
    {
        "id":          "challenger_comparison",
        "title":       "Challenger Comparison",
        "description": "Baseline vs challenger model factories, ranked by Gini lift. Writes the comparison table consumed by the 'Proof of lift' chart on this page.",
        "tags":        ["comparison", "lift attribution"],
        "is_featured": False,
        "status":      "built",
        "path":        f"{_BUNDLE_BASE}/challenger_comparison",
    },
    # ---- on-request: model types the platform supports but no canned example ----
    {"id": "tweedie_glm",       "title": "Tweedie GLM (single-stage pure premium)",
     "description": "One-shot frequency × severity via a Tweedie distribution — useful when the two-part decomposition isn't adding value.",
     "tags": ["GLM", "Tweedie", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "gam_frequency",     "title": "Generalised Additive Model (GAM)",
     "description": "Non-linear main effects with partial smoothers — interpretable alternative to GBMs when regulator wants transparency.",
     "tags": ["GAM", "splines", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "xgboost_tweedie",   "title": "XGBoost Tweedie",
     "description": "XGBoost with a Tweedie objective — drop-in alternative to LightGBM on pure-premium targets.",
     "tags": ["GBM", "XGBoost", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "quantile_glm",      "title": "Quantile Regression",
     "description": "Models the 90th / 99th percentile of the loss distribution — used for tail-risk rating and reinsurance.",
     "tags": ["regression", "tail", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "survival_renewal",  "title": "Survival Analysis (renewal timing)",
     "description": "Cox PH or accelerated failure time on how long a policy lives. Feeds a lifetime-value model.",
     "tags": ["survival", "lifetime", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "pymc_bayesian",     "title": "Bayesian Hierarchical GLM (PyMC)",
     "description": "Partial-pooling GLM using PyMC — useful when credibility is thin in some segments (new postcodes, rare SICs).",
     "tags": ["Bayesian", "PyMC", "credibility", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "isolation_forest",  "title": "Isolation Forest (outlier pricing)",
     "description": "Unsupervised anomaly detection on quotes / policies — flag risks that look unlike anything in the book.",
     "tags": ["unsupervised", "anomaly", "on request"],
     "is_featured": False, "status": "on_request"},
    {"id": "monotonic_gbm",     "title": "Monotonic GBM (regulatory constraints)",
     "description": "LightGBM / XGBoost with monotonic constraints per feature — keeps the non-linearity but guarantees direction of effect (critical for fairness reviews).",
     "tags": ["GBM", "monotonic", "fairness", "on request"],
     "is_featured": False, "status": "on_request"},
]

# ---------------------------------------------------------------------------
# Runtime library pins — shown on the Libraries panel.
# These are the versions the Databricks Serverless ML runtime pins by default.
# Not introspected live because app runtime ≠ notebook runtime.
# ---------------------------------------------------------------------------

RUNTIME_LIBRARIES = [
    {"name": "mlflow",                        "version": "≥ 2.9",   "purpose": "Experiment tracking, model registry, fe.log_model"},
    {"name": "databricks-feature-engineering","version": "≥ 0.8",   "purpose": "FeatureLookup binding for serving-time feature fetch"},
    {"name": "statsmodels",                   "version": "≥ 0.14",  "purpose": "GLMs, GAMs, coefficient-level diagnostics for actuaries"},
    {"name": "lightgbm",                      "version": "≥ 4.1",   "purpose": "GBM for frequency/severity/demand/uplift models"},
    {"name": "scikit-learn",                  "version": "≥ 1.4",   "purpose": "Train/test splits, metrics, sklearn-wrapped GLM logging"},
    {"name": "shap",                          "version": "≥ 0.45",  "purpose": "Tree-SHAP explanations for GBMs (fairness + regulation)"},
    {"name": "pandas",                        "version": "≥ 2.1",   "purpose": "In-memory data manipulation between Spark and models"},
    {"name": "pyspark",                       "version": "runtime", "purpose": "Lakehouse access; reads the Modelling Mart as a Delta table"},
]

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

def _notebook_open_url(path: str) -> str:
    """Build the workspace URL that opens the notebook in the user's browser.
    Uses the /#workspace{path} fragment form — works as long as the logged-in
    user (not the app SP) has access. We deliberately don't resolve to the
    /#notebook/{object_id} form because that would require the app SP to have
    read access on the user's personal bundle directory, which it doesn't.

    Databricks workspace paths are prefixed with /Workspace internally but the
    URL fragment doesn't include that prefix — strip it before building."""
    host = get_workspace_host()
    # Strip the /Workspace prefix; the URL fragment starts directly with /Users/...
    rel = path[len("/Workspace"):] if path.startswith("/Workspace/") else path
    return f"{host}/#workspace{rel}"


@router.get("/notebooks")
async def list_notebooks() -> dict:
    """Return the notebook catalogue — featured + model-library tiles — with each
    card's workspace URL pre-resolved so the browser opens it directly."""
    out = []
    for nb in NOTEBOOKS:
        entry = {**nb}
        if nb.get("path"):
            entry["workspace_url"] = _notebook_open_url(nb["path"])
            entry["git_url"]       = f"https://github.com/wryszka/pricing-workbench/blob/main/src/04_models/{nb['id']}.py"
        out.append(entry)
    return {"notebooks": out, "libraries": RUNTIME_LIBRARIES}


class OpenRequest(BaseModel):
    notebook_id: str


@router.post("/open-notebook")
async def open_notebook(req: OpenRequest) -> dict:
    """Record an audit event then return the workspace URL the client navigates to.
    The client opens the URL in a new tab."""
    nb = next((n for n in NOTEBOOKS if n["id"] == req.notebook_id), None)
    if not nb:
        raise HTTPException(404, f"Unknown notebook: {req.notebook_id}")
    if nb.get("status") != "built":
        raise HTTPException(400, "This model type is 'on request' — no runnable notebook yet.")

    url = _notebook_open_url(nb["path"])

    await log_audit_event(
        event_type  = "notebook_opened",
        entity_type = "notebook",
        entity_id   = req.notebook_id,
        details     = {"title": nb["title"], "path": nb["path"], "workspace_url": url},
    )
    return {"notebook_id": req.notebook_id, "workspace_url": url, "title": nb["title"]}


@router.get("/recent-runs")
async def recent_runs(limit: int = 10) -> dict:
    """Pull the most recent MLflow runs across any experiment whose path contains
    'pricing_workbench'. Uses the workspace-client's MLflow helper."""
    try:
        w = get_workspace_client()
        # Find matching experiments
        from mlflow.tracking import MlflowClient
        mlclient = MlflowClient()
        exps = mlclient.search_experiments(
            filter_string="name LIKE '%pricing_workbench%'",
            max_results=50,
        )
    except Exception as e:
        logger.warning("recent_runs: could not list experiments: %s", e)
        return {"runs": [], "error": str(e)[:300]}

    if not exps:
        return {"runs": []}

    try:
        runs = mlclient.search_runs(
            experiment_ids = [e.experiment_id for e in exps],
            order_by       = ["attributes.start_time DESC"],
            max_results    = int(limit),
        )
    except Exception as e:
        logger.warning("recent_runs: search_runs failed: %s", e)
        return {"runs": [], "error": str(e)[:300]}

    host = get_workspace_host()
    out = []
    exp_by_id = {e.experiment_id: e for e in exps}
    for r in runs:
        exp = exp_by_id.get(r.info.experiment_id)
        metrics = r.data.metrics or {}
        # Pick the most "interesting" metric to surface on the row — Gini / RMSE
        # / AUC in that priority order. Runs without any of these show "—".
        key_metric = None
        for candidate in ("gini", "gini_on_incurred", "rmse", "auc", "aic"):
            if candidate in metrics:
                key_metric = {"name": candidate, "value": round(float(metrics[candidate]), 4)}
                break

        out.append({
            "run_id":         r.info.run_id,
            "run_name":       r.data.tags.get("mlflow.runName") or r.info.run_name or "(unnamed)",
            "experiment_id":  r.info.experiment_id,
            "experiment_name":(exp.name if exp else None),
            "status":         r.info.status,
            "start_time":     datetime.fromtimestamp((r.info.start_time or 0) / 1000).isoformat() if r.info.start_time else None,
            "user":           r.data.tags.get("mlflow.user"),
            "key_metric":     key_metric,
            "url":            f"{host}/ml/experiments/{r.info.experiment_id}/runs/{r.info.run_id}",
        })

    return {"runs": out, "experiments_matched": len(exps)}
