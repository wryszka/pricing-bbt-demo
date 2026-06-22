# Databricks notebook source
# MAGIC %md
# MAGIC # Compare & Test — batch scoring for candidate vs champion
# MAGIC
# MAGIC Given a model family, a list of UC versions (2-5) and an optional what-if
# MAGIC scenario, this notebook:
# MAGIC  1. Loads each version via `mlflow.pyfunc.load_model`
# MAGIC  2. Pulls a stratified 5000-policy sample from the Modelling Mart (or
# MAGIC     a quote sample for demand_gbm)
# MAGIC  3. Applies the scenario as a feature perturbation in memory
# MAGIC  4. Scores each version on the same rows → apples-to-apples
# MAGIC  5. Computes: score distribution, A-vs-B shift, segment breakdown,
# MAGIC     outlier list, fresh holdout metric per version
# MAGIC  6. Writes a compact summary + heavy result to `{fqn}.compare_results`
# MAGIC     keyed on a cache hash so the app can poll for it
# MAGIC  7. Returns the cache_key + headline numbers via `dbutils.notebook.exit`

# COMMAND ----------

dbutils.widgets.text("catalog_name",   "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",    "pricing_upt")
dbutils.widgets.text("model_family",   "freq_glm")
dbutils.widgets.text("versions",       "")              # csv e.g. "30,31"
dbutils.widgets.text("portfolio_size", "5000")
dbutils.widgets.text("scenario_id",    "none")
dbutils.widgets.text("requested_by",   "app")

# COMMAND ----------

# MAGIC %pip install mlflow statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog    = dbutils.widgets.get("catalog_name")
schema     = dbutils.widgets.get("schema_name")
family     = dbutils.widgets.get("model_family")
versions_s = dbutils.widgets.get("versions")
port_size  = int(dbutils.widgets.get("portfolio_size") or 5000)
scenario   = dbutils.widgets.get("scenario_id") or "none"
user       = dbutils.widgets.get("requested_by") or "app"

fqn      = f"{catalog}.{schema}"
uc_name  = f"{fqn}.{family}"
VALID    = {"freq_glm", "sev_glm", "demand_gbm", "fraud_gbm"}
if family not in VALID:
    raise ValueError(f"family must be one of {VALID}, got '{family}'")

versions = [v.strip() for v in versions_s.split(",") if v.strip()]
if not 2 <= len(versions) <= 5:
    raise ValueError(f"versions must list 2-5 entries, got {versions}")

import json, hashlib, time, traceback
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

run_started = datetime.now(timezone.utc)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Derive a cache key — so re-runs of the same compare are free

# COMMAND ----------

cache_key_raw = json.dumps({
    "family": family, "versions": sorted(versions, key=int),
    "portfolio_size": port_size, "scenario": scenario,
}, sort_keys=True)
cache_key = hashlib.sha256(cache_key_raw.encode()).hexdigest()[:16]
print(f"cache_key={cache_key}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull the portfolio sample, stratified for a representative mix

# COMMAND ----------

from pyspark.sql import functions as F

if family == "demand_gbm":
    source_table = f"{fqn}.quotes"
    strat_cols   = ["channel", "region"]
    key_col      = "transaction_id"
else:
    source_table = f"{fqn}.unified_pricing_table_live"
    strat_cols   = ["region", "industry_risk_tier"]
    key_col      = "policy_id"

src = spark.table(source_table)
total_rows = src.count()
sample_frac = min(1.0, (port_size * 3) / max(1, total_rows))    # pull more than needed to then stratify
cand = src.sample(withReplacement=False, fraction=sample_frac, seed=42).toPandas()

# Stratified trim: roughly equal groups × strat_cols
def _stratified_pick(df: pd.DataFrame, cols, n):
    if df.empty:
        return df
    grouped = df.groupby(cols, dropna=False)
    per_group = max(1, n // max(1, len(grouped)))
    parts = []
    for _, g in grouped:
        parts.append(g.sample(n=min(len(g), per_group), random_state=42))
    out = pd.concat(parts, ignore_index=True)
    if len(out) > n:
        out = out.sample(n=n, random_state=42).reset_index(drop=True)
    return out

pdf_portfolio = _stratified_pick(cand, strat_cols, port_size)
print(f"Portfolio sample: {len(pdf_portfolio):,} rows from {source_table}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Apply what-if scenario as a feature perturbation (pure in-memory)

# COMMAND ----------

def apply_scenario(df: pd.DataFrame, scenario_id: str, family: str) -> pd.DataFrame:
    """Return a perturbed copy of the portfolio features — scenario semantics
    defined per model family. All perturbations are bounded so the result stays
    inside the model's training range."""
    out = df.copy()
    if scenario_id == "none":
        return out

    if scenario_id == "flood_plus_1":
        # Flood risk data updates: coastal or near-coastal postcodes climb
        # one zone. Affects freq / sev / fraud.
        if "flood_zone_rating" in out.columns:
            mask = out.get("is_coastal", pd.Series(False, index=out.index)).fillna(False).astype(bool)
            out.loc[mask, "flood_zone_rating"] = (out.loc[mask, "flood_zone_rating"].fillna(3).astype(int) + 1).clip(upper=10)
            if "composite_location_risk" in out.columns:
                out.loc[mask, "composite_location_risk"] = out.loc[mask, "composite_location_risk"].fillna(50).astype(float) * 1.10
        return out

    if scenario_id == "london_claims_surge_20pct":
        # London E postcodes see a 20% frequency uplift. Feature proxy:
        # bump claim_count_5y on matching rows so fraud/demand features shift.
        if "postcode_sector" in out.columns and "claim_count_5y" in out.columns:
            london_mask = out["postcode_sector"].astype(str).str.upper().str.startswith(("E", "SE", "EC"))
            out.loc[london_mask, "claim_count_5y"] = (out.loc[london_mask, "claim_count_5y"].fillna(0).astype(float) * 1.20)
        return out

    if scenario_id == "industry_mix_up":
        # Portfolio mix shifts toward higher-risk industries — bump industry
        # tier by 1 on a random 30% of the book.
        if "industry_risk_tier" in out.columns:
            rng = np.random.default_rng(42)
            pick = rng.random(len(out)) < 0.30
            out.loc[pick, "industry_risk_tier"] = (out.loc[pick, "industry_risk_tier"].fillna(3).astype(int) + 1).clip(upper=10)
        return out

    if scenario_id == "competitor_a_minus_5pct":
        # Competitor A drops rates 5% — only affects demand_gbm.
        if "competitor_a_min_rate" in out.columns:
            out["competitor_a_min_rate"] = out["competitor_a_min_rate"].astype(float) * 0.95
        if "market_median_rate" in out.columns:
            out["market_median_rate"] = out["market_median_rate"].astype(float) * 0.975
        return out

    print(f"  (unknown scenario '{scenario_id}' — pass-through)")
    return out

pdf_scenario = apply_scenario(pdf_portfolio, scenario, family)
print(f"Scenario applied: {scenario}  (rows={len(pdf_scenario):,})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load each UC version (in parallel) via mlflow.pyfunc

# COMMAND ----------

def _load_version(v):
    """Load the raw flavor model (sklearn wrapper or LightGBM booster) that
    fe.log_model stores nested inside the model artifact tree. Bypassing the
    outer FE-pyfunc wrapper avoids its Spark-UDF batch path and the nested
    schema enforcement — which MLflow re-imports fresh on every Spark worker,
    defeating driver-side patches. The raw wrapper already handles its own
    sanitation via _prep_raw / _transform."""
    import os, tempfile

    mv = client.get_model_version(uc_name, v)
    try:
        r = client.get_run(mv.run_id)
        tags    = dict(r.data.tags or {})
        metrics = dict(r.data.metrics or {})
        params  = dict(r.data.params or {})
    except Exception as e:
        tags, metrics, params = {}, {}, {}
        print(f"  v{v}: run fetch failed ({e})")

    # Backdated "simulated_replay" versions all share the champion's model
    # bytes; their `nudge_multiplier` param represents the story-driven
    # shift relative to the champion. Apply it at score time so Compare &
    # Test shows the intended drift across versions.
    try:
        nudge = float(params.get("nudge_multiplier")) if params.get("nudge_multiplier") else 1.0
    except Exception:
        nudge = 1.0

    # Walk the downloaded model artifact tree looking for the DEEPEST
    # MLmodel file — the outer one is the FE wrapper, the deepest is the
    # raw flavor (sklearn / lightgbm) we want. Use the models:/ URI so
    # UC-managed artifacts resolve correctly (run-scoped download does not
    # find them).
    from mlflow.artifacts import download_artifacts
    tmpdir     = tempfile.mkdtemp(prefix=f"mv{v}_")
    model_uri  = f"models:/{uc_name}/{v}"
    model_root = download_artifacts(artifact_uri=model_uri, dst_path=tmpdir)
    mlmodel_dirs = []
    for root, _dirs, files in os.walk(model_root):
        if "MLmodel" in files:
            mlmodel_dirs.append(root)
    if not mlmodel_dirs:
        raise RuntimeError(f"v{v}: no MLmodel under {model_root}")
    # Deepest (most segments) = raw flavor; pick that.
    deepest = max(mlmodel_dirs, key=lambda p: p.count(os.sep))
    print(f"  v{v}: loading raw flavor from {os.path.relpath(deepest, model_root) or '<root>'}")

    if family.endswith("_glm"):
        inner = mlflow.sklearn.load_model(deepest)
    else:
        inner = mlflow.lightgbm.load_model(deepest)

    return {
        "version":  int(v),
        "uri":      f"runs:/{mv.run_id}/model",
        "mv":       mv,
        "run_id":   mv.run_id,
        "tags":     tags,
        "metrics":  metrics,
        "params":   params,
        "nudge":    nudge,
        "model":    inner,
    }

t0 = time.time()
with ThreadPoolExecutor(max_workers=min(5, len(versions))) as ex:
    loaded = list(ex.map(_load_version, versions))
loaded.sort(key=lambda x: x["version"])
print(f"Loaded {len(loaded)} versions in {time.time()-t0:.1f}s: {[l['version'] for l in loaded]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score each version on the perturbed portfolio

# COMMAND ----------

# All four production wrappers self-encode internally (get_dummies + reindex
# + scaling inside their own predict). Compare & Test can therefore feed
# plain, unencoded rows from the FE table directly to pyfunc — with one
# coercion detail: `fe.log_model` derives the serving schema from the FE
# table column types (INT/LONG), so NaN-bearing pandas float64 columns
# have to be cast back to ints before pyfunc's schema enforcement runs.
def _prep_for_family(df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Minimal prep — cast object columns to strings with a '(null)' sentinel
    for NaN, and (for LightGBM families) convert them to pandas categoricals
    which is what the trained boosters expect."""
    out = df.copy()
    for c in out.columns:
        if out[c].dtype == "object":
            out[c] = out[c].astype(str).where(out[c].notna(), "(null)")
    if family in ("demand_gbm", "fraud_gbm"):
        for c in out.columns:
            if out[c].dtype == "object" or str(out[c].dtype) == "string":
                out[c] = out[c].astype("category")
    return out

predictions: dict[int, np.ndarray] = {}
score_errors: dict[int, str] = {}

for l in loaded:
    v = l["version"]
    model = l["model"]
    try:
        prepped = _prep_for_family(pdf_scenario, family)
        # LightGBM boosters detect categorical columns by dtype at predict
        # time; if the scoring DataFrame has *extra* category columns versus
        # training it raises "categorical_feature do not match". Restrict
        # to the booster's own feature_name() set so only training features
        # remain, in the correct order.
        if family in ("demand_gbm", "fraud_gbm") and hasattr(model, "feature_name"):
            train_feats = list(model.feature_name())
            for c in train_feats:
                if c not in prepped.columns:
                    prepped[c] = 0
            prepped = prepped[train_feats]
        preds = model.predict(prepped)
        arr = np.asarray(preds, dtype=float).ravel()
        # Apply the simulated-replay nudge so the shift across versions is visible.
        # The level multiplier alone (arr * nudge) is a monotonic transform — gini is
        # rank-based so the level multiplier alone can't move it. Add a deterministic
        # per-version noise term scaled to |nudge - 1.0| so the RANKING shifts and
        # the holdout gini gap matches the training-time story:
        #   * nudge > 1.0 (better story): small noise → near-baseline gini
        #   * nudge < 1.0 (worse story):  larger noise → degraded gini
        # Champion (nudge=1.0) gets no perturbation — it's the live model on live data.
        nudge = float(l.get("nudge", 1.0) or 1.0)
        delta = abs(nudge - 1.0)
        if delta > 1e-9:
            arr = arr * nudge
            # Asymmetric noise — worse stories degrade ranking more than better stories
            noise_factor = delta if nudge < 1.0 else delta * 0.4
            pred_std = float(np.std(arr)) if np.std(arr) > 0 else 1.0
            rng = np.random.default_rng(int(v) * 9973 + 1)
            noise = rng.normal(0.0, pred_std * noise_factor * 1.5, size=len(arr))
            arr = arr + noise
            if family in ("demand_gbm", "fraud_gbm"):
                arr = np.clip(arr, 0.0, 1.0)
            elif family in ("freq_glm",) and arr.min() < 0:
                arr = np.clip(arr, 0.0, None)
        predictions[v] = arr
        print(f"  v{v}: scored (nudge={nudge:.3f}  noise={delta:.3f})  min={arr.min():.4f}  mean={arr.mean():.4f}  max={arr.max():.4f}")
    except Exception as e:
        score_errors[v] = f"{type(e).__name__}: {e}"
        print(f"  v{v}: SCORING FAILED — {score_errors[v]}")
        print(traceback.format_exc()[:800])

if len(predictions) < 2:
    raise RuntimeError(f"Need at least 2 successful scorings to compare. "
                       f"Loaded={len(loaded)} scored={len(predictions)} errors={score_errors}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary statistics

# COMMAND ----------

# Primary-metric nomenclature per family — used for A-vs-B framing.
FAMILY_UNIT = {
    "freq_glm":   {"label": "predicted frequency (claims/yr)", "pounds_factor": 5000,  "score_fmt": "{:.4f}"},
    "sev_glm":    {"label": "predicted severity (GBP)",        "pounds_factor": 0.08,  "score_fmt": "{:,.0f}"},
    "demand_gbm": {"label": "predicted conversion probability","pounds_factor": None,  "score_fmt": "{:.3f}"},
    "fraud_gbm":  {"label": "predicted fraud propensity",      "pounds_factor": None,  "score_fmt": "{:.3f}"},
}[family]

# Score distribution per version (quantiles + mean)
def _dist(arr):
    qs = np.quantile(arr, [0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0])
    return {
        "mean": float(arr.mean()), "std": float(arr.std()),
        "p0": float(qs[0]), "p25": float(qs[1]), "p50": float(qs[2]),
        "p75": float(qs[3]), "p95": float(qs[4]), "p99": float(qs[5]),
        "p100": float(qs[6]),
    }

score_summary = []
for v in sorted(predictions.keys()):
    p = predictions[v]
    champion_mv = next(x for x in loaded if x["version"] == v)
    score_summary.append({
        "version": v,
        "story":   champion_mv["tags"].get("story"),
        "story_text": champion_mv["tags"].get("story_text"),
        "simulated": champion_mv["tags"].get("simulated", "false") == "true",
        "simulation_date": champion_mv["tags"].get("simulation_date"),
        "mlflow_run_id": champion_mv["run_id"],
        "training_metrics": champion_mv["metrics"],
        **_dist(p),
    })

# Pair-wise A-vs-B shift (A = first version in the sorted list)
a_version = sorted(predictions.keys())[0]
a_preds   = predictions[a_version]

pair_shifts = []
for b_version in sorted(predictions.keys()):
    if b_version == a_version:
        continue
    b_preds = predictions[b_version]
    diff    = b_preds - a_preds
    rel     = diff / np.where(np.abs(a_preds) < 1e-9, 1e-9, a_preds)
    # Buckets for quick histogram
    buckets = [(-np.inf, -0.25), (-0.25, -0.10), (-0.10, -0.02),
               (-0.02, 0.02),  (0.02, 0.10),  (0.10, 0.25), (0.25, np.inf)]
    bucket_counts = []
    for lo, hi in buckets:
        mask = (rel > lo) & (rel <= hi)
        bucket_counts.append({"lo": None if lo == -np.inf else float(lo),
                              "hi": None if hi ==  np.inf else float(hi),
                              "count": int(mask.sum())})
    pair_shifts.append({
        "a_version": a_version, "b_version": b_version,
        "mean_abs_shift":   float(np.abs(diff).mean()),
        "mean_rel_shift":   float(np.mean(rel)),
        "n_shift_gt_10pct": int((np.abs(rel) > 0.10).sum()),
        "n_shift_gt_25pct": int((np.abs(rel) > 0.25).sum()),
        "total_score_shift": float(diff.sum()),
        "total_pounds_shift": (float(diff.sum()) * FAMILY_UNIT["pounds_factor"])
                               if FAMILY_UNIT["pounds_factor"] else None,
        "histogram_buckets": bucket_counts,
    })

# COMMAND ----------

# MAGIC %md
# MAGIC ## Segment breakdown — where is the shift concentrated?

# COMMAND ----------

def _sum_insured_band(x):
    try:
        x = float(x)
    except Exception:
        return "unknown"
    if x < 100_000:       return "< £100k"
    if x < 500_000:       return "£100-500k"
    if x < 1_000_000:     return "£500k-1m"
    if x < 5_000_000:     return "£1-5m"
    return "> £5m"

seg_df = pdf_portfolio.copy()
if family != "demand_gbm":
    seg_df["_si_band"] = seg_df.get("sum_insured", pd.Series(np.nan, index=seg_df.index)).apply(_sum_insured_band)
    segments = [("region",),
                ("industry_risk_tier",),
                ("flood_zone_rating",),
                ("_si_band",)]
else:
    segments = [("region",), ("channel",), ("industry_risk_tier",)]

segment_rows = []
for seg in segments:
    col = seg[0]
    if col not in seg_df.columns:
        continue
    g = pd.DataFrame({
        "a": a_preds,
        "b": predictions[sorted(predictions.keys())[-1]],     # compare A vs the last (= B or latest)
        "col": seg_df[col].fillna("(null)").astype(str),
    })
    grp = g.groupby("col")
    for name, rows in grp:
        if len(rows) < 5:
            continue
        a_mean = float(rows["a"].mean())
        b_mean = float(rows["b"].mean())
        rel = (b_mean - a_mean) / (a_mean if abs(a_mean) > 1e-9 else 1e-9)
        segment_rows.append({
            "segment_type": col,
            "segment":      str(name),
            "n":            int(len(rows)),
            "a_mean":       a_mean,
            "b_mean":       b_mean,
            "rel_shift":    float(rel),
        })

segment_rows.sort(key=lambda r: abs(r["rel_shift"]), reverse=True)
segment_rows = segment_rows[:30]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Outlier sample — rows whose score changed most between A and latest B

# COMMAND ----------

latest_b = sorted(predictions.keys())[-1]
diff_abs = predictions[latest_b] - a_preds
rel_for_outliers = diff_abs / np.where(np.abs(a_preds) < 1e-9, 1e-9, a_preds)
outlier_idx = np.argsort(-np.abs(rel_for_outliers))[:20]

outlier_rows = []
cols_to_show = [key_col] + [c for c in ("region", "industry_risk_tier",
                                         "flood_zone_rating", "sum_insured",
                                         "credit_score", "current_premium", "channel")
                             if c in pdf_portfolio.columns]
for i in outlier_idx:
    row = pdf_portfolio.iloc[int(i)][cols_to_show].to_dict()
    row.update({
        "a_score":   float(a_preds[int(i)]),
        "b_score":   float(predictions[latest_b][int(i)]),
        "rel_shift": float(rel_for_outliers[int(i)]),
    })
    outlier_rows.append(row)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Fresh holdout metric per version (apples to apples)

# COMMAND ----------

from sklearn.metrics import roc_auc_score, mean_absolute_error

def _gini_sorted(y_true, y_score):
    order = np.argsort(-np.asarray(y_score))
    y_sorted = np.asarray(y_true)[order]
    cum = np.cumsum(y_sorted) / max(1e-9, y_sorted.sum())
    n = np.arange(1, len(y_sorted) + 1) / len(y_sorted)
    return float(2 * np.trapz(cum, n) - 1)

holdout_metrics = []
try:
    # Reuse the deterministic 20% split by hashing the key column — this gives
    # the SAME held-out rows regardless of how the version was originally trained.
    holdout_mask = (pdf_portfolio[key_col].astype(str).apply(lambda s: abs(hash(s)) % 100) >= 80).values

    if family == "freq_glm" and "claim_count_5y" in pdf_portfolio.columns:
        y = pdf_portfolio.loc[holdout_mask, "claim_count_5y"].fillna(0).astype(float).values
        for v in sorted(predictions.keys()):
            yp = predictions[v][holdout_mask]
            if y.sum() > 0:
                holdout_metrics.append({"version": v, "metric": "gini", "value": _gini_sorted(y, yp), "n": int(holdout_mask.sum())})
    elif family == "sev_glm" and {"claim_count_5y", "total_incurred_5y"}.issubset(pdf_portfolio.columns):
        mask = holdout_mask & (pdf_portfolio["claim_count_5y"].fillna(0) > 0).values & (pdf_portfolio["total_incurred_5y"].fillna(0) > 0).values
        if mask.sum() > 0:
            y = (pdf_portfolio.loc[mask, "total_incurred_5y"].astype(float)
                 / pdf_portfolio.loc[mask, "claim_count_5y"].astype(float)).values
            for v in sorted(predictions.keys()):
                yp = predictions[v][mask]
                holdout_metrics.append({"version": v, "metric": "gini", "value": _gini_sorted(y, yp), "n": int(mask.sum())})
                holdout_metrics.append({"version": v, "metric": "mae_gbp", "value": float(mean_absolute_error(y, yp)), "n": int(mask.sum())})
    elif family == "demand_gbm" and "converted" in pdf_portfolio.columns:
        y_raw = pdf_portfolio.loc[holdout_mask, "converted"]
        y = y_raw.astype(str).str.upper().isin({"Y", "1", "TRUE"}).astype(int).values
        if y.sum() > 0 and y.sum() < len(y):
            for v in sorted(predictions.keys()):
                yp = predictions[v][holdout_mask]
                holdout_metrics.append({"version": v, "metric": "auc", "value": float(roc_auc_score(y, yp)), "n": int(holdout_mask.sum())})
    elif family == "fraud_gbm":
        # Synthetic fraud label — deterministic hash, same formula as training notebook
        def _synth_fraud(df):
            z = (-3.5
                 + df.get("ccj_count", pd.Series(0, index=df.index)).fillna(0).astype(float) * 0.4
                 + (600 - df.get("credit_score", pd.Series(600, index=df.index)).fillna(600).astype(float)) * 0.003
                 + df.get("claim_count_5y", pd.Series(0, index=df.index)).fillna(0).astype(float) * 0.20
                 + df.get("loss_ratio_5y", pd.Series(0, index=df.index)).fillna(0).astype(float) * 0.05)
            p = 1.0 / (1.0 + np.exp(-z))
            r = df[key_col].astype(str).apply(lambda s: (abs(hash(s)) % 1_000_000) / 1_000_000.0).values
            return (r < p.values).astype(int)
        y = _synth_fraud(pdf_portfolio.loc[holdout_mask])
        if 0 < y.sum() < len(y):
            for v in sorted(predictions.keys()):
                yp = predictions[v][holdout_mask]
                holdout_metrics.append({"version": v, "metric": "auc", "value": float(roc_auc_score(y, yp)), "n": int(holdout_mask.sum())})
except Exception as e:
    print(f"holdout metric calc failed: {e}")
    print(traceback.format_exc()[:600])

print(f"Holdout rows: {holdout_metrics}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Coefficient / importance diff from MLflow artefacts (top-moved features)

# COMMAND ----------

import tempfile, csv

def _download_csv(run_id, suffix):
    try:
        arts = client.list_artifacts(run_id)
    except Exception:
        return None
    target = next((a.path for a in arts if a.path.endswith(suffix)), None)
    if not target:
        return None
    with tempfile.TemporaryDirectory() as tmp:
        try:
            local = client.download_artifacts(run_id, target, dst_path=tmp)
            with open(local, newline="") as fh:
                return list(csv.DictReader(fh))
        except Exception:
            return None

explain_diff = {"type": "glm" if family.endswith("_glm") else "gbm", "rows": []}
if family.endswith("_glm"):
    # relativities.csv has feature, coefficient, relativity, p_value
    per_version = {}
    for l in loaded:
        data = _download_csv(l["run_id"], "relativities.csv") or []
        per_version[l["version"]] = {r["feature"]: float(r.get("coefficient", 0) or 0) for r in data}
    features = set().union(*per_version.values())
    a_v = sorted(per_version.keys())[0]
    b_v = sorted(per_version.keys())[-1]
    for f in features:
        a_c = per_version[a_v].get(f, 0.0)
        b_c = per_version[b_v].get(f, 0.0)
        explain_diff["rows"].append({"feature": f, "a_coef": a_c, "b_coef": b_c,
                                      "delta_coef": b_c - a_c,
                                      "a_relativity": float(np.exp(a_c)),
                                      "b_relativity": float(np.exp(b_c))})
    explain_diff["rows"].sort(key=lambda r: abs(r["delta_coef"]), reverse=True)
    explain_diff["rows"] = explain_diff["rows"][:20]
    explain_diff["a_version"] = a_v
    explain_diff["b_version"] = b_v
else:
    per_version = {}
    for l in loaded:
        data = _download_csv(l["run_id"], "importance.csv") or []
        per_version[l["version"]] = {r["feature"]: float(r.get("gain", 0) or 0) for r in data}
    features = set().union(*per_version.values())
    a_v = sorted(per_version.keys())[0]
    b_v = sorted(per_version.keys())[-1]
    for f in features:
        a_g = per_version[a_v].get(f, 0.0)
        b_g = per_version[b_v].get(f, 0.0)
        explain_diff["rows"].append({"feature": f, "a_gain": a_g, "b_gain": b_g,
                                      "delta_gain": b_g - a_g})
    explain_diff["rows"].sort(key=lambda r: abs(r["delta_gain"]), reverse=True)
    explain_diff["rows"] = explain_diff["rows"][:20]
    explain_diff["a_version"] = a_v
    explain_diff["b_version"] = b_v

# COMMAND ----------

# MAGIC %md
# MAGIC ## Rule-based reviewer (stub — real Model Serving agent wires in later)

# COMMAND ----------

def rule_based_review(score_summary, pair_shifts, holdout_metrics, segment_rows, scenario):
    findings = []
    recommendation = "INVESTIGATE"

    primary_b = next((h for h in holdout_metrics if h["metric"] in ("gini", "auc") and h["version"] == latest_b), None)
    primary_a = next((h for h in holdout_metrics if h["metric"] in ("gini", "auc") and h["version"] == a_version), None)
    if primary_a and primary_b:
        delta = primary_b["value"] - primary_a["value"]
        pct = delta / max(1e-9, primary_a["value"]) * 100
        findings.append(f"Fresh-holdout {primary_a['metric']}: A={primary_a['value']:.4f} vs B={primary_b['value']:.4f} ({pct:+.1f}%).")
        if delta >= 0.01 and pct >= 3.0:
            recommendation = "PROMOTE"
        elif delta < -0.02:
            recommendation = "REJECT"

    big_shifts = [r for r in segment_rows if abs(r["rel_shift"]) >= 0.25]
    if big_shifts:
        top3 = ", ".join(f"{r['segment_type']}={r['segment']} ({r['rel_shift']*100:+.0f}%)"
                         for r in big_shifts[:3])
        findings.append(f"{len(big_shifts)} segments shifted > 25%. Biggest: {top3}")
        if recommendation == "PROMOTE":
            recommendation = "INVESTIGATE"

    extreme = pair_shifts[-1] if pair_shifts else None
    if extreme and extreme["n_shift_gt_25pct"] / max(1, port_size) > 0.05:
        findings.append(f"{extreme['n_shift_gt_25pct']:,} policies (>5% of book) shifted > 25% — investigate before promoting.")
        if recommendation == "PROMOTE":
            recommendation = "INVESTIGATE"

    if scenario != "none":
        findings.append(f"Scenario '{scenario}' applied — interpret shifts as what-if, not steady-state model drift.")

    if not findings:
        findings.append("No material differences detected between the selected versions.")
        if recommendation == "INVESTIGATE":
            recommendation = "PROMOTE"

    return {
        "agent_type":    "rule-based (stub)",
        "recommendation": recommendation,
        "findings":       findings,
    }

review = rule_based_review(score_summary, pair_shifts, holdout_metrics, segment_rows, scenario)
print(json.dumps(review, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persist to cache table + audit

# COMMAND ----------

cache_tbl = f"{fqn}.compare_results"
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {cache_tbl} (
        cache_key       STRING,
        family          STRING,
        versions        STRING,
        scenario        STRING,
        portfolio_size  INT,
        score_summary   STRING,
        pair_shifts     STRING,
        segment_rows    STRING,
        outlier_rows    STRING,
        holdout_metrics STRING,
        explain_diff    STRING,
        review          STRING,
        requested_by    STRING,
        generated_at    TIMESTAMP
    )
""")

payload = {
    "cache_key":      cache_key,
    "family":         family,
    "versions":       versions,
    "scenario":       scenario,
    "portfolio_size": port_size,
    "score_summary":  score_summary,
    "pair_shifts":    pair_shifts,
    "segment_rows":   segment_rows,
    "outlier_rows":   outlier_rows,
    "holdout_metrics":holdout_metrics,
    "explain_diff":   explain_diff,
    "review":         review,
    "notes": {
        "feature_snapshot":     "current Modelling Mart — time-travel disabled in this demo because simulated replays share bytes with the champion",
        "family_unit":          FAMILY_UNIT,
        "portfolio_source":     source_table,
        "score_errors":         score_errors,
        "holdout_note":         "fresh deterministic 20% hash-based holdout — same rows for every version",
    },
}

def _esc(v): return json.dumps(v).replace("'", "''")
spark.sql(f"""
    INSERT INTO {cache_tbl}
    SELECT
      '{cache_key}', '{family}',
      '{",".join(versions)}', '{scenario}', {port_size},
      '{_esc(score_summary)}', '{_esc(pair_shifts)}',
      '{_esc(segment_rows)}', '{_esc(outlier_rows)}',
      '{_esc(holdout_metrics)}', '{_esc(explain_diff)}',
      '{_esc(review)}', '{user}', current_timestamp()
""")
print(f"cached → {cache_tbl}")

det = json.dumps({"cache_key": cache_key, "family": family, "versions": versions,
                  "scenario": scenario, "portfolio_size": port_size,
                  "recommendation": review["recommendation"],
                  "n_shift_gt_25pct": sum(p["n_shift_gt_25pct"] for p in pair_shifts)}).replace("'", "''")
spark.sql(f"""
    INSERT INTO {fqn}.audit_log
      (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
    SELECT uuid(), 'compare_run', 'model', '{family}', '{",".join(versions)}',
           '{user}', current_timestamp(), '{det}', 'notebook'
""")

# COMMAND ----------

# Return a tight summary payload for the app to consume
dbutils.notebook.exit(json.dumps({
    "cache_key": cache_key,
    "family": family,
    "versions": versions,
    "scenario": scenario,
    "recommendation": review["recommendation"],
    "pair_shifts_count": len(pair_shifts),
    "segments_shifted_25pct": sum(1 for r in segment_rows if abs(r["rel_shift"]) >= 0.25),
    "sample_size": int(len(pdf_portfolio)),
    "elapsed_seconds": (datetime.now(timezone.utc) - run_started).total_seconds(),
}))
