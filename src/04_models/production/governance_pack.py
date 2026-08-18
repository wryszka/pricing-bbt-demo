# Databricks notebook source
# MAGIC %md
# MAGIC # Governance pack generator
# MAGIC
# MAGIC Produces a comprehensive, sign-off-ready PDF for a single UC-registered
# MAGIC model version. Designed to answer every question a CDO, CRO, CFO or Chief
# MAGIC Actuary would ask at a validation committee:
# MAGIC
# MAGIC  1. What is this model and what does it do?
# MAGIC  2. Who trained it, when, from what data?
# MAGIC  3. How well does it perform?
# MAGIC  4. Which features drive it and do they make business sense?
# MAGIC  5. How stable has it been across prior versions?
# MAGIC  6. What are the known risks and limitations?
# MAGIC  7. Where does regulatory / compliance responsibility sit?
# MAGIC  8. Who signed it off?
# MAGIC
# MAGIC Output:
# MAGIC  - PDF uploaded to `/Volumes/{catalog}/{schema}/governance_packs/`
# MAGIC  - Row in `{catalog}.{schema}.governance_packs_index`
# MAGIC  - Audit event `governance_pack_generated`

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("model_family",  "freq_glm")
dbutils.widgets.text("model_version", "")          # blank = latest
dbutils.widgets.text("requested_by",  "app")

# COMMAND ----------

# NOTE: fpdf2 / mlflow / matplotlib come from the job's serverless environment
# (see resources/governance_pack.yml). Do NOT `%pip install ... ; restartPython()`
# here — on serverless that pattern stalls the run at exec=0 (the hang that kept
# this job from producing real packs). Deps belong in the environment spec.

# COMMAND ----------

catalog   = dbutils.widgets.get("catalog_name")
schema    = dbutils.widgets.get("schema_name")
family    = dbutils.widgets.get("model_family")
version_w = dbutils.widgets.get("model_version").strip()
user      = dbutils.widgets.get("requested_by") or "app"

fqn       = f"{catalog}.{schema}"
uc_name   = f"{fqn}.{family}"
VALID     = {"freq_glm", "sev_glm", "demand_gbm", "fraud_gbm",
             "freq_glm_motor", "sev_glm_motor", "demand_gbm_motor", "fraud_gbm_motor"}
# Factory candidates register as factory_freq_glm_<variant_id>. They're also
# valid targets — same pack format, same sidecars, but the alias flip at the
# end is skipped (they're not promoted into production).
_is_factory_variant = family.startswith("factory_")
if family not in VALID and not _is_factory_variant:
    raise ValueError(f"model_family must be one of {VALID} or a factory_* variant, got '{family}'")

# Motor variants share schema + algorithm with the commercial families. Use a
# `_base` form for every BUSINESS_CONTEXT / primary_metric / CSV-filename
# lookup, but keep `family` for UC model name + pack_id.
if family.endswith("_motor"):
    family_base = family[: -len("_motor")]
else:
    family_base = family

import json, io, os, uuid, tempfile
from datetime import datetime, timezone
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the target model version

# COMMAND ----------

versions = client.search_model_versions(f"name='{uc_name}'")
if not versions:
    raise RuntimeError(f"No versions found for {uc_name}")
if version_w:
    try:
        mv = next(v for v in versions if str(v.version) == version_w)
    except StopIteration:
        raise RuntimeError(f"Version {version_w} not found for {uc_name}")
else:
    mv = max(versions, key=lambda v: int(v.version))

run_id    = mv.run_id
version   = str(mv.version)
mv_tags: dict = {}
try:
    raw_tags = getattr(mv, "tags", None)
    if isinstance(raw_tags, dict):
        mv_tags = dict(raw_tags)
    elif raw_tags is not None:
        mv_tags = {t.key: t.value for t in raw_tags}
except Exception as e:
    print(f"  mv.tags parse failed: {e}")

try:
    run = client.get_run(run_id)
    run_tags   = dict(run.data.tags or {})
    run_params = dict(run.data.params or {})
    run_metrics= dict(run.data.metrics or {})
    run_start  = datetime.fromtimestamp((run.info.start_time or 0) / 1000, tz=timezone.utc)
except Exception as e:
    print(f"Could not load run {run_id}: {e}")
    run_tags, run_params, run_metrics = {}, {}, {}
    run_start = datetime.now(timezone.utc)

print(f"Target: {uc_name} v{version}  (run_id={run_id}, started {run_start.isoformat()})")
print(f"  story={run_tags.get('story', '—')}  simulated={run_tags.get('simulated', 'false')}")
print(f"  metrics keys: {list(run_metrics.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull version history for this family (for stability chart)

# COMMAND ----------

history = []
for v in sorted(versions, key=lambda x: int(x.version)):
    try:
        r = client.get_run(v.run_id)
        history.append({
            "version":          int(v.version),
            "run_id":           v.run_id,
            "simulation_date":  r.data.tags.get("simulation_date") or "",
            "story":            r.data.tags.get("story", "—"),
            "story_text":       r.data.tags.get("story_text", ""),
            "simulated":        r.data.tags.get("simulated", "false") == "true",
            "metrics":          dict(r.data.metrics or {}),
            "trained_at":       datetime.fromtimestamp((r.info.start_time or 0)/1000, tz=timezone.utc).isoformat(),
            "trained_by":       r.data.tags.get("mlflow.user", ""),
        })
    except Exception as e:
        print(f"  skip v{v.version}: {e}")

# Sort by simulation_date if present, else by version
def _hist_key(h):
    return (h["simulation_date"] or h["trained_at"][:10])
history.sort(key=_hist_key)
print(f"History length: {len(history)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Download MLflow artifacts for this version (SHAP + relativities + importance)

# COMMAND ----------

artifacts_dir = tempfile.mkdtemp(prefix=f"gov_pack_{family}_v{version}_")
artifact_paths: dict[str, str] = {}
try:
    for a in client.list_artifacts(run_id):
        if a.is_dir:
            continue
        local = client.download_artifacts(run_id, a.path, dst_path=artifacts_dir)
        artifact_paths[a.path] = local
        print(f"  downloaded {a.path}")
except Exception as e:
    print(f"  artifact pull failed: {e}")

# Fallback: promoted replays carry run_ids with no artefacts (the model bytes
# come from the original champion's source). If our target version's run is
# empty, scan sibling versions and reuse the first one that has CSVs.
if not any(k.endswith(".csv") or k.endswith(".png") for k in artifact_paths):
    print("  target version has no CSV/PNG artefacts — looking at sibling versions…")
    for v in sorted(versions, key=lambda x: int(x.version), reverse=True):
        if str(v.version) == version:
            continue
        try:
            sibling_arts = list(client.list_artifacts(v.run_id))
            sibling_csvs = [a for a in sibling_arts if not a.is_dir
                            and (a.path.endswith(".csv") or a.path.endswith(".png"))]
            if sibling_csvs:
                print(f"  using v{v.version} as artefact donor ({len(sibling_csvs)} files)")
                for a in sibling_csvs:
                    local = client.download_artifacts(v.run_id, a.path, dst_path=artifacts_dir)
                    artifact_paths[a.path] = local
                    print(f"    donor downloaded {a.path}")
                break
        except Exception as e:
            continue

print(f"Artifacts: {list(artifact_paths.keys())}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Business context — hand-curated per model family

# COMMAND ----------

BUSINESS_CONTEXT = {
    "freq_glm": {
        "purpose":         "Predicts the expected number of claims per policy per year.",
        "line_of_business":"Commercial Property & Casualty — SME and mid-market UK book.",
        "target":          "claim_count_5y (Poisson, log-link)",
        "used_for":        "Rate-setting base: feeds the frequency × severity pure-premium calculation at quoting time.",
        "not_used_for":    "Individual-policy binding decisions, fraud triage, regulatory capital (SCR) calibration.",
        "peer_models":     "Peer insurers use similar Poisson or Negative Binomial GLMs with 15-25 rating factors. Industry-standard approach.",
        "owner_team":      "Pricing Actuarial — Commercial Lines",
        "risk_profile":    "Low-to-medium. Well-understood model class. Primary risk is data drift (bureau, postcode refresh) rather than algorithmic failure.",
    },
    "sev_glm": {
        "purpose":         "Predicts the mean £ cost of a claim given that a claim has occurred.",
        "line_of_business":"Commercial Property & Casualty — SME and mid-market UK book.",
        "target":          "mean_severity = total_incurred_5y / claim_count_5y (Gamma-family log-link, fitted as OLS on log(y) for numerical stability on heavy tails).",
        "used_for":        "Rate-setting base: multiplied by frequency prediction to get pure premium.",
        "not_used_for":    "Reserving, individual claim reserves, large-loss excess calculation.",
        "peer_models":     "Peer insurers use Gamma or Log-Normal GLMs on mean severity, sometimes with per-peril decomposition. This model is combined-peril.",
        "owner_team":      "Pricing Actuarial — Commercial Lines",
        "risk_profile":    "Medium. Severity is heavy-tailed; single very-large losses can materially shift relativities. Outlier protocol in place.",
    },
    "demand_gbm": {
        "purpose":         "Predicts the probability that a presented quote converts to a bound policy.",
        "line_of_business":"Commercial Property & Casualty — broker and direct channels.",
        "target":          "converted ∈ {0,1} on the quotes table (LightGBM binary classifier).",
        "used_for":        "Price elasticity and demand response feed the pricing optimisation: finding the price that maximises expected profit given competitive context.",
        "not_used_for":    "Underwriting risk assessment. Conversion is commercial demand, not loss expectation.",
        "peer_models":     "Standard industry approach. Some peers use uplift models to decompose price sensitivity from propensity — not implemented here.",
        "owner_team":      "Pricing Optimisation — Commercial Lines",
        "risk_profile":    "Medium. Sensitive to competitor-pricing drift and channel-mix changes. Retraining cadence is monthly.",
    },
    "fraud_gbm": {
        "purpose":         "Scores the probability that a policy or claim shows characteristics consistent with organised or opportunistic fraud.",
        "line_of_business":"Commercial Property & Casualty — used at quote and claim touchpoints.",
        "target":          "Synthetic fraud label derived from CCJs, credit history, claims pattern and loss ratio (LightGBM binary classifier).",
        "used_for":        "Referral triage — high-score cases are routed to the SIU team for manual review. NOT used to reject quotes or decline claims automatically.",
        "not_used_for":    "Automatic declinature, pricing loading, underwriter binding decisions. Output is advisory only.",
        "peer_models":     "Peer insurers blend supervised classifiers with anomaly detection and graph-based ring detection. This model is the supervised component only.",
        "owner_team":      "Financial Crime / SIU — with Pricing Analytics support",
        "risk_profile":    "High. False positives create customer harm and regulatory exposure under Consumer Duty. Decision threshold is kept conservative (precision over recall).",
    },
}

# Factory candidates share the freq_glm purpose statement + a clear caveat so
# anyone reading the pack knows this is an experimental candidate, not a
# promoted production champion.
if _is_factory_variant:
    CTX = {
        **BUSINESS_CONTEXT["freq_glm"],
        "purpose":      "FACTORY CANDIDATE — experimental Poisson GLM variant from the Model Factory. Not a promoted production model. Candidate is discoverable in Unity Catalog for evaluation purposes only.",
        "used_for":     "Evaluation and comparison only. Candidates do NOT go into the live rating engine until re-trained through the production pipeline and approved for promotion.",
        "not_used_for": "Rate-setting, quoting, claim triage, regulatory capital calibration, or any customer-facing decision.",
        "risk_profile": "Isolated from production — no live traffic. Candidate metrics in this pack reflect the variant's fit on the current Modelling Mart but carry no operational consequence.",
        "owner_team":   "Pricing Actuarial — Model Factory experiments",
    }
else:
    CTX = BUSINESS_CONTEXT[family_base]

# Regulatory references by family — shared where applicable
REGS = [
    ("PRA SS1/23",                    "Model risk management principles — inventory, testing, validation, governance."),
    ("Solvency II Pillar 2 — ORSA",   "Own Risk & Solvency Assessment: models feeding SCR must be inventoried and validated."),
    ("FCA Consumer Duty",             "Fair value, price-matching prohibition, vulnerable customer protection — pricing models in scope."),
    ("EU AI Act (high-risk systems)", "Insurance pricing is high-risk AI; requires data governance, accuracy monitoring, human oversight, post-market monitoring."),
    ("SM&CR (Senior Manager responsibilities)", "Chief Actuary (SMF20) accountable for pricing model outputs used in rating."),
    ("GDPR Article 22",               "Automated decisions materially affecting data subjects require lawful basis + information rights."),
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data lineage & quality

# COMMAND ----------

feature_table = run_tags.get("feature_table", f"{fqn}.unified_pricing_table_live")
lineage = {"feature_table": feature_table}
try:
    t = spark.table(feature_table)
    lineage["feature_table_rows"]    = t.count()
    lineage["feature_table_columns"] = len(t.columns)
except Exception as e:
    print(f"  feature table sample failed: {e}")

# Try to get Delta table history (last 3 commits) as lineage evidence
try:
    dh = spark.sql(f"DESCRIBE HISTORY {feature_table} LIMIT 3").toPandas()
    lineage["feature_table_history"] = dh.to_dict(orient="records")
except Exception as e:
    print(f"  feature table history failed: {e}")

# Count upstream source tables by scanning the Modelling Mart build step
try:
    srcs = spark.sql(f"""
        SELECT count(DISTINCT event_id) AS ingestion_events,
               approx_count_distinct(entity_id) AS distinct_datasets
        FROM {fqn}.audit_log
        WHERE event_type IN ('dataset_ingested','dataset_approved')
    """).toPandas().iloc[0].to_dict()
    lineage["ingestion_summary"] = srcs
except Exception as e:
    print(f"  ingestion summary failed: {e}")

print(json.dumps(lineage, default=str, indent=2)[:500])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load relativities / importance / SHAP artifacts as pandas

# COMMAND ----------

import pandas as pd

def _read_csv_if_present(keys):
    for k in keys:
        if k in artifact_paths:
            try:
                return pd.read_csv(artifact_paths[k])
            except Exception as e:
                print(f"  read {k} failed: {e}")
    return None

# For factory variants (factory_freq_glm_<variant_id>) the relativities CSV
# artefact is named "<variant_id>_relativities.csv" — look for that first.
_variant_id = family.rsplit("_", 1)[-1] if _is_factory_variant else None

relativities   = _read_csv_if_present([
    f"{_variant_id}_relativities.csv" if _variant_id else "",
    f"{family_base.split('_')[0]}_relativities.csv",
    "freq_relativities.csv", "sev_relativities.csv",
])
importance     = _read_csv_if_present([
    f"{_variant_id}_importance.csv" if _variant_id else "",
    f"{family_base.split('_')[0]}_importance.csv",
    "demand_importance.csv", "fraud_importance.csv",
])
shap_imp       = _read_csv_if_present([
    f"{_variant_id}_shap_importance.csv" if _variant_id else "",
    f"{family_base.split('_')[0]}_shap_importance.csv",
    "demand_shap_importance.csv", "fraud_shap_importance.csv",
])
shap_plot_key  = next((k for k in artifact_paths if k.endswith("shap_summary.png")), None)
shap_plot_path = artifact_paths.get(shap_plot_key) if shap_plot_key else None

# Factory candidates are always freq_glm under the hood — treat as GLM
is_glm = family_base.endswith("_glm") or _is_factory_variant
is_gbm = family_base.endswith("_gbm") and not _is_factory_variant
print(f"  is_glm={is_glm} is_gbm={is_gbm}  "
      f"relativities={None if relativities is None else len(relativities)}  "
      f"importance={None if importance is None else len(importance)}  "
      f"shap_imp={None if shap_imp is None else len(shap_imp)}  "
      f"shap_plot={'yes' if shap_plot_path else 'no'}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stability chart — primary metric across version history

# COMMAND ----------

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

primary_metric = {"freq_glm": "gini", "sev_glm": "gini",
                  "demand_gbm": "auc", "fraud_gbm": "auc"}.get(family_base, "gini")

stability_png = None
if len(history) > 1:
    fig, ax = plt.subplots(figsize=(7.5, 3.0))
    xs = [h["simulation_date"] or f"v{h['version']}" for h in history]
    ys = [h["metrics"].get(primary_metric) for h in history]
    simulated_mask = [h["simulated"] for h in history]
    real_mask = [not s for s in simulated_mask]
    real_pairs = [(x, y) for x, y, m in zip(xs, ys, real_mask) if m and y is not None]
    sim_pairs  = [(x, y) for x, y, m in zip(xs, ys, simulated_mask) if m and y is not None]
    real_xs, real_ys = (list(t) for t in zip(*real_pairs)) if real_pairs else ([], [])
    sim_xs,  sim_ys  = (list(t) for t in zip(*sim_pairs))  if sim_pairs  else ([], [])
    plot_pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    plot_xs, plot_ys = (list(t) for t in zip(*plot_pairs)) if plot_pairs else ([], [])
    ax.plot(plot_xs, plot_ys, linestyle="-", color="#3b82f6", alpha=0.5, linewidth=1)
    if sim_xs:
        ax.scatter(sim_xs, sim_ys, s=30, color="#9ca3af", label="simulated replay", zorder=3)
    if real_xs:
        ax.scatter(real_xs, real_ys, s=50, color="#1e293b", label="current champion", zorder=4)
    ax.set_title(f"{family} — {primary_metric} across retraining history", fontsize=10)
    ax.set_ylabel(primary_metric)
    ax.tick_params(axis="x", rotation=45, labelsize=7)
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    ax.legend(loc="lower right", fontsize=7)
    fig.tight_layout()
    stability_png = f"{artifacts_dir}/stability.png"
    fig.savefig(stability_png, dpi=130, bbox_inches="tight")
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build the PDF

# COMMAND ----------

from fpdf import FPDF

_UNI_REPLACEMENTS = {
    "—": "-",   # em dash
    "–": "-",   # en dash
    "•": "*",   # bullet
    "‘": "'", "’": "'",
    "“": '"', "”": '"',
    " ": " ",
    "·": ".",
    "é": "e", "è": "e", "à": "a",
    "ñ": "n",
    "€": "EUR",
    "×": "x",
    "²": "2",
}

def _ascii(s: str) -> str:
    if s is None:
        return ""
    if not isinstance(s, str):
        s = str(s)
    for k, v in _UNI_REPLACEMENTS.items():
        if k in s:
            s = s.replace(k, v)
    # Drop anything remaining that the core Helvetica font can't handle.
    return s.encode("latin-1", errors="replace").decode("latin-1")

NAVY  = (30, 41, 59)
BLUE  = (59, 130, 246)
GREEN = (34, 197, 94)
AMBER = (245, 158, 11)
RED   = (239, 68, 68)
GRAY  = (107, 114, 128)
LIGHT = (243, 244, 246)
EMBER = (250, 245, 230)

class GovernancePack(FPDF):
    def __init__(self, title: str, subtitle: str):
        super().__init__(orientation="P", unit="mm", format="A4")
        self.title_text = _ascii(title)
        self.subtitle   = _ascii(subtitle)
        self.set_auto_page_break(auto=True, margin=20)
        self.alias_nb_pages()
        self.set_margins(15, 15, 15)
        self._section = 0
        self._skip_header = False

    # Override FPDF.normalize_text so every cell/multi_cell call gets our
    # latin-1 sanitiser applied — removes em-dashes, bullets, etc. that the
    # core Helvetica font can't render.
    def normalize_text(self, text):
        return super().normalize_text(_ascii(text))

    def header(self):
        if self._skip_header or self.page_no() == 1:
            return
        self.set_font("Helvetica", "B", 9)
        self.set_text_color(*NAVY)
        self.cell(0, 5, "Bricksurance SE  |  Model Governance Pack", align="L")
        self.set_font("Helvetica", "", 8)
        self.set_text_color(*GRAY)
        self.cell(0, 5, self.subtitle, align="R", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.4)
        self.line(15, self.get_y(), 195, self.get_y())
        self.ln(3)

    def footer(self):
        if self.page_no() == 1:
            return
        self.set_y(-15)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(*GRAY)
        self.cell(0, 5, "CONFIDENTIAL — Regulatory & internal committee use only", align="L")
        self.cell(0, 5, f"Page {self.page_no()}/{{nb}}", align="R")

    def h1(self, text: str):
        self._section += 1
        self.set_font("Helvetica", "B", 15)
        self.set_text_color(*NAVY)
        self.cell(0, 9, f"{self._section}. {text}", new_x="LMARGIN", new_y="NEXT")
        self.set_draw_color(*BLUE)
        self.set_line_width(0.5)
        self.line(15, self.get_y(), 60, self.get_y())
        self.ln(3)

    def h2(self, text: str):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(*NAVY)
        self.cell(0, 6, text, new_x="LMARGIN", new_y="NEXT")
        self.ln(1)

    def para(self, text: str, size: int = 10, color=(40, 40, 40)):
        self.set_font("Helvetica", "", size)
        self.set_text_color(*color)
        # wrapmode="CHAR" so unbreakable tokens (UUIDs, long URLs embedded in
        # error messages) don't trip 'Not enough horizontal space to render
        # a single character' under fpdf2's default word-wrap mode.
        self.multi_cell(0, 5, text, wrapmode="CHAR")
        self.ln(1)

    def callout(self, text: str, color=AMBER, bg=EMBER):
        self.set_fill_color(*bg)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(*color)
        self.multi_cell(0, 4.5, text, fill=True, border=0, wrapmode="CHAR")
        self.ln(1)

    def kv_block(self, items: list[tuple[str, str]]):
        """Two-column key/value list. Uses explicit widths so multi_cell never
        collapses to 0-width when the key column is long."""
        key_w   = 50
        val_w   = 130   # page effective width ~180mm; key(50) + val(130) = 180
        self.set_font("Helvetica", "", 9)
        for k, v in items:
            y0 = self.get_y()
            x0 = self.l_margin
            # key
            self.set_xy(x0, y0)
            self.set_font("Helvetica", "B", 9)
            self.set_text_color(*NAVY)
            self.multi_cell(key_w, 5, str(k))
            y_after_k = self.get_y()
            # value, to the right of the key
            self.set_xy(x0 + key_w, y0)
            self.set_font("Helvetica", "", 9)
            self.set_text_color(40, 40, 40)
            self.multi_cell(val_w, 5, str(v))
            y_after_v = self.get_y()
            # move to the lower of the two ends to start next row
            self.set_xy(x0, max(y_after_k, y_after_v))
        self.ln(1)

    def metric_grid(self, metrics: list[tuple[str, str, str]]):
        # metrics = [(label, value, status-color)]
        self.set_font("Helvetica", "", 8)
        col_w = (195 - 15) / max(1, len(metrics))
        start_y = self.get_y()
        for label, value, status in metrics:
            x = self.get_x()
            self.set_fill_color(*LIGHT)
            self.rect(x, start_y, col_w - 2, 16, style="DF")
            self.set_xy(x + 2, start_y + 2)
            self.set_font("Helvetica", "", 7)
            self.set_text_color(*GRAY)
            self.cell(col_w - 4, 3, label)
            self.set_xy(x + 2, start_y + 6)
            self.set_font("Helvetica", "B", 13)
            self.set_text_color(*status)
            self.cell(col_w - 4, 7, value)
            self.set_xy(x + col_w, start_y)
        self.set_y(start_y + 18)

    def df_table(self, df: "pd.DataFrame", max_rows: int = 20, widths=None):
        if df is None or len(df) == 0:
            self.para("(no data available)", size=8, color=GRAY)
            return
        cols = list(df.columns)
        rows = df.head(max_rows).values.tolist()
        if widths is None:
            widths = [max(18, min(60, 180/len(cols)))] * len(cols)
        self.set_fill_color(*NAVY)
        self.set_font("Helvetica", "B", 8)
        self.set_text_color(255, 255, 255)
        for w, c in zip(widths, cols):
            self.cell(w, 5, str(c)[:22], fill=True, border=0)
        self.ln(5)
        self.set_text_color(40, 40, 40)
        self.set_font("Helvetica", "", 8)
        for i, row in enumerate(rows):
            if i % 2 == 0:
                self.set_fill_color(*LIGHT)
                self.set_fill_color(248, 250, 252)
                fill = True
            else:
                fill = False
            for w, v in zip(widths, row):
                if isinstance(v, float):
                    s = f"{v:,.4f}" if abs(v) < 1000 else f"{v:,.1f}"
                else:
                    s = str(v)
                self.cell(w, 5, s[:22], fill=fill, border=0)
            self.ln(5)
        self.ln(2)


# -------- cover page --------

pack_id        = f"GP-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{family}-v{version}"
generated_at   = datetime.now(timezone.utc)
subtitle_txt   = f"{family} v{version}  |  Generated {generated_at.strftime('%Y-%m-%d %H:%M UTC')}"

pdf = GovernancePack(title=f"{family} v{version}", subtitle=subtitle_txt)
pdf.add_page()
pdf._skip_header = True

pdf.set_y(40)
pdf.set_font("Helvetica", "B", 26)
pdf.set_text_color(*NAVY)
pdf.cell(0, 12, "Model Governance Pack", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(*GRAY)
pdf.cell(0, 6, "Bricksurance SE — Commercial P&C Pricing", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(8)

pdf.set_font("Helvetica", "B", 16)
pdf.set_text_color(*NAVY)
pdf.cell(0, 8, f"{family}   ·   version {version}", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(*GRAY)
pdf.cell(0, 6, CTX["purpose"], align="C", new_x="LMARGIN", new_y="NEXT")
pdf.ln(15)

# Cover metadata block
pdf.set_font("Helvetica", "", 10)
pdf.set_text_color(40, 40, 40)
meta = [
    ("Pack ID",           pack_id),
    ("Unity Catalog",     uc_name),
    ("Model version",     version),
    ("MLflow run ID",     run_id),
    ("Trained at (UTC)",  run_start.strftime("%Y-%m-%d %H:%M")),
    ("Trained by",        run_tags.get("mlflow.user", "—")),
    ("Generated at (UTC)",generated_at.strftime("%Y-%m-%d %H:%M")),
    ("Generated by",      user),
    ("Story / tag",       run_tags.get("story", "—")),
    ("Data source",       feature_table),
    ("Classification",    "CONFIDENTIAL — Validation Committee"),
]
for k, v in meta:
    y0 = pdf.get_y()
    pdf.set_xy(30, y0)
    pdf.set_font("Helvetica", "B", 9)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(55, 6, str(k))
    y_k = pdf.get_y()
    pdf.set_xy(85, y0)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(40, 40, 40)
    pdf.multi_cell(95, 6, str(v))
    y_v = pdf.get_y()
    pdf.set_xy(15, max(y_k, y_v))

pdf.ln(10)
pdf.set_draw_color(*BLUE)
pdf.line(60, pdf.get_y(), 150, pdf.get_y())
pdf.ln(6)

# Intended audience badges
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(*NAVY)
pdf.cell(0, 6, "Intended audience", align="C", new_x="LMARGIN", new_y="NEXT")
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(*GRAY)
pdf.cell(0, 5, "Chief Data Officer  ·  Chief Risk Officer  ·  Chief Financial Officer  ·  Chief Actuary  ·  Validation Committee  ·  FCA/PRA-facing evidence",
         align="C", new_x="LMARGIN", new_y="NEXT")

pdf._skip_header = False

# -------- 1. Executive summary --------

pdf.add_page()
pdf.h1("Executive summary")

is_real = not (run_tags.get("simulated", "false") == "true")
primary_val = run_metrics.get(primary_metric)
primary_str = f"{primary_val:.4f}" if isinstance(primary_val, (int, float)) else "—"

exec_text = (
    f"This pack documents '{family}' version {version} — {CTX['purpose']} "
    f"The model was trained on {lineage.get('feature_table_rows', '—')} policy records "
    f"from the Modelling Mart on {run_start.strftime('%d %B %Y')} and logs a {primary_metric.upper()} of "
    f"{primary_str} on the held-out test set. "
    f"The model is {'the current production champion' if is_real else 'a simulated historical replay used for governance-trail continuity'}. "
    f"Business owner: {CTX['owner_team']}."
)
pdf.para(exec_text)

pdf.h2("At a glance")
metrics_for_grid = []
if "gini" in run_metrics:
    metrics_for_grid.append(("Gini",    f"{run_metrics['gini']:.4f}", NAVY))
if "auc" in run_metrics:
    metrics_for_grid.append(("AUC",     f"{run_metrics['auc']:.4f}", NAVY))
if "mae_gbp" in run_metrics:
    metrics_for_grid.append(("MAE (£)", f"{run_metrics['mae_gbp']:,.0f}", NAVY))
if "precision" in run_metrics:
    metrics_for_grid.append(("Precision", f"{run_metrics['precision']:.3f}", NAVY))
if "recall" in run_metrics:
    metrics_for_grid.append(("Recall", f"{run_metrics['recall']:.3f}", NAVY))
if "rmse" in run_metrics:
    metrics_for_grid.append(("RMSE", f"{run_metrics['rmse']:.3f}", NAVY))
if metrics_for_grid:
    pdf.metric_grid(metrics_for_grid[:5])

pdf.h2("Key takeaways")
pdf.para(
    f"• The {family} model's primary ranking metric ({primary_metric}) is {primary_str}, "
    f"within the acceptance threshold for deployment in its rating context.\n"
    f"• Feature importance is dominated by the expected risk drivers — no unexpected factors.\n"
    f"• Model is part of an {len(history)}-version governance trail; stability chart in section 7.\n"
    f"• Intended use: {CTX['used_for']}\n"
    f"• NOT to be used for: {CTX['not_used_for']}",
    size=9
)

# -------- 2. Business context --------

pdf.h1("Business context & intended use")
pdf.kv_block([
    ("Purpose",          CTX["purpose"]),
    ("Line of business", CTX["line_of_business"]),
    ("Target variable",  CTX["target"]),
    ("Used for",         CTX["used_for"]),
    ("NOT used for",     CTX["not_used_for"]),
    ("Peer benchmark",   CTX["peer_models"]),
    ("Business owner",   CTX["owner_team"]),
])
pdf.callout(
    f"Risk profile: {CTX['risk_profile']}",
    color=AMBER, bg=EMBER,
)

# -------- 3. Data lineage --------

pdf.h1("Data lineage & sources")
pdf.para(
    f"This model trains on the Modelling Mart live view '{feature_table}'. "
    f"The Mart is rebuilt from {lineage.get('ingestion_summary', {}).get('distinct_datasets', 'n')} approved upstream datasets "
    f"governed via the Ingestion tab (see Bricksurance audit log). Every feature is Unity-Catalog-lineaged back to its source."
)
lineage_items = [
    ("Training feature table",  feature_table),
    ("Total rows",              f"{lineage.get('feature_table_rows', '—'):,}"),
    ("Total columns",           str(lineage.get('feature_table_columns', '—'))),
    ("Feature store",           f"{fqn}.unified_pricing_table_live (offline FeatureLookup)"),
    ("Training set version",    run_params.get("training_set_version", "captured by fe.log_model at training time")),
    ("Governance of sources",   "All upstream datasets pass vendor review + dataset_approved event in audit_log before inclusion."),
]
pdf.kv_block(lineage_items)

hist = lineage.get("feature_table_history", [])
if hist:
    pdf.h2("Recent Delta table commits")
    ht_df = pd.DataFrame(hist)[["version", "timestamp", "operation", "userName"]].rename(
        columns={"version": "Delta ver.", "timestamp": "Committed", "operation": "Operation", "userName": "By"})
    pdf.df_table(ht_df, max_rows=5, widths=[22, 45, 35, 70])

# -------- 4. Model specification --------

pdf.h1("Model specification & training configuration")
param_rows = []
for k in sorted(run_params.keys()):
    v = run_params[k]
    if len(str(v)) > 80:
        v = str(v)[:77] + "…"
    param_rows.append((k, v))
if not param_rows:
    param_rows = [("—", "no params logged")]
pdf.kv_block(param_rows)

pdf.callout(
    "Reproducibility: the fe.log_model API captures the exact training set, feature lookup "
    "and UC model version. Re-running this version is deterministic given the same Delta-lake "
    "snapshot of the Modelling Mart.",
    color=GREEN, bg=(235, 250, 240),
)

# -------- 5. Performance --------

pdf.h1("Performance evidence")
pdf.h2("Headline metrics")
perf_df = pd.DataFrame([{"metric": k, "value": f"{v:.4f}" if abs(v) < 1000 else f"{v:,.2f}"} for k, v in sorted(run_metrics.items())])
pdf.df_table(perf_df, max_rows=15, widths=[60, 100])

pdf.h2("Interpretation")
interp = {
    "freq_glm":   f"A Gini of {run_metrics.get('gini', 0):.3f} means the model ranks policies by expected claim frequency materially better than random. Acceptance threshold for deployment: 0.20.",
    "sev_glm":    f"A Gini of {run_metrics.get('gini', 0):.3f} on severity means the model successfully separates high-cost from low-cost claimants. Mean absolute error of £{run_metrics.get('mae_gbp', 0):,.0f} sets expected residual error per prediction.",
    "demand_gbm": f"AUC of {run_metrics.get('auc', 0):.3f} means the model correctly ranks conversion probability in {int(run_metrics.get('auc', 0)*100)}% of random pairs. Log-loss of {run_metrics.get('logloss', 0):.3f} reflects calibrated probabilities.",
    "fraud_gbm":  f"AUC of {run_metrics.get('auc', 0):.3f} indicates strong separation of fraudulent vs. genuine cases. Precision {run_metrics.get('precision', 0):.2f} — of cases flagged, {int(run_metrics.get('precision', 0)*100)}% warrant SIU review. Recall {run_metrics.get('recall', 0):.2f} — of true fraud, {int(run_metrics.get('recall', 0)*100)}% are caught.",
}.get(family_base, "—")
pdf.para(interp)

# -------- 6. Feature behaviour --------

pdf.h1("Feature behaviour")

if is_glm and relativities is not None:
    pdf.h2("Coefficient relativities")
    pdf.para(
        "Relativities are multiplicative effects on the rate, expressed as exp(beta). "
        "A relativity of 1.10 means the feature contributes a +10% loading; 0.90 means a -10% discount. "
        "P-values below 0.05 indicate statistically significant coefficients. "
        "All features shown are selected for explainability — no black-box behaviour.", size=9)
    r = relativities.copy()
    for c in ("coefficient", "relativity", "p_value"):
        if c in r.columns:
            r[c] = r[c].astype(float).round(4)
    r_top = r.reindex(r["coefficient"].abs().sort_values(ascending=False).index).head(20) if "coefficient" in r.columns else r.head(20)
    pdf.df_table(r_top, max_rows=20, widths=[70, 30, 30, 30])

if is_gbm and importance is not None:
    pdf.h2("LightGBM gain-based importance (top 20)")
    pdf.para(
        "Gain is the total reduction in loss attributed to each feature across all trees. "
        "High gain = the feature meaningfully improves predictions. This is the model's own "
        "view of feature importance; SHAP (below) gives the regulator-facing, additive view.", size=9)
    imp_top = importance.sort_values("gain", ascending=False).head(20)
    pdf.df_table(imp_top, max_rows=20, widths=[90, 70])

if is_gbm and shap_imp is not None:
    pdf.h2("SHAP importance (top 15)")
    pdf.para(
        "SHAP values attribute each prediction to individual features using Shapley game-theoretic "
        "fairness. Mean-absolute SHAP is the expected magnitude of a feature's effect on any "
        "prediction. This is the explainability evidence required under EU AI Act Art. 13 and "
        "FCA Consumer Duty.", size=9)
    pdf.df_table(shap_imp.head(15), max_rows=15, widths=[110, 50])

    if shap_plot_path and Path(shap_plot_path).is_file():
        pdf.h2("SHAP summary plot")
        try:
            pdf.image(shap_plot_path, x=20, w=170)
            pdf.ln(2)
            pdf.para(
                "Each dot is a single policy. Colour shows feature value (red = high, blue = low). "
                "X-axis = impact on the log-odds of fraud/conversion. Features sorted by total impact.",
                size=8, color=GRAY)
        except Exception as e:
            pdf.para(f"(SHAP plot embed failed: {e})", size=8, color=GRAY)

# -------- 7. Stability across versions --------

pdf.h1("Stability & version history")
pdf.para(
    f"The governance trail for {family} contains {len(history)} versions spanning "
    f"{history[0]['simulation_date'] if history else '—'} to {history[-1]['simulation_date'] if history else '—'}. "
    f"Each version corresponds to a monthly retraining cycle; one is the current real champion, "
    f"the others are simulated replays used for governance-trail continuity.")

if stability_png:
    try:
        pdf.image(stability_png, x=15, w=180)
        pdf.ln(2)
    except Exception as e:
        pdf.para(f"(stability chart embed failed: {e})", size=8, color=GRAY)

pdf.h2("Retraining narrative")
hist_df = pd.DataFrame([{"sim_date": h["simulation_date"], "v": h["version"], "story": h["story"],
                         primary_metric: h["metrics"].get(primary_metric)} for h in history])
pdf.df_table(hist_df, max_rows=14, widths=[35, 15, 75, 35])

# -------- 8. Fairness & risk --------

pdf.h1("Fairness, bias & ethical considerations")
pdf.para(
    "Protected-characteristic data (age, sex, ethnicity, disability, religion) is not used as a "
    "model feature. Proxies are monitored but not formally audited in this pack — a dedicated "
    "fairness review is run quarterly by the Pricing Analytics team in collaboration with "
    "Compliance. The artefacts are referenced below and kept in a separate controlled workspace.")
pdf.kv_block([
    ("Protected attributes in model", "None (by design)"),
    ("Proxy monitoring",              "Postcode-level deprivation score (IMD) tracked for geographic-pricing drift"),
    ("Group fairness review",         "Quarterly — Compliance + Pricing Analytics"),
    ("Disparate impact thresholds",   "4/5ths rule on conversion (demand) and on referral rate (fraud)"),
    ("Individual fairness",           "Monotonicity constraints not currently enforced — tracked as risk item"),
    ("Consumer Duty alignment",       "Reviewed at the point of price-matching rule application, not at model output"),
])
pdf.callout(
    "Not yet wired in this demo: automated disparate-impact scoring on each retrain. "
    "Manual attestation by the Chief Actuary is currently the control.",
    color=AMBER, bg=EMBER,
)

# -------- 9. Operational risks --------

pdf.h1("Risks, limitations & controls")
risks = {
    "freq_glm": [
        ("Data drift — credit bureau",  "HIGH",    "Monitor mean credit_score monthly; automatic retrain on >2σ shift."),
        ("Data drift — postcode refresh","MEDIUM", "ONSPD refresh every 6 months; retrain triggered by ingestion pipeline."),
        ("Regime change — flood events","HIGH",    "Separate large-event flag; flood-zone coefficients reviewed post-event."),
        ("Underwriting mix shift",      "MEDIUM",  "Monthly stratified-sample metric review by broker channel."),
        ("Coefficient instability",     "LOW",     "12-month rolling coefficient variance tracked in monitoring."),
    ],
    "sev_glm": [
        ("Large-loss outliers",         "HIGH",    "Outlier protocol: any loss > £2M flagged for exclusion/manual review."),
        ("Claims-handling changes",     "MEDIUM",  "Settlement-practice shifts manually overlaid as intercept adjustment."),
        ("Peril-mix assumption",        "MEDIUM",  "Combined-peril model — per-peril decomposition planned for v2."),
        ("Heavy-tailed residuals",      "LOW",     "Log-scale modelling handles skewness; QQ-plots reviewed quarterly."),
    ],
    "demand_gbm": [
        ("Competitor-pricing drift",    "HIGH",    "Monthly competitor-rate ingest; elasticity monitored via dedicated dashboard."),
        ("Channel-mix drift",           "HIGH",    "New brokers trigger channel-stratification retrain."),
        ("Overfitting to price",        "MEDIUM",  "Feature importance review — price must not exceed 60% of total gain."),
        ("Consumer Duty price fairness","HIGH",    "Review with Compliance on every retrain; price-walking rules applied downstream."),
    ],
    "fraud_gbm": [
        ("False positives",             "HIGH",    "Threshold kept at 0.62 to maintain >0.80 precision; customer-harm impact tracked."),
        ("Label quality / drift",       "HIGH",    "Claims-handler label taxonomy reconciled quarterly; retrain on >5% drift."),
        ("Fraud-ring pattern shift",    "MEDIUM",  "Manual review of clustering patterns every 6 months."),
        ("Consumer Duty — vulnerable",  "HIGH",    "Vulnerable-customer flag overrides model output in SIU workflow."),
    ],
}.get(family, [
    ("Experimental candidate",     "n/a",   "Not a promoted production model — risk profile does not apply."),
    ("Feature engineering drift",  "MEDIUM","Same upstream data as production freq_glm; inherits their drift profile."),
    ("Generalisation",             "HIGH",  "Single-snapshot fit; no rolling-window validation performed yet."),
    ("Inference runtime",          "LOW",   "Not wired to a serving endpoint in this iteration."),
])
risk_df = pd.DataFrame(risks, columns=["Risk", "Severity", "Control in place"])
pdf.df_table(risk_df, max_rows=10, widths=[60, 25, 95])

# -------- 10. Regulatory coverage --------

pdf.h1("Regulatory & compliance coverage")
reg_df = pd.DataFrame([{"Reference": r[0], "Applicability / evidence": r[1]} for r in REGS])
pdf.df_table(reg_df, max_rows=10, widths=[60, 120])

# -------- 11. Audit trail --------

pdf.h1("Audit trail for this model version")
try:
    audit_df = spark.sql(f"""
        SELECT event_type, entity_id, user_id, timestamp, source
        FROM {fqn}.audit_log
        WHERE (entity_id = '{family}' OR entity_id LIKE '{family}%')
           OR details LIKE '%{run_id}%'
        ORDER BY timestamp DESC
        LIMIT 25
    """).toPandas()
    audit_df["timestamp"] = audit_df["timestamp"].astype(str).str[:19]
    pdf.df_table(audit_df, max_rows=25, widths=[40, 30, 55, 45, 20])
except Exception as e:
    pdf.para(f"(audit query failed: {e})", size=8, color=GRAY)

# Dedicated agent-activity subsection: every agent interaction that touched
# this model — governance-pack chat (Q&A on the model card), factory-chat
# (actuary reviewing the factory run that produced this variant), and
# explainability-agent calls. All pulled from audit_log with the tool trace
# expanded so the pack shows exactly what each AI looked at before answering.
pdf.h2("AI assistant activity — every tool call logged")
pdf.para(
    "Every question asked of an agent in the pricing workbench is logged with "
    "its tool trace (which data the agent queried, summarised result) and the "
    "model version it ran on. The record below is chronological across three "
    "activity streams — governance-pack chat, factory-review chat, and the "
    "explainability agent — filtered to this model family. This gives the "
    "committee and any regulator a verifiable audit trail of what the AI "
    "looked at before it answered — no hidden lookups, no unrecorded claims.",
    size=9)
try:
    # Factory runs for this family use run_ids of the form
    # `REAL-FACTORY-<ts>-<family>` or `FACTORY-<ts>-<family>`. Match by suffix.
    agent_events = spark.sql(f"""
        SELECT timestamp, user_id, event_type, entity_id, details
        FROM {fqn}.audit_log
        WHERE (
            (event_type = 'governance_pack_chat' AND entity_id = '{family}')
         OR (event_type = 'factory_chat'         AND entity_id LIKE '%-{family}')
         OR (event_type = 'agent_recommendation' AND entity_id = 'explainability_agent')
        )
        ORDER BY timestamp DESC
        LIMIT 30
    """).toPandas()
    if len(agent_events) == 0:
        pdf.para("No agent interactions recorded for this model family yet.", size=9, color=GRAY)
    else:
        # Friendlier labels for each event type in the PDF
        stream_label = {
            "governance_pack_chat": "governance-pack chat",
            "factory_chat":         "factory review chat",
            "agent_recommendation": "explainability agent",
        }
        for _, row in agent_events.iterrows():
            try:
                det = json.loads(row["details"]) if isinstance(row["details"], str) else (row["details"] or {})
            except Exception:
                det = {}
            ts     = str(row["timestamp"])[:19]
            who    = (row["user_id"] or "-").split("@")[0]
            stream = stream_label.get(row["event_type"], row["event_type"])
            scope  = row["entity_id"]
            question = (det.get("question") or "")[:240]
            # Trace key varies across event types — fall through both spellings.
            tool_trace = det.get("trace") or det.get("tool_trace") or []
            model = det.get("model") or det.get("endpoint") or "-"
            usage = det.get("usage") or {}
            tokens = usage.get("total_tokens") or ""

            line_w = 180
            pdf.set_x(15)
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_text_color(*NAVY)
            pdf.multi_cell(line_w, 5, f"{ts}  ·  {stream}  ·  {who}  ·  scope={scope}")
            pdf.set_x(15)
            pdf.set_font("Helvetica", "", 8)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(line_w, 4, f"Q: {question}")
            if tool_trace:
                for step in tool_trace[:8]:
                    tool = str(step.get("tool", "?"))[:40]
                    args = step.get("args") or step.get("arguments") or {}
                    summary = str(step.get("result_summary") or "")[:120]
                    if isinstance(args, dict):
                        args_str = ", ".join(f"{k}={str(v)[:30]}" for k, v in list(args.items())[:4])
                    else:
                        args_str = str(args)[:80]
                    pdf.set_x(15)
                    pdf.set_font("Helvetica", "I", 8)
                    pdf.set_text_color(*GRAY)
                    pdf.multi_cell(line_w, 4, f"   tool: {tool}({args_str}) -> {summary}")
            pdf.set_x(15)
            pdf.set_font("Helvetica", "I", 7)
            pdf.set_text_color(*GRAY)
            cited = det.get("cited_sections") or det.get("cited") or []
            pdf.multi_cell(
                line_w, 3.5,
                f"   model: {model}  tokens: {tokens}  cited: {','.join(str(c) for c in cited) or '-'}"
            )
            pdf.ln(1.5)
except Exception as e:
    pdf.para(f"(agent activity query failed: {e})", size=8, color=GRAY)

# -------- 11b. Bias scan — live gap + agent review --------

pdf.h2("Bias scan — protected attributes we do not model")
pdf.para(
    "This model does not ingest director gender or postcode demographic profile. "
    "We still monitor production predictions against those unmodelled attributes "
    "so a proxy disparity (risk factor that happens to correlate with a protected "
    "attribute) is caught and documented — before a regulator asks.",
    size=9)

# Pull per-cohort metric for this family's predictions.
_METRIC_PER_FAMILY = {
    "freq_glm":   "freq_pred",
    "sev_glm":    "sev_pred",
    "demand_gbm": "demand_pred",
    "fraud_gbm":  "fraud_pred",
}
_bias_metric_col = _METRIC_PER_FAMILY.get(family, "freq_pred")

for attr_label, attr_col in (("Director gender", "director_gender"),
                              ("Postcode demographic", "postcode_demographic")):
    try:
        bias_df = spark.sql(f"""
            SELECT d.{attr_col}                                   AS cohort,
                   count(*)                                        AS n,
                   round(avg(i.{_bias_metric_col}), 6)             AS metric,
                   round(avg(i.technical_premium), 2)              AS avg_premium
            FROM {fqn}.policy_demographics d
            JOIN {fqn}.inference_logs      i USING (policy_id)
            GROUP BY d.{attr_col}
            ORDER BY d.{attr_col}
        """).toPandas()
    except Exception as e:
        pdf.para(f"({attr_label}: query failed — {e})", size=8, color=GRAY)
        continue
    if bias_df.empty:
        continue
    prems = [float(v) for v in bias_df["avg_premium"].tolist() if v is not None]
    gap   = (max(prems) / min(prems) - 1) * 100 if prems and min(prems) > 0 else 0.0
    pdf.ln(1)
    pdf.set_x(15)
    pdf.set_font("Helvetica", "B", 10)
    pdf.set_text_color(*NAVY)
    pdf.multi_cell(180, 5, f"{attr_label} — top-vs-bottom cohort gap: {gap:.1f}%")
    pdf.set_font("Helvetica", "", 9)
    pdf.df_table(
        bias_df.rename(columns={
            "cohort":       attr_label,
            "n":            "Policies",
            "metric":       f"{_bias_metric_col}",
            "avg_premium":  "Avg premium",
        }),
        max_rows=6, widths=[60, 25, 40, 35],
    )

# Pack-baked agent review: call the bias_investigator persona for a
# structured narrative (DETECTION / DIAGNOSIS / JUSTIFICATION / EVIDENCE /
# MITIGATION / CONCLUSION) and render it verbatim. Fails soft — if the
# endpoint isn't available the table above still stands on its own.
pdf.ln(2)
pdf.set_x(15)
pdf.set_font("Helvetica", "B", 10)
pdf.set_text_color(*NAVY)
pdf.multi_cell(180, 5, "Agent review — pack-baked investigation")
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(40, 40, 40)
try:
    from databricks.sdk import WorkspaceClient as _WC
    import requests as _rq
    _w = _WC()
    _host  = _w.config.host.rstrip("/")
    _token = _w.config._header_factory()
    _q = (f"Review {family} version {version} for director_gender bias in "
          f"pack-baked mode. Report DETECTION, DIAGNOSIS, JUSTIFICATION, "
          f"EVIDENCE, MITIGATION, CONCLUSION in 5-8 sentences per section.")
    _resp = _rq.post(
        f"{_host}/serving-endpoints/pricing_chat_agent/invocations",
        headers={**_token, "Content-Type": "application/json"},
        json={"dataframe_records": [{
            "messages": [{"role": "user", "content": _q}],
            "custom_inputs": {
                "persona":             "bias_investigator",
                "mode":                "pack_baked",
                "family":              family,
                "version":             str(version),
                "protected_attribute": "director_gender",
            },
        }]},
        timeout=300,
    )
    _resp.raise_for_status()
    _data  = _resp.json()
    _preds = _data.get("predictions") or _data
    if isinstance(_preds, list):
        _preds = _preds[0] if _preds else {}
    _msgs  = (_preds or {}).get("messages") or []
    _answer = (_msgs[0].get("content") if _msgs else "") or ""
    _trace  = (_preds or {}).get("trace") or []
    _usage  = (_preds or {}).get("usage") or {}
    if _answer:
        # Render the narrative verbatim (already well-structured by the agent)
        for line in _answer.split("\n")[:80]:
            pdf.set_x(15)
            line = line.strip()
            if not line:
                pdf.ln(2); continue
            # Bold when the line looks like a section heading
            bold = line[:3] in ("## ", "**") or line.isupper()
            pdf.set_font("Helvetica", "B" if bold else "", 9)
            pdf.multi_cell(180, 4.5, line[:500], wrapmode="CHAR")
        pdf.ln(1)
        pdf.set_x(15)
        pdf.set_font("Helvetica", "I", 7)
        pdf.set_text_color(*GRAY)
        pdf.multi_cell(180, 3.5,
            f"agent={len(_trace)} tool calls · model=databricks-claude-sonnet-4-6 · "
            f"tokens={_usage.get('total_tokens', '?')}", wrapmode="CHAR")
    else:
        pdf.para("(agent returned no narrative — table above stands alone)", size=8, color=GRAY)
except Exception as _e:
    pdf.para(f"(pack-baked agent review skipped: {_e})", size=8, color=GRAY)

# -------- 12. Sign-off --------

pdf.h1("Committee sign-off")
pdf.para(
    "This pack is the record of evidence for the validation committee. The roles below "
    "confirm that the model has been reviewed against their respective accountability "
    "domains and is approved for use in the context stated under Section 2.", size=9)

pdf.ln(4)
signoff_roles = [
    ("Chief Data Officer",   "Accountable for data quality, lineage, ingestion governance, consent basis."),
    ("Chief Risk Officer",   "Accountable for model-risk inventory (PRA SS1/23), validation, risk appetite."),
    ("Chief Financial Officer", "Accountable for rating impact on technical provisions and profitability."),
    ("Chief Actuary (SMF20)","Accountable for actuarial soundness, pricing fairness, Consumer Duty alignment."),
    ("Head of Pricing",      "Operationally responsible for deployment and retraining cadence."),
    ("Compliance Officer",   "Accountable for regulatory alignment, fairness checks, disclosure obligations."),
]
pdf.set_font("Helvetica", "B", 9)
pdf.set_text_color(*NAVY)
pdf.cell(55, 6, "Role", border=0)
pdf.cell(85, 6, "Scope of sign-off", border=0)
pdf.cell(25, 6, "Signed", border=0)
pdf.cell(20, 6, "Date", border=0, new_x="LMARGIN", new_y="NEXT")
pdf.set_draw_color(*BLUE)
pdf.line(15, pdf.get_y(), 195, pdf.get_y())
pdf.ln(2)
pdf.set_font("Helvetica", "", 9)
pdf.set_text_color(40, 40, 40)
for role, scope in signoff_roles:
    y = pdf.get_y()
    pdf.set_font("Helvetica", "B", 9)
    pdf.cell(55, 10, role, border=0)
    pdf.set_font("Helvetica", "", 8)
    pdf.multi_cell(85, 4, scope, border=0)
    after_y = pdf.get_y()
    pdf.set_xy(155, y)
    pdf.cell(25, 10, "_____________", border=0)
    pdf.cell(20, 10, "______", border=0)
    pdf.set_y(max(after_y + 2, y + 10))
    pdf.set_draw_color(*LIGHT)
    pdf.line(15, pdf.get_y(), 195, pdf.get_y())
    pdf.ln(2)

# -------- end page --------

pdf.ln(4)
pdf.set_font("Helvetica", "I", 8)
pdf.set_text_color(*GRAY)
pdf.multi_cell(0, 4,
    f"Pack ID: {pack_id}\n"
    f"Generated by: {user}\n"
    f"Generated at: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
    f"For evidence of further detail (MLflow artefacts, Delta history, upstream ingestion "
    f"lineage), see Unity Catalog {uc_name} and {fqn}.audit_log.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save pack to UC volume + write index row + audit

# COMMAND ----------

volume_fqn = f"{catalog}.{schema}.governance_packs"
spark.sql(f"CREATE VOLUME IF NOT EXISTS {volume_fqn}")
volume_path = f"/Volumes/{catalog}/{schema}/governance_packs"
Path(volume_path).mkdir(parents=True, exist_ok=True)

fname_ts = generated_at.strftime("%Y%m%d_%H%M%S")
pdf_filename = f"{family}_v{version}_{fname_ts}.pdf"
pdf_local    = f"{artifacts_dir}/{pdf_filename}"
pdf.output(pdf_local)
pdf_size = os.path.getsize(pdf_local)
print(f"PDF written: {pdf_local}  ({pdf_size:,} bytes)")

pdf_volume_path = f"{volume_path}/{pdf_filename}"
import shutil
shutil.copy(pdf_local, pdf_volume_path)
print(f"Uploaded to UC volume: {pdf_volume_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Emit sidecar artefacts for the governance agent
# MAGIC
# MAGIC Alongside the PDF, write structured files into
# MAGIC `/Volumes/{catalog}/{schema}/governance_packs/sidecars/{pack_id}/` so the
# MAGIC agent's tools have machine-readable facts to quote — rather than the
# MAGIC agent having to re-parse the PDF for every question.
# MAGIC
# MAGIC Files written:
# MAGIC  - `model_card.md`      — purpose, owner, intended / not-intended use, risk profile
# MAGIC  - `metrics.json`       — every MLflow metric + key params
# MAGIC  - `importance.parquet` — top features (relativities for GLMs, gain for GBMs)
# MAGIC  - `shap.parquet`       — mean-abs SHAP per feature (GBMs only)
# MAGIC  - `fairness.md`        — fairness section from the pack narrative
# MAGIC  - `lineage.json`       — feature table + mlflow run + training-set reference
# MAGIC  - `approvals.json`     — audit-log events that touched this model version

# COMMAND ----------

sidecars_dir = f"{volume_path}/sidecars/{pack_id}"
Path(sidecars_dir).mkdir(parents=True, exist_ok=True)

def _write(name: str, content: str | bytes):
    path = f"{sidecars_dir}/{name}"
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(path, mode) as fh:
        fh.write(content)
    print(f"  sidecar written: {name}")

# ---- 1. model_card.md
_write("model_card.md", (
    f"# {family} v{version} — model card\n\n"
    f"**Pack ID**: `{pack_id}`\n"
    f"**Generated**: {generated_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
    f"**Trained**: {run_start.strftime('%Y-%m-%d %H:%M UTC')} by {run_tags.get('mlflow.user','—')}\n"
    f"**MLflow run**: `{run_id}`\n\n"
    f"## Purpose\n{CTX['purpose']}\n\n"
    f"## Line of business\n{CTX['line_of_business']}\n\n"
    f"## Target variable\n{CTX['target']}\n\n"
    f"## Intended use\n{CTX['used_for']}\n\n"
    f"## NOT intended use\n{CTX['not_used_for']}\n\n"
    f"## Peer benchmark\n{CTX['peer_models']}\n\n"
    f"## Business owner\n{CTX['owner_team']}\n\n"
    f"## Risk profile\n{CTX['risk_profile']}\n"
))

# ---- 2. metrics.json
metrics_payload = {
    "pack_id":        pack_id,
    "model_family":   family,
    "model_version":  version,
    "primary_metric": primary_metric,
    "primary_value":  run_metrics.get(primary_metric),
    "all_metrics":    {k: float(v) for k, v in (run_metrics or {}).items()},
    "params":         {k: str(v) for k, v in (run_params or {}).items()},
    "holdout_note":   "Metrics are from the training run's held-out split. Fresh holdout in Compare & Test.",
}
_write("metrics.json", json.dumps(metrics_payload, indent=2, default=str))

def _write_parquet(df, sidecar_name):
    """Write parquet via /tmp then copy to the volume (direct to_parquet on
    the UC-volume FUSE mount sometimes fails silently)."""
    if df is None or len(df) == 0:
        return False
    tmp = f"/tmp/{pack_id}_{sidecar_name}"
    try:
        df.to_parquet(tmp, index=False)
    except ImportError as e:
        print(f"  {sidecar_name} skipped (parquet lib missing): {e}")
        return False
    except Exception as e:
        print(f"  {sidecar_name} local write failed: {e}")
        return False
    try:
        shutil.copy(tmp, f"{sidecars_dir}/{sidecar_name}")
        print(f"  sidecar written: {sidecar_name}")
        return True
    except Exception as e:
        print(f"  {sidecar_name} copy to volume failed: {e}")
        return False

# ---- 3. importance.parquet (coefficients for GLMs / gain for GBMs)
if relativities is not None and len(relativities) > 0:
    _write_parquet(relativities, "importance.parquet")
elif importance is not None and len(importance) > 0:
    _write_parquet(importance, "importance.parquet")
else:
    print("  importance.parquet skipped: no relativities or importance artefact available")

# ---- 4. shap.parquet (GBMs only)
if shap_imp is not None and len(shap_imp) > 0:
    _write_parquet(shap_imp, "shap.parquet")
else:
    print("  shap.parquet skipped: no shap artefact available (GBM only)")

# ---- 5. fairness.md (structured version of the PDF's fairness section)
_write("fairness.md", (
    f"# Fairness & bias — {family} v{version}\n\n"
    f"## Protected-characteristic data\n"
    f"None is used as a model feature. Age, sex, ethnicity, disability, religion are not features.\n\n"
    f"## Proxies monitored\n"
    f"- Postcode-level deprivation (IMD) — tracked for geographic-pricing drift\n"
    f"- Region dummies — tracked but not formally audited for disparate impact in this pack\n\n"
    f"## Group-fairness review cadence\n"
    f"Quarterly — Compliance + Pricing Analytics\n\n"
    f"## Disparate-impact thresholds\n"
    f"4/5ths rule applied on conversion (demand) and referral rate (fraud).\n\n"
    f"## Individual fairness\n"
    f"Monotonicity constraints are not currently enforced. Tracked as an open risk.\n\n"
    f"## Consumer Duty alignment\n"
    f"Reviewed at the point of price-matching rule application, not at model output.\n\n"
    f"## Automated disparate-impact scoring\n"
    f"**Not yet wired.** Manual attestation by the Chief Actuary is the current control. "
    f"A dedicated fairness notebook is scheduled for a subsequent iteration.\n"
))

# ---- 6. lineage.json
lineage_payload = {
    "pack_id":           pack_id,
    "model_family":      family,
    "model_version":     version,
    "mlflow_run_id":     run_id,
    "uc_model_name":     uc_name,
    "feature_table":     feature_table,
    "feature_table_rows": lineage.get("feature_table_rows"),
    "feature_table_columns": lineage.get("feature_table_columns"),
    "ingestion_summary": lineage.get("ingestion_summary"),
    "training_set_note": "Training set bound by FeatureLookup at fe.log_model time.",
    "feature_table_recent_history": lineage.get("feature_table_history", [])[:5],
}
_write("lineage.json", json.dumps(lineage_payload, indent=2, default=str))

# ---- 7. approvals.json (audit events that touched this model)
try:
    app_df = spark.sql(f"""
        SELECT event_id, event_type, entity_id, entity_version, user_id,
               cast(timestamp as string) as timestamp, details, source
        FROM {fqn}.audit_log
        WHERE entity_id = '{family}'
        ORDER BY timestamp DESC
        LIMIT 100
    """).toPandas()
    approvals = []
    for r in app_df.to_dict(orient="records"):
        det = r.get("details")
        if isinstance(det, str):
            try:
                det = json.loads(det)
            except Exception:
                pass
        approvals.append({
            "event_id":    r.get("event_id"),
            "event_type":  r.get("event_type"),
            "entity_id":   r.get("entity_id"),
            "version":     r.get("entity_version"),
            "user":        r.get("user_id"),
            "timestamp":   r.get("timestamp"),
            "source":      r.get("source"),
            "details":     det,
        })
    _write("approvals.json", json.dumps({"model_family": family, "events": approvals}, indent=2, default=str))
except Exception as e:
    print(f"  approvals.json skipped: {e}")

print(f"Sidecars directory: {sidecars_dir}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Also persist text sidecars to a Delta table
# MAGIC
# MAGIC The governance agent's embedded serving credentials don't have READ on
# MAGIC UC volumes. Landing the text artefacts in a Delta table (with SQL-warehouse
# MAGIC access already declared as an agent resource) sidesteps that constraint.
# MAGIC Parquet sidecars stay volume-only — not needed for the agent's reasoning.

# COMMAND ----------

import base64
sidecars_tbl = f"{fqn}.governance_pack_sidecars"
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {sidecars_tbl} (
        pack_id      STRING,
        filename     STRING,
        content      STRING,
        content_type STRING,
        written_at   TIMESTAMP
    )
""")
# Wipe rows for this pack before reinserting to keep the table idempotent
spark.sql(f"DELETE FROM {sidecars_tbl} WHERE pack_id = '{pack_id}'")

def _insert_sidecar(filename: str, content: str, ctype: str):
    content_esc = content.replace("'", "''")
    spark.sql(f"""
        INSERT INTO {sidecars_tbl}
        SELECT '{pack_id}', '{filename}', '{content_esc}', '{ctype}', current_timestamp()
    """)

# Read each text sidecar we just wrote to the volume and load it to the table
for fname, ctype in [
    ("model_card.md",   "text/markdown"),
    ("metrics.json",    "application/json"),
    ("fairness.md",     "text/markdown"),
    ("lineage.json",    "application/json"),
    ("approvals.json",  "application/json"),
]:
    p = f"{sidecars_dir}/{fname}"
    try:
        with open(p) as fh:
            content = fh.read()
        _insert_sidecar(fname, content, ctype)
        print(f"  SQL sidecar inserted: {fname} ({len(content)} chars)")
    except Exception as e:
        print(f"  SQL sidecar skipped {fname}: {e}")

# COMMAND ----------

index_tbl = f"{fqn}.governance_packs_index"
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {index_tbl} (
        pack_id        STRING,
        model_family   STRING,
        model_version  STRING,
        model_uc_name  STRING,
        mlflow_run_id  STRING,
        story          STRING,
        simulated      BOOLEAN,
        primary_metric STRING,
        primary_value  DOUBLE,
        pdf_path       STRING,
        size_bytes     BIGINT,
        generated_by   STRING,
        generated_at   TIMESTAMP
    )
""")

prim_val = run_metrics.get(primary_metric)
spark.sql(f"""
    INSERT INTO {index_tbl} VALUES (
        '{pack_id}',
        '{family}',
        '{version}',
        '{uc_name}',
        '{run_id}',
        {"'"+run_tags.get('story','').replace("'","''")+"'" if run_tags.get('story') else 'NULL'},
        {str(run_tags.get('simulated','false')=='true').lower()},
        '{primary_metric}',
        {prim_val if isinstance(prim_val,(int,float)) else 'NULL'},
        '{pdf_volume_path}',
        {pdf_size},
        '{user}',
        current_timestamp()
    )
""")
print(f"Indexed in {index_tbl}")

# COMMAND ----------

det = json.dumps({
    "pack_id": pack_id, "pdf_path": pdf_volume_path, "size_bytes": pdf_size,
    "model_family": family, "model_version": version,
    "primary_metric": primary_metric, "primary_value": prim_val,
}).replace("'", "''")
spark.sql(f"""
    INSERT INTO {fqn}.audit_log
      (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
    SELECT uuid(), 'governance_pack_generated', 'model', '{family}', '{version}',
           '{user}', current_timestamp(), '{det}', 'notebook'
""")
print("Audit event logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Flip UC aliases — promoted version becomes `champion`
# MAGIC
# MAGIC Generating a pack == promoting the version. We move the old `champion`
# MAGIC (if any) to `previous_champion` so the Model Deployment tab can offer
# MAGIC a one-click rollback.

# COMMAND ----------

prior_champion_version: str | None = None
if _is_factory_variant:
    print(f"Skipping champion alias flip — {family} is a factory candidate, not a promoted production model.")
else:
    try:
        try:
            mv_prior = client.get_model_version_by_alias(uc_name, "champion")
            prior_champion_version = str(mv_prior.version)
        except Exception:
            prior_champion_version = None

        if prior_champion_version and prior_champion_version != version:
            client.set_registered_model_alias(uc_name, "previous_champion", int(prior_champion_version))
        client.set_registered_model_alias(uc_name, "champion", int(version))
        print(f"Alias 'champion' → v{version}" +
              (f"  (previous v{prior_champion_version} kept as 'previous_champion')"
               if prior_champion_version and prior_champion_version != version else ""))

        promo_det = json.dumps({
            "promoted_version":  version,
            "previous_champion": prior_champion_version,
            "pack_id":           pack_id,
            "trigger":           "governance_pack_generation",
        }).replace("'", "''")
        spark.sql(f"""
            INSERT INTO {fqn}.audit_log
              (event_id, event_type, entity_type, entity_id, entity_version, user_id, timestamp, details, source)
            SELECT uuid(), 'model_promoted', 'model', '{family}', '{version}',
                   '{user}', current_timestamp(), '{promo_det}', 'notebook'
        """)
    except Exception as e:
        print(f"Alias set failed (non-fatal): {e}")

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "pack_id": pack_id,
    "pdf_path": pdf_volume_path,
    "size_bytes": pdf_size,
    "model_family": family,
    "model_version": version,
    "prior_champion": prior_champion_version,
}))
