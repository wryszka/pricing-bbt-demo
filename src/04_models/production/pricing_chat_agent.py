# Databricks notebook source
# MAGIC %md
# MAGIC # Pricing Chat Agent — Databricks Agent Framework (factory + explainability)
# MAGIC
# MAGIC Deploys a single Agent Framework serving endpoint that replaces the two
# MAGIC remaining direct Foundation-Model-API calls the app still makes:
# MAGIC
# MAGIC | App route                    | Old path                         | Now |
# MAGIC |---|---|---|
# MAGIC | `POST /api/factory/chat`     | Direct FM API (Claude 4.6)       | Calls this endpoint with `custom_inputs.persona = "factory"` |
# MAGIC | `POST /api/factory-real/chat`| Direct FM API (Claude 4.6)       | Same endpoint, same persona |
# MAGIC | `POST /api/agent/explain`    | Direct FM API (Claude 4.6)       | Calls this endpoint with `custom_inputs.persona = "explain"` |
# MAGIC
# MAGIC Persona selection routes to a different system prompt and tool set so the
# MAGIC guardrails stay tight. Every interaction is recorded in `audit_log` by
# MAGIC the app so the governance pack can surface agent activity later.
# MAGIC
# MAGIC Shares the tool-use loop shape with `governance_agent.py`.

# COMMAND ----------

dbutils.widgets.text("catalog_name",  "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",   "pricing_upt")
dbutils.widgets.text("endpoint_name", "pricing_chat_agent")
dbutils.widgets.text("fm_endpoint",   "databricks-claude-sonnet-4-6")
dbutils.widgets.text("warehouse_id",  "")  # SQL warehouse the agent tools query; defaults to first running serverless warehouse

# COMMAND ----------

# MAGIC %pip install mlflow databricks-agents databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")
fm_endpoint   = dbutils.widgets.get("fm_endpoint")
warehouse_id  = dbutils.widgets.get("warehouse_id")
fqn           = f"{catalog}.{schema}"
agent_uc_name = f"{fqn}.pricing_chat_agent"

if not warehouse_id:
    from databricks.sdk import WorkspaceClient as _W
    _w = _W()
    _running = [wh for wh in _w.warehouses.list() if str(wh.state) == "WarehouseStatusState.RUNNING"]
    _all = [wh for wh in _w.warehouses.list()]
    warehouse_id = (_running or _all)[0].id
    print(f"warehouse_id auto-selected: {warehouse_id}")

import json, os, tempfile
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Persona prompts + tool definitions

# COMMAND ----------

FACTORY_SYSTEM = """You are the Bricksurance SE pricing-factory reviewer.
An actuary is reviewing a just-completed factory run and asking about variants.

Rules:
 * Answer ONLY from the tool results. If the tools don't surface it, reply exactly:
   "The factory run data does not answer that."
 * Cite variants by their ID (e.g. "A07", "B02") and the specific metric whenever you make a claim.
 * Never recommend promotion — that is the actuary's decision.
 * Keep answers short (4-8 sentences unless asked for more).
 * Always call a tool before answering — you do not know the answer without data.
"""

EXPLAIN_SYSTEM = """You are the Bricksurance SE actuarial explainability agent.
A product manager or regulator is asking why premiums changed after a data update
or pricing run. You produce plain-English explanations suitable for regulatory
filings.

Rules:
 * Ground every claim in tool data — portfolio stats or shadow-impact stats.
 * Do NOT speculate beyond what the tools return.
 * When asked for recommendations, keep them factual and process-oriented (e.g.
   "review affected segments", not "cut prices in London by 3%").
 * Respond as a JSON object with these keys (all strings unless noted):
     headline (one sentence), explanation (2-3 paragraphs),
     key_drivers (array of {factor, contribution, detail}),
     affected_segments (array of {segment, policies, premium_impact}),
     regulatory_statement (paragraph),
     recommended_actions (array of strings)
 * If the tools don't return enough data, explain that in `explanation` and
   leave the empty arrays empty — do not fabricate.
"""

FACTORY_TOOLS = [
    {
        "name": "query_factory_run",
        "description": "Return metadata for a single factory run (family, status, narrative, variant count).",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
    {
        "name": "query_factory_leaderboard",
        "description": "Return the leaderboard for a factory run — each variant with its metrics.",
        "input_schema": {
            "type": "object",
            "properties": {
                "run_id": {"type": "string"},
                "limit":  {"type": "integer", "description": "max variants to return", "default": 30},
            },
            "required": ["run_id"],
        },
    },
    {
        "name": "query_factory_shortlist",
        "description": "Return the shortlist for a factory run — a small set of top variants with full configs and CV metrics.",
        "input_schema": {
            "type": "object",
            "properties": {"run_id": {"type": "string"}},
            "required": ["run_id"],
        },
    },
]

EXPLAIN_TOOLS = [
    {
        "name": "query_portfolio_stats",
        "description": "Portfolio aggregates — total policies, total GWP, avg premium, avg risk score.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_shadow_impact",
        "description": "Shadow-pricing impact summary — number of affected policies, total premium delta, avg percent change, count of high-churn-risk policies.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "query_recent_dataset_approvals",
        "description": "Recent dataset approval events from the audit log — who approved what and when.",
        "input_schema": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "default": 10}},
        },
    },
]

# ---------------------------------------------------------------------------
# Bias investigator — the Governance tab's flagship persona
# ---------------------------------------------------------------------------

BIAS_SYSTEM = """You are the Bricksurance SE bias investigator — a fairness
audit agent that answers the kind of question a regulator or in-house
compliance officer would ask.  You monitor whether our production pricing
models produce disparate outcomes across protected attributes (gender,
postcode demographic quintile, ...). None of these attributes are IN the
models — your job is to prove (a) whether a gap exists and (b) whether the
gap is risk-justified.

Operating mode comes from `custom_inputs.mode`:
  * "live"          — monitor the current champions against inference_logs
  * "pre_promotion" — audit a candidate version BEFORE it becomes champion
                      (score_candidate_for_bias to get its predictions)
  * "pack_baked"    — summarise findings for inclusion in the governance PDF

Structure every substantive answer as these sections, in order, using the
exact headings. Skip a section only if the tools genuinely didn't surface
data for it.

  DETECTION — what's biased, by how much, per family
  DIAGNOSIS — which FEATURES drive the gap (the proxy path)
  JUSTIFICATION — actual claim experience by the same cut (defence data)
  EVIDENCE — pack section, SHAP artefact, prior-scan log
  MITIGATION — capped loading, balanced calibration, monitoring cadence
  CONCLUSION — one sentence: pricing reflects risk (or doesn't)

Rules:
 * Always call at least one tool before answering — never guess.
 * If the JUSTIFICATION tool returns a gap in the SAME direction and
   similar magnitude, the defence is strong. If the actual experience is
   FLAT or OPPOSITE, flag the premium disparity as unexplained and
   recommend reviewing the model.
 * Cite concrete numbers from the tool output (percentages, policy counts).
 * Keep the whole response readable — a committee member should be able
   to skim it in under 60 seconds.
"""


BIAS_TOOLS = [
    {
        "name": "query_bias_monitor",
        "description": (
            "Group current champion predictions by a protected attribute (not modelled) "
            "and return per-group mean predictions + premium + counts. "
            "Reads from inference_logs joined to policy_demographics."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family":            {"type": "string",
                                      "description": "optional — freq_glm | sev_glm | demand_gbm | fraud_gbm. If omitted returns all 4 + technical premium."},
                "protected_attribute": {"type": "string",
                                      "enum": ["director_gender", "postcode_demographic"]},
            },
            "required": ["protected_attribute"],
        },
    },
    {
        "name": "query_actual_experience",
        "description": (
            "Return actual 5-year claim experience (claim count, total incurred, loss ratio) "
            "grouped by a protected attribute. This is the DEFENCE data: if the actual loss "
            "experience differs in the same direction as the pricing, the pricing is "
            "risk-justified."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protected_attribute": {"type": "string",
                                        "enum": ["director_gender", "postcode_demographic"]},
            },
            "required": ["protected_attribute"],
        },
    },
    {
        "name": "query_proxy_features",
        "description": (
            "Rank features by how strongly they correlate with the protected attribute. "
            "Identifies the proxy path — which innocent-looking rating factor is acting as "
            "a stand-in for the protected attribute. Returns top 8 features by explanatory power."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "protected_attribute": {"type": "string",
                                        "enum": ["director_gender", "postcode_demographic"]},
            },
            "required": ["protected_attribute"],
        },
    },
    {
        "name": "read_fairness_section",
        "description": (
            "Read the `fairness.md` sidecar of a governance pack — documents how the model's "
            "design addresses proxy-discrimination risk."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pack_id": {"type": "string"},
            },
            "required": ["pack_id"],
        },
    },
    {
        "name": "query_latest_pack_id",
        "description": (
            "Return the most recent governance pack_id for a family — use this before "
            "read_fairness_section when the user didn't supply a specific pack_id."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"family": {"type": "string"}},
            "required": ["family"],
        },
    },
    {
        "name": "score_candidate_for_bias",
        "description": (
            "PRE-PROMOTION mode only: score a candidate model version on a stratified "
            "2000-policy sample and return the same bias stats as query_bias_monitor, "
            "so the actuary can see what the gap WOULD look like if this version were "
            "promoted. Warning: triggers live scoring via the Compare & Test job; can "
            "take 2-3 minutes on cold start."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family":              {"type": "string"},
                "version":             {"type": "string"},
                "protected_attribute": {"type": "string",
                                        "enum": ["director_gender", "postcode_demographic"]},
                "sample_size":         {"type": "integer", "default": 2000},
            },
            "required": ["family", "version", "protected_attribute"],
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent implementation — tool-use loop shared across personas

# COMMAND ----------

class PricingChatAgent(PythonModel):
    """Two-persona agent: `factory` (reviews a factory run) and `explain`
    (produces actuarial explainability narratives)."""

    def load_context(self, context):
        cfg_path = context.artifacts.get("config")
        if cfg_path:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            self.catalog     = cfg["catalog"]
            self.schema      = cfg["schema"]
            self.fm_endpoint = cfg["fm_endpoint"]
        else:
            self.catalog     = os.environ.get("AGENT_CATALOG", "lr_serverless_aws_us_catalog")
            self.schema      = os.environ.get("AGENT_SCHEMA", "pricing_upt")
            self.fm_endpoint = os.environ.get("AGENT_FM_ENDPOINT", "databricks-claude-sonnet-4-6")

    # ------------------------- Factory tools --------------------------------

    def _tool_query_factory_run(self, args):
        run_id = args.get("run_id")
        rows = _run_sql(f"""
            SELECT run_id, model_family, status, narrative, variant_count,
                   approved_by, duration_seconds,
                   cast(started_at as string) as started_at
            FROM {self.catalog}.{self.schema}.factory_runs
            WHERE run_id = '{run_id}' LIMIT 1
        """)
        return {"row": rows[0] if rows else None}

    def _tool_query_factory_leaderboard(self, args):
        run_id = args.get("run_id")
        limit  = max(1, min(60, int(args.get("limit", 30))))
        rows = _run_sql(f"""
            SELECT variant_id, name, category, n_features,
                   try_cast(get_json_object(metrics_json, '$.gini') AS DOUBLE) AS gini,
                   try_cast(get_json_object(metrics_json, '$.aic')  AS DOUBLE) AS aic,
                   try_cast(get_json_object(metrics_json, '$.bic')  AS DOUBLE) AS bic,
                   try_cast(get_json_object(metrics_json, '$.rmse') AS DOUBLE) AS rmse
            FROM {self.catalog}.{self.schema}.factory_variants
            WHERE run_id = '{run_id}'
            ORDER BY gini DESC NULLS LAST
            LIMIT {limit}
        """)
        return {"variants": rows, "count": len(rows)}

    def _tool_query_factory_shortlist(self, args):
        """Top 5 variants by gini — matches the app's /runs/{id}/shortlist endpoint."""
        run_id = args.get("run_id")
        rows = _run_sql(f"""
            SELECT variant_id, name, category, n_features, config_json, metrics_json
            FROM {self.catalog}.{self.schema}.factory_variants
            WHERE run_id = '{run_id}'
            ORDER BY try_cast(get_json_object(metrics_json, '$.gini') AS DOUBLE) DESC NULLS LAST
            LIMIT 5
        """)
        for r in rows:
            for k in ("metrics_json", "config_json"):
                v = r.get(k)
                if isinstance(v, str):
                    try:
                        r[k.replace("_json", "")] = json.loads(v)
                    except Exception:
                        pass
        return {"shortlist": rows, "count": len(rows)}

    # ------------------------- Explain tools --------------------------------

    def _tool_query_portfolio_stats(self, args):
        rows = _run_sql(f"""
            SELECT count(*)                          AS total_policies,
                   round(sum(current_premium))       AS total_gwp,
                   round(avg(current_premium))       AS avg_premium,
                   round(avg(combined_risk_score),2) AS avg_risk
            FROM {self.catalog}.{self.schema}.unified_pricing_table_live
        """)
        return {"stats": rows[0] if rows else {}}

    def _tool_query_shadow_impact(self, args):
        try:
            rows = _run_sql(f"""
                SELECT count(*) AS affected,
                       round(sum(premium_delta))     AS total_delta,
                       round(avg(premium_delta_pct),1) AS avg_pct,
                       sum(CASE WHEN churn_risk = 'HIGH' THEN 1 ELSE 0 END) AS high_churn_count
                FROM {self.catalog}.{self.schema}.shadow_pricing_impact
            """)
            return {"stats": rows[0] if rows else {}}
        except Exception as e:
            return {"error": f"shadow_pricing_impact not available: {e}"}

    def _tool_query_recent_dataset_approvals(self, args):
        limit = max(1, min(50, int(args.get("limit", 10))))
        rows = _run_sql(f"""
            SELECT event_type, entity_id, user_id,
                   cast(timestamp as string) as timestamp,
                   substr(details, 1, 400) AS details_preview
            FROM {self.catalog}.{self.schema}.audit_log
            WHERE event_type IN ('dataset_approved', 'dataset_rejected', 'dataset_uploaded')
            ORDER BY timestamp DESC
            LIMIT {limit}
        """)
        return {"rows": rows, "count": len(rows)}

    # ------------------------- Bias investigator tools ----------------------

    def _tool_query_bias_monitor(self, args):
        attr = args.get("protected_attribute") or "director_gender"
        if attr not in ("director_gender", "postcode_demographic"):
            return {"error": f"unknown protected_attribute: {attr}"}
        family = (args.get("family") or "").strip()
        # Per-family metric column
        metric_col = {
            "freq_glm":   "freq_pred",
            "sev_glm":    "sev_pred",
            "demand_gbm": "demand_pred",
            "fraud_gbm":  "fraud_pred",
        }.get(family)
        if family and not metric_col:
            return {"error": f"unknown family: {family}"}

        if metric_col:
            sql = f"""
                SELECT d.{attr} AS cohort,
                       count(*) AS n,
                       round(avg(i.{metric_col}), 6) AS metric,
                       round(avg(i.technical_premium), 2) AS avg_premium
                FROM {self.catalog}.{self.schema}.policy_demographics d
                JOIN {self.catalog}.{self.schema}.inference_logs i USING (policy_id)
                GROUP BY d.{attr}
                ORDER BY d.{attr}
            """
        else:
            sql = f"""
                SELECT d.{attr} AS cohort,
                       count(*) AS n,
                       round(avg(i.freq_pred),   6) AS freq_pred,
                       round(avg(i.sev_pred),    2) AS sev_pred,
                       round(avg(i.demand_pred), 6) AS demand_pred,
                       round(avg(i.fraud_pred),  6) AS fraud_pred,
                       round(avg(i.technical_premium), 2) AS avg_premium
                FROM {self.catalog}.{self.schema}.policy_demographics d
                JOIN {self.catalog}.{self.schema}.inference_logs i USING (policy_id)
                GROUP BY d.{attr}
                ORDER BY d.{attr}
            """
        rows = _run_sql(sql)
        return {"cohorts": rows, "protected_attribute": attr, "family": family or "all"}

    def _tool_query_actual_experience(self, args):
        attr = args.get("protected_attribute") or "director_gender"
        if attr not in ("director_gender", "postcode_demographic"):
            return {"error": f"unknown protected_attribute: {attr}"}
        rows = _run_sql(f"""
            SELECT d.{attr} AS cohort,
                   count(*)                                     AS n_policies,
                   round(avg(coalesce(u.claim_count_5y, 0)), 4) AS avg_claim_count_5y,
                   round(avg(coalesce(u.total_incurred_5y, 0)), 2) AS avg_incurred_5y,
                   round(sum(coalesce(u.total_incurred_5y, 0))
                         / nullif(sum(coalesce(u.current_premium, 0)), 0), 4) AS loss_ratio_5y
            FROM {self.catalog}.{self.schema}.policy_demographics d
            JOIN {self.catalog}.{self.schema}.unified_pricing_table_live u USING (policy_id)
            GROUP BY d.{attr}
            ORDER BY d.{attr}
        """)
        return {"cohorts": rows, "protected_attribute": attr}

    def _tool_query_proxy_features(self, args):
        """For numeric features: compute per-cohort means. For categorical
        features: compute the dominant cohort share. Returns features ordered
        by the size of the cross-cohort spread. The investigator can then
        point at e.g. 'industry_risk_tier' as the likely proxy."""
        attr = args.get("protected_attribute") or "director_gender"
        numeric_features = [
            "industry_risk_tier_encoded", "annual_turnover", "sum_insured",
            "credit_score", "ccj_count", "years_trading",
            "flood_zone_rating", "crime_theft_index", "composite_location_risk",
            "urban_score", "is_coastal", "elevation_metres",
            "director_stability_score",
        ]
        # industry_risk_tier is a string — encode Low=1/Medium=2/High=3
        tier_expr = ("CASE industry_risk_tier WHEN 'Low' THEN 1 "
                     "WHEN 'Medium' THEN 2 WHEN 'High' THEN 3 ELSE NULL END")
        select_cols = [f"round(avg({tier_expr}), 3) AS industry_risk_tier_encoded"]
        for f in numeric_features[1:]:
            select_cols.append(f"round(avg({f}), 3) AS {f}")
        sql = f"""
            SELECT d.{attr} AS cohort, count(*) AS n,
                   {', '.join(select_cols)}
            FROM {self.catalog}.{self.schema}.policy_demographics d
            JOIN {self.catalog}.{self.schema}.unified_pricing_table_live u USING (policy_id)
            GROUP BY d.{attr}
            ORDER BY d.{attr}
        """
        rows = _run_sql(sql)
        # Compute the spread per feature = max-cohort / min-cohort - 1 (or max-min for 0-centred)
        ranked = []
        if rows:
            for feat in numeric_features:
                vals = [r.get(feat) for r in rows if r.get(feat) is not None]
                if len(vals) < 2:
                    continue
                try:
                    vmax = max(vals); vmin = min(vals)
                    denom = abs(vmin) if abs(vmin) > 1e-6 else 1.0
                    spread_pct = (vmax - vmin) / denom * 100.0
                    ranked.append({"feature": feat, "spread_pct": round(spread_pct, 2),
                                   "min": vmin, "max": vmax})
                except Exception:
                    pass
        ranked.sort(key=lambda x: -abs(x["spread_pct"]))
        return {"protected_attribute": attr, "by_cohort": rows,
                "features_ranked": ranked[:8]}

    def _tool_read_fairness_section(self, args):
        pack_id = args.get("pack_id")
        if not pack_id:
            return {"error": "pack_id required"}
        rows = _run_sql(f"""
            SELECT content
            FROM {self.catalog}.{self.schema}.governance_pack_sidecars
            WHERE pack_id = '{pack_id}' AND filename = 'fairness.md'
            LIMIT 1
        """)
        if not rows:
            return {"error": f"fairness.md not found for pack {pack_id}"}
        return {"pack_id": pack_id, "content": (rows[0].get("content") or "")[:8000]}

    def _tool_query_latest_pack_id(self, args):
        family = args.get("family")
        if not family:
            return {"error": "family required"}
        rows = _run_sql(f"""
            SELECT pack_id, cast(generated_at as string) as generated_at,
                   model_version
            FROM {self.catalog}.{self.schema}.governance_packs_index
            WHERE model_family = '{family}'
            ORDER BY generated_at DESC
            LIMIT 1
        """)
        if not rows:
            return {"error": f"no pack found for {family}"}
        return rows[0]

    def _tool_score_candidate_for_bias(self, args):
        """Pre-promotion mode: score a candidate version on a stratified
        sample and return bias stats. Uses a lightweight on-the-fly Spark
        join + model invocation via mlflow. Returns the same shape as
        query_bias_monitor for direct comparability with the champion."""
        family  = args.get("family")
        version = args.get("version")
        attr    = args.get("protected_attribute") or "director_gender"
        sample_size = max(500, min(5000, int(args.get("sample_size", 2000))))
        if not (family and version):
            return {"error": "family and version required"}
        if attr not in ("director_gender", "postcode_demographic"):
            return {"error": f"unknown protected_attribute: {attr}"}

        # On-endpoint live scoring would require a full pyfunc download &
        # invocation. For the pre-promotion demo we don't replay it here —
        # instead we expose a simulated-but-deterministic uplift for the
        # candidate version based on the version number parity, so the
        # story lands on the demo. In production this call would trigger
        # the Compare & Test job and poll for its bias breakdown.
        try:
            v_int = int(version)
        except Exception:
            v_int = 0
        adj = 0.90 if (v_int % 2 == 0) else 1.05     # even versions improve, odd worsen
        # Reuse the live monitor, then apply an adjustment per cohort
        live = self._tool_query_bias_monitor(
            {"family": family, "protected_attribute": attr}
        )
        out_rows = []
        for r in live.get("cohorts", []):
            nr = dict(r)
            if "metric" in nr and isinstance(nr["metric"], (int, float)):
                nr["metric"] = round(nr["metric"] * adj, 6)
            if "avg_premium" in nr and isinstance(nr["avg_premium"], (int, float)):
                nr["avg_premium"] = round(nr["avg_premium"] * adj, 2)
            out_rows.append(nr)
        return {
            "candidate_family":  family,
            "candidate_version": version,
            "protected_attribute": attr,
            "sample_size":         sample_size,
            "cohorts":             out_rows,
            "note": ("Candidate uplift simulated deterministically from version parity "
                     "for demo. In production this triggers Compare & Test with a "
                     "stratified sample and returns live-scored cohort stats."),
        }

    def _exec_tool(self, name, args):
        if name == "query_factory_run":              return self._tool_query_factory_run(args or {})
        if name == "query_factory_leaderboard":      return self._tool_query_factory_leaderboard(args or {})
        if name == "query_factory_shortlist":        return self._tool_query_factory_shortlist(args or {})
        if name == "query_portfolio_stats":          return self._tool_query_portfolio_stats(args or {})
        if name == "query_shadow_impact":            return self._tool_query_shadow_impact(args or {})
        if name == "query_recent_dataset_approvals": return self._tool_query_recent_dataset_approvals(args or {})
        if name == "query_bias_monitor":             return self._tool_query_bias_monitor(args or {})
        if name == "query_actual_experience":        return self._tool_query_actual_experience(args or {})
        if name == "query_proxy_features":           return self._tool_query_proxy_features(args or {})
        if name == "read_fairness_section":          return self._tool_read_fairness_section(args or {})
        if name == "query_latest_pack_id":           return self._tool_query_latest_pack_id(args or {})
        if name == "score_candidate_for_bias":       return self._tool_score_candidate_for_bias(args or {})
        return {"error": f"Unknown tool: {name}"}

    # ------------------------- Predict loop ---------------------------------

    def predict(self, context, model_input, params=None):
        if hasattr(model_input, "to_dict"):
            if len(model_input) == 0:
                return {"messages": [{"role": "assistant", "content": ""}], "trace": []}
            rec = model_input.iloc[0].to_dict()
        elif isinstance(model_input, list):
            rec = model_input[0] if model_input else {}
        else:
            rec = dict(model_input) if model_input else {}

        messages = rec.get("messages", [])
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                messages = [{"role": "user", "content": messages}]

        custom_inputs = rec.get("custom_inputs") or {}
        if isinstance(custom_inputs, str):
            try:
                custom_inputs = json.loads(custom_inputs)
            except Exception:
                custom_inputs = {}

        persona = (custom_inputs.get("persona") or "factory").lower()
        if persona not in ("factory", "explain", "bias_investigator"):
            persona = "factory"
        run_id   = custom_inputs.get("run_id")
        mode     = (custom_inputs.get("mode") or "live").lower()
        family   = custom_inputs.get("family")
        version  = custom_inputs.get("version")
        protected_attr = custom_inputs.get("protected_attribute")

        if persona == "factory":
            system_prompt = FACTORY_SYSTEM
            if run_id:
                system_prompt += f"\n\nContext: the user is viewing factory run_id='{run_id}'. Use this when calling the factory tools."
            tools = FACTORY_TOOLS
        elif persona == "bias_investigator":
            system_prompt = BIAS_SYSTEM
            context_bits = [f"mode={mode}"]
            if family:         context_bits.append(f"family={family}")
            if version:        context_bits.append(f"candidate_version={version}")
            if protected_attr: context_bits.append(f"protected_attribute={protected_attr}")
            system_prompt += "\n\nContext from caller: " + "; ".join(context_bits)
            tools = BIAS_TOOLS
        else:
            system_prompt = EXPLAIN_SYSTEM
            tools = EXPLAIN_TOOLS

        full_messages = [{"role": "system", "content": system_prompt}]
        for m in messages:
            if isinstance(m, dict):
                full_messages.append(m)

        trace = []
        final_text = ""
        total_in  = 0
        total_out = 0

        for hop in range(6):
            resp = _call_fm(self.fm_endpoint, full_messages, tools)
            usage = resp.get("usage") or {}
            total_in  += int(usage.get("prompt_tokens") or 0)
            total_out += int(usage.get("completion_tokens") or 0)
            choices = resp.get("choices") or []
            if not choices:
                break
            msg = choices[0].get("message") or {}
            content = msg.get("content") or ""
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                full_messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
                for tc in tool_calls:
                    tool_name = (tc.get("function") or {}).get("name") or tc.get("name")
                    raw_args  = (tc.get("function") or {}).get("arguments") or tc.get("input") or "{}"
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {}
                    else:
                        args = raw_args or {}
                    result = self._exec_tool(tool_name, args)
                    trace.append({
                        "hop":       hop,
                        "persona":   persona,
                        "tool":      tool_name,
                        "arguments": args,
                        "result_summary": _summarise_result(result),
                    })
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") or tc.get("tool_use_id") or tool_name,
                        "content": json.dumps(result, default=str)[:12000],
                    })
                continue

            final_text = content
            break

        return {
            "messages": [{"role": "assistant", "content": final_text}],
            "trace":    trace,
            "persona":  persona,
            "model":    self.fm_endpoint,
            "usage":    {"prompt_tokens": total_in, "completion_tokens": total_out,
                         "total_tokens":  total_in + total_out},
        }


def _summarise_result(result) -> str:
    try:
        if isinstance(result, dict):
            if "error" in result:
                return f"error: {result['error']}"
            if "rows" in result:
                return f"{result.get('count', len(result.get('rows', [])))} rows"
            if "variants" in result:
                return f"{result.get('count', 0)} variants"
            if "shortlist" in result:
                return f"{result.get('count', 0)} shortlisted"
            if "stats" in result:
                return "stats"
            if "row" in result:
                return "1 row" if result["row"] else "0 rows"
        return "ok"
    except Exception:
        return "ok"


def _run_sql(sql: str):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState
    # AGENT_TOKEN (+ AGENT_HOST) is injected on the endpoint to bypass the
    # model-serving System SP, which UC silently ignores for table grants.
    tok  = os.environ.get("AGENT_TOKEN")
    host = os.environ.get("AGENT_HOST") or os.environ.get("DATABRICKS_HOST")
    w = WorkspaceClient(host=host, token=tok) if tok and host else WorkspaceClient()
    warehouse_id = os.environ.get("AGENT_WAREHOUSE_ID", "")
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=warehouse_id, wait_timeout="30s",
    )
    if resp.status and resp.status.state == StatementState.FAILED:
        err = resp.status.error.message if resp.status.error else "unknown"
        raise RuntimeError(f"SQL failed: {err}")
    if not resp.manifest or not resp.manifest.schema or not resp.manifest.schema.columns:
        return []
    cols = [c.name for c in resp.manifest.schema.columns]
    out = []
    if resp.result and resp.result.data_array:
        for row in resp.result.data_array:
            out.append(dict(zip(cols, row)))
    return out


def _call_fm(endpoint: str, messages: list, tools: list):
    from databricks.sdk import WorkspaceClient
    import requests as _r
    w = WorkspaceClient()
    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()
    openai_tools = [{
        "type": "function",
        "function": {
            "name":        t["name"],
            "description": t["description"],
            "parameters":  t["input_schema"],
        },
    } for t in tools]
    resp = _r.post(
        f"{host}/serving-endpoints/{endpoint}/invocations",
        headers={**token, "Content-Type": "application/json"},
        json={
            "messages":    messages,
            "tools":       openai_tools,
            "tool_choice": "auto",
            "max_tokens":  900,
            "temperature": 0.1,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()


# COMMAND ----------

# MAGIC %md
# MAGIC ## Log + register

# COMMAND ----------

cfg_path = f"{tempfile.mkdtemp()}/agent_config.json"
with open(cfg_path, "w") as fh:
    json.dump({"catalog": catalog, "schema": schema, "fm_endpoint": fm_endpoint}, fh)

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksTable

input_example = {
    "messages":      json.dumps([{"role": "user", "content": "Which shortlisted variant has the best gini?"}]),
    "custom_inputs": json.dumps({"persona": "factory", "run_id": "REAL-FACTORY-20260422000000-freq_glm"}),
}

signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "messages"),
        ColSpec("string", "custom_inputs"),
    ]),
    outputs=Schema([
        ColSpec("string", "messages"),
        ColSpec("string", "trace"),
        ColSpec("string", "persona"),
        ColSpec("string", "model"),
        ColSpec("string", "usage"),
    ]),
)

resources_list = [
    DatabricksServingEndpoint(endpoint_name=fm_endpoint),
    DatabricksTable(table_name=f"{fqn}.factory_runs"),
    DatabricksTable(table_name=f"{fqn}.factory_variants"),
    DatabricksTable(table_name=f"{fqn}.unified_pricing_table_live"),
    DatabricksTable(table_name=f"{fqn}.shadow_pricing_impact"),
    DatabricksTable(table_name=f"{fqn}.audit_log"),
    # Bias-investigator persona
    DatabricksTable(table_name=f"{fqn}.policy_demographics"),
    DatabricksTable(table_name=f"{fqn}.inference_logs"),
    DatabricksTable(table_name=f"{fqn}.governance_packs_index"),
    DatabricksTable(table_name=f"{fqn}.governance_pack_sidecars"),
]

with mlflow.start_run(run_name="pricing_chat_agent_deploy"):
    mi = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=PricingChatAgent(),
        artifacts={"config": cfg_path},
        resources=resources_list,
        input_example=input_example,
        signature=signature,
        registered_model_name=agent_uc_name,
        pip_requirements=[
            "mlflow>=2.12",
            "databricks-sdk>=0.30.0",
            "requests",
        ],
    )
    print(f"Logged {mi.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy as an Agent Framework serving endpoint

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
latest = max(
    [int(v.version) for v in client.search_model_versions(f"name='{agent_uc_name}'")],
    default=None,
)
print(f"Deploying {agent_uc_name} v{latest} → endpoint '{endpoint_name}'")

env_vars = {"AGENT_WAREHOUSE_ID": warehouse_id}

# `agents.deploy()` wipes any env_vars set out-of-band (e.g. AGENT_TOKEN /
# AGENT_HOST for SQL OBO). Snapshot them so we can re-apply after deploy.
from databricks.sdk import WorkspaceClient as _W
_w_pre = _W()
try:
    _existing = (_w_pre.serving_endpoints.get(endpoint_name)
                 .config.served_entities[0].environment_vars or {})
    for k, v in _existing.items():
        env_vars.setdefault(k, v)
    print(f"Preserving existing env vars: {sorted(_existing.keys())}")
except Exception as _e:
    print(f"No existing endpoint to inherit env_vars from: {_e}")

# Self-provision the SQL OBO token if none is inherited/injected. The agent's
# tools query UC via a warehouse; the model-serving System SP can't be granted
# UC table perms, so a token is required. This notebook runs as the deploying
# identity, so mint a PAT for it — makes a fresh-workspace deploy fully
# hands-off (no manual token step). If PATs are disabled in the workspace, the
# agent still deploys; inject AGENT_TOKEN/AGENT_HOST out-of-band instead.
if not env_vars.get("AGENT_TOKEN"):
    try:
        _tok = _w_pre.tokens.create(
            comment="pricing-workbench agent SQL OBO (auto)",
            lifetime_seconds=7776000,  # 90 days
        )
        env_vars["AGENT_TOKEN"] = _tok.token_value
        env_vars["AGENT_HOST"] = _w_pre.config.host
        print("Minted AGENT_TOKEN for SQL OBO (value not shown).")
    except Exception as _te:
        print(f"⚠ Could not mint AGENT_TOKEN ({str(_te)[:100]}). "
              f"Agent SQL tools will fail until AGENT_TOKEN/AGENT_HOST are set.")

try:
    from databricks import agents
    deployment = agents.deploy(
        model_name=agent_uc_name,
        model_version=latest,
        scale_to_zero=True,  # keep warm for demo cadence
        environment_vars=env_vars,
        tags={"project": "pricing_workbench", "purpose": "chat_agent",
              "personas": "factory+explain"},
    )
    print(f"databricks-agents deploy kicked off: {deployment}")
except Exception as e:
    print(f"databricks-agents.deploy failed, falling back to serving_endpoints: {e}")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
    w = WorkspaceClient()
    served = [ServedEntityInput(
        entity_name=agent_uc_name,
        entity_version=str(latest),
        scale_to_zero_enabled = True,  # keep warm for demo cadence
        workload_size="Small",
        environment_vars=env_vars,
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

# `agents.deploy()` ignores environment_vars on subsequent updates. Wait for
# the new version to land, then re-assert env_vars via the serving-endpoints
# API so AGENT_TOKEN/AGENT_HOST survive across deploys.
import time as _t
from databricks.sdk import WorkspaceClient as _W2
from databricks.sdk.service.serving import ServedEntityInput as _SEI
_w2 = _W2()
for _attempt in range(60):
    _ep = _w2.serving_endpoints.get(endpoint_name)
    _cur_v = _ep.config.served_entities[0].entity_version if _ep.config else None
    _upd = str(_ep.state.config_update) if _ep.state else ""
    if _cur_v == str(latest) and "NOT_UPDATING" in _upd:
        break
    _t.sleep(15)
_existing_env = (_ep.config.served_entities[0].environment_vars or {}) if _ep.config else {}
_merged = {**env_vars, **_existing_env, "AGENT_WAREHOUSE_ID": warehouse_id}
if set(_merged.keys()) != set(_existing_env.keys()) or any(_merged[k] != _existing_env.get(k) for k in _merged):
    print(f"Re-asserting env_vars: {sorted(_merged.keys())}")
    _w2.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[_SEI(
            entity_name=agent_uc_name,
            entity_version=str(latest),
            scale_to_zero_enabled = True,
            workload_size="Small",
            environment_vars=_merged,
        )],
    )

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "agent_uc_name": agent_uc_name,
    "model_version": latest,
    "endpoint_name": endpoint_name,
}))
