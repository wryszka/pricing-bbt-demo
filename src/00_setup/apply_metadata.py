# Databricks notebook source
# MAGIC %md
# MAGIC # Apply Metadata — descriptions on everything
# MAGIC
# MAGIC One-shot, idempotent pass over the Pricing Workbench schema that sets:
# MAGIC
# MAGIC - **Schema** comment
# MAGIC - **Volume** comments (`external_landing`, `saved_payloads`)
# MAGIC - **Table** comments for every table the app / pipeline references
# MAGIC - **Column** comments for the high-traffic tables (UPT, quotes, feature_catalog, mf_leaderboard)
# MAGIC - **Registered model** comments (UC) for the pricing + impact-study models
# MAGIC
# MAGIC Safe to re-run — every statement is guarded. Missing tables are skipped
# MAGIC with a note, never break the run.
# MAGIC
# MAGIC Wired into `setup_demo` (runs at the end) and available on its own as the
# MAGIC `apply_metadata` bundle job.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")

catalog = dbutils.widgets.get("catalog_name")
schema  = dbutils.widgets.get("schema_name")
fqn     = f"{catalog}.{schema}"

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def _esc(s: str) -> str:
    return (s or "").replace("'", "''")

def _table_exists(tbl: str) -> bool:
    try:
        spark.table(f"{fqn}.{tbl}").schema  # cheap — raises if missing
        return True
    except Exception:
        return False

def _set_table_comment(tbl: str, comment: str) -> None:
    if not _table_exists(tbl):
        print(f"  [skip] {tbl} does not exist")
        return
    try:
        spark.sql(f"COMMENT ON TABLE {fqn}.{tbl} IS '{_esc(comment)}'")
        print(f"  ✓ {tbl}")
    except Exception as e:
        print(f"  [err] {tbl}: {str(e)[:120]}")

def _set_column_comments(tbl: str, col_comments: dict) -> None:
    if not _table_exists(tbl):
        return
    existing_cols = {c.name for c in spark.table(f"{fqn}.{tbl}").schema.fields}
    for col, comment in col_comments.items():
        if col not in existing_cols:
            continue
        try:
            spark.sql(f"ALTER TABLE {fqn}.{tbl} ALTER COLUMN {col} COMMENT '{_esc(comment)}'")
        except Exception as e:
            print(f"  [err] {tbl}.{col}: {str(e)[:80]}")

def _set_volume_comment(vol: str, comment: str) -> None:
    try:
        spark.sql(f"COMMENT ON VOLUME {fqn}.{vol} IS '{_esc(comment)}'")
        print(f"  ✓ volume {vol}")
    except Exception as e:
        print(f"  [err] volume {vol}: {str(e)[:120]}")

def _set_model_comment(model: str, comment: str) -> None:
    full_name = f"{fqn}.{model}"
    try:
        w.registered_models.update(full_name=full_name, comment=comment)
        print(f"  ✓ model {model}")
    except Exception as e:
        print(f"  [skip] model {model}: {str(e)[:100]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Schema

# COMMAND ----------

SCHEMA_COMMENT = (
    "Pricing Workbench demo — end-to-end commercial P&C pricing on Databricks. "
    "The Pricing Feature Table (unified_pricing_table_live) is built from every "
    "approved source in this schema: the 3 ingested vendor feeds (silver_*), "
    "2 internal systems of record (policies + claims), the real UK postcode "
    "enrichment (ONSPD + IMD 2019 + ONS RUC), plus derived factors. Also contains "
    "the quote stream, Model Factory artefacts, and the New Data Impact notebook "
    "study tables. Synthetic data throughout — Bricksurance SE is fictional."
)
try:
    spark.sql(f"COMMENT ON SCHEMA {fqn} IS '{_esc(SCHEMA_COMMENT)}'")
    print(f"✓ schema {fqn}")
except Exception as e:
    print(f"[err] schema: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Volumes

# COMMAND ----------

VOLUMES = {
    "external_landing":
        "CSV landing zone for synthetic external vendor data (market pricing benchmark, "
        "geospatial hazard, credit bureau, economic indicators). Populated by setup.py, "
        "ingested by the DLT Silver pipeline.",
    "saved_payloads":
        "Quote Review operator exports — when an investigator clicks 'Save to UC volume' "
        "on a JSON payload, the file lands here with a timestamped filename.",
    "raw_data":
        "Holds the downloaded ONSPD + IMD 2019 CSV/ZIP files for the New Data Impact "
        "postcode enrichment build. Idempotent — 00a skips download if files already exist.",
    "reports":
        "Governance PDF reports exported from the app (per-model validation reports, "
        "factory-run logs, regulatory submissions).",
}
for v, c in VOLUMES.items():
    _set_volume_comment(v, c)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Table comments — all tables the app or pipeline touches

# COMMAND ----------

TABLES = {
    # ── Internal core ────────────────────────────────────────────────────
    "internal_commercial_policies":
        "Bricksurance SE's commercial property policy book — 50,000 rows at scale=1. "
        "One row per policy with SIC, postcode sector, sums insured, and current premium. "
        "The labelled population the pricing models train against (joined with claims).",
    "internal_claims_history":
        "Claim ledger — one row per claim, keyed on claim_id and linked to policy_id. "
        "Per-peril breakdowns (fire, flood, theft, etc.) become aggregated features in the UPT.",
    "internal_quote_history":
        "(Deprecated — replaced by the unified `quotes` table). Legacy flat quote history "
        "from the earlier demo version.",
    "audit_log":
        "Immutable audit trail for all Pricing Workbench events — dataset approvals, model "
        "decisions, AI agent interactions, factory run submissions. One row per event with a "
        "JSON details blob.",
    "dataset_approvals":
        "Per-dataset-version approval decisions captured via the Data Ingestion tab of the "
        "app. References the dataset_name and the actuary who decided.",

    # ── External vendor raw (bronze) ────────────────────────────────────
    "raw_market_pricing_benchmark":
        "Raw ingested competitor pricing benchmark CSV (SIC × region aggregates). Loaded "
        "from external_landing volume by 01_ingestion/ingest_market_pricing.py.",
    "raw_geospatial_hazard_enrichment":
        "Raw ingested location risk scores per postcode sector (flood, fire distance, crime "
        "theft index, subsidence). Loaded from external_landing volume.",
    "raw_credit_bureau_summary":
        "Raw ingested company credit + financial health CSV (credit score, CCJs, years "
        "trading, director changes). Loaded from external_landing volume.",

    # ── External silver (cleansed via DLT expectations) ─────────────────
    "silver_market_pricing_benchmark":
        "Silver (DLT-cleansed) market pricing benchmark. Rows failing DQ expectations are "
        "dropped; a composite match_key_sic_region is added for joining to policies.",
    "silver_geospatial_hazard_enrichment":
        "Silver location risk with composite_location_risk score and High/Medium/Low tier.",
    "silver_credit_bureau_summary":
        "Silver credit bureau with credit_risk_tier (Prime/Standard/Sub-Standard/High Risk) "
        "and business_stability_score (0–100 composite).",

    # ── Real UK public-data enrichment ──────────────────────────────────
    "postcode_enrichment":
        "~1.5M real English postcodes built from the ONS Postcode Directory (ONSPD) joined "
        "to the English Indices of Deprivation 2019 and the ONS Rural-Urban Classification. "
        "Columns include lat/long, LSOA code, IMD deciles (overall + crime/income/health/"
        "living environment), is_urban, is_coastal. Built by new_data_impact/00a.",

    # ── Gold / Training feature store ───────────────────────────────────
    "unified_pricing_table_live":
        "PRICING FEATURE TABLE — engineered artifact built from every approved source by the "
        "build_upt pipeline. Policies is just one of 8 contributing sources alongside claims, "
        "market pricing benchmark, geospatial hazard, credit bureau, real UK postcode "
        "enrichment, derived factors, and quotes. policy_id is the grain (one feature vector "
        "per policy), not the identity. 50K rows x ~100 columns, PK policy_id (registered UC "
        "feature table). Consumed by Frequency/Severity GLMs, GBMs, and the challenger "
        "comparison. Can be promoted to the online store (Lakebase) for sub-10ms serving.",
    "derived_factors":
        "Postcode-sector-level derived factors sourced from real UK public data: "
        "urban_score, is_coastal, deprivation_composite, plus the claims-driven "
        "neighbourhood_claim_frequency (Bühlmann credibility-weighted, K=100). "
        "Left-joined into the UPT by build_upt.py.",
    "feature_catalog":
        "ONE ROW PER FEATURE in the training feature store. Captures source tables, "
        "source columns, transformation formula, owner, regulatory_sensitive + PII flags. "
        "Drives the Feature Catalog panel in the app and the foundation for feature-lineage / "
        "audit bolt-ons.",

    # ── Quote stream ────────────────────────────────────────────────────
    "quotes":
        "Canonical flat quote table — one row per transaction. ~120K rows; BOUND quotes "
        "carry a real policy_id linking back to internal_commercial_policies. Training data "
        "for the Demand GBM and the serving-time feature shape. Supersedes the old "
        "internal_quote_history.",
    "quote_payload_sales":
        "Captured JSON of the sales-channel request for the subset of transactions where "
        "has_payload=true (~1,000 rows). Keyed on transaction_id.",
    "quote_payload_engine_request":
        "Captured JSON of the request sent to the rating engine for the captured subset. "
        "Keyed on transaction_id.",
    "quote_payload_engine_response":
        "Captured JSON of the rating engine's response (pricing, loadings, discounts). "
        "Absent for ABANDONED transactions.",

    # ── Model Factory ────────────────────────────────────────────────────
    "mf_training_plan":
        "Planned training configurations — one row per (factory_run_id, model_config_id). "
        "Populated either by the manual MF pipeline (mf_01) or by the Agentic Planner in the "
        "app. Consumed by mf_02 to drive training.",
    "mf_training_log":
        "Per-config training outcomes — status (SUCCESS/FAILED), metrics, MLflow run_id, "
        "training duration. Written by mf_02_automated_training.",
    "mf_leaderboard":
        "Ranked evaluation across every model in a factory run. AIC/BIC/Gini/Deviance + "
        "regulatory suitability + recommended_action (RECOMMEND_APPROVE / REVIEW / REJECT). "
        "Powers the Model Factory leaderboard and the governance PDFs.",
    "mf_actuary_decisions":
        "Actuary approve/reject decisions for each model in a factory run, with reviewer, "
        "reviewer notes, timestamp, and regulatory sign-off flag.",
    "mf_audit_log":
        "Factory-scoped audit events — events tied to a specific factory_run_id. Complements "
        "the workspace-wide audit_log with factory-specific detail.",
    "mf_feature_profile":
        "Feature statistics (null rates, cardinality, basic distribution) profiled at the "
        "start of a factory run. Used by the AI agent to reason about feature availability.",
    "mf_run_log":
        "Per-run log for the Agentic Planner — intent (target, family, scope), Claude "
        "feature-analysis narrative, proposal summary, configs JSON, status + summary "
        "metrics. Consumed by the full run-log PDF export.",

    # ── Challenger comparison ───────────────────────────────────────────
    "challenger_comparison_latest":
        "Latest baseline-vs-challenger Gini comparison — one row per cohort along an "
        "ablation ladder (baseline, +urban_score, +is_coastal, +deprivation_composite, "
        "+neighbourhood_claim_frequency). Consumed by the Challenger panel on Model Development.",

    # ── New Data Impact study tables (notebook-driven) ──────────────────
    "impact_portfolio":
        "New Data Impact study: 200K synthetic home insurance portfolio. Every policy "
        "assigned a real English postcode from postcode_enrichment. Source of impact_train_set / "
        "impact_test_set.",
    "impact_train_set":
        "70% train split of impact_portfolio, one-hot encoded. Used to fit the standard + "
        "enriched frequency GLMs and severity GBMs.",
    "impact_test_set":
        "30% test split of impact_portfolio, one-hot encoded. Held-out evaluation population.",
    "impact_severity_train_set":
        "Claimants-only slice of impact_train_set. Input to the severity GBM training "
        "(Gamma distribution).",
    "impact_severity_test_set":
        "Claimants-only slice of impact_test_set for severity evaluation.",
    "impact_model_comparison":
        "Side-by-side AIC/BIC/Gini/Deviance Explained for the standard vs enriched "
        "frequency GLMs. The 'scoreboard' the executive narrative shows.",
    "impact_glm_coefficients":
        "GLM coefficients + std error + z-score + p-value + 95% CI for both frequency "
        "models. Powers the coefficient forest plot.",
    "impact_loss_ratio_by_decile":
        "Loss-ratio stability across premium deciles for each frequency model. Tighter = "
        "fairer pricing = less adverse selection.",
    "impact_priced_portfolio":
        "Test set scored by both frequency models (frequency-only quotes).",
    "impact_severity_model_comparison":
        "Severity metrics — MAE, RMSE, MAPE, Gini, bias — for the standard vs enriched "
        "severity GBMs.",
    "impact_severity_feature_importance":
        "LightGBM gain-based feature importance for both severity models.",
    "impact_severity_priced_portfolio":
        "Test set with full burning-cost quotes (frequency × severity) for each model.",
    "impact_severity_loss_ratio_by_decile":
        "Loss-ratio stability on the full burning-cost quotes, by premium decile.",
    "impact_model_factory_results":
        "All 50 GLM-spec factory outcomes ranked by AIC, BIC, Gini. Reproduces what an "
        "actuary would do manually in WTW Radar, at scale.",
    "impact_model_factory_feature_impact":
        "Average AIC improvement per enrichment feature across the 50-spec factory — "
        "the 'which feature is most valuable' view.",
    "impact_model_governance_summary":
        "Consolidated governance facts — model inventory, rationale, feature justification, "
        "performance evidence. Powers notebook 04's governance PDF.",

    # ── Endpoint latency test results ───────────────────────────────────
    "endpoint_latency":
        "Measured latency metrics for the deployed pricing serving endpoint — single "
        "lookup avg/P50/P99 + batch 100 keys. Populated by 07_serving/test_endpoint.py.",
    "online_store_latency":
        "Measured latency for online feature store lookups — used by the Feature Store "
        "page to surface real-world-ish serving performance numbers.",
}

print(f"Applying comments to {len(TABLES)} tables:")
for tbl, comment in TABLES.items():
    _set_table_comment(tbl, comment)

# Monthly UPT snapshots — discover + comment dynamically
try:
    snapshots = spark.sql(f"SHOW TABLES IN {fqn} LIKE 'unified_pricing_table_20*'").collect()
    for row in snapshots:
        snap_name = row["tableName"]
        _set_table_comment(
            snap_name,
            f"Monthly DEEP CLONE snapshot of unified_pricing_table_live. Used by UC2 "
            f"point-in-time backtesting for reproducing model training against a frozen "
            f"feature set.",
        )
except Exception as e:
    print(f"  [skip] snapshots: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Column comments — high-traffic tables

# COMMAND ----------

# UPT — most comprehensive. build_upt.py already sets many; here we fill gaps
# + the new real-UK enrichment columns.
_set_column_comments("unified_pricing_table_live", {
    "policy_id":                    "Unique policy identifier — primary key",
    "sic_code":                     "Standard Industrial Classification code (4-digit)",
    "postcode_sector":              "UK postcode sector for the insured premises (e.g. EC1A)",
    "annual_turnover":              "Declared gross revenue of the business (GBP)",
    "sum_insured":                  "Total sum insured under the policy (GBP)",
    "current_premium":              "Current annual premium charged (GBP)",
    "construction_type":            "ISO construction class (Fire Resistive, Non-Combustible, etc.)",
    "year_built":                   "Original construction year",
    "building_age_years":           "Age of the primary building in years (2026 - year_built)",
    "claim_count_5y":               "Total number of claims in the last 5 years",
    "total_incurred_5y":            "Total incurred claim amount over 5 years (GBP)",
    "loss_ratio_5y":                "5-year loss ratio: total_incurred_5y / (current_premium * 5)",
    "market_median_rate":           "Market median premium rate per £1k sum insured",
    "flood_zone_rating":            "Flood risk score (1=low, 10=high) from vendor",
    "credit_score":                 "Company credit score (200-900) from credit bureau",
    "urban_score":                  "Real-UK composite: 0.60 * frac_urban + 0.40 * (10 - living_env_decile)/9",
    "is_coastal":                   "Real-UK coastal flag derived from ONS local authority codes",
    "deprivation_composite":        "Real-UK IMD composite (crime + income + health + living-env, 0–1, 1=most deprived)",
    "imd_decile":                   "Real-UK IMD 2019 decile averaged to postcode area (1=most deprived)",
    "crime_decile":                 "Real-UK IMD 2019 crime sub-decile averaged to postcode area",
    "neighbourhood_claim_frequency":"Bühlmann credibility-weighted postcode claim frequency (K=100)",
    "combined_risk_score":          "Blended risk score across location, credit, industry, claims",
    "upt_build_timestamp":          "When this UPT row was built",
})

# quotes
_set_column_comments("quotes", {
    "transaction_id":   "Unique quote identifier — primary key",
    "policy_id":        "Linked policy_id — populated only for BOUND quotes",
    "created_at":       "Timestamp of the quote request from the sales channel",
    "channel":          "Sales channel (Direct / Broker / Aggregator / Renewal)",
    "company_name":     "Business name declared in the quote",
    "sic_code":         "Industry SIC code (4-digit)",
    "postcode_sector":  "Postcode outcode — joins to policy features",
    "gross_premium":    "Quoted gross premium including IPT (GBP)",
    "quote_status":     "BOUND / QUOTED / ABANDONED",
    "converted":        "Y/N flag derived from quote_status — Y iff BOUND",
    "competitor_quoted":"Y/N — a competitor is known to have quoted this prospect",
    "is_outlier":       "True for seeded anomalous quotes (e.g. £48M bakery) used in the Quote Review investigation demo",
    "has_payload":      "True if the 3 JSON payloads are captured for this transaction (subset used in Quote Review)",
    "model_version":    "Pricing model version that produced this quote",
})

# feature_catalog
_set_column_comments("feature_catalog", {
    "feature_name":         "Column name in the UPT — primary key of this catalog",
    "feature_group":        "Classification: rating_factor / enrichment / claim_derived / quote_derived / derived / audit / key",
    "data_type":            "Feature Spark data type",
    "description":          "Plain-English description for actuaries + regulators",
    "source_tables":        "Upstream source tables feeding this feature",
    "source_columns":       "Upstream source columns feeding this feature",
    "transformation":       "Plain-English transformation formula (visible to regulators)",
    "owner":                "Team or role accountable for this feature",
    "regulatory_sensitive": "Flag — regulators may scrutinise this feature (e.g. crime_decile, IMD-based)",
    "pii":                  "Flag — this feature contains or derives from personally identifiable information",
})

# mf_leaderboard
_set_column_comments("mf_leaderboard", {
    "factory_run_id":                "Links to the factory run (mf_run_log + mf_training_plan)",
    "model_config_id":               "Unique ID within the factory run",
    "target_column":                 "Target variable the model was trained against",
    "model_type":                    "GLM_Poisson / GLM_Gamma / GBM_Classifier / GBM_Regressor",
    "model_family":                  "Higher-level grouping (GLM / GBM / etc.)",
    "feature_count":                 "Number of features used in this model",
    "gini":                          "Normalised Gini coefficient on held-out data",
    "aic":                           "Akaike Information Criterion — lower is better (GLMs)",
    "bic":                           "Bayesian Information Criterion — lower is better (GLMs)",
    "rmse":                          "Root Mean Squared Error",
    "mae":                           "Mean Absolute Error",
    "regulatory_suitability_score":  "Internal 0–100 composite of transparency + stability + interpretability",
    "recommended_action":            "RECOMMEND_APPROVE / REVIEW / REJECT",
})

# mf_run_log
_set_column_comments("mf_run_log", {
    "factory_run_id":       "Run identifier — matches the key in mf_training_plan and mf_leaderboard",
    "proposal_source":      "Where the plan came from: 'agent' (Agentic Planner), 'manual', 'scheduled'",
    "intent_target":        "Target variable chosen by the user in the Agentic Planner",
    "intent_model_family":  "Model family chosen in the planner dropdowns",
    "intent_feature_scope": "Feature scope chosen: all / baseline_only / plus_real_uk / exclude_regulatory",
    "intent_sweep_size":    "Number of configs the user asked Claude to generate",
    "intent_focus":         "Focus: exploration / interaction_terms / hyperparam_sweep / feature_ablation",
    "user_note":            "Optional free-form note appended by the user to the agent",
    "feature_analysis":     "Snapshot of Claude's feature-catalog analysis at the time of the run",
    "plan_summary":         "Claude's 2-3 sentence summary of the proposed sweep strategy",
    "configs_json":         "Full JSON array of the proposed configs — preserved for audit",
    "status":               "PROPOSED / RUNNING / COMPLETED / FAILED",
    "summary_metrics":      "JSON with best_gini, median_gini, best_aic, n_configs_evaluated",
})

# challenger_comparison_latest
_set_column_comments("challenger_comparison_latest", {
    "cohort":              "Ablation cohort: baseline / plus_urban / plus_coastal / plus_deprivation / plus_claim_freq",
    "n_features":          "Feature count for this cohort",
    "gini":                "Normalised Gini on held-out test sample",
    "lift_vs_baseline":    "Gini delta between this cohort and the baseline cohort",
    "lift_vs_prev":        "Gini delta between this cohort and the previous cohort (per-factor lift)",
    "attribution_factor":  "The real-UK factor added at this step",
    "run_id":              "MLflow run ID for the cohort's trained model",
    "upt_delta_version":   "Delta version of unified_pricing_table_live at training time",
})

# postcode_enrichment
_set_column_comments("postcode_enrichment", {
    "postcode":             "Full UK postcode (e.g. SW1A 1AA)",
    "lsoa_code":            "Lower-layer Super Output Area code (ONSPD)",
    "region_code":          "ONS region code",
    "local_authority_code": "ONS local authority code — used for is_coastal derivation",
    "urban_rural_code":     "ONS RUC 2011 numeric classification",
    "urban_rural_band":     "Textual urban/rural band label",
    "is_urban":             "0/1 flag — 1 if the ONS RUC category is urban",
    "is_coastal":           "0/1 flag — 1 if the local_authority_code is a coastal English LA",
    "imd_decile":           "English IMD 2019 overall decile (1=most deprived, 10=least)",
    "imd_score":            "Raw IMD 2019 overall deprivation score",
    "crime_decile":         "IMD 2019 crime sub-decile",
    "income_decile":        "IMD 2019 income sub-decile",
    "health_decile":        "IMD 2019 health + disability sub-decile",
    "living_env_decile":    "IMD 2019 living environment sub-decile",
})

# internal_commercial_policies
_set_column_comments("internal_commercial_policies", {
    "policy_id":         "Primary key — unique policy identifier",
    "sic_code":          "Standard Industrial Classification (4-digit)",
    "postcode_sector":   "UK postcode sector of the insured premises",
    "annual_turnover":   "Declared gross revenue (GBP)",
    "sum_insured":       "Sum insured on the policy (GBP)",
    "current_premium":   "Current annual premium (GBP)",
    "construction_type": "ISO construction class",
    "year_built":        "Year the primary building was constructed",
    "renewal_date":      "Next renewal date",
})

# internal_claims_history
_set_column_comments("internal_claims_history", {
    "claim_id":        "Primary key — unique claim identifier",
    "policy_id":       "Foreign key to internal_commercial_policies",
    "peril":           "Claim peril — Fire / Flood / Theft / Liability / Storm / Subsidence / Escape of Water",
    "loss_date":       "Date the loss occurred",
    "incurred_amount": "Total incurred amount for the claim (paid + reserve), GBP",
    "paid_amount":     "Amount already paid on the claim, GBP",
    "reserve":         "Remaining reserve for the claim, GBP",
    "status":          "Open / Closed / Reopened",
})

# audit_log
_set_column_comments("audit_log", {
    "event_id":       "UUID for the event",
    "event_type":     "Category: dataset_approved / model_rejected / agent_action / manual_download / etc.",
    "entity_type":    "What the event is about: dataset / model / feature / endpoint / factory_run",
    "entity_id":      "Identifier of the affected entity",
    "entity_version": "Version stamp if relevant",
    "user_id":        "Who triggered the event",
    "timestamp":      "UTC timestamp of the event",
    "details":        "JSON blob with flexible event-specific metadata",
    "source":         "Where the event came from: app / notebook / api",
})

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Registered models — UC model comments

# COMMAND ----------

MODELS = {
    # Core pricing models (commercial track)
    "pricing_frequency_glm":  "Poisson GLM on claim_count_5y — primary frequency model. Transparent "
                              "coefficients with relativities suitable for regulatory submission.",
    "pricing_severity_glm":   "Gamma GLM on total_incurred_5y — primary severity model. "
                              "Technical price = Frequency * Severity.",
    "pricing_demand_gbm":     "LightGBM classifier on the quote conversion outcome — drives the "
                              "commercial overlay on the technical price.",
    "pricing_risk_uplift_gbm":"GBM on GLM residuals — captures non-linear interactions the GLM missed.",
    "pricing_fraud_model":    "Binary classifier for claims fraud detection (~3% prevalence).",
    "pricing_retention_model":"Binary classifier for policy non-renewal prediction.",

    # Impact study models (residential / new data impact track)
    "impact_glm_frequency_standard": "New Data Impact study — Poisson GLM on claim_count using standard "
                                     "rating factors only. Baseline for the lift study.",
    "impact_glm_frequency_enriched": "New Data Impact study — Poisson GLM on claim_count using standard "
                                     "rating factors + real UK enrichment (IMD, is_coastal, urban_score). "
                                     "Materially better Gini on the 200K portfolio.",
}
for model, comment in MODELS.items():
    _set_model_comment(model, comment)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Done

# COMMAND ----------

print(f"""
Metadata applied to {fqn}.
  Schema:  comment set
  Volumes: {len(VOLUMES)} commented
  Tables:  {len(TABLES)} comments applied (missing tables skipped)
  Models:  {len(MODELS)} UC models commented (missing models skipped)

Catalog Explorer at {fqn} will now show descriptions on every registered object.
""")
