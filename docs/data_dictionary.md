# Data Dictionary — Pricing UPT Accelerator

All tables reside in `{catalog}.{schema}` (default: `lr_serverless_aws_us_catalog.pricing_upt`).

## Internal Tables (Pre-existing)

These represent data the insurer already has on their Databricks platform.

### `internal_commercial_policies`

The insurer's book of business — one row per active policy.

| Column | Type | Description |
|--------|------|-------------|
| `policy_id` | STRING | Unique policy identifier (POL-NNNNNN) |
| `sic_code` | STRING | Standard Industrial Classification (4-digit) |
| `postcode_sector` | STRING | UK postcode sector of insured premises |
| `annual_turnover` | LONG | Declared gross revenue (GBP) |
| `construction_type` | STRING | ISO building class (Non-Combustible, Joisted Masonry, Fire Resistive, Frame, Heavy Timber) |
| `year_built` | INT | Original construction year (1920-2024) |
| `sum_insured` | LONG | Total coverage amount (GBP) |
| `claims_history_5y` | LONG | Total claims value in last 5 years (GBP, 0 = no claims) |
| `inception_date` | STRING | Policy start date (YYYY-MM-DD) |
| `renewal_date` | STRING | Next renewal date (YYYY-MM-DD) |
| `current_premium` | LONG | Current annual premium (GBP) |

**Rows:** 50,000 × SCALE_FACTOR

### `internal_claims_history`

Individual claim records linked to policies.

| Column | Type | Description |
|--------|------|-------------|
| `claim_id` | STRING | Unique claim identifier (CLM-NNNNNNN) |
| `policy_id` | STRING | FK to internal_commercial_policies |
| `peril` | STRING | Cause of loss (Fire, Flood, Theft, Liability, Storm, Subsidence, Escape of Water) |
| `incurred_amount` | LONG | Total incurred cost (GBP) |
| `paid_amount` | LONG | Amount paid to date (GBP) |
| `reserve` | LONG | Outstanding reserve (incurred - paid) |
| `loss_date` | STRING | Date of loss (YYYY-MM-DD) |
| `status` | STRING | Claim status (Open, Closed) |

**Rows:** ~50,000 × SCALE_FACTOR (1-5 claims per policy with claims)

### `internal_quote_history`

Historical quotes — both converted and declined.

| Column | Type | Description |
|--------|------|-------------|
| `quote_id` | STRING | Unique quote identifier (QTE-NNNNNN) |
| `policy_id` | STRING | FK if converted, NULL if declined |
| `sic_code` | STRING | Industry of quoted business |
| `postcode_sector` | STRING | Location of quoted premises |
| `annual_turnover` | LONG | Declared revenue at time of quote (GBP) |
| `sum_insured` | LONG | Quoted coverage amount (GBP) |
| `quoted_premium` | LONG | Premium quoted (GBP) |
| `competitor_quoted` | STRING | Whether competitor also quoted (Y/N) |
| `converted` | STRING | Whether quote became a policy (Y/N, ~38% conversion) |
| `quote_date` | STRING | Date quote was generated (YYYY-MM-DD) |

**Rows:** 120,000 × SCALE_FACTOR

---

## External Datasets (Ingested via Volume)

These represent data from external vendors, ingested through the medallion pipeline.

### `raw_market_pricing_benchmark` / `silver_market_pricing_benchmark`

Aggregated competitor pricing data by industry and region.

| Column | Type | Description |
|--------|------|-------------|
| `match_key_sic_region` | STRING | Composite key: {SIC_CODE}_{REGION} |
| `market_median_rate` | DOUBLE | Market median premium per £1k sum insured |
| `competitor_a_min_premium` | DOUBLE | Lowest observed competitor rate |
| `price_index_trend` | DOUBLE | Quarterly market price change (%) |
| `_ingested_at` | TIMESTAMP | Ingestion timestamp (raw only) |
| `_source_file` | STRING | Source file path (raw only) |

**Silver additions:** `sic_code`, `region` (parsed from key), `competitor_ratio`

**Rows:** ~150 (15 SIC codes × 10 regions)

### `raw_geospatial_hazard_enrichment` / `silver_geospatial_hazard_enrichment`

Location-based risk scores from environmental data providers.

| Column | Type | Description |
|--------|------|-------------|
| `postcode_sector` | STRING | Join key to policies |
| `flood_zone_rating` | INT | Flood risk (1=low, 10=high) |
| `proximity_to_fire_station_km` | DOUBLE | Distance to nearest fire station (km) |
| `crime_theft_index` | DOUBLE | Local business crime index (0-100) |
| `subsidence_risk` | DOUBLE | Ground subsidence score (0-10) |

**Silver additions:** `composite_location_risk`, `location_risk_tier` (High/Medium/Low)

**Rows:** ~300 (one per postcode sector)

**DQ expectations (silver):** flood 1-10, fire distance ≥0, crime not null, subsidence 0-10

### `raw_credit_bureau_summary` / `silver_credit_bureau_summary`

Company financial health from credit reference agencies.

| Column | Type | Description |
|--------|------|-------------|
| `company_id` | STRING | Bureau company reference (CMP-NNNNNN) |
| `policy_id` | STRING | FK to policies |
| `credit_score` | INT | Company credit score (200-900) |
| `ccj_count` | INT | County Court Judgments |
| `years_trading` | INT | Years since incorporation |
| `director_changes` | INT | Director changes in last 3 years |

**Silver additions:** `credit_risk_tier` (Prime/Standard/Sub-Standard/High Risk), `business_stability_score` (0-100)

**DQ expectations (silver):** credit score 200-900, CCJ ≥0, years trading not null

**Rows:** 50,000 × SCALE_FACTOR

### `economic_indicators` (NEW)

Regional economic context for pricing adjustments.

| Column | Type | Description |
|--------|------|-------------|
| `region` | STRING | UK region name |
| `year` | INT | Calendar year |
| `quarter` | INT | Calendar quarter (1-4) |
| `gdp_growth_pct` | DOUBLE | Regional GDP growth (%) |
| `unemployment_rate_pct` | DOUBLE | Unemployment rate (%) |
| `cpi_inflation_pct` | DOUBLE | CPI inflation (%) |
| `construction_cost_index` | DOUBLE | Construction cost index (base=100) |

**Rows:** ~200 (10 regions × 5 years × 4 quarters)

---

## Gold Layer

### `unified_pricing_table_live`

The **Unified Pricing Table** — single wide denormalized feature table for model training.

**Primary Key:** `policy_id`
**Feature Count:** ~90 columns
**Registered as:** UC Feature Table (PK constraint)
**Online Store:** Lakebase (pricing-upt-online-store)

| Category | Columns | Origin |
|----------|---------|--------|
| Policy core | policy_id, sic_code, postcode_sector, annual_turnover, sum_insured, current_premium, construction_type, year_built, inception_date, renewal_date | internal_commercial_policies |
| Claims aggregation | claim_count_5y, total_incurred_5y, total_paid_5y, total_reserve_5y, distinct_perils, last_claim_date, open_claims_count, fire_incurred, flood_incurred, theft_incurred, liability_incurred, storm_incurred, subsidence_incurred, water_incurred | internal_claims_history (aggregated) |
| Quote aggregation | quote_count, avg_quoted_premium, min_quoted_premium, max_quoted_premium, competitor_quote_count, last_quote_date | internal_quote_history (aggregated) |
| Market intelligence | market_median_rate, competitor_a_min_premium, price_index_trend, competitor_ratio | silver_market_pricing_benchmark |
| Geospatial hazard | flood_zone_rating, proximity_to_fire_station_km, crime_theft_index, subsidence_risk, composite_location_risk, location_risk_tier | silver_geospatial_hazard_enrichment |
| Credit bureau | credit_score, ccj_count, years_trading, director_changes, credit_risk_tier, business_stability_score | silver_credit_bureau_summary |
| Derived features | loss_ratio_5y, rate_per_1k_si, market_position_ratio, building_age_years, combined_risk_score, industry_risk_tier, region | Computed in build_upt.py |
| Synthetic bureau | credit_default_probability, director_stability_score, payment_history_score, ... (20 features) | Deterministic hash simulation |
| Synthetic geo | distance_to_coast_km, local_unemployment_rate_pct, traffic_density_index, ... (20 features) | Deterministic hash simulation |
| Audit metadata | last_updated_by, approval_timestamp, source_version, upt_build_timestamp | System-generated |

---

## Use Case Tables

| Table | Description |
|-------|-------------|
| `shadow_pricing_impact` | UC1: Policy-level premium deltas from shadow pricing simulation |
| `pit_drift_summary` | UC2: Feature drift between Delta versions |
| `staging_subsidence_risk_index` | UC3: New dataset staging for subsidence enrichment |
| `new_dataset_subsidence_impact` | UC3: Pricing impact of adding subsidence data |
| `online_store_latency` | Online store latency test results |
| `endpoint_latency` | Model serving endpoint latency test results |

---

## Governance Tables

| Table | Description |
|-------|-------------|
| `audit_log` | Unified audit trail — every governance event |
| `dataset_approvals` | Dataset approval/rejection decisions |
| `mf_audit_log` | Model factory event log |
| `mf_leaderboard` | Model factory evaluation rankings |
| `mf_actuary_decisions` | Model approval decisions |
| `mf_training_plan` | Model training configurations |
| `mf_training_log` | Model training execution log |
| `mf_feature_profile` | Feature inspection results |

---

## Scale Factor

| Scale | Policies | Claims | Quotes | Setup Time |
|-------|----------|--------|--------|-----------|
| 1 (default) | 50K | ~50K | 120K | ~2 min |
| 10 | 500K | ~500K | 1.2M | ~10 min |
| 100 | 5M | ~5M | 12M | ~60 min |

All ratios (claims/policy, quote conversion, fraud rate, churn rate) are maintained at every scale.
