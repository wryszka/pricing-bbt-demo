-- Seed the demo approval state so the Ingestion tab tells a story:
-- two of the three vendor feeds are already approved and merged; only the
-- Geospatial Hazard update is awaiting actuary review. This matches the
-- demo narrative: "the actuary just got a Slack notification that a new
-- geo hazard version has landed".
--
-- Re-run safe: clears any prior approvals on these datasets, then writes
-- a fresh "approved" row for market_pricing_benchmark and credit_bureau_summary.
-- Leaves geospatial_hazard_enrichment alone so it stays in "pending review".

-- Make sure the approvals table exists (the FastAPI handler usually creates
-- it lazily; doing it here so the seed can run even before the app starts).
CREATE TABLE IF NOT EXISTS lr_serverless_aws_us_catalog.pricing_upt.dataset_approvals (
    approval_id STRING,
    dataset_name STRING,
    decision STRING,
    reviewer STRING,
    reviewer_notes STRING,
    reviewed_at TIMESTAMP,
    raw_row_count BIGINT,
    silver_row_count BIGINT,
    rows_dropped_by_dq INT
);

-- Wipe any stale approval rows for these two so the "latest" approval is
-- the one we're about to insert.
DELETE FROM lr_serverless_aws_us_catalog.pricing_upt.dataset_approvals
WHERE dataset_name IN ('market_pricing_benchmark', 'credit_bureau_summary');

-- Seed "approved" for both — three weeks ago so it feels like real history.
INSERT INTO lr_serverless_aws_us_catalog.pricing_upt.dataset_approvals
(approval_id, dataset_name, decision, reviewer, reviewer_notes,
 reviewed_at, raw_row_count, silver_row_count, rows_dropped_by_dq)
SELECT
    'seed_market_pricing_benchmark',
    'market_pricing_benchmark',
    'approved',
    current_user(),
    'Q1 2026 market benchmark refresh — reviewed and approved.',
    current_timestamp() - INTERVAL 21 DAYS,
    (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_market_pricing_benchmark),
    (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_market_pricing_benchmark),
    (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_market_pricing_benchmark)
    - (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_market_pricing_benchmark);

INSERT INTO lr_serverless_aws_us_catalog.pricing_upt.dataset_approvals
(approval_id, dataset_name, decision, reviewer, reviewer_notes,
 reviewed_at, raw_row_count, silver_row_count, rows_dropped_by_dq)
SELECT
    'seed_credit_bureau_summary',
    'credit_bureau_summary',
    'approved',
    current_user(),
    'Monthly bureau refresh — all DQ checks passed.',
    current_timestamp() - INTERVAL 14 DAYS,
    (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_credit_bureau_summary),
    (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_credit_bureau_summary),
    (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_credit_bureau_summary)
    - (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_credit_bureau_summary);

-- Clear any prior approval on geo_hazard so it stays in pending review —
-- this is the dataset the actuary was just notified about.
DELETE FROM lr_serverless_aws_us_catalog.pricing_upt.dataset_approvals
WHERE dataset_name = 'geospatial_hazard_enrichment';
