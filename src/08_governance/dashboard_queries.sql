-- Databricks AI/BI Dashboard Queries — Pricing Governance
--
-- These queries power the governance dashboard. Create an AI/BI Dashboard
-- in Databricks and add each query as a dataset, then build widgets.
--
-- To use: replace {catalog} and {schema} with your values
-- Default: lr_serverless_aws_us_catalog.pricing_upt

-- ============================================================
-- PANEL 1: Data Lineage Timeline
-- Shows the full chain from raw → silver → gold with versions
-- ============================================================

-- Dataset: data_lineage
SELECT
  'Raw → Silver → Gold' AS pipeline,
  table_name,
  operation,
  version,
  timestamp,
  userName AS modified_by,
  operationMetrics
FROM (
  SELECT 'unified_pricing_table_live' AS table_name, * FROM (DESCRIBE HISTORY lr_serverless_aws_us_catalog.pricing_upt.unified_pricing_table_live LIMIT 20)
  UNION ALL
  SELECT 'silver_geospatial_hazard_enrichment' AS table_name, * FROM (DESCRIBE HISTORY lr_serverless_aws_us_catalog.pricing_upt.silver_geospatial_hazard_enrichment LIMIT 10)
  UNION ALL
  SELECT 'silver_market_pricing_benchmark' AS table_name, * FROM (DESCRIBE HISTORY lr_serverless_aws_us_catalog.pricing_upt.silver_market_pricing_benchmark LIMIT 10)
  UNION ALL
  SELECT 'silver_credit_bureau_summary' AS table_name, * FROM (DESCRIBE HISTORY lr_serverless_aws_us_catalog.pricing_upt.silver_credit_bureau_summary LIMIT 10)
)
ORDER BY timestamp DESC
LIMIT 50;


-- ============================================================
-- PANEL 2: Approval Activity
-- Recent approvals/rejections from unified audit trail
-- ============================================================

-- Dataset: approval_activity
SELECT
  event_id,
  event_type,
  entity_type,
  entity_id,
  user_id,
  timestamp,
  source,
  details
FROM lr_serverless_aws_us_catalog.pricing_upt.audit_log
ORDER BY timestamp DESC
LIMIT 100;


-- ============================================================
-- PANEL 3: Audit Events by Type
-- Aggregated view of governance events
-- ============================================================

-- Dataset: events_by_type
SELECT
  event_type,
  entity_type,
  COUNT(*) AS event_count,
  MAX(timestamp) AS last_occurrence,
  COUNT(DISTINCT user_id) AS unique_users
FROM lr_serverless_aws_us_catalog.pricing_upt.audit_log
GROUP BY event_type, entity_type
ORDER BY event_count DESC;


-- ============================================================
-- PANEL 4: Data Quality — Row counts across layers
-- Shows data flow and DQ drop rates
-- ============================================================

-- Dataset: data_quality
SELECT 'Market Pricing' AS dataset,
  (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_market_pricing_benchmark) AS raw_rows,
  (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_market_pricing_benchmark) AS silver_rows
UNION ALL
SELECT 'Geospatial Hazard',
  (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_geospatial_hazard_enrichment),
  (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_geospatial_hazard_enrichment)
UNION ALL
SELECT 'Credit Bureau',
  (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.raw_credit_bureau_summary),
  (SELECT COUNT(*) FROM lr_serverless_aws_us_catalog.pricing_upt.silver_credit_bureau_summary);


-- ============================================================
-- PANEL 5: Online Store & Serving Health
-- Latency metrics from test results
-- ============================================================

-- Dataset: serving_latency
SELECT metric, ROUND(value, 1) AS value
FROM lr_serverless_aws_us_catalog.pricing_upt.online_store_latency
ORDER BY metric;


-- ============================================================
-- PANEL 6: LLM / Agent Usage
-- AI agent interactions from audit_log
-- ============================================================

-- Dataset: agent_usage
SELECT
  event_id,
  user_id,
  timestamp,
  details
FROM lr_serverless_aws_us_catalog.pricing_upt.audit_log
WHERE event_type = 'agent_recommendation'
ORDER BY timestamp DESC;


-- ============================================================
-- PANEL 7: Feature Table Health
-- UPT metadata and tags
-- ============================================================

-- Dataset: feature_table_tags
SELECT tag_name, tag_value
FROM lr_serverless_aws_us_catalog.information_schema.table_tags
WHERE schema_name = 'pricing_upt'
  AND table_name = 'unified_pricing_table_live';
