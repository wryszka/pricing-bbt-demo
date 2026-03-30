-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Silver: Market Pricing Benchmark
-- MAGIC Cleansed market intelligence data with DQ expectations.

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW silver_market_pricing_benchmark(
  CONSTRAINT valid_median_rate EXPECT (market_median_rate IS NOT NULL AND market_median_rate > 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_competitor_min EXPECT (competitor_a_min_premium IS NOT NULL AND competitor_a_min_premium > 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_price_trend EXPECT (price_index_trend BETWEEN -50.0 AND 50.0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_match_key EXPECT (match_key_sic_region IS NOT NULL)
    ON VIOLATION DROP ROW
)
AS
SELECT
  match_key_sic_region,
  -- Split composite key into SIC code and region for downstream joins
  SUBSTRING(match_key_sic_region, 1, INSTR(match_key_sic_region, '_') - 1) AS sic_code,
  SUBSTRING(match_key_sic_region, INSTR(match_key_sic_region, '_') + 1) AS region,
  CAST(market_median_rate AS DOUBLE) AS market_median_rate,
  CAST(competitor_a_min_premium AS DOUBLE) AS competitor_a_min_premium,
  CAST(price_index_trend AS DOUBLE) AS price_index_trend,
  -- Derived: competitiveness ratio
  ROUND(competitor_a_min_premium / NULLIF(market_median_rate, 0), 3) AS competitor_ratio,
  _ingested_at,
  _source_file
FROM raw_market_pricing_benchmark
