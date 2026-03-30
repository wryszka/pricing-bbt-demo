-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Silver: Credit Bureau Summary
-- MAGIC Cleansed bureau data with DQ expectations.

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW silver_credit_bureau_summary(
  CONSTRAINT valid_company_id EXPECT (company_id IS NOT NULL)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_policy_id EXPECT (policy_id IS NOT NULL)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_credit_score EXPECT (credit_score BETWEEN 200 AND 900)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_ccj_count EXPECT (ccj_count >= 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_years_trading EXPECT (years_trading IS NOT NULL AND years_trading >= 0)
    ON VIOLATION DROP ROW
)
AS
SELECT
  company_id,
  policy_id,
  CAST(credit_score AS INT) AS credit_score,
  CAST(ccj_count AS INT) AS ccj_count,
  CAST(years_trading AS INT) AS years_trading,
  CAST(director_changes AS INT) AS director_changes,
  -- Derived: credit risk tier
  CASE
    WHEN credit_score >= 750 AND ccj_count = 0 THEN 'Prime'
    WHEN credit_score >= 550 AND ccj_count <= 1 THEN 'Standard'
    WHEN credit_score >= 400 THEN 'Sub-Standard'
    ELSE 'High Risk'
  END AS credit_risk_tier,
  -- Derived: business stability score (0-100)
  ROUND(
    LEAST(100,
      (LEAST(years_trading, 50) / 50.0 * 40) +
      (LEAST(credit_score, 900) / 900.0 * 40) +
      (GREATEST(0, 20 - (ccj_count * 5 + director_changes * 3)))
    ),
    1
  ) AS business_stability_score,
  _ingested_at,
  _source_file
FROM raw_credit_bureau_summary
