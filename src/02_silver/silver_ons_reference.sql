-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Silver: ONS Reference Data
-- MAGIC Cleansed postcode-level reference data (ONS + NHS open data) with DQ expectations.
-- MAGIC Used downstream by the derived-factors notebook to compute urban_score.

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW silver_ons_reference(
  CONSTRAINT valid_postcode EXPECT (postcode_sector IS NOT NULL)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_population_density EXPECT (population_density_per_km2 IS NOT NULL AND population_density_per_km2 > 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_urban_class EXPECT (urban_classification_score BETWEEN 1 AND 6)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_gp_density EXPECT (gp_density_per_10k >= 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_imd EXPECT (deprivation_decile BETWEEN 1 AND 10)
    ON VIOLATION DROP ROW
)
COMMENT 'Cleansed ONS + NHS reference data at postcode sector level. Raw inputs for the urban_score derived factor.'
AS
SELECT
  postcode_sector,
  CAST(population_density_per_km2 AS DOUBLE) AS population_density_per_km2,
  CAST(urban_classification_score AS INT)    AS urban_classification_score,
  CAST(gp_density_per_10k AS DOUBLE)         AS gp_density_per_10k,
  CAST(deprivation_decile AS INT)            AS deprivation_decile,
  _ingested_at,
  _source_file
FROM raw_ons_reference
