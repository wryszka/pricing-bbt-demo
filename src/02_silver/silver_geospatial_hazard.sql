-- Databricks notebook source
-- MAGIC %md
-- MAGIC # Silver: Geospatial Hazard Enrichment
-- MAGIC Cleansed location risk data with DQ expectations.

-- COMMAND ----------

CREATE OR REFRESH MATERIALIZED VIEW silver_geospatial_hazard_enrichment(
  CONSTRAINT valid_postcode EXPECT (postcode_sector IS NOT NULL)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_flood_zone EXPECT (flood_zone_rating BETWEEN 1 AND 10)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_fire_distance EXPECT (proximity_to_fire_station_km >= 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_crime_index EXPECT (crime_theft_index IS NOT NULL AND crime_theft_index >= 0)
    ON VIOLATION DROP ROW,
  CONSTRAINT valid_subsidence EXPECT (subsidence_risk BETWEEN 0 AND 10)
    ON VIOLATION DROP ROW
)
AS
SELECT
  postcode_sector,
  CAST(flood_zone_rating AS INT) AS flood_zone_rating,
  CAST(proximity_to_fire_station_km AS DOUBLE) AS proximity_to_fire_station_km,
  CAST(crime_theft_index AS DOUBLE) AS crime_theft_index,
  CAST(subsidence_risk AS DOUBLE) AS subsidence_risk,
  -- Derived: composite location risk score (weighted average)
  ROUND(
    (flood_zone_rating * 0.30) +
    (LEAST(proximity_to_fire_station_km, 25.0) / 25.0 * 10 * 0.20) +
    (crime_theft_index / 10.0 * 0.25) +
    (subsidence_risk * 0.25),
    2
  ) AS composite_location_risk,
  -- Risk tier based on composite
  CASE
    WHEN (flood_zone_rating * 0.30 + LEAST(proximity_to_fire_station_km, 25.0) / 25.0 * 10 * 0.20 + crime_theft_index / 10.0 * 0.25 + subsidence_risk * 0.25) >= 6.0 THEN 'High'
    WHEN (flood_zone_rating * 0.30 + LEAST(proximity_to_fire_station_km, 25.0) / 25.0 * 10 * 0.20 + crime_theft_index / 10.0 * 0.25 + subsidence_risk * 0.25) >= 3.5 THEN 'Medium'
    ELSE 'Low'
  END AS location_risk_tier,
  _ingested_at,
  _source_file
FROM raw_geospatial_hazard_enrichment
