-- Seed a realistic "Flood Risk v2" vendor refresh into the raw geospatial table.
--
-- Why:
--   Demo state out of the box has raw == silver, so the Impact tab shows
--   "no pricing impact". To demonstrate shadow pricing end-to-end, we need
--   the raw (incoming) version to genuinely differ from silver (approved):
--     • some shared postcodes have updated flood / crime / subsidence scores
--     • any new postcodes that weren't in silver stay in raw (adds coverage)
--
-- What this does (deterministic, hash-based so reruns are identical):
--   • ~15% of postcodes: flood zone +1..+3  (new flood mapping — worsened)
--   • ~15% of postcodes: flood zone -1..-2  (reassessed — improved)
--   • ~20% of postcodes: crime index × 1.15 (crime uptick)
--   • ~15% of postcodes: crime index × 0.90 (improved policing)
--   •  ~8% of postcodes: subsidence +1.5    (new sinkhole survey)
--   • All other postcodes unchanged
--   • Any postcodes in raw but not in silver are preserved unchanged
--
-- Re-run safe: CREATE OR REPLACE TABLE rebuilds the raw table in place.
-- To reset to a clean match with silver, re-run the ingestion notebook
-- (src/01_ingestion/ingest_geospatial_hazard.py) instead.

CREATE OR REPLACE TABLE lr_serverless_aws_us_catalog.pricing_upt.raw_geospatial_hazard_enrichment AS
WITH shifted_silver AS (
    SELECT
        postcode_sector,
        CASE
            WHEN abs(hash(concat(postcode_sector, 'flood_v2'))) % 100 < 15
                THEN LEAST(10, flood_zone_rating + (abs(hash(postcode_sector)) % 3 + 1))
            WHEN abs(hash(concat(postcode_sector, 'flood_v2'))) % 100 < 30
                THEN GREATEST(1, flood_zone_rating - (abs(hash(postcode_sector)) % 2 + 1))
            ELSE flood_zone_rating
        END AS flood_zone_rating,
        proximity_to_fire_station_km,
        CASE
            WHEN abs(hash(concat(postcode_sector, 'crime_v2'))) % 100 < 20
                THEN ROUND(crime_theft_index * 1.15, 1)
            WHEN abs(hash(concat(postcode_sector, 'crime_v2'))) % 100 < 35
                THEN ROUND(crime_theft_index * 0.90, 1)
            ELSE crime_theft_index
        END AS crime_theft_index,
        CASE
            WHEN abs(hash(concat(postcode_sector, 'subs_v2'))) % 100 < 8
                THEN ROUND(LEAST(10.0, subsidence_risk + 1.5), 1)
            ELSE subsidence_risk
        END AS subsidence_risk,
        current_timestamp()                              AS _ingested_at,
        'manual_upload:vendor_refresh_2026_q2.csv'       AS _source_file
    FROM lr_serverless_aws_us_catalog.pricing_upt.silver_geospatial_hazard_enrichment
),
new_postcodes_kept AS (
    SELECT
        r.postcode_sector, r.flood_zone_rating, r.proximity_to_fire_station_km,
        r.crime_theft_index, r.subsidence_risk, r._ingested_at, r._source_file
    FROM lr_serverless_aws_us_catalog.pricing_upt.raw_geospatial_hazard_enrichment r
    LEFT ANTI JOIN lr_serverless_aws_us_catalog.pricing_upt.silver_geospatial_hazard_enrichment s
         ON r.postcode_sector = s.postcode_sector
)
SELECT * FROM shifted_silver
UNION ALL
SELECT * FROM new_postcodes_kept;
