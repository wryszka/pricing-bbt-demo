"""Dataset ingestion review and approval routes.

Provides endpoints for:
1. Listing external datasets available for review
2. Diff between current and pending versions (new/changed/removed rows)
3. Impact analytics (pricing impact simulation)
4. Data quality expectations and freshness
5. Approve/reject workflow
6. Manual CSV download and upload with audit trail
"""

import csv
import hashlib
import io
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, UploadFile, File, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from server.audit import log_audit_event

from server.config import fqn, get_current_user, get_catalog, get_schema
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# Datasets on the Ingestion tab. The `category` field drives the UI badge +
# the backend short-circuits approval/upload/impact for non-external rows via
# the existing `is_reference: True` flag.
EXTERNAL_DATASETS = {
    # --- internal systems of record (no review required) ---
    "internal_commercial_policies": {
        "display_name": "Commercial Policies",
        "category": "internal",
        "source": "Policy admin system (internal)",
        "join_key": "policy_id",
        "raw_table": "internal_commercial_policies",
        "silver_table": "internal_commercial_policies",
        "description": "Active commercial book — 50K in-force policies with risk, premium, and renewal information. Internal system of record; no actuary approval required.",
        "expected_columns": ["policy_id", "company_id", "postcode_sector", "sic_code", "industry_risk_tier", "construction_type", "sum_insured", "current_premium", "renewal_date", "region"],
        "is_reference": True,
    },
    "internal_claims_history": {
        "display_name": "Claims History",
        "category": "internal",
        "source": "Claims system (internal)",
        "join_key": "policy_id",
        "raw_table": "internal_claims_history",
        "silver_table": "internal_claims_history",
        "description": "Full claims history against the book — frequency and severity per policy. Internal system of record; no actuary approval required.",
        "expected_columns": ["claim_id", "policy_id", "claim_date", "peril", "paid_amount", "status"],
        "is_reference": True,
    },
    # --- external vendor feeds (HITL approval workflow) ---
    "market_pricing_benchmark": {
        "display_name": "Market Pricing Benchmark",
        "category": "external_vendor",
        "source": "External Vendor (PCW)",
        "join_key": "sic_code + region",
        "raw_table": "raw_market_pricing_benchmark",
        "silver_table": "silver_market_pricing_benchmark",
        "description": "Aggregated competitor pricing data by industry and region",
        "expected_columns": ["match_key_sic_region", "market_median_rate", "competitor_a_min_premium", "price_index_trend"],
    },
    "geospatial_hazard_enrichment": {
        "display_name": "Geospatial Hazard Enrichment",
        "category": "external_vendor",
        "source": "External Vendor (OS/EA)",
        "join_key": "postcode_sector",
        "raw_table": "raw_geospatial_hazard_enrichment",
        "silver_table": "silver_geospatial_hazard_enrichment",
        "description": "Location-based risk scores: flood, fire, crime, subsidence",
        "expected_columns": ["postcode_sector", "flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"],
    },
    "credit_bureau_summary": {
        "display_name": "Credit Bureau Summary",
        "category": "external_vendor",
        "source": "Bureau (D&B/Experian)",
        "join_key": "policy_id",
        "raw_table": "raw_credit_bureau_summary",
        "silver_table": "silver_credit_bureau_summary",
        "description": "Company financial health: credit score, CCJs, years trading",
        "expected_columns": ["company_id", "policy_id", "credit_score", "ccj_count", "years_trading", "director_changes"],
    },
    # --- real public reference data (no approval — one-shot builds) ---
    "postcode_enrichment": {
        "display_name": "Postcode Enrichment (real UK public data)",
        "category": "reference_data",
        "source": "ONS Postcode Directory + IMD 2019 + ONS Rural-Urban Classification",
        "join_key": "postcode (aggregated to postcode area for the UPT)",
        "raw_table": "postcode_enrichment",
        "silver_table": "postcode_enrichment",
        "description": "~1.5M English postcodes with IMD deprivation deciles, urban/rural band, coastal flags. Built by src/new_data_impact/00a. Feeds urban_score, is_coastal, deprivation_composite in the UPT.",
        "expected_columns": ["postcode", "lsoa_code", "region_code", "is_urban", "is_coastal", "imd_decile", "crime_decile", "income_decile", "health_decile", "living_env_decile"],
        "is_reference": True,
    },
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class ApprovalRequest(BaseModel):
    decision: str  # "approved" or "rejected"
    reviewer_notes: str = ""


# ---------------------------------------------------------------------------
# Ensure approvals tracking table exists
# ---------------------------------------------------------------------------

async def ensure_approvals_table():
    await execute_query(f"""
        CREATE TABLE IF NOT EXISTS {fqn('dataset_approvals')} (
            approval_id STRING,
            dataset_name STRING,
            decision STRING,
            reviewer STRING,
            reviewer_notes STRING,
            reviewed_at TIMESTAMP,
            raw_row_count BIGINT,
            silver_row_count BIGINT,
            rows_dropped_by_dq INT
        )
    """)


# ---------------------------------------------------------------------------
# 1. List datasets
# ---------------------------------------------------------------------------

@router.get("/meta")
async def list_datasets_meta():
    """Static metadata for every dataset — no SQL. Instant response so the
    Ingestion page can render its card frames before the slow stats come in
    from the `/api/datasets` endpoint."""
    return [
        {"id": ds_id, **ds_info,
         "raw_row_count":      None, "silver_row_count":   None,
         "rows_dropped_by_dq": None, "last_ingested":      None,
         "approval":           None}
        for ds_id, ds_info in EXTERNAL_DATASETS.items()
    ]


@router.get("")
async def list_datasets():
    """List all external datasets with their current status.
    Reference datasets (is_reference=True) are one-shot builds with no raw→silver
    split and no approval workflow — they just report a row count.

    Fans all per-dataset SQL queries out in parallel via asyncio.gather so the
    response comes back in roughly the slowest-single-query time, not the sum."""
    import asyncio

    async def _stats_for(ds_id: str, ds_info: dict):
        is_reference = bool(ds_info.get("is_reference"))
        raw_count = silver_count = 0
        last_ingested = None
        approval_row: dict | None = None
        try:
            if is_reference:
                stats = await execute_query(f"""
                    SELECT count(*) as row_count FROM {fqn(ds_info['silver_table'])}
                """)
                raw_count = silver_count = int(stats[0]["row_count"]) if stats else 0
            else:
                raw_stats, silver_stats, approvals = await asyncio.gather(
                    execute_query(f"""
                        SELECT count(*) as row_count,
                               max(_ingested_at) as last_ingested
                        FROM {fqn(ds_info['raw_table'])}
                    """),
                    execute_query(f"""
                        SELECT count(*) as row_count FROM {fqn(ds_info['silver_table'])}
                    """),
                    execute_query(f"""
                        SELECT decision, reviewer, reviewed_at, reviewer_notes
                        FROM {fqn('dataset_approvals')}
                        WHERE dataset_name = '{ds_id}'
                        ORDER BY reviewed_at DESC LIMIT 1
                    """),
                )
                raw_count      = int(raw_stats[0]["row_count"]) if raw_stats else 0
                silver_count   = int(silver_stats[0]["row_count"]) if silver_stats else 0
                last_ingested  = raw_stats[0].get("last_ingested") if raw_stats else None
                approval_row   = approvals[0] if approvals else None
        except Exception as e:
            logger.warning("Failed to query stats for %s: %s", ds_id, e)
        return {
            "id": ds_id,
            **ds_info,
            "raw_row_count":       raw_count,
            "silver_row_count":    silver_count,
            "rows_dropped_by_dq":  max(0, raw_count - silver_count),
            "last_ingested":       last_ingested,
            "approval":            approval_row,
        }

    return await asyncio.gather(*[
        _stats_for(ds_id, info) for ds_id, info in EXTERNAL_DATASETS.items()
    ])


# ---------------------------------------------------------------------------
# 2. Diff: new/changed/removed rows between raw and silver
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/diff")
async def get_dataset_diff(dataset_id: str):
    """Show difference between raw (pending) and silver (current approved) data."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    import asyncio

    ds = EXTERNAL_DATASETS[dataset_id]
    if ds.get("is_reference"):
        return {
            "is_reference": True,
            "message": "This is a reference dataset with no raw→silver diff workflow.",
            "raw_count": 0, "silver_count": 0,
            "new_rows": [], "changed_rows": [], "removed_rows": [],
            "summary_stats": {},
        }
    raw_table = fqn(ds["raw_table"])
    silver_table = fqn(ds["silver_table"])

    # Dataset-specific key and comparison logic
    if dataset_id == "market_pricing_benchmark":
        key_col = "match_key_sic_region"
        compare_cols = ["market_median_rate", "competitor_a_min_premium", "price_index_trend"]
    elif dataset_id == "geospatial_hazard_enrichment":
        key_col = "postcode_sector"
        compare_cols = ["flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk"]
    else:  # credit_bureau
        key_col = "policy_id"
        compare_cols = ["credit_score", "ccj_count", "years_trading", "director_changes"]

    compare_conditions = " OR ".join(
        f"CAST(r.{c} AS STRING) != CAST(s.{c} AS STRING)" for c in compare_cols
    )
    changed_select = ", ".join(
        f"s.{c} as old_{c}, r.{c} as new_{c}" for c in compare_cols
    )

    sql_summary = f"""
        SELECT
            (SELECT count(*) FROM {raw_table})    as raw_total,
            (SELECT count(*) FROM {silver_table}) as silver_total,
            (SELECT count(*) FROM {raw_table} r
             LEFT ANTI JOIN {silver_table} s ON r.{key_col} = s.{key_col}) as new_rows,
            (SELECT count(*) FROM {silver_table} s
             LEFT ANTI JOIN {raw_table} r ON s.{key_col} = r.{key_col}) as removed_rows
    """
    sql_changed = f"""
        SELECT r.{key_col}, {changed_select}
        FROM {raw_table} r
        INNER JOIN {silver_table} s ON r.{key_col} = s.{key_col}
        WHERE {compare_conditions}
        LIMIT 50
    """
    sql_new = f"""
        SELECT r.*
        FROM {raw_table} r
        LEFT ANTI JOIN {silver_table} s ON r.{key_col} = s.{key_col}
        LIMIT 20
    """
    sql_removed = f"""
        SELECT s.*
        FROM {silver_table} s
        LEFT ANTI JOIN {raw_table} r ON s.{key_col} = r.{key_col}
        LIMIT 20
    """

    async def _safe(q):
        try: return await execute_query(q)
        except Exception as e:
            logger.warning("diff query failed: %s", str(e)[:120]); return None

    summary, changed_sample, new_sample, removed_sample = await asyncio.gather(
        _safe(sql_summary), _safe(sql_changed), _safe(sql_new), _safe(sql_removed),
    )

    return {
        "dataset_id": dataset_id,
        "key_column": key_col,
        "compare_columns": compare_cols,
        "summary": (summary or [{}])[0] if summary else {},
        "changed_rows":  changed_sample or [],
        "new_rows":      new_sample     or [],
        "removed_rows":  removed_sample or [],
        "changed_count": len(changed_sample or []),
    }


# ---------------------------------------------------------------------------
# 3. Impact analytics — "Shadow Pricing" simulation
#
# When new data is staged, we automatically re-rate every affected policy
# using a proxy pricing formula and show the actuary the exact financial
# impact BEFORE they approve the merge.
#
# Proxy pricing formula (mirrors a real rating engine):
#   Technical_Price = BASE_RATE × (sum_insured / 1000) × industry_factor
#                     × flood_factor × crime_factor × construction_factor
# ---------------------------------------------------------------------------

# SQL fragments for the proxy pricing formula (reused across queries)
_PRICING_SQL = """
    CASE p.industry_risk_tier
        WHEN 'High' THEN 1.8 WHEN 'Medium' THEN 1.2 ELSE 0.85
    END AS industry_factor,
    CASE p.construction_type
        WHEN 'Fire Resistive' THEN 0.7 WHEN 'Non-Combustible' THEN 0.85
        WHEN 'Joisted Masonry' THEN 1.0 WHEN 'Heavy Timber' THEN 1.15
        WHEN 'Frame' THEN 1.4 ELSE 1.0
    END AS construction_factor,
    ROUND(0.8 + COALESCE(p.crime_theft_index, 50) / 100.0 * 0.7, 2) AS crime_factor
"""


@router.get("/{dataset_id}/impact")
async def get_dataset_impact(dataset_id: str):
    """Run a shadow pricing simulation for this dataset against the portfolio."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    ds = EXTERNAL_DATASETS[dataset_id]
    if ds.get("is_reference"):
        return {
            "is_reference": True,
            "message": "Reference dataset — feeds derived_factors and the UPT via a postcode-area join, so 'portfolio impact' is best understood through the challenger comparison on the Model Development page.",
            "policies_affected": 0, "portfolio_change_pct": 0, "risk_summary": [],
        }
    upt = fqn("unified_pricing_table_live")
    raw = fqn(ds["raw_table"])
    silver = fqn(ds["silver_table"])

    # ------------------------------------------------------------------
    # Section 1: Data Diff — statistical shift in the incoming data
    # ------------------------------------------------------------------
    numeric_cols = [c for c in ds["expected_columns"]
                    if c not in ("match_key_sic_region", "postcode_sector", "company_id", "policy_id")]

    stats_parts = []
    for c in numeric_cols:
        stats_parts.append(f"""
            ROUND(AVG(CAST(r.{c} AS DOUBLE)), 3) AS new_{c}_mean,
            ROUND(STDDEV(CAST(r.{c} AS DOUBLE)), 3) AS new_{c}_std,
            ROUND(MIN(CAST(r.{c} AS DOUBLE)), 3) AS new_{c}_min,
            ROUND(MAX(CAST(r.{c} AS DOUBLE)), 3) AS new_{c}_max
        """)
    for c in numeric_cols:
        stats_parts.append(f"""
            ROUND(AVG(CAST(s.{c} AS DOUBLE)), 3) AS old_{c}_mean,
            ROUND(STDDEV(CAST(s.{c} AS DOUBLE)), 3) AS old_{c}_std
        """)

    if dataset_id == "market_pricing_benchmark":
        join_key = "match_key_sic_region"
    elif dataset_id == "geospatial_hazard_enrichment":
        join_key = "postcode_sector"
    else:
        join_key = "policy_id"

    diff_stats_sql = f"""
        SELECT {', '.join(stats_parts)},
            (SELECT COUNT(*) FROM {raw}) AS raw_count,
            (SELECT COUNT(*) FROM {silver}) AS silver_count,
            (SELECT COUNT(*) FROM {raw} r LEFT ANTI JOIN {silver} s ON r.{join_key} = s.{join_key}) AS new_rows,
            (SELECT COUNT(*) FROM {silver} s LEFT ANTI JOIN {raw} r ON s.{join_key} = r.{join_key}) AS removed_rows
        FROM {raw} r FULL OUTER JOIN {silver} s ON r.{join_key} = s.{join_key}
    """
    diff_raw = await execute_query(diff_stats_sql)
    d = diff_raw[0] if diff_raw else {}

    column_shifts = []
    for c in numeric_cols:
        old_mean = _f(d.get(f"old_{c}_mean"))
        new_mean = _f(d.get(f"new_{c}_mean"))
        shift_pct = round((new_mean - old_mean) / old_mean * 100, 1) if old_mean else 0
        column_shifts.append({
            "column": c,
            "old_mean": old_mean, "new_mean": new_mean,
            "old_std": _f(d.get(f"old_{c}_std")), "new_std": _f(d.get(f"new_{c}_std")),
            "new_min": _f(d.get(f"new_{c}_min")), "new_max": _f(d.get(f"new_{c}_max")),
            "shift_pct": shift_pct,
            "severity": "high" if abs(shift_pct) > 10 else ("medium" if abs(shift_pct) > 5 else "low"),
        })

    data_diff = {
        "raw_count": _i(d.get("raw_count")),
        "silver_count": _i(d.get("silver_count")),
        "new_rows": _i(d.get("new_rows")),
        "removed_rows": _i(d.get("removed_rows")),
        "column_shifts": column_shifts,
    }

    # ------------------------------------------------------------------
    # Section 2: Portfolio Impact — re-rate affected policies
    # ------------------------------------------------------------------
    portfolio_impact = await _compute_portfolio_impact(dataset_id, join_key, upt, raw, silver)

    # ------------------------------------------------------------------
    # Section 3: Risk Summary — tier migration and score distribution
    # ------------------------------------------------------------------
    risk_summary = await _compute_risk_summary(dataset_id, join_key, upt, raw, silver)

    return {
        "dataset_id": dataset_id,
        "display_name": ds["display_name"],
        "data_diff": data_diff,
        "portfolio_impact": portfolio_impact,
        "risk_summary": risk_summary,
    }


def _f(v) -> float:
    """Safe float cast."""
    try:
        return round(float(v), 3) if v is not None else 0.0
    except (ValueError, TypeError):
        return 0.0


def _i(v) -> int:
    """Safe int cast."""
    try:
        return int(v) if v is not None else 0
    except (ValueError, TypeError):
        return 0


def _impact_repriced_cte(dataset_id: str, upt: str, raw: str, silver: str) -> str:
    """Return the WITH ... `repriced` CTE for a given dataset_id. The CTE
    has columns: policy_id, postcode_sector, industry_risk_tier, region,
    current_premium, old_price, new_price, premium_delta, delta_pct."""
    if dataset_id == "geospatial_hazard_enrichment":
        return f"""
            WITH changes AS (
                SELECT r.postcode_sector,
                    CAST(s.flood_zone_rating AS DOUBLE) AS old_flood,
                    CAST(r.flood_zone_rating AS DOUBLE) AS new_flood,
                    CAST(s.crime_theft_index AS DOUBLE) AS old_crime,
                    CAST(r.crime_theft_index AS DOUBLE) AS new_crime
                FROM {raw} r JOIN {silver} s ON r.postcode_sector = s.postcode_sector
                WHERE CAST(r.flood_zone_rating AS STRING) != CAST(s.flood_zone_rating AS STRING)
                   OR CAST(r.crime_theft_index AS STRING) != CAST(s.crime_theft_index AS STRING)
                   OR CAST(r.subsidence_risk AS STRING) != CAST(s.subsidence_risk AS STRING)
            ),
            repriced AS (
                SELECT p.policy_id, p.postcode_sector,
                    COALESCE(p.industry_risk_tier, 'Unknown') AS industry_risk_tier,
                    COALESCE(p.region, substr(p.postcode_sector, 1, 2), 'Unknown') AS region,
                    p.current_premium,
                    ROUND(5.0 * (p.sum_insured / 1000.0)
                        * CASE p.industry_risk_tier WHEN 'High' THEN 1.8 WHEN 'Medium' THEN 1.2 ELSE 0.85 END
                        * (0.7 + (COALESCE(c.old_flood, 5) - 1) * 0.2)
                        * (0.8 + COALESCE(c.old_crime, 50) / 100.0 * 0.7)
                        * CASE p.construction_type WHEN 'Fire Resistive' THEN 0.7 WHEN 'Non-Combustible' THEN 0.85 WHEN 'Heavy Timber' THEN 1.15 WHEN 'Frame' THEN 1.4 ELSE 1.0 END
                    , 0) AS old_price,
                    ROUND(5.0 * (p.sum_insured / 1000.0)
                        * CASE p.industry_risk_tier WHEN 'High' THEN 1.8 WHEN 'Medium' THEN 1.2 ELSE 0.85 END
                        * (0.7 + (COALESCE(c.new_flood, 5) - 1) * 0.2)
                        * (0.8 + COALESCE(c.new_crime, 50) / 100.0 * 0.7)
                        * CASE p.construction_type WHEN 'Fire Resistive' THEN 0.7 WHEN 'Non-Combustible' THEN 0.85 WHEN 'Heavy Timber' THEN 1.15 WHEN 'Frame' THEN 1.4 ELSE 1.0 END
                    , 0) AS new_price
                FROM {upt} p JOIN changes c ON p.postcode_sector = c.postcode_sector
            )
        """
    if dataset_id == "market_pricing_benchmark":
        return f"""
            WITH changes AS (
                SELECT r.match_key_sic_region,
                    CAST(s.market_median_rate AS DOUBLE) AS old_rate,
                    CAST(r.market_median_rate AS DOUBLE) AS new_rate
                FROM {raw} r JOIN {silver} s ON r.match_key_sic_region = s.match_key_sic_region
                WHERE CAST(r.market_median_rate AS STRING) != CAST(s.market_median_rate AS STRING)
            ),
            repriced AS (
                SELECT p.policy_id, p.postcode_sector,
                    COALESCE(p.industry_risk_tier, 'Unknown') AS industry_risk_tier,
                    COALESCE(p.region, substr(p.postcode_sector, 1, 2), 'Unknown') AS region,
                    p.current_premium,
                    p.current_premium AS old_price,
                    p.current_premium AS new_price
                FROM {upt} p
                JOIN (SELECT DISTINCT sic_code FROM {silver}) sk ON p.sic_code = sk.sic_code
                JOIN changes c ON c.match_key_sic_region = CONCAT(p.sic_code, '_', p.region)
            )
        """
    # credit_bureau
    return f"""
        WITH changes AS (
            SELECT r.policy_id,
                CAST(s.credit_score AS INT) AS old_score,
                CAST(r.credit_score AS INT) AS new_score
            FROM {raw} r JOIN {silver} s ON r.policy_id = s.policy_id
            WHERE CAST(r.credit_score AS STRING) != CAST(s.credit_score AS STRING)
               OR CAST(r.ccj_count AS STRING) != CAST(s.ccj_count AS STRING)
        ),
        repriced AS (
            SELECT p.policy_id, p.postcode_sector,
                COALESCE(p.industry_risk_tier, 'Unknown') AS industry_risk_tier,
                COALESCE(p.region, substr(p.postcode_sector, 1, 2), 'Unknown') AS region,
                p.current_premium,
                p.current_premium AS old_price,
                ROUND(p.current_premium * (1.0 + (c.old_score - c.new_score) / 100.0 * 0.05), 0) AS new_price
            FROM {upt} p JOIN changes c ON p.policy_id = c.policy_id
        )
    """


async def _compute_portfolio_impact(dataset_id, join_key, upt, raw, silver):
    """Shadow-price affected policies via 4 small server-side aggregates.

    The previous implementation pulled row-level reprice data into the app
    process and aggregated in Python — which blew past the 25 MiB inline
    cap for the geospatial dataset (millions of policies × wide columns)
    and EXTERNAL_LINKS isn't reachable from the Apps egress zone.
    """
    cte = _impact_repriced_cte(dataset_id, upt, raw, silver)

    # 1. Overall totals + histogram counts (1 row, ~20 columns)
    totals = await execute_query(f"""
        {cte},
        scored AS (
          SELECT current_premium,
                 (new_price - old_price) AS premium_delta,
                 ROUND(CASE WHEN old_price > 0 THEN (new_price - old_price) / old_price * 100 ELSE 0 END, 1) AS delta_pct
          FROM repriced
        )
        SELECT
          COUNT(*)                                                AS affected,
          SUM(current_premium)                                    AS affected_gwp,
          SUM(premium_delta)                                      AS delta_total,
          AVG(premium_delta)                                      AS delta_avg,
          percentile_approx(premium_delta, 0.5)                   AS delta_median,
          SUM(CASE WHEN premium_delta > 0 THEN 1 ELSE 0 END)      AS pol_increase,
          SUM(CASE WHEN premium_delta < 0 THEN 1 ELSE 0 END)      AS pol_decrease,
          SUM(CASE WHEN premium_delta = 0 THEN 1 ELSE 0 END)      AS pol_unchanged,
          SUM(CASE WHEN delta_pct < -10 THEN 1 ELSE 0 END)        AS h_lt_n10,
          SUM(CASE WHEN delta_pct >= -10 AND delta_pct < -5 THEN 1 ELSE 0 END) AS h_n10_n5,
          SUM(CASE WHEN delta_pct >= -5  AND delta_pct < 0  THEN 1 ELSE 0 END) AS h_n5_0,
          SUM(CASE WHEN delta_pct = 0 THEN 1 ELSE 0 END)          AS h_0,
          SUM(CASE WHEN delta_pct > 0   AND delta_pct < 5  THEN 1 ELSE 0 END)  AS h_0_5,
          SUM(CASE WHEN delta_pct >= 5  AND delta_pct < 10 THEN 1 ELSE 0 END)  AS h_5_10,
          SUM(CASE WHEN delta_pct >= 10 THEN 1 ELSE 0 END)        AS h_gt_10,
          SUM(CASE WHEN ABS(delta_pct) > 10 THEN 1 ELSE 0 END)    AS flagged_count
        FROM scored
    """)
    if not totals or _i(totals[0].get("affected", 0)) == 0:
        return {"affected_policies": 0, "total_policies": 0}
    t = totals[0]
    affected = _i(t.get("affected", 0))

    total_q = await execute_query(f"SELECT COUNT(*) AS cnt FROM {upt}")
    total   = _i(total_q[0]["cnt"]) if total_q else 0

    # 2. Per-industry
    by_industry_rows = await execute_query(f"""
        {cte}
        SELECT industry_risk_tier AS industry,
               COUNT(*)                                  AS policies,
               SUM(current_premium)                      AS gwp,
               SUM(new_price - old_price)                AS total_delta
        FROM repriced
        GROUP BY industry_risk_tier
        ORDER BY ABS(SUM(new_price - old_price)) DESC
    """)
    by_industry = [{
        "industry":    r.get("industry") or "Unknown",
        "policies":    _i(r.get("policies", 0)),
        "gwp":         _f(r.get("gwp", 0)),
        "total_delta": _f(r.get("total_delta", 0)),
    } for r in by_industry_rows]

    # 3. Per-region (top 15)
    by_region_rows = await execute_query(f"""
        {cte}
        SELECT region,
               COUNT(*)                                  AS policies,
               SUM(current_premium)                      AS gwp,
               SUM(new_price - old_price)                AS total_delta
        FROM repriced
        GROUP BY region
        ORDER BY ABS(SUM(new_price - old_price)) DESC
        LIMIT 15
    """)
    by_region = [{
        "region":      r.get("region") or "Unknown",
        "policies":    _i(r.get("policies", 0)),
        "gwp":         _f(r.get("gwp", 0)),
        "total_delta": _f(r.get("total_delta", 0)),
    } for r in by_region_rows]

    # 4. Flagged policies (top 30 by |delta_pct| over 10%)
    flagged_rows = await execute_query(f"""
        {cte},
        scored AS (
          SELECT policy_id, postcode_sector, industry_risk_tier, current_premium,
                 (new_price - old_price) AS premium_delta,
                 ROUND(CASE WHEN old_price > 0 THEN (new_price - old_price) / old_price * 100 ELSE 0 END, 1) AS delta_pct
          FROM repriced
        )
        SELECT policy_id, postcode_sector, industry_risk_tier,
               current_premium, premium_delta, delta_pct
        FROM scored
        WHERE ABS(delta_pct) > 10
        ORDER BY ABS(delta_pct) DESC
        LIMIT 30
    """)
    flagged = [{
        "policy_id":       r.get("policy_id"),
        "postcode":        r.get("postcode_sector"),
        "industry":        r.get("industry_risk_tier"),
        "current_premium": _f(r.get("current_premium")),
        "premium_delta":   _f(r.get("premium_delta")),
        "delta_pct":       _f(r.get("delta_pct")),
    } for r in flagged_rows]

    histogram = [
        {"bucket": "< -10%",     "count": _i(t.get("h_lt_n10", 0))},
        {"bucket": "-10 to -5%", "count": _i(t.get("h_n10_n5", 0))},
        {"bucket": "-5 to 0%",   "count": _i(t.get("h_n5_0",   0))},
        {"bucket": "0%",         "count": _i(t.get("h_0",      0))},
        {"bucket": "0 to 5%",    "count": _i(t.get("h_0_5",    0))},
        {"bucket": "5 to 10%",   "count": _i(t.get("h_5_10",   0))},
        {"bucket": "> 10%",      "count": _i(t.get("h_gt_10",  0))},
    ]

    return {
        "total_policies":      total,
        "affected_policies":   affected,
        "affected_pct":        round(affected / total * 100, 1) if total else 0,
        "total_gwp":           round(_f(t.get("affected_gwp", 0))),
        "affected_gwp":        round(_f(t.get("affected_gwp", 0))),
        "premium_delta_total": round(_f(t.get("delta_total", 0))),
        "premium_delta_avg":   round(_f(t.get("delta_avg", 0))),
        "premium_delta_median":round(_f(t.get("delta_median", 0))),
        "policies_increase":   _i(t.get("pol_increase", 0)),
        "policies_decrease":   _i(t.get("pol_decrease", 0)),
        "policies_unchanged":  _i(t.get("pol_unchanged", 0)),
        "histogram":           histogram,
        "by_industry":         by_industry,
        "by_region":           by_region,
        "flagged_policies":    flagged,
        "flagged_count":       _i(t.get("flagged_count", 0)),
    }


async def _compute_risk_summary(dataset_id, join_key, upt, raw, silver):
    """Compute risk tier migration and score distribution shift."""

    if dataset_id == "geospatial_hazard_enrichment":
        migration = await execute_query(f"""
            SELECT
                CASE WHEN CAST(s.flood_zone_rating AS INT) >= 7 THEN 'High'
                     WHEN CAST(s.flood_zone_rating AS INT) >= 4 THEN 'Medium' ELSE 'Low' END AS old_tier,
                CASE WHEN CAST(r.flood_zone_rating AS INT) >= 7 THEN 'High'
                     WHEN CAST(r.flood_zone_rating AS INT) >= 4 THEN 'Medium' ELSE 'Low' END AS new_tier,
                COUNT(*) AS postcodes
            FROM {raw} r JOIN {silver} s ON r.postcode_sector = s.postcode_sector
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        score_shift = await execute_query(f"""
            SELECT
                ROUND(AVG(CAST(s.flood_zone_rating AS DOUBLE)), 2) AS old_avg_score,
                ROUND(AVG(CAST(r.flood_zone_rating AS DOUBLE)), 2) AS new_avg_score,
                SUM(CASE WHEN CAST(r.flood_zone_rating AS INT) > CAST(s.flood_zone_rating AS INT) THEN 1 ELSE 0 END) AS worsened,
                SUM(CASE WHEN CAST(r.flood_zone_rating AS INT) < CAST(s.flood_zone_rating AS INT) THEN 1 ELSE 0 END) AS improved,
                SUM(CASE WHEN CAST(r.flood_zone_rating AS INT) = CAST(s.flood_zone_rating AS INT) THEN 1 ELSE 0 END) AS unchanged
            FROM {raw} r JOIN {silver} s ON r.postcode_sector = s.postcode_sector
        """)
        return {
            "score_type": "Flood Zone Rating",
            "tier_migration": migration,
            "score_shift": score_shift[0] if score_shift else {},
        }

    elif dataset_id == "credit_bureau_summary":
        migration = await execute_query(f"""
            SELECT
                CASE WHEN CAST(s.credit_score AS INT) >= 750 THEN 'Prime'
                     WHEN CAST(s.credit_score AS INT) >= 550 THEN 'Standard'
                     WHEN CAST(s.credit_score AS INT) >= 400 THEN 'Sub-Standard' ELSE 'High Risk' END AS old_tier,
                CASE WHEN CAST(r.credit_score AS INT) >= 750 THEN 'Prime'
                     WHEN CAST(r.credit_score AS INT) >= 550 THEN 'Standard'
                     WHEN CAST(r.credit_score AS INT) >= 400 THEN 'Sub-Standard' ELSE 'High Risk' END AS new_tier,
                COUNT(*) AS companies
            FROM {raw} r JOIN {silver} s ON r.policy_id = s.policy_id
            GROUP BY 1, 2
            ORDER BY 1, 2
        """)
        score_shift = await execute_query(f"""
            SELECT
                ROUND(AVG(CAST(s.credit_score AS DOUBLE)), 1) AS old_avg_score,
                ROUND(AVG(CAST(r.credit_score AS DOUBLE)), 1) AS new_avg_score,
                SUM(CASE WHEN CAST(r.credit_score AS INT) < CAST(s.credit_score AS INT) THEN 1 ELSE 0 END) AS worsened,
                SUM(CASE WHEN CAST(r.credit_score AS INT) > CAST(s.credit_score AS INT) THEN 1 ELSE 0 END) AS improved,
                SUM(CASE WHEN CAST(r.credit_score AS INT) = CAST(s.credit_score AS INT) THEN 1 ELSE 0 END) AS unchanged
            FROM {raw} r JOIN {silver} s ON r.policy_id = s.policy_id
        """)
        return {
            "score_type": "Credit Score",
            "tier_migration": migration,
            "score_shift": score_shift[0] if score_shift else {},
        }

    else:  # market
        score_shift = await execute_query(f"""
            SELECT
                ROUND(AVG(CAST(s.market_median_rate AS DOUBLE)), 2) AS old_avg_score,
                ROUND(AVG(CAST(r.market_median_rate AS DOUBLE)), 2) AS new_avg_score,
                SUM(CASE WHEN CAST(r.market_median_rate AS DOUBLE) > CAST(s.market_median_rate AS DOUBLE) THEN 1 ELSE 0 END) AS worsened,
                SUM(CASE WHEN CAST(r.market_median_rate AS DOUBLE) < CAST(s.market_median_rate AS DOUBLE) THEN 1 ELSE 0 END) AS improved,
                SUM(CASE WHEN CAST(r.market_median_rate AS DOUBLE) = CAST(s.market_median_rate AS DOUBLE) THEN 1 ELSE 0 END) AS unchanged
            FROM {raw} r JOIN {silver} s ON r.match_key_sic_region = s.match_key_sic_region
        """)
        return {
            "score_type": "Market Median Rate",
            "tier_migration": [],
            "score_shift": score_shift[0] if score_shift else {},
        }


# Download impact report
@router.get("/{dataset_id}/impact/download")
async def download_impact_report(dataset_id: str):
    """Export the policy-level impact analysis as CSV."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    # Get the full impact data
    impact_data = await get_dataset_impact(dataset_id)
    pi = impact_data.get("portfolio_impact", {})
    flagged = pi.get("flagged_policies", [])
    by_industry = pi.get("by_industry", [])
    by_region = pi.get("by_region", [])

    output = io.StringIO()
    output.write(f"Impact Report: {impact_data.get('display_name', dataset_id)}\n")
    output.write(f"Generated: {datetime.utcnow().isoformat()}\n\n")

    output.write(f"Total Policies,{pi.get('total_policies', 0)}\n")
    output.write(f"Affected Policies,{pi.get('affected_policies', 0)}\n")
    output.write(f"Affected %,{pi.get('affected_pct', 0)}%\n")
    output.write(f"Total Premium Delta,{pi.get('premium_delta_total', 0)}\n")
    output.write(f"Avg Premium Delta,{pi.get('premium_delta_avg', 0)}\n\n")

    if by_industry:
        output.write("By Industry\n")
        output.write("Industry,Policies,GWP,Total Delta\n")
        for row in by_industry:
            output.write(f"{row['industry']},{row['policies']},{round(row['gwp'])},{round(row['total_delta'])}\n")
        output.write("\n")

    if flagged:
        output.write("Flagged Policies (>10% change)\n")
        output.write("Policy ID,Postcode,Industry,Current Premium,Delta,Delta %\n")
        for row in flagged:
            output.write(f"{row['policy_id']},{row['postcode']},{row['industry']},"
                         f"{row['current_premium']},{row['premium_delta']},{row['delta_pct']}%\n")

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"impact_report_{dataset_id}_{datetime.utcnow().strftime('%Y%m%d')}.csv"

    await log_audit_event(
        event_type="manual_download",
        entity_type="dataset",
        entity_id=dataset_id,
        entity_version="impact_report",
        details={"report_type": "impact_analysis", "filename": filename,
                 "affected_policies": pi.get("affected_policies", 0)},
    )

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# 4. Data quality expectations and freshness
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/quality")
async def get_dataset_quality(dataset_id: str):
    """Return DQ expectations results, freshness, and completeness metrics."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    ds = EXTERNAL_DATASETS[dataset_id]
    if ds.get("is_reference"):
        # Reference/internal datasets: no raw→silver split, so we just report
        # a row count + per-column completeness on the (single) table.
        table = fqn(ds['silver_table'])
        cat, sch, tbl = table.split('.')
        cols_q = await execute_query(f"""
            SELECT column_name FROM information_schema.columns
            WHERE table_catalog='{cat}' AND table_schema='{sch}' AND table_name='{tbl}'
              AND column_name NOT LIKE '\\_%'
            ORDER BY ordinal_position
        """)
        columns = [c["column_name"] for c in cols_q][:15]
        row_q = await execute_query(f"SELECT count(*) AS n FROM {table}")
        row_count = int(row_q[0]["n"]) if row_q else 0
        completeness = {}
        if columns and row_count > 0:
            null_checks = ", ".join(
                [f"round(count({c}) / count(*) * 100, 1) as `{c}`" for c in columns]
            )
            comp_q = await execute_query(f"SELECT {null_checks} FROM {table}")
            completeness = comp_q[0] if comp_q else {}
        return {
            "is_reference": True,
            "category":          ds.get("category", "reference_data"),
            "raw_row_count":     row_count,
            "silver_row_count":  row_count,
            "rows_dropped":      0,
            "dq_pass_rate":      100,
            "last_ingested":     None,
            "freshness_status":  "fresh",
            "completeness":      completeness,
            "expectations":      [],
        }
    raw_table = fqn(ds["raw_table"])
    silver_table = fqn(ds["silver_table"])

    # Row counts and pass rate
    counts = await execute_query(f"""
        SELECT
            (SELECT count(*) FROM {raw_table}) as raw_count,
            (SELECT count(*) FROM {silver_table}) as silver_count,
            (SELECT max(_ingested_at) FROM {raw_table}) as last_ingested
    """)

    raw_count = int(counts[0]["raw_count"]) if counts else 0
    silver_count = int(counts[0]["silver_count"]) if counts else 0
    last_ingested = counts[0].get("last_ingested") if counts else None
    dq_pass_rate = round(silver_count / raw_count * 100, 1) if raw_count > 0 else 0

    # Column-level completeness from silver
    cols_result = await execute_query(f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = '{silver_table.split('.')[0]}'
          AND table_schema = '{silver_table.split('.')[1]}'
          AND table_name = '{silver_table.split('.')[2]}'
          AND column_name NOT LIKE '\\_%'
        ORDER BY ordinal_position
    """)
    silver_columns = [r["column_name"] for r in cols_result]

    # Completeness per column
    null_checks = ", ".join(
        [f"round(count({c}) / count(*) * 100, 1) as `{c}`" for c in silver_columns[:15]]
    )
    completeness = await execute_query(f"SELECT {null_checks} FROM {silver_table}")

    # Dataset-specific DQ expectations
    expectations = _get_expectations(dataset_id, raw_count, silver_count)

    return {
        "dataset_id": dataset_id,
        "raw_row_count": raw_count,
        "silver_row_count": silver_count,
        "rows_dropped": raw_count - silver_count,
        "dq_pass_rate": dq_pass_rate,
        "last_ingested": last_ingested,
        "freshness_status": "fresh" if last_ingested else "stale",
        "completeness": completeness[0] if completeness else {},
        "expectations": expectations,
    }


def _get_expectations(dataset_id: str, raw_count: int, silver_count: int) -> list[dict]:
    """Return the DQ expectations applied to this dataset."""
    dropped = raw_count - silver_count

    if dataset_id == "market_pricing_benchmark":
        return [
            {"name": "valid_median_rate", "rule": "market_median_rate IS NOT NULL AND > 0", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_competitor_min", "rule": "competitor_a_min_premium IS NOT NULL AND > 0", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_price_trend", "rule": "price_index_trend BETWEEN -50 AND 50", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_match_key", "rule": "match_key_sic_region IS NOT NULL", "action": "DROP ROW", "status": "enforced"},
        ]
    elif dataset_id == "geospatial_hazard_enrichment":
        return [
            {"name": "valid_postcode", "rule": "postcode_sector IS NOT NULL", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_flood_zone", "rule": "flood_zone_rating BETWEEN 1 AND 10", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_fire_distance", "rule": "proximity_to_fire_station_km >= 0", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_crime_index", "rule": "crime_theft_index IS NOT NULL AND >= 0", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_subsidence", "rule": "subsidence_risk BETWEEN 0 AND 10", "action": "DROP ROW", "status": "enforced"},
        ]
    else:
        return [
            {"name": "valid_company_id", "rule": "company_id IS NOT NULL", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_policy_id", "rule": "policy_id IS NOT NULL", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_credit_score", "rule": "credit_score BETWEEN 200 AND 900", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_ccj_count", "rule": "ccj_count >= 0", "action": "DROP ROW", "status": "enforced"},
            {"name": "valid_years_trading", "rule": "years_trading IS NOT NULL AND >= 0", "action": "DROP ROW", "status": "enforced"},
        ]


# ---------------------------------------------------------------------------
# 5. Approve / reject dataset
# ---------------------------------------------------------------------------

@router.post("/{dataset_id}/approve")
async def approve_dataset(dataset_id: str, req: ApprovalRequest):
    """Record approval/rejection decision for a dataset version."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")
    if EXTERNAL_DATASETS[dataset_id].get("is_reference"):
        raise HTTPException(400, "Reference datasets do not use the approval workflow.")

    if req.decision not in ("approved", "rejected"):
        raise HTTPException(400, "Decision must be 'approved' or 'rejected'")

    await ensure_approvals_table()

    ds = EXTERNAL_DATASETS[dataset_id]
    reviewer = get_current_user()
    now = datetime.utcnow().isoformat()
    approval_id = f"{dataset_id}_{now.replace(':', '').replace('-', '').replace('.', '')}"

    # Get current counts
    raw_stats = await execute_query(f"SELECT count(*) as cnt FROM {fqn(ds['raw_table'])}")
    silver_stats = await execute_query(f"SELECT count(*) as cnt FROM {fqn(ds['silver_table'])}")
    raw_count = int(raw_stats[0]["cnt"]) if raw_stats else 0
    silver_count = int(silver_stats[0]["cnt"]) if silver_stats else 0

    await execute_query(f"""
        INSERT INTO {fqn('dataset_approvals')} VALUES (
            '{approval_id}',
            '{dataset_id}',
            '{req.decision}',
            '{reviewer}',
            '{req.reviewer_notes.replace("'", "''")}',
            current_timestamp(),
            {raw_count},
            {silver_count},
            {raw_count - silver_count}
        )
    """)

    # Capture Delta versions for audit lineage
    upt_version = None
    silver_version = None
    try:
        upt_hist = await execute_query(
            f"SELECT max(version) as v FROM (DESCRIBE HISTORY {fqn('unified_pricing_table_live')} LIMIT 1)"
        )
        upt_version = upt_hist[0]["v"] if upt_hist else None
    except Exception:
        pass
    try:
        silver_hist = await execute_query(
            f"SELECT max(version) as v FROM (DESCRIBE HISTORY {fqn(ds['silver_table'])} LIMIT 1)"
        )
        silver_version = silver_hist[0]["v"] if silver_hist else None
    except Exception:
        pass

    await log_audit_event(
        event_type=f"dataset_{req.decision}",
        entity_type="dataset",
        entity_id=dataset_id,
        entity_version=approval_id,
        user_id=reviewer,
        details={
            "approval_id": approval_id,
            "raw_table": ds["raw_table"],
            "silver_table": ds["silver_table"],
            "raw_row_count": raw_count,
            "silver_row_count": silver_count,
            "rows_dropped_by_dq": raw_count - silver_count,
            "upt_delta_version": upt_version,
            "silver_delta_version": silver_version,
            "reviewer_notes": req.reviewer_notes,
        },
    )

    return {
        "approval_id": approval_id,
        "dataset_id": dataset_id,
        "decision": req.decision,
        "reviewer": reviewer,
        "upt_delta_version": upt_version,
        "silver_delta_version": silver_version,
        "message": f"Dataset {dataset_id} has been {req.decision}."
        + (" Data is ready to merge into the Unified Pricing Table." if req.decision == "approved" else ""),
    }


# ---------------------------------------------------------------------------
# Approval history
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/approvals")
async def get_approval_history(dataset_id: str):
    """Get approval history for a dataset."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    try:
        await ensure_approvals_table()
        history = await execute_query(f"""
            SELECT * FROM {fqn('dataset_approvals')}
            WHERE dataset_name = '{dataset_id}'
            ORDER BY reviewed_at DESC
            LIMIT 20
        """)
    except Exception:
        history = []

    return history


# ---------------------------------------------------------------------------
# 6. Download dataset as CSV
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/download")
async def download_dataset(dataset_id: str, layer: str = Query("silver", enum=["raw", "silver"])):
    """Export the current dataset version as CSV. Logs a manual_download audit event."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    ds = EXTERNAL_DATASETS[dataset_id]
    table = fqn(ds["raw_table"] if layer == "raw" else ds["silver_table"])

    rows = await execute_query(f"SELECT * FROM {table}")
    if not rows:
        raise HTTPException(404, "No data found in table")

    # Build CSV in memory
    output = io.StringIO()
    # Filter out metadata columns for a clean export
    all_cols = list(rows[0].keys())
    cols = [c for c in all_cols if not c.startswith("_")]
    writer = csv.DictWriter(output, fieldnames=cols, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: row.get(c) for c in cols})

    csv_bytes = output.getvalue().encode("utf-8")
    filename = f"{dataset_id}_{layer}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"

    await log_audit_event(
        event_type="manual_download",
        entity_type="dataset",
        entity_id=dataset_id,
        entity_version=layer,
        details={
            "layer": layer,
            "table": table,
            "row_count": len(rows),
            "columns": cols,
            "filename": filename,
        },
    )

    return StreamingResponse(
        io.BytesIO(csv_bytes),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


# ---------------------------------------------------------------------------
# 7. Upload CSV to bronze layer
# ---------------------------------------------------------------------------

@router.post("/{dataset_id}/upload/validate")
async def validate_upload(dataset_id: str, file: UploadFile = File(...)):
    """Validate an uploaded CSV against the expected schema. Returns a preview."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")
    if EXTERNAL_DATASETS[dataset_id].get("is_reference"):
        raise HTTPException(400, "Reference datasets are read-only; uploads are not supported.")

    ds = EXTERNAL_DATASETS[dataset_id]
    expected_cols = ds["expected_columns"]

    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(400, "File is not valid UTF-8 text")

    reader = csv.DictReader(io.StringIO(text))
    csv_cols = reader.fieldnames or []

    # Check columns match
    missing = [c for c in expected_cols if c not in csv_cols]
    extra = [c for c in csv_cols if c not in expected_cols]
    columns_matched = len(missing) == 0

    # Read all rows for count, preview first 20
    all_rows = list(reader)
    preview = all_rows[:20]

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "file_hash": file_hash,
        "row_count": len(all_rows),
        "csv_columns": csv_cols,
        "expected_columns": expected_cols,
        "missing_columns": missing,
        "extra_columns": extra,
        "columns_matched": columns_matched,
        "preview": preview,
        "valid": columns_matched and len(all_rows) > 0,
    }


@router.post("/{dataset_id}/upload/confirm")
async def confirm_upload(
    dataset_id: str,
    file: UploadFile = File(...),
    mode: str = Query("replace", enum=["replace", "append"]),
):
    """Write the uploaded CSV to the bronze table. Logs a manual_upload audit event."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    ds = EXTERNAL_DATASETS[dataset_id]
    expected_cols = ds["expected_columns"]
    raw_table = fqn(ds["raw_table"])

    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()
    text = content.decode("utf-8")

    reader = csv.DictReader(io.StringIO(text))
    csv_cols = reader.fieldnames or []
    missing = [c for c in expected_cols if c not in csv_cols]

    if missing:
        raise HTTPException(400, f"Missing required columns: {missing}")

    all_rows = list(reader)
    if not all_rows:
        raise HTTPException(400, "CSV file contains no data rows")

    # Build INSERT statements in batches
    write_mode = "OVERWRITE" if mode == "replace" else "INTO"
    col_list = ", ".join(expected_cols)

    # Write via INSERT with VALUES (batch of 100)
    inserted = 0
    batch_size = 100
    for i in range(0, len(all_rows), batch_size):
        batch = all_rows[i:i + batch_size]
        values_list = []
        for row in batch:
            vals = []
            for c in expected_cols:
                v = row.get(c, "")
                if v is None or v == "":
                    vals.append("NULL")
                else:
                    vals.append(f"'{v.replace(chr(39), chr(39)+chr(39))}'")
            values_list.append(f"({', '.join(vals)})")

        values_sql = ",\n".join(values_list)
        if inserted == 0 and mode == "replace":
            # First batch: overwrite
            await execute_query(f"""
                INSERT OVERWRITE {raw_table} ({col_list}, _ingested_at, _source_file)
                SELECT {col_list}, current_timestamp() as _ingested_at,
                       'manual_upload:{file.filename}' as _source_file
                FROM (VALUES {values_sql}) AS t({col_list})
            """)
        else:
            await execute_query(f"""
                INSERT INTO {raw_table} ({col_list}, _ingested_at, _source_file)
                SELECT {col_list}, current_timestamp() as _ingested_at,
                       'manual_upload:{file.filename}' as _source_file
                FROM (VALUES {values_sql}) AS t({col_list})
            """)
        inserted += len(batch)

    reviewer = get_current_user()
    await log_audit_event(
        event_type="manual_upload",
        entity_type="dataset",
        entity_id=dataset_id,
        details={
            "original_filename": file.filename,
            "file_hash": file_hash,
            "row_count": len(all_rows),
            "columns_matched": True,
            "upload_mode": mode,
            "target_table": raw_table,
        },
    )

    return {
        "dataset_id": dataset_id,
        "filename": file.filename,
        "file_hash": file_hash,
        "row_count": len(all_rows),
        "mode": mode,
        "target_table": raw_table,
        "message": f"Uploaded {len(all_rows)} rows to {raw_table} ({mode} mode). "
                   "Run the ingestion pipeline to promote to silver.",
    }


# ---------------------------------------------------------------------------
# 8. Upload history (from audit_log)
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/uploads")
async def get_upload_history(dataset_id: str):
    """Get recent manual upload events for this dataset from the audit log."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    try:
        uploads = await execute_query(f"""
            SELECT event_id, event_type, user_id, timestamp, details, source
            FROM {fqn('audit_log')}
            WHERE event_type = 'manual_upload' AND entity_id = '{dataset_id}'
            ORDER BY timestamp DESC
            LIMIT 20
        """)
    except Exception:
        uploads = []

    return uploads
