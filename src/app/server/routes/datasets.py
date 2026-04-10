"""Dataset ingestion review and approval routes.

Provides endpoints for:
1. Listing external datasets available for review
2. Diff between current and pending versions (new/changed/removed rows)
3. Impact analytics (pricing impact simulation)
4. Data quality expectations and freshness
5. Approve/reject workflow
"""

import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event

from server.config import fqn, get_current_user
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/datasets", tags=["datasets"])

# The three external datasets we ingest
EXTERNAL_DATASETS = {
    "market_pricing_benchmark": {
        "display_name": "Market Pricing Benchmark",
        "source": "External Vendor (PCW)",
        "join_key": "sic_code + region",
        "raw_table": "raw_market_pricing_benchmark",
        "silver_table": "silver_market_pricing_benchmark",
        "description": "Aggregated competitor pricing data by industry and region",
    },
    "geospatial_hazard_enrichment": {
        "display_name": "Geospatial Hazard Enrichment",
        "source": "External Vendor (OS/EA)",
        "join_key": "postcode_sector",
        "raw_table": "raw_geospatial_hazard_enrichment",
        "silver_table": "silver_geospatial_hazard_enrichment",
        "description": "Location-based risk scores: flood, fire, crime, subsidence",
    },
    "credit_bureau_summary": {
        "display_name": "Credit Bureau Summary",
        "source": "Bureau (D&B/Experian)",
        "join_key": "policy_id",
        "raw_table": "raw_credit_bureau_summary",
        "silver_table": "silver_credit_bureau_summary",
        "description": "Company financial health: credit score, CCJs, years trading",
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

@router.get("")
async def list_datasets():
    """List all external datasets with their current status."""
    results = []
    for ds_id, ds_info in EXTERNAL_DATASETS.items():
        # Get row counts and freshness
        try:
            raw_stats = await execute_query(f"""
                SELECT count(*) as row_count,
                       max(_ingested_at) as last_ingested
                FROM {fqn(ds_info['raw_table'])}
            """)
            silver_stats = await execute_query(f"""
                SELECT count(*) as row_count
                FROM {fqn(ds_info['silver_table'])}
            """)
            # Get latest approval
            approval = await execute_query(f"""
                SELECT decision, reviewer, reviewed_at, reviewer_notes
                FROM {fqn('dataset_approvals')}
                WHERE dataset_name = '{ds_id}'
                ORDER BY reviewed_at DESC
                LIMIT 1
            """)
        except Exception as e:
            logger.warning("Failed to query stats for %s: %s", ds_id, e)
            raw_stats = [{"row_count": "0", "last_ingested": None}]
            silver_stats = [{"row_count": "0"}]
            approval = []

        raw_count = int(raw_stats[0]["row_count"]) if raw_stats else 0
        silver_count = int(silver_stats[0]["row_count"]) if silver_stats else 0
        last_ingested = raw_stats[0].get("last_ingested") if raw_stats else None

        results.append({
            "id": ds_id,
            **ds_info,
            "raw_row_count": raw_count,
            "silver_row_count": silver_count,
            "rows_dropped_by_dq": raw_count - silver_count,
            "last_ingested": last_ingested,
            "approval": approval[0] if approval else None,
        })

    return results


# ---------------------------------------------------------------------------
# 2. Diff: new/changed/removed rows between raw and silver
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/diff")
async def get_dataset_diff(dataset_id: str):
    """Show difference between raw (pending) and silver (current approved) data."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    ds = EXTERNAL_DATASETS[dataset_id]
    raw_table = fqn(ds["raw_table"])
    silver_table = fqn(ds["silver_table"])

    # Get column lists (exclude metadata columns)
    cols_result = await execute_query(f"""
        SELECT column_name FROM information_schema.columns
        WHERE table_catalog = '{raw_table.split('.')[0]}'
          AND table_schema = '{raw_table.split('.')[1]}'
          AND table_name = '{raw_table.split('.')[2]}'
          AND column_name NOT LIKE '\\_%'
        ORDER BY ordinal_position
    """)
    raw_columns = [r["column_name"] for r in cols_result]

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

    # Summary counts
    summary = await execute_query(f"""
        SELECT
            (SELECT count(*) FROM {raw_table}) as raw_total,
            (SELECT count(*) FROM {silver_table}) as silver_total,
            (SELECT count(*) FROM {raw_table} r
             LEFT ANTI JOIN {silver_table} s ON r.{key_col} = s.{key_col}) as new_rows,
            (SELECT count(*) FROM {silver_table} s
             LEFT ANTI JOIN {raw_table} r ON s.{key_col} = r.{key_col}) as removed_rows
    """)

    # Sample of changed rows (where values differ)
    compare_conditions = " OR ".join(
        [f"CAST(r.{c} AS STRING) != CAST(s.{c} AS STRING)" for c in compare_cols]
    )
    changed_sample = await execute_query(f"""
        SELECT r.{key_col},
               {', '.join([f"s.{c} as old_{c}, r.{c} as new_{c}" for c in compare_cols])}
        FROM {raw_table} r
        INNER JOIN {silver_table} s ON r.{key_col} = s.{key_col}
        WHERE {compare_conditions}
        LIMIT 50
    """)

    # Sample of new rows
    new_sample = await execute_query(f"""
        SELECT r.*
        FROM {raw_table} r
        LEFT ANTI JOIN {silver_table} s ON r.{key_col} = s.{key_col}
        LIMIT 20
    """)

    # Sample of removed rows
    removed_sample = await execute_query(f"""
        SELECT s.*
        FROM {silver_table} s
        LEFT ANTI JOIN {raw_table} r ON s.{key_col} = r.{key_col}
        LIMIT 20
    """)

    return {
        "dataset_id": dataset_id,
        "key_column": key_col,
        "compare_columns": compare_cols,
        "summary": summary[0] if summary else {},
        "changed_rows": changed_sample,
        "new_rows": new_sample,
        "removed_rows": removed_sample,
        "changed_count": len(changed_sample),
    }


# ---------------------------------------------------------------------------
# 3. Impact analytics — simulated pricing impact
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/impact")
async def get_dataset_impact(dataset_id: str):
    """Simulate the pricing impact of merging this dataset into the UPT."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    upt_table = fqn("unified_pricing_table_live")
    policies_table = fqn("internal_commercial_policies")

    if dataset_id == "market_pricing_benchmark":
        # Impact: how market position changes affect competitiveness
        impact = await execute_query(f"""
            WITH current_book AS (
                SELECT
                    count(*) as total_policies,
                    sum(current_premium) as total_gwp,
                    avg(current_premium) as avg_premium,
                    count(CASE WHEN renewal_date <= date_add(current_date(), 90) THEN 1 END) as renewals_next_90d,
                    sum(CASE WHEN renewal_date <= date_add(current_date(), 90) THEN current_premium ELSE 0 END) as renewal_gwp_next_90d
                FROM {policies_table}
            ),
            market_position AS (
                SELECT
                    count(*) as policies_with_market_data,
                    avg(market_position_ratio) as avg_market_position,
                    count(CASE WHEN market_position_ratio > 1.2 THEN 1 END) as overpriced_count,
                    count(CASE WHEN market_position_ratio < 0.8 THEN 1 END) as underpriced_count,
                    sum(CASE WHEN market_position_ratio > 1.2 THEN current_premium ELSE 0 END) as overpriced_gwp,
                    sum(CASE WHEN market_position_ratio < 0.8 THEN current_premium ELSE 0 END) as underpriced_gwp
                FROM {upt_table}
                WHERE market_position_ratio IS NOT NULL
            )
            SELECT * FROM current_book CROSS JOIN market_position
        """)

        return {
            "dataset_id": dataset_id,
            "impact_type": "Market Competitiveness",
            "summary": impact[0] if impact else {},
            "insights": [
                {
                    "title": "Competitive Positioning",
                    "description": "Market pricing data enables identification of over/under-priced segments",
                    "severity": "high",
                },
                {
                    "title": "Renewal Risk",
                    "description": "Policies renewing in next 90 days can be re-rated with fresh market intelligence",
                    "severity": "medium",
                },
                {
                    "title": "New Business Targeting",
                    "description": "Identifies segments where we can competitively price to win business",
                    "severity": "medium",
                },
            ],
        }

    elif dataset_id == "geospatial_hazard_enrichment":
        # Impact: how location risk changes affect the book
        impact = await execute_query(f"""
            SELECT
                count(*) as total_policies,
                sum(current_premium) as total_gwp,
                location_risk_tier,
                count(*) as tier_count,
                avg(current_premium) as avg_premium,
                avg(composite_location_risk) as avg_risk_score,
                sum(CASE WHEN flood_zone_rating >= 7 THEN 1 ELSE 0 END) as high_flood_risk,
                sum(CASE WHEN subsidence_risk >= 7 THEN 1 ELSE 0 END) as high_subsidence_risk
            FROM {upt_table}
            WHERE location_risk_tier IS NOT NULL
            GROUP BY location_risk_tier
            ORDER BY location_risk_tier
        """)

        # Overall portfolio impact
        portfolio = await execute_query(f"""
            SELECT
                count(*) as total_policies,
                sum(current_premium) as total_gwp,
                count(CASE WHEN composite_location_risk >= 6.0 THEN 1 END) as high_risk_policies,
                sum(CASE WHEN composite_location_risk >= 6.0 THEN current_premium ELSE 0 END) as high_risk_gwp,
                avg(composite_location_risk) as avg_location_risk,
                count(CASE WHEN flood_zone_rating >= 8 AND current_premium < 5000 THEN 1 END) as potentially_underpriced
            FROM {upt_table}
            WHERE composite_location_risk IS NOT NULL
        """)

        return {
            "dataset_id": dataset_id,
            "impact_type": "Location Risk Assessment",
            "by_tier": impact,
            "portfolio": portfolio[0] if portfolio else {},
            "insights": [
                {
                    "title": "Hidden Risk Exposure",
                    "description": "Geospatial data reveals policies in high-risk zones that may be underpriced",
                    "severity": "high",
                },
                {
                    "title": "Premium Adequacy",
                    "description": "Location scoring enables granular risk-based pricing adjustments",
                    "severity": "medium",
                },
                {
                    "title": "Portfolio Concentration",
                    "description": "Identifies geographic concentration risk for reinsurance planning",
                    "severity": "low",
                },
            ],
        }

    else:  # credit_bureau
        impact = await execute_query(f"""
            SELECT
                count(*) as total_policies,
                sum(current_premium) as total_gwp,
                credit_risk_tier,
                count(*) as tier_count,
                avg(current_premium) as avg_premium,
                avg(credit_score) as avg_credit_score,
                avg(business_stability_score) as avg_stability
            FROM {upt_table}
            WHERE credit_risk_tier IS NOT NULL
            GROUP BY credit_risk_tier
            ORDER BY credit_risk_tier
        """)

        portfolio = await execute_query(f"""
            SELECT
                count(*) as total_policies,
                sum(current_premium) as total_gwp,
                count(CASE WHEN credit_risk_tier = 'High Risk' THEN 1 END) as high_risk_count,
                sum(CASE WHEN credit_risk_tier = 'High Risk' THEN current_premium ELSE 0 END) as high_risk_gwp,
                count(CASE WHEN credit_risk_tier = 'Prime' THEN 1 END) as prime_count,
                avg(business_stability_score) as avg_stability
            FROM {upt_table}
            WHERE credit_risk_tier IS NOT NULL
        """)

        return {
            "dataset_id": dataset_id,
            "impact_type": "Credit Risk Profiling",
            "by_tier": impact,
            "portfolio": portfolio[0] if portfolio else {},
            "insights": [
                {
                    "title": "Default Risk Segmentation",
                    "description": "Bureau data enables credit-based pricing differentiation",
                    "severity": "high",
                },
                {
                    "title": "Collection Risk",
                    "description": "Identifies policyholders with high CCJ counts for premium collection risk",
                    "severity": "medium",
                },
                {
                    "title": "Cross-sell Opportunity",
                    "description": "Prime-rated businesses are candidates for expanded coverage",
                    "severity": "low",
                },
            ],
        }


# ---------------------------------------------------------------------------
# 4. Data quality expectations and freshness
# ---------------------------------------------------------------------------

@router.get("/{dataset_id}/quality")
async def get_dataset_quality(dataset_id: str):
    """Return DQ expectations results, freshness, and completeness metrics."""
    if dataset_id not in EXTERNAL_DATASETS:
        raise HTTPException(404, f"Unknown dataset: {dataset_id}")

    ds = EXTERNAL_DATASETS[dataset_id]
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
