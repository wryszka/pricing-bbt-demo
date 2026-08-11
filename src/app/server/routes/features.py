"""Feature Store status, catalog, and online-store lifecycle routes."""

import asyncio
import logging

from fastapi import APIRouter, HTTPException

from server.config import fqn, get_catalog, get_schema, get_workspace_client, get_workspace_host
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/features", tags=["features"])

ONLINE_STORE_NAME = "pricing-upt-online-store"
UPT_TABLE_KEY = "unified_pricing_table_live"


@router.get("/status")
async def feature_store_status():
    """Get the status of the online feature store and UPT.
    All SQL + SDK lookups run concurrently via asyncio.gather."""
    import asyncio

    upt_table = fqn("unified_pricing_table_live")
    host = get_workspace_host()
    cat, sch, tbl_name = upt_table.split(".", 2)

    sql_upt_stats  = f"SELECT count(*) AS row_count, count(DISTINCT policy_id) AS unique_policies FROM {upt_table}"
    sql_history    = f"DESCRIBE HISTORY {upt_table} LIMIT 1"
    sql_col_count  = (f"SELECT count(*) AS cnt FROM information_schema.columns "
                      f"WHERE table_catalog = '{cat}' AND table_schema = '{sch}' AND table_name = '{tbl_name}'")
    sql_latency    = f"SELECT metric, value FROM {fqn('online_store_latency')}"
    sql_tags       = (f"SELECT tag_name, tag_value FROM {cat}.information_schema.table_tags "
                      f"WHERE schema_name = '{sch}' AND table_name = '{tbl_name}'")

    async def _online_store():
        try:
            w = get_workspace_client()
            store = await asyncio.to_thread(w.feature_store.get_online_store, "pricing-upt-online-store")
            return {
                "name":     store.name,
                "state":    str(store.state).split(".")[-1] if store.state else "UNKNOWN",
                "capacity": store.capacity,
                "created":  store.creation_time,
            }
        except Exception as e:
            return {"name": "pricing-upt-online-store", "state": "NOT_CREATED",
                    "message": str(e)[:100]}

    async def _safe(q):
        try: return await execute_query(q)
        except Exception: return None

    upt_stats, history, cols, lat_results, tag_results, online_store = await asyncio.gather(
        _safe(sql_upt_stats), _safe(sql_history), _safe(sql_col_count),
        _safe(sql_latency), _safe(sql_tags), _online_store(),
    )

    upt_row_count = int(upt_stats[0]["row_count"])         if upt_stats else 0
    upt_policies  = int(upt_stats[0]["unique_policies"])   if upt_stats else 0
    delta_version = history[0]["version"]                   if history else "?"
    last_modified = history[0]["timestamp"]                 if history else "?"
    col_count     = int(cols[0]["cnt"])                     if cols else 0
    latency       = {r["metric"]: float(r["value"]) for r in (lat_results or [])}
    tags          = {r["tag_name"]: r["tag_value"] for r in (tag_results or [])}

    return {
        "upt": {
            "table": upt_table,
            "row_count": upt_row_count,
            "unique_policies": upt_policies,
            "column_count": col_count,
            "delta_version": delta_version,
            "last_modified": last_modified,
            "primary_key": "policy_id",
            "tags": tags,
            "catalog_url": f"{host}/explore/data/{upt_table.replace('.', '/')}",
        },
        "online_store": online_store,
        "latency": latency,
    }


# ---------------------------------------------------------------------------
# Feature catalog — metadata for every feature in the training feature store
# ---------------------------------------------------------------------------

@router.get("/sources")
async def feature_sources():
    """Every upstream that contributes to the Pricing Feature Table. Shows
    what was approved, row count, freshness — so the "approved source →
    feature table" story is visible.
    Each source has a `kind`:
      - ingested  — external CSV via HITL approval
      - internal  — system-of-record (policies, claims)
      - enrichment — reference data (real UK postcode + derived factors)
    """
    sources: list[dict[str, Any]] = [
        # Ingested external vendors — these carry an approval state
        {"id": "market_pricing_benchmark",       "kind": "ingested",   "title": "Market Pricing Benchmark",
         "table": "silver_market_pricing_benchmark",
         "features_feed": ["market_median_rate", "competitor_a_min_premium", "price_index_trend", "market_position_ratio"]},
        {"id": "geospatial_hazard_enrichment",   "kind": "ingested",   "title": "Geospatial Hazard Enrichment",
         "table": "silver_geospatial_hazard_enrichment",
         "features_feed": ["flood_zone_rating", "proximity_to_fire_station_km", "crime_theft_index", "subsidence_risk", "composite_location_risk"]},
        {"id": "credit_bureau_summary",          "kind": "ingested",   "title": "Credit Bureau Summary",
         "table": "silver_credit_bureau_summary",
         "features_feed": ["credit_score", "ccj_count", "years_trading", "credit_risk_tier", "business_stability_score"]},

        # Internal systems of record
        {"id": "internal_commercial_policies",   "kind": "internal",   "title": "Commercial Policies",
         "table": "internal_commercial_policies",
         "features_feed": ["sic_code", "postcode_sector", "annual_turnover", "sum_insured", "current_premium", "construction_type", "year_built", "building_age_years", "industry_risk_tier"]},
        {"id": "internal_claims_history",        "kind": "internal",   "title": "Claims History",
         "table": "internal_claims_history",
         "features_feed": ["claim_count_5y", "total_incurred_5y", "loss_ratio_5y", "fire_incurred", "flood_incurred", "theft_incurred"]},
        {"id": "quotes",                         "kind": "internal",   "title": "Quote Stream",
         "table": "quotes",
         "features_feed": ["quote_count", "avg_quoted_premium", "competitor_quote_count"]},

        # Reference / enrichment
        {"id": "postcode_enrichment",            "kind": "enrichment", "title": "Postcode Enrichment (real UK public data)",
         "table": "postcode_enrichment",
         "features_feed": ["urban_score", "is_coastal", "deprivation_composite", "imd_decile", "crime_decile"]},
        {"id": "derived_factors",                "kind": "enrichment", "title": "Derived Factors",
         "table": "derived_factors",
         "features_feed": ["urban_score", "neighbourhood_claim_frequency", "deprivation_composite", "is_coastal"]},
    ]

    # Enrich each source with row count + approval state in parallel.
    import asyncio

    async def _enrich(s: dict):
        tbl = s["table"]
        try:
            rows = await execute_query(f"SELECT count(*) AS n FROM {fqn(tbl)}")
            s["row_count"] = int(rows[0]["n"]) if rows else 0
        except Exception:
            s["row_count"] = None

        if s["kind"] == "ingested":
            try:
                apr = await execute_query(f"""
                    SELECT decision, reviewer, reviewed_at
                    FROM {fqn('dataset_approvals')}
                    WHERE dataset_name = '{s['id']}'
                    ORDER BY reviewed_at DESC LIMIT 1
                """)
                s["approval"] = apr[0] if apr else None
            except Exception:
                s["approval"] = None
        else:
            s["approval"] = {"decision": "system_of_record"} if s["kind"] == "internal" else {"decision": "reference"}
        return s

    await asyncio.gather(*[_enrich(s) for s in sources])

    return {
        "sources":      sources,
        "target_table": fqn("unified_pricing_table_live"),
        "target_label": "Pricing Feature Table",
        "note":         "Sources are joined and transformed by the build_upt pipeline into the feature table. Policy_id is the grain, not the identity — the feature table draws from all of these.",
    }


@router.post("/rebuild")
async def rebuild_feature_table():
    """Trigger the build_upt job — runs derive_factors + build_upt + build_feature_catalog.
    This is the 'approved sources → feature table' flow made live."""
    from server.audit import log_audit_event

    def _find_and_run() -> tuple:
        w = get_workspace_client()
        # Match by suffix — the dev bundle target prefixes job names with
        # "[dev <whoami>] ", so an exact lookup on the bare name misses. This
        # works regardless of who deployed or which target.
        jobs = [j for j in w.jobs.list() if j.settings and "Build Unified Pricing Table" in (j.settings.name or "")]
        if not jobs:
            raise HTTPException(404, "build_upt job not found in this workspace")
        job = jobs[0]
        run = w.jobs.run_now(job_id=job.job_id)
        return job, run

    try:
        job, run = await asyncio.to_thread(_find_and_run)
        host = get_workspace_host()
        run_url = f"{host}/jobs/{job.job_id}/runs/{run.run_id}"
        await log_audit_event(
            event_type="feature_table_rebuild",
            entity_type="feature_store",
            entity_id="unified_pricing_table_live",
            details={"triggered_by": "app", "job_id": job.job_id, "run_id": run.run_id},
        )
        return {
            "success":  True,
            "job_id":   job.job_id,
            "run_id":   run.run_id,
            "run_url":  run_url,
            "message":  "build_upt job submitted — the feature table will be rebuilt in ~2-3 minutes.",
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("rebuild failed")
        raise HTTPException(500, f"Rebuild failed: {str(e)[:300]}")


@router.get("/catalog")
async def feature_catalog():
    """Return the feature_catalog table — one row per feature with full provenance.
    Foundation for feature-lineage and audit bolt-ons."""
    try:
        rows = await execute_query(f"""
            SELECT
                feature_name, feature_group, data_type, description,
                source_tables, source_columns, transformation, owner,
                regulatory_sensitive, pii
            FROM {fqn('feature_catalog')}
            ORDER BY feature_group, feature_name
        """)
        groups: dict = {}
        for r in rows:
            g = r.get("feature_group") or "other"
            groups[g] = groups.get(g, 0) + 1
        return {
            "features":    rows,
            "counts_by_group": groups,
            "total":       len(rows),
        }
    except Exception as e:
        logger.warning("feature_catalog query failed: %s", e)
        return {
            "features": [], "counts_by_group": {}, "total": 0,
            "error": f"feature_catalog table missing — run build_feature_catalog. ({str(e)[:120]})",
        }


# ---------------------------------------------------------------------------
# Online store lifecycle — promote (create) / pause (delete)
# ---------------------------------------------------------------------------

@router.post("/online/promote")
async def promote_online():
    """Promote the UPT to the online feature store (Lakebase key-value).
    Creates the online store if it doesn't exist and kicks off a SNAPSHOT publish
    of the UPT. Idempotent."""
    from databricks.sdk.service.ml import OnlineStore, PublishSpec, PublishSpecPublishMode

    upt_table = fqn(UPT_TABLE_KEY)
    steps = []

    try:
        w = get_workspace_client()

        # --- Step 1: ensure store exists ---
        try:
            store = w.feature_store.get_online_store(ONLINE_STORE_NAME)
            state = str(store.state).split(".")[-1] if store.state else "UNKNOWN"
            steps.append(f"Store exists (state: {state}).")
        except Exception:
            store = w.feature_store.create_online_store(
                online_store=OnlineStore(name=ONLINE_STORE_NAME, capacity="CU_1")
            )
            state = str(store.state).split(".")[-1] if store.state else "PROVISIONING"
            steps.append(f"Created online store ({state}) — CU_1 capacity.")

        # --- Step 2: publish UPT to online store (SNAPSHOT) ---
        try:
            result = w.feature_store.publish_table(
                source_table_name=upt_table,
                publish_spec=PublishSpec(
                    online_store=ONLINE_STORE_NAME,
                    online_table_name=upt_table,
                    publish_mode=PublishSpecPublishMode.SNAPSHOT,
                ),
            )
            steps.append(f"Published {upt_table} to {ONLINE_STORE_NAME} (SNAPSHOT).")
        except Exception as pub_err:
            err_s = str(pub_err).lower()
            if "already" in err_s:
                steps.append("UPT was already published to the online store.")
            else:
                steps.append(f"Publish failed: {str(pub_err)[:200]}")

        return {
            "status":       "ok",
            "online_store": ONLINE_STORE_NAME,
            "state":        state,
            "steps":        steps,
            "message":      "Online serving enabled — lookups by policy_id will hit Lakebase.",
        }
    except Exception as e:
        logger.exception("promote_online failed")
        raise HTTPException(500, f"Promote failed: {str(e)[:300]}")


@router.post("/online/pause")
async def pause_online():
    """Delete the online feature store to stop cost. The offline UPT is untouched —
    the online copy can be re-promoted later."""
    try:
        w = get_workspace_client()
        w.feature_store.delete_online_store(ONLINE_STORE_NAME)
        return {
            "status":       "deleted",
            "online_store": ONLINE_STORE_NAME,
            "message":      "Online store deleted. Offline UPT unchanged. Promote again to re-enable low-latency serving.",
        }
    except Exception as e:
        logger.warning("pause_online — assuming already absent: %s", e)
        return {
            "status":       "not_present",
            "online_store": ONLINE_STORE_NAME,
            "message":      "Online store was not provisioned; nothing to pause.",
            "error":        str(e)[:200],
        }


# ---------------------------------------------------------------------------
# Mart Profile — one-shot payload for the Modelling Mart Overview tab.
#
# Returns:
#   • headline tiles (row count, policy count, date range, last refresh,
#     column count, contributing upstream feeds)
#   • factor-group composition (from feature_catalog)
#   • top-10 features by missingness (null rate) on the live mart
#   • coverage breakdown: policies by region + industry_risk_tier
#   • claims sanity block: total claims, frequency, mean severity, loss
#     ratio per industry tier
#   • recent refresh activity (last 5 Delta commits)
#
# All SQL runs through the shared warehouse; designed to complete inside
# the app's fetch timeout even on a cold warehouse.
# ---------------------------------------------------------------------------

# The 6 vendor/internal feeds that compose the mart.
UPT_CONTRIBUTING_FEEDS = [
    "internal_commercial_policies",
    "internal_claims_history",
    "silver_market_pricing_benchmark",
    "silver_geospatial_hazard_enrichment",
    "silver_credit_bureau_summary",
    "postcode_enrichment",
]


@router.get("/mart-profile")
async def mart_profile():
    """Dashboard payload for the Overview tab on the Modelling Mart page.
    All independent SQL queries run concurrently via asyncio.gather."""
    import asyncio

    upt_table = fqn("unified_pricing_table_live")
    catalog_name, schema_name = get_catalog(), get_schema()

    async def _safe(sql):
        try: return await execute_query(sql)
        except Exception as e:
            logger.warning("mart-profile q failed: %s", str(e)[:120]); return None

    sql_headline = f"""
        SELECT COUNT(*)                              AS total_rows,
               COUNT(DISTINCT policy_id)             AS unique_policies,
               MIN(TRY_CAST(inception_date AS DATE)) AS policy_date_min,
               MAX(TRY_CAST(renewal_date   AS DATE)) AS policy_date_max
        FROM {upt_table}
    """
    sql_col_count = f"""
        SELECT COUNT(*) AS n FROM system.information_schema.columns
        WHERE table_catalog = '{catalog_name}' AND table_schema = '{schema_name}'
          AND table_name = 'unified_pricing_table_live'
          AND column_name NOT LIKE '\\_%'
    """
    sql_hist = f"""
        SELECT version, timestamp, userName, operation
        FROM (DESCRIBE HISTORY {upt_table})
        ORDER BY version DESC LIMIT 5
    """
    sql_groups = f"""
        SELECT feature_group, COUNT(*) AS n
        FROM {fqn('feature_catalog')}
        GROUP BY feature_group ORDER BY n DESC
    """
    sql_col_names = f"""
        SELECT column_name FROM system.information_schema.columns
        WHERE table_catalog = '{catalog_name}' AND table_schema = '{schema_name}'
          AND table_name = 'unified_pricing_table_live'
          AND column_name NOT LIKE '\\_%' AND column_name NOT IN ('policy_id')
        ORDER BY ordinal_position
    """
    sql_by_region = f"""
        SELECT COALESCE(region, 'Unknown') AS region, COUNT(*) AS n
        FROM {upt_table} GROUP BY region ORDER BY n DESC
    """
    sql_by_tier = f"""
        SELECT COALESCE(industry_risk_tier, 'Unknown') AS tier, COUNT(*) AS n
        FROM {upt_table} GROUP BY industry_risk_tier ORDER BY n DESC
    """

    # Six independent queries fire in parallel.
    (headline, col_stat, hist, groups, col_rows, by_region, by_tier) = await asyncio.gather(
        _safe(sql_headline), _safe(sql_col_count), _safe(sql_hist),
        _safe(sql_groups), _safe(sql_col_names),
        _safe(sql_by_region), _safe(sql_by_tier),
    )

    head = (headline[0] if headline else {}) or {}
    head["column_count"]         = int(col_stat[0]["n"]) if col_stat else 0
    head["last_refresh"]         = hist[0]["timestamp"]          if hist else None
    head["last_refresh_version"] = hist[0]["version"]            if hist else None
    head["last_refresh_user"]    = hist[0].get("userName")       if hist else None
    head["upstream_feeds_count"] = len(UPT_CONTRIBUTING_FEEDS)

    groups    = groups    or []
    by_region = by_region or []
    by_tier   = by_tier   or []

    # Missingness — chain after col_rows. Exclude claim-tied columns: they're
    # NULL by design for the ~32k policies with no 5-year claim history. We
    # detect them by null-count equality to the canonical claim_count_5y
    # baseline — catches cols the feature_catalog doesn't tag (storm_incurred,
    # last_claim_date, etc.). Without this, the top-10 chart is dominated by
    # one tied bucket and hides the real variety in vendor / enrichment data.
    feat_missing = []
    try:
        cols = [r["column_name"] for r in (col_rows or [])
                if r["column_name"] not in {"policy_id", "_ingested_at"}][:90]
        if cols:
            select_parts = ", ".join(
                f"SUM(CASE WHEN `{c}` IS NULL THEN 1 ELSE 0 END) AS `{c}`" for c in cols
            )
            null_row = await execute_query(
                f"SELECT {select_parts}, COUNT(*) AS __total FROM {upt_table}"
            )
            row = null_row[0] if null_row else {}
            total = int(row.get("__total") or 0) or 1

            # Anchor the "no claim history" baseline — every claim-tied column
            # shares this exact null count. Filter them out.
            claim_baseline = int(row.get("claim_count_5y") or 0)

            # Dedupe by null_count: cols sharing the exact same null count
            # come from a single upstream coverage gap (e.g., five market-data
            # cols all missing on the same ~25k policies because the vendor
            # only licenses half the book). Showing one representative per
            # bucket gives a clean diagnostic gradient instead of plateaus.
            seen_buckets: set[int] = set()
            feat_missing = []
            ranked = sorted(
                ((c, int(row.get(c) or 0)) for c in cols
                 if int(row.get(c) or 0) > 0
                 and int(row.get(c) or 0) != claim_baseline),
                key=lambda x: x[1], reverse=True,
            )
            for c, n in ranked:
                if n in seen_buckets:
                    continue
                seen_buckets.add(n)
                feat_missing.append({
                    "feature_name": c,
                    "null_count":   n,
                    "null_rate":    round(n / total, 4),
                })
                if len(feat_missing) >= 10:
                    break
    except Exception as e:
        logger.warning("mart-profile missingness failed: %s", e)

    # ----- claims sanity: totals, frequency, severity, loss ratio per tier -----
    # IMPORTANT: use the AGGREGATE (premium-weighted) loss ratio, not the mean
    # of per-policy ratios. Per-policy `loss_ratio_5y` has extreme outliers (a
    # £100 policy with £10M claims → LR = 100,000%) that blow up the mean.
    # The actuarially correct portfolio number is total claims £ / total premium £.
    try:
        claims = await execute_query(f"""
            SELECT
                SUM(COALESCE(claim_count_5y, 0))        AS total_claims,
                SUM(COALESCE(total_incurred_5y, 0))     AS total_incurred,
                SUM(COALESCE(current_premium, 0))       AS total_premium,
                AVG(COALESCE(claim_count_5y, 0))        AS avg_freq_5y,
                CASE WHEN SUM(COALESCE(claim_count_5y, 0)) > 0
                     THEN SUM(COALESCE(total_incurred_5y, 0)) * 1.0 / SUM(COALESCE(claim_count_5y, 0))
                     ELSE 0
                END AS mean_severity,
                CASE WHEN SUM(COALESCE(current_premium, 0)) > 0
                     THEN SUM(COALESCE(total_incurred_5y, 0)) * 1.0 / SUM(COALESCE(current_premium, 0))
                     ELSE 0
                END AS portfolio_loss_ratio_5y
            FROM {upt_table}
        """)
        c = claims[0] if claims else {}
    except Exception:
        c = {}

    try:
        # Per-tier aggregate loss ratio — same premium-weighted formula
        lr_by_tier = await execute_query(f"""
            SELECT COALESCE(industry_risk_tier, 'Unknown') AS tier,
                   COUNT(*)                              AS n,
                   SUM(COALESCE(total_incurred_5y, 0))   AS total_incurred,
                   SUM(COALESCE(current_premium, 0))     AS total_premium,
                   ROUND(
                       CASE WHEN SUM(COALESCE(current_premium, 0)) > 0
                            THEN SUM(COALESCE(total_incurred_5y, 0)) * 1.0 / SUM(COALESCE(current_premium, 0))
                            ELSE 0
                       END, 4
                   ) AS loss_ratio,
                   SUM(COALESCE(claim_count_5y, 0))      AS total_claims
            FROM {upt_table}
            GROUP BY industry_risk_tier
            ORDER BY n DESC
        """)
    except Exception:
        lr_by_tier = []

    return {
        "upt_table": upt_table,
        "headline": head,
        "factor_groups": groups,
        "feature_health": {
            "top_missingness": feat_missing,
            "evaluated_columns": len(cols) if 'cols' in dir() else 0,
        },
        "coverage": {
            "by_region": by_region,
            "by_industry_tier": by_tier,
        },
        "claims": {
            "total_claims":            int(c.get("total_claims") or 0),
            "total_incurred":          float(c.get("total_incurred") or 0),
            "total_premium":           float(c.get("total_premium") or 0),
            "avg_freq_5y":             float(c.get("avg_freq_5y") or 0),
            "mean_severity":           float(c.get("mean_severity") or 0),
            "portfolio_loss_ratio_5y": float(c.get("portfolio_loss_ratio_5y") or 0),
            "loss_ratio_by_tier":      lr_by_tier,
        },
        "recent_activity": {
            "refreshes": [
                {"version": h.get("version"),  "timestamp": h.get("timestamp"),
                 "user": h.get("userName"),    "operation": h.get("operation")}
                for h in (hist or [])
            ],
        },
        "upstream_feeds": UPT_CONTRIBUTING_FEEDS,
    }
