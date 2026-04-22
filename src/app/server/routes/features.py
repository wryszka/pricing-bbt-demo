"""Feature Store status, catalog, and online-store lifecycle routes."""

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
    """Get the status of the online feature store and UPT."""

    upt_table = fqn("unified_pricing_table_live")
    host = get_workspace_host()

    # UPT stats
    try:
        upt_stats = await execute_query(f"""
            SELECT count(*) as row_count,
                   count(DISTINCT policy_id) as unique_policies
            FROM {upt_table}
        """)
        upt_row_count = int(upt_stats[0]["row_count"]) if upt_stats else 0
        upt_policies = int(upt_stats[0]["unique_policies"]) if upt_stats else 0
    except Exception:
        upt_row_count = 0
        upt_policies = 0

    # Delta version
    try:
        history = await execute_query(f"DESCRIBE HISTORY {upt_table} LIMIT 1")
        delta_version = history[0]["version"] if history else "?"
        last_modified = history[0]["timestamp"] if history else "?"
    except Exception:
        delta_version = "?"
        last_modified = "?"

    # Column count
    try:
        cols = await execute_query(f"""
            SELECT count(*) as cnt FROM information_schema.columns
            WHERE table_catalog = '{upt_table.split('.')[0]}'
              AND table_schema = '{upt_table.split('.')[1]}'
              AND table_name = '{upt_table.split('.')[2]}'
        """)
        col_count = int(cols[0]["cnt"]) if cols else 0
    except Exception:
        col_count = 0

    # Online store status
    online_store = None
    try:
        w = get_workspace_client()
        store = w.feature_store.get_online_store("pricing-upt-online-store")
        online_store = {
            "name": store.name,
            "state": str(store.state).split(".")[-1] if store.state else "UNKNOWN",
            "capacity": store.capacity,
            "created": store.creation_time,
        }
    except Exception as e:
        online_store = {
            "name": "pricing-upt-online-store",
            "state": "NOT_CREATED",
            "message": str(e)[:100],
        }

    # Latency results (from test notebook)
    latency = {}
    try:
        lat_results = await execute_query(
            f"SELECT metric, value FROM {fqn('online_store_latency')}"
        )
        for r in lat_results:
            latency[r["metric"]] = float(r["value"])
    except Exception:
        pass

    # Tags
    tags = {}
    try:
        tag_results = await execute_query(f"""
            SELECT tag_name, tag_value
            FROM {upt_table.split('.')[0]}.information_schema.table_tags
            WHERE schema_name = '{upt_table.split('.')[1]}'
              AND table_name = '{upt_table.split('.')[2]}'
        """)
        for r in tag_results:
            tags[r["tag_name"]] = r["tag_value"]
    except Exception:
        pass

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

    # Enrich each with a row count + approval state + last modified
    for s in sources:
        tbl = s["table"]
        try:
            rows = await execute_query(f"SELECT count(*) AS n FROM {fqn(tbl)}")
            s["row_count"] = int(rows[0]["n"]) if rows else 0
        except Exception:
            s["row_count"] = None

        # Approval state only meaningful for ingested sources
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
            # Not gated by an approval workflow — these are always available
            s["approval"] = {"decision": "system_of_record"} if s["kind"] == "internal" else {"decision": "reference"}

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
    try:
        w = get_workspace_client()
        # Find the build_upt job by name
        jobs = list(w.jobs.list(name="[dev laurence_ryszka] Build Unified Pricing Table (Gold)"))
        if not jobs:
            # Fall back: match by short name prefix
            jobs = [j for j in w.jobs.list() if j.settings and "Build Unified Pricing Table" in (j.settings.name or "")]
        if not jobs:
            raise HTTPException(404, "build_upt job not found in this workspace")
        job = jobs[0]
        run = w.jobs.run_now(job_id=job.job_id)
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
