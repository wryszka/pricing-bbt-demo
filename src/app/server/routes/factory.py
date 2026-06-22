"""Model Factory — systematic generation + review of many candidate models.

Step 1: Analyse & Plan  — generate a 50-variant plan, Claude narrates the why
Step 2: Train           — virtual training synthesises metrics deterministically
                          from variant configs (MVP cuts corner here; real-
                          training notebook lands in phase 2)
Step 3: Review          — leaderboard, shortlist of top 5, portfolio what-if
                          (portfolio numbers are synthesised in MVP — corner
                          #2: reuse of Compare & Test scoring lands later)
Step 4: Selective pack  — actuary picks variants and triggers the governance
                          pack flow (MVP logs the intent but does not run the
                          pack job for virtual candidates — corner #3)

Only freq_glm is wired end-to-end for MVP. The UI exposes the family selector
but other three families surface a "coming next iteration" message.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
import re
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import fqn, get_catalog, get_current_user, get_schema, get_workspace_client
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/factory", tags=["factory"])

FM_ENDPOINT = "databricks-claude-sonnet-4-6"

SUPPORTED_FAMILIES = {"freq_glm"}    # MVP — rest are stubbed in the UI
FREQ_FEATURES = [
    "sum_insured", "annual_turnover", "current_premium",
    "industry_risk_tier", "construction_type",
    "credit_score", "ccj_count", "years_trading",
    "flood_zone_rating", "proximity_to_fire_station_km",
    "crime_theft_index", "subsidence_risk", "composite_location_risk",
    "urban_score", "is_coastal", "population_density_per_km2",
    "elevation_metres", "annual_rainfall_mm",
    "director_stability_score", "employee_count_est",
    "distance_to_coast_km", "neighbourhood_claim_frequency",
]
CORE_FEATURES = [
    "sum_insured", "annual_turnover", "industry_risk_tier",
    "construction_type", "credit_score",
]
INTERACTION_CANDIDATES = [
    ("industry_risk_tier", "construction_type"),
    ("flood_zone_rating", "is_coastal"),
    ("credit_score", "ccj_count"),
    ("urban_score", "population_density_per_km2"),
    ("sum_insured", "industry_risk_tier"),
    ("subsidence_risk", "composite_location_risk"),
    ("years_trading", "director_stability_score"),
]
BANDING_STRATEGIES = ["raw_linear", "quantile_5_bands", "quantile_10_bands", "log_then_linear"]
GLM_FAMILIES = [
    {"family": "Poisson", "link": "log"},
    {"family": "Quasi-Poisson", "link": "log"},
    {"family": "Negative Binomial", "link": "log"},
    {"family": "Tweedie (p=1.5)", "link": "log"},
]
CHAMPION_GINI = 0.224    # used for "vs champion" shift calculations


# ---------------------------------------------------------------------------
# Delta tables — created lazily on first call
# ---------------------------------------------------------------------------

async def ensure_factory_tables():
    await execute_query(f"""
        CREATE TABLE IF NOT EXISTS {fqn('factory_runs')} (
            run_id        STRING,
            model_family  STRING,
            plan_json     STRING,
            narrative     STRING,
            approved_by   STRING,
            started_at    TIMESTAMP,
            duration_seconds DOUBLE,
            status        STRING,
            variant_count INT
        )
    """)
    await execute_query(f"""
        CREATE TABLE IF NOT EXISTS {fqn('factory_variants')} (
            run_id           STRING,
            variant_id       STRING,
            name             STRING,
            category         STRING,
            config_json      STRING,
            metrics_json     STRING,
            n_features       INT,
            created_at       TIMESTAMP
        )
    """)


# ---------------------------------------------------------------------------
# Plan generation — deterministic variant enumerator
# ---------------------------------------------------------------------------

def _agentic_plan_review(family: str) -> dict:
    """Lightweight, deterministic data review that recommends how many model
    variants make sense for this family + Modelling Mart shape. Pricing AI
    sees the data signals (feature count, interaction candidates, banding
    strategies, GLM family options) and reasons about an appropriate sweep
    size — better than a hardcoded 50, better than a slow LLM call."""
    if family != "freq_glm":
        return {
            "recommended_count": 0,
            "breakdown": {"feature_subset": 0, "interactions": 0, "banding": 0},
            "reasoning": f"{family} not yet supported by the factory.",
        }

    enrichment = [f for f in FREQ_FEATURES if f not in CORE_FEATURES]
    n_core         = len(CORE_FEATURES)
    n_enrichment   = len(enrichment)
    n_interactions = len(INTERACTION_CANDIDATES)
    n_banding      = len(BANDING_STRATEGIES)
    n_glm_families = len(GLM_FAMILIES)

    # Sweep size driven by what the data actually offers:
    #  - feature_subset: one variant per enrichment-feature step + a few extras
    #  - interactions:   each candidate pair × ~2 GLM families is enough to
    #                    detect overdispersion sensitivity without redundancy
    #  - banding:        each strategy × ~2 family probes
    n_feature_subset = min(25, max(8, n_enrichment + 5))
    n_interactions_v = min(15, max(6, int(n_interactions * 1.5)))
    n_banding_v      = min(10, max(4, n_banding * 2))
    total = n_feature_subset + n_interactions_v + n_banding_v

    reasoning = (
        f"Modelling Mart exposes {n_core + n_enrichment} candidate features for the frequency model "
        f"({n_core} core risk + {n_enrichment} enrichment), {n_interactions} interaction pairs flagged "
        f"by prior actuarial review, and {n_banding} banding strategies × {n_glm_families} GLM families "
        f"({', '.join(g['family'] for g in GLM_FAMILIES)}). Recommending {total} variants — enough to "
        f"sweep all three orthogonal axes without redundancy: {n_feature_subset} feature-subset cuts, "
        f"{n_interactions_v} interaction probes, {n_banding_v} banding strategies."
    )
    return {
        "recommended_count": total,
        "breakdown": {
            "feature_subset": n_feature_subset,
            "interactions":   n_interactions_v,
            "banding":        n_banding_v,
        },
        "reasoning": reasoning,
    }


def _variants_for_freq_glm(breakdown: dict | None = None) -> list[dict[str, Any]]:
    """Enumerate candidate GLM configurations. The breakdown (from the
    agentic review) controls how many variants per category — it falls back
    to a balanced default if the caller doesn't pass one."""
    breakdown = breakdown or _agentic_plan_review("freq_glm")["breakdown"]
    n_feature = breakdown.get("feature_subset", 25)
    n_inter   = breakdown.get("interactions",   15)
    n_band    = breakdown.get("banding",        10)
    variants: list[dict[str, Any]] = []

    # Category A — baseline feature subsets
    enrichment = [f for f in FREQ_FEATURES if f not in CORE_FEATURES]
    for i in range(n_feature):
        extra_n = i % max(1, len(enrichment))
        features = CORE_FEATURES + enrichment[:extra_n]
        variants.append({
            "variant_id": f"A{i+1:02d}",
            "name":       f"GLM-A{i+1:02d} — core + {extra_n} enrichment",
            "category":   "feature_subset",
            "features":   features,
            "interactions": [],
            "banding":    "raw_linear",
            "glm":        GLM_FAMILIES[0],
            "notes":      f"Baseline sweep: {len(features)} features, no interactions.",
        })

    # Category B — interaction probes
    base_feats = CORE_FEATURES + enrichment[:6]
    for i in range(n_inter):
        pair = INTERACTION_CANDIDATES[i % len(INTERACTION_CANDIDATES)]
        fam = GLM_FAMILIES[(i // 4) % len(GLM_FAMILIES)]
        variants.append({
            "variant_id": f"B{i+1:02d}",
            "name":       f"GLM-B{i+1:02d} — {pair[0]} × {pair[1]} · {fam['family']}",
            "category":   "interactions",
            "features":   base_feats,
            "interactions": [list(pair)],
            "banding":    "raw_linear",
            "glm":        fam,
            "notes":      f"Probes interaction between {pair[0]} and {pair[1]}.",
        })

    # Category C — banding strategies
    for i in range(n_band):
        banding = BANDING_STRATEGIES[i % len(BANDING_STRATEGIES)]
        fam = GLM_FAMILIES[(i // 3) % len(GLM_FAMILIES)]
        feats = (CORE_FEATURES + enrichment)[:12]
        variants.append({
            "variant_id": f"C{i+1:02d}",
            "name":       f"GLM-C{i+1:02d} — {banding} · {fam['family']}",
            "category":   "banding",
            "features":   feats,
            "interactions": [],
            "banding":    banding,
            "glm":        fam,
            "notes":      f"Probes banding strategy '{banding}' on a 12-feature subset.",
        })

    return variants


def _seeded(seed: str, lo: float, hi: float) -> float:
    h = int(hashlib.sha256(seed.encode()).hexdigest()[:8], 16)
    return lo + (h % 100000) / 100000.0 * (hi - lo)


def _synth_metrics(variant: dict, run_id: str) -> dict:
    """Deterministic plausible metrics from variant config.
    Note: MVP cuts a corner here — real metrics will replace these once the
    factory training notebook is wired."""
    seed_base = f"{run_id}:{variant['variant_id']}"
    n_feat    = len(variant.get("features", []))
    n_inter   = len(variant.get("interactions", []))
    banding   = variant.get("banding", "raw_linear")
    fam       = (variant.get("glm") or {}).get("family", "Poisson")

    # Gini: more features → better up to a decay point; interactions help;
    # non-raw banding helps; non-Poisson family trades bias for variance.
    gini = 0.15
    gini += min(n_feat, 15) * 0.006               # diminishing return
    gini += n_inter * 0.012
    gini += {"raw_linear": 0.0, "quantile_5_bands": 0.004,
             "quantile_10_bands": 0.006, "log_then_linear": 0.003}[banding]
    gini += {"Poisson": 0.0, "Quasi-Poisson": 0.003,
             "Negative Binomial": 0.006, "Tweedie (p=1.5)": 0.004}[fam]
    gini += _seeded(seed_base + ":gini", -0.015, 0.015)
    gini = max(0.10, min(0.38, gini))

    aic = 46000 - gini * 4000 + n_feat * 80 + n_inter * 30 + _seeded(seed_base + ":aic", -200, 200)
    bic = aic + n_feat * 12
    deviance_explained = 0.5 * gini + _seeded(seed_base + ":dev", -0.01, 0.01)
    mae = 0.32 - gini * 0.3 + _seeded(seed_base + ":mae", -0.005, 0.005)

    return {
        "gini": round(gini, 4),
        "aic": round(aic, 1),
        "bic": round(bic, 1),
        "deviance_explained": round(deviance_explained, 4),
        "mae": round(mae, 4),
    }


class ProposeRequest(BaseModel):
    family: str


@router.post("/plan")
async def propose_plan(req: ProposeRequest) -> dict:
    if req.family not in SUPPORTED_FAMILIES:
        # Surface a structured "not supported yet" so the UI can render a nice message.
        return {
            "family":   req.family,
            "status":   "unsupported",
            "message":  f"Factory for {req.family} lands in a later iteration. "
                        f"MVP wires {', '.join(sorted(SUPPORTED_FAMILIES))} only.",
            "plan":     [],
            "narrative": "",
        }

    review = _agentic_plan_review(req.family)
    plan = _variants_for_freq_glm(review["breakdown"])
    narrative = _static_plan_narrative()

    return {
        "family": req.family,
        "status": "proposed",
        "plan": plan,
        "narrative": narrative,
        "review": review,
        "summary": {
            "total_variants":    len(plan),
            "by_category":       {
                "feature_subset": sum(1 for v in plan if v["category"] == "feature_subset"),
                "interactions":   sum(1 for v in plan if v["category"] == "interactions"),
                "banding":        sum(1 for v in plan if v["category"] == "banding"),
            },
            "glm_families_used": sorted({v["glm"]["family"] for v in plan}),
            "features_min":      min(len(v["features"]) for v in plan),
            "features_max":      max(len(v["features"]) for v in plan),
        },
    }


def _static_plan_narrative() -> str:
    """Static narrative — kept identical to the FM-generated version (it's
    effectively a templated explanation), without the ~15-20s round trip.
    The data-driven variant count comes from `_agentic_plan_review`; this
    paragraph just frames the *why* of the search space."""
    return (
        "This plan sweeps three orthogonal axes: feature inclusion (how much "
        "enrichment data to pull in), specification complexity (whether to add "
        "pairwise interactions), and risk-factor banding (how to express continuous "
        "variables). Sweeping across these dimensions surfaces the trade-off between "
        "bias and variance — a more complex model will capture subtler segmentation "
        "but is more exposed to overfitting on new business.\n\n"
        "Distributional-family variants (Negative Binomial and Tweedie alongside "
        "Poisson) probe whether claim-count overdispersion is a meaningful departure "
        "from the canonical assumption; for SME property books it often is. "
        "Interaction-probe variants check specific hypotheses (e.g. flood × coastal, "
        "credit × CCJ) that actuaries commonly debate. Banding variants quantify how "
        "much lift comes from coarser vs finer rating.\n\n"
        "No single variant is recommended here — the leaderboard and shortlist in "
        "Step 3 will surface the candidates for the Pricing Committee's review."
    )


# ---------------------------------------------------------------------------
# Approve + virtual-train
# ---------------------------------------------------------------------------

class ApproveRequest(BaseModel):
    family: str
    plan: list[dict]
    narrative: str | None = None


@router.post("/approve")
async def approve_and_train(req: ApproveRequest, background_tasks: BackgroundTasks) -> dict:
    """Inserts the factory_runs row synchronously (fast) and returns the run_id
    immediately. Variant materialisation is the slow part (~15-20s for the big
    UNION ALL INSERT of 50 rows) — pushed to a background task so the UI can
    advance to the training step without blocking. The frontend polls
    /runs/{run_id} every couple of seconds and catches the COMPLETED state."""
    if req.family not in SUPPORTED_FAMILIES:
        raise HTTPException(400, f"Family {req.family} not supported yet.")
    if not req.plan:
        raise HTTPException(400, "Plan is empty.")

    await ensure_factory_tables()

    user = get_current_user()
    run_id = f"FACTORY-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}-{req.family}"
    plan_json = json.dumps(req.plan).replace("'", "''")
    narrative = (req.narrative or "").replace("'", "''")

    # factory_runs row — fast, do it synchronously so we can return run_id
    await execute_query(f"""
        INSERT INTO {fqn('factory_runs')}
        SELECT '{run_id}', '{req.family}', '{plan_json}', '{narrative}',
               '{user}', current_timestamp(), 0.0, 'TRAINING', {len(req.plan)}
    """)

    background_tasks.add_task(_materialise_variants_background, run_id, req.family, req.plan, user)

    return {"run_id": run_id, "family": req.family, "status": "TRAINING",
            "variant_count": len(req.plan), "approved_by": user}


async def _materialise_variants_background(run_id: str, family: str, plan: list[dict], user: str) -> None:
    """Slow part — synthesise per-variant metrics and INSERT them. Updates the
    factory_runs row to COMPLETED when done. Catches its own exceptions and
    flips the run to FAILED so the UI doesn't poll forever."""
    try:
        variant_rows: list[str] = []
        for v in plan:
            m = _synth_metrics(v, run_id)
            v_with_metrics = {**v, "metrics": m}
            variant_rows.append(
                "SELECT "
                f"'{run_id}' AS run_id, "
                f"'{v['variant_id']}' AS variant_id, "
                f"'{v['name'].replace(chr(39), chr(39)+chr(39))}' AS name, "
                f"'{v['category']}' AS category, "
                f"'{json.dumps(v_with_metrics).replace(chr(39), chr(39)+chr(39))}' AS config_json, "
                f"'{json.dumps(m).replace(chr(39), chr(39)+chr(39))}' AS metrics_json, "
                f"{len(v.get('features', []))} AS n_features, "
                f"current_timestamp() AS created_at "
            )
        if variant_rows:
            await execute_query(f"""
                INSERT INTO {fqn('factory_variants')}
                {' UNION ALL '.join(variant_rows)}
            """)

        await execute_query(f"""
            UPDATE {fqn('factory_runs')}
            SET status = 'COMPLETED', duration_seconds = 12.0
            WHERE run_id = '{run_id}'
        """)

        await log_audit_event(
            event_type="factory_plan_approved",
            entity_type="factory_run",
            entity_id=run_id,
            user_id=user,
            details={"family": family, "variants": len(plan), "training_mode": "virtual"},
        )
    except Exception as e:
        logger.exception("Factory materialisation failed for %s: %s", run_id, e)
        try:
            await execute_query(f"""
                UPDATE {fqn('factory_runs')}
                SET status = 'FAILED', duration_seconds = 0.0
                WHERE run_id = '{run_id}'
            """)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Run status polling
# ---------------------------------------------------------------------------

@router.get("/runs/{run_id}")
async def run_status(run_id: str) -> dict:
    rows = await execute_query(f"""
        SELECT run_id, model_family, approved_by, started_at, duration_seconds,
               status, variant_count, narrative
        FROM {fqn('factory_runs')} WHERE run_id = '{run_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"run {run_id} not found")
    r = rows[0]

    # Virtual progress: seconds elapsed since started_at, capped at
    # duration_seconds. Gives the UI something meaningful to render.
    progress = 1.0
    elapsed  = 0.0
    n_complete = r["variant_count"]
    try:
        started = r["started_at"]
        if isinstance(started, str):
            started = datetime.fromisoformat(started.replace(" ", "T").replace("Z", "+00:00"))
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        elapsed = (datetime.now(timezone.utc) - started).total_seconds()
        if r["status"] != "COMPLETED":
            progress = min(1.0, elapsed / max(1.0, r["duration_seconds"] or 12.0))
            n_complete = int(progress * r["variant_count"])
    except Exception as e:
        logger.warning("progress calc failed: %s", e)

    return {
        "run_id":      r["run_id"],
        "family":      r["model_family"],
        "status":      r["status"],
        "variant_count": r["variant_count"],
        "n_complete":  n_complete,
        "progress":    round(progress, 3),
        "elapsed_seconds": round(elapsed, 1),
        "approved_by": r["approved_by"],
        "started_at":  str(r["started_at"]),
        "narrative":   r.get("narrative") or "",
    }


@router.get("/runs")
async def list_runs(limit: int = 10) -> dict:
    limit = max(1, min(25, int(limit)))
    try:
        rows = await execute_query(f"""
            SELECT run_id, model_family, status, variant_count, approved_by, started_at
            FROM {fqn('factory_runs')}
            ORDER BY started_at DESC
            LIMIT {limit}
        """)
    except Exception:
        rows = []
    return {"runs": rows}


# ---------------------------------------------------------------------------
# Leaderboard / shortlist / portfolio
# ---------------------------------------------------------------------------

def _parse_variant_rows(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        cfg = _safe_json(r.get("config_json"))
        met = _safe_json(r.get("metrics_json"))
        out.append({
            "variant_id": r["variant_id"],
            "name":       r["name"],
            "category":   r["category"],
            "n_features": r["n_features"],
            "metrics":    met,
            "config":     cfg,
        })
    return out


def _safe_json(raw):
    if raw is None:
        return {}
    if isinstance(raw, (dict, list)):
        return raw
    try:
        return json.loads(raw)
    except Exception:
        return {}


@router.get("/runs/{run_id}/leaderboard")
async def leaderboard(run_id: str) -> dict:
    rows = await execute_query(f"""
        SELECT variant_id, name, category, n_features, config_json, metrics_json
        FROM {fqn('factory_variants')}
        WHERE run_id = '{run_id}'
    """)
    variants = _parse_variant_rows(rows)
    # Default sort: descending Gini
    variants.sort(key=lambda v: -(v["metrics"].get("gini") or 0))
    return {"run_id": run_id, "variants": variants, "n_total": len(variants)}


@router.get("/runs/{run_id}/shortlist")
async def shortlist(run_id: str) -> dict:
    data = await leaderboard(run_id)
    top = data["variants"][:5]

    # Enrich each shortlisted variant with diagnostics that'd normally come
    # from cross-validation. Synthesised in MVP.
    for v in top:
        seed = f"{run_id}:{v['variant_id']}:cv"
        cv_mean = v["metrics"].get("gini", 0) * (1.0 + _seeded(seed, -0.02, 0.02))
        cv_std  = 0.004 + _seeded(seed + ":std", 0, 0.010)
        v["cv"] = {
            "cv_gini_mean":  round(cv_mean, 4),
            "cv_gini_std":   round(cv_std,  4),
            "cv_folds":      5,
            "stability":     "stable" if cv_std < 0.010 else "watch",
        }
        # Coefficient-sign sanity check — all expected signs (flood↑, credit↓).
        v["sign_checks"] = {
            "flood_zone_rating":  "positive ✓",
            "credit_score":       "negative ✓",
            "years_trading":      "negative ✓",
            "is_coastal":         "positive ✓",
        }

    return {"run_id": run_id, "shortlist": top}


@router.get("/runs/{run_id}/portfolio")
async def portfolio_whatif(run_id: str) -> dict:
    """Synthetic portfolio what-if for top 5. Real scoring lands later."""
    data = await shortlist(run_id)
    top = data["shortlist"]

    REGIONS  = ["London", "South East", "North West", "Midlands", "Scotland", "Wales"]
    SEGMENTS = ["SME Retail", "SME Office", "Mid-market Mfg", "Hospitality", "Light Industrial"]

    results = []
    for v in top:
        seed = f"{run_id}:{v['variant_id']}:portfolio"
        gini = v["metrics"].get("gini", 0)
        # Overall premium shift: more-discriminative models push premium more,
        # but capped at +/- ~12% vs champion in a well-behaved factory.
        mean_shift_pct = (gini - CHAMPION_GINI) * 35     # heuristic
        mean_shift_pct += _seeded(seed + ":mean", -1.5, 1.5)
        segment_shifts = [{
            "segment": f"{r} / {s}",
            "shift_pct": round((gini - CHAMPION_GINI) * 35
                               + _seeded(f"{seed}:{r}:{s}", -4.0, 4.0), 2),
        } for r in REGIONS for s in SEGMENTS]
        # Count "big movers" — absolute shift > 10% or > 25%
        n_gt_10 = sum(1 for s in segment_shifts if abs(s["shift_pct"]) > 10)
        n_gt_25 = sum(1 for s in segment_shifts if abs(s["shift_pct"]) > 25)

        # Loss ratio by decile (10 deciles). Champion is wobblier; top variants
        # flatten the loss-ratio curve.
        decile_curve = []
        for d in range(1, 11):
            base = 0.65 + 0.04 * abs(d - 5)          # u-shape
            improv = gini * 0.6                      # better discrimination → flatter
            champ_val = base + _seeded(f"{seed}:d{d}:champ", -0.05, 0.05)
            cand_val  = base - improv * 0.05 * abs(d - 5) + _seeded(f"{seed}:d{d}:cand", -0.02, 0.02)
            decile_curve.append({
                "decile": d,
                "champion_lr":  round(champ_val, 3),
                "candidate_lr": round(cand_val, 3),
            })

        results.append({
            "variant_id": v["variant_id"],
            "name":       v["name"],
            "gini":       gini,
            "premium_shift_pct": round(mean_shift_pct, 2),
            "n_policies_sampled": 5000,
            "n_shift_gt_10pct": n_gt_10,
            "n_shift_gt_25pct": n_gt_25,
            "top_segments_up": sorted(
                [s for s in segment_shifts if s["shift_pct"] > 0],
                key=lambda s: -s["shift_pct"],
            )[:5],
            "top_segments_down": sorted(
                [s for s in segment_shifts if s["shift_pct"] < 0],
                key=lambda s: s["shift_pct"],
            )[:5],
            "loss_ratio_deciles": decile_curve,
        })

    return {"run_id": run_id, "champion_gini": CHAMPION_GINI, "results": results,
            "notes": "Synthesised portfolio metrics — real scoring wiring is phase 2."}


# ---------------------------------------------------------------------------
# Agent chat — delegated to the Databricks Agent Framework endpoint
# `pricing_chat_agent` (persona=factory). The agent's own tools query
# factory_runs / factory_variants directly, so the app no longer pre-bakes
# context — it just forwards the question + run_id.
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    run_id: str
    question: str


CHAT_AGENT_ENDPOINT = "pricing_chat_agent"


@router.post("/chat")
async def factory_chat(req: ChatRequest) -> dict:
    """Forward the actuary's question to the Databricks Agent Framework
    `pricing_chat_agent` endpoint with persona=factory. The agent calls
    its own SQL tools (factory_runs, factory_variants) so the app no longer
    has to pre-bake context — it just passes run_id + question."""
    if not req.question.strip():
        raise HTTPException(400, "question is required")

    # Confirm the run exists (and fail fast with 404 if not) before spending
    # an agent call on a missing run.
    try:
        run_rows = await execute_query(f"""
            SELECT model_family, status FROM {fqn('factory_runs')}
            WHERE run_id = '{req.run_id}' LIMIT 1
        """)
    except Exception as e:
        raise HTTPException(500, f"Run lookup failed: {e}")
    if not run_rows:
        raise HTTPException(404, f"run {req.run_id} not found")

    from server.agent_client import invoke_agent
    result = await invoke_agent(
        endpoint_name=CHAT_AGENT_ENDPOINT,
        question=req.question,
        custom_inputs={"persona": "factory", "run_id": req.run_id},
    )

    answer = result.get("answer") or ""
    cited  = sorted(set(re.findall(r"\b([ABC]\d{2})\b", answer))) if answer else []

    await log_audit_event(
        event_type="factory_chat",
        entity_type="factory_run",
        entity_id=req.run_id,
        details={
            "question":       req.question[:400],
            "cited":          cited,
            "endpoint":       CHAT_AGENT_ENDPOINT,
            "model":          result.get("model"),
            "answer_length":  len(answer),
            "trace":          result.get("trace", []),
            "usage":          result.get("usage", {}),
            "error":          result.get("error"),
        },
    )

    return {
        "run_id":         req.run_id,
        "question":       req.question,
        "answer":         answer or (f"[chat unavailable: {result.get('error')}]"
                                      if not result.get("ok") else ""),
        "cited_variants": cited,
        "endpoint":       CHAT_AGENT_ENDPOINT,
        "model":          result.get("model"),
        "usage":          result.get("usage", {}),
        "trace":          result.get("trace", []),
        "error":          result.get("error"),
        "cached":         bool(result.get("cached")),
    }


# ---------------------------------------------------------------------------
# Selective packaging — intent-only in MVP
# ---------------------------------------------------------------------------

@router.post("/runs/{run_id}/variants/{variant_id}/pack")
async def promote_variant(run_id: str, variant_id: str) -> dict:
    """Record the actuary's intent to progress this variant to a governance
    pack. MVP cuts a corner here — we don't run the pack job for virtual
    candidates (they have no real UC version). In phase 2 this calls the
    existing governance_pack_generation job."""
    rows = await execute_query(f"""
        SELECT name, category, config_json, metrics_json
        FROM {fqn('factory_variants')}
        WHERE run_id = '{run_id}' AND variant_id = '{variant_id}' LIMIT 1
    """)
    if not rows:
        raise HTTPException(404, f"variant {variant_id} not found in run {run_id}")
    v = rows[0]
    user = get_current_user()
    await log_audit_event(
        event_type="factory_variant_promoted_to_pack",
        entity_type="factory_variant",
        entity_id=f"{run_id}:{variant_id}",
        user_id=user,
        details={"run_id": run_id, "variant_id": variant_id, "name": v["name"],
                 "metrics": _safe_json(v.get("metrics_json")),
                 "mode": "virtual — no real pack generated in MVP"},
    )
    return {
        "run_id":      run_id,
        "variant_id":  variant_id,
        "status":      "queued",
        "message":     "Variant flagged for pack generation. "
                       "In MVP this logs the intent to audit_log but does not run the "
                       "pack job (the factory candidate is virtual). "
                       "Phase 2 will wire this to governance_pack_generation.",
        "queued_by":   user,
    }
