"""Model Deployment routes — registered models, serving endpoints, metrics, and live scoring."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import fqn, get_catalog, get_current_user, get_schema, get_workspace_client, get_workspace_host
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/deployment", tags=["deployment"])

# The 4 production model families tracked on the Production Models tab.
PRODUCTION_FAMILIES = [
    {"key": "freq_glm",   "label": "Frequency (GLM)"},
    {"key": "sev_glm",    "label": "Severity (GLM)"},
    {"key": "demand_gbm", "label": "Demand (GBM)"},
    {"key": "fraud_gbm",  "label": "Fraud (GBM)"},
]
CHAMPION_ALIAS   = "champion"
PREV_ALIAS       = "previous_champion"


@router.get("/models")
async def list_registered_models():
    """List all models registered in UC for this schema."""
    host = get_workspace_host()
    catalog = get_catalog()
    schema = get_schema()

    # SDK-first path; SQL fallback if SDK errors. All blocking SDK calls run
    # off the event loop on the thread pool — registered_models.list +
    # model_versions.list × N families fan out concurrently.
    def _sdk_pull() -> list[dict]:
        w = get_workspace_client()
        out: list[dict] = []
        models_list = list(w.registered_models.list(catalog_name=catalog, schema_name=schema))
        for m in models_list:
            full_name = f"{catalog}.{schema}.{m.name}"
            versions: list[dict] = []
            try:
                vs = list(w.model_versions.list(full_name=full_name))
                for v in sorted(vs, key=lambda x: int(x.version), reverse=True)[:5]:
                    versions.append({
                        "version": v.version,
                        "run_id": v.run_id,
                        "status": str(v.status).split(".")[-1] if v.status else "?",
                        "created_at": v.created_at,
                        "created_by": v.created_by,
                    })
            except Exception:
                pass
            out.append({
                "name": m.name,
                "full_name": full_name,
                "comment": m.comment,
                "created_at": m.created_at,
                "created_by": m.created_by,
                "updated_at": m.updated_at,
                "updated_by": m.updated_by,
                "versions": versions,
                "latest_version": versions[0] if versions else None,
                "catalog_url": f"{host}/explore/data/models/{catalog}/{schema}/{m.name}",
            })
        return out

    results: list[dict] = []
    try:
        results = await asyncio.to_thread(_sdk_pull)
    except Exception as e:
        logger.warning("SDK model list failed (%s), trying SQL fallback", e)
        # SQL fallback — query information_schema for models
        try:
            rows = await execute_query(f"""
                SELECT model_name, comment, created, created_by, last_altered, last_altered_by
                FROM {catalog}.information_schema.registered_models
                WHERE schema_name = '{schema}'
                ORDER BY model_name
            """)
            for r in rows:
                results.append({
                    "name": r.get("model_name", ""),
                    "full_name": f"{catalog}.{schema}.{r.get('model_name', '')}",
                    "comment": r.get("comment"),
                    "created_at": r.get("created"),
                    "created_by": r.get("created_by"),
                    "updated_at": r.get("last_altered"),
                    "updated_by": r.get("last_altered_by"),
                    "versions": [],
                    "latest_version": None,
                    "catalog_url": f"{host}/explore/data/models/{catalog}/{schema}/{r.get('model_name', '')}",
                })
        except Exception as e2:
            logger.warning("SQL model list also failed: %s", e2)

    return results


# ---------------------------------------------------------------------------
# Production Models tab — champion aliases across the 4 families
# ---------------------------------------------------------------------------

def _get_alias_version(w, full_name: str, alias: str) -> str | None:
    try:
        mv = w.model_versions.get_by_alias(full_name=full_name, alias=alias)
        return str(mv.version) if mv else None
    except Exception as e:
        logger.debug("alias %s on %s not found: %s", alias, full_name, e)
        return None


def _version_detail(w, full_name: str, version: str | None) -> dict[str, Any] | None:
    if not version:
        return None
    try:
        v = w.model_versions.get(full_name=full_name, version=int(version))
    except Exception as e:
        logger.warning("model_versions.get(%s, %s) failed: %s", full_name, version, e)
        return None
    created_iso = None
    if v.created_at:
        try:
            created_iso = datetime.fromtimestamp(v.created_at / 1000).isoformat()
        except Exception:
            created_iso = None
    return {
        "version":     str(v.version),
        "run_id":      v.run_id,
        "status":      str(v.status).split(".")[-1] if v.status else None,
        "created_at":  created_iso,
        "created_by":  v.created_by,
    }


@router.get("/champions")
async def list_champions(require_pack: bool = True) -> dict:
    """Return champion + previous_champion per family, joined with the latest
    governance pack. By default we only surface families that have a generated
    pack (the Production tab only shows models cleared for promotion). Set
    `require_pack=false` to include pre-pack families too."""
    w       = get_workspace_client()
    catalog = get_catalog()
    schema  = get_schema()
    host    = get_workspace_host()

    # Latest pack per family (single query)
    packs_by_family: dict[str, dict[str, Any]] = {}
    try:
        pack_rows = await execute_query(f"""
            SELECT model_family, pack_id, pdf_path, generated_by, generated_at
            FROM (
                SELECT model_family, pack_id, pdf_path, generated_by, generated_at,
                       row_number() OVER (PARTITION BY model_family ORDER BY generated_at DESC) AS rn
                FROM {fqn('governance_packs_index')}
            )
            WHERE rn = 1
        """)
        for r in pack_rows:
            packs_by_family[r["model_family"]] = r
    except Exception as e:
        logger.info("governance_packs_index not available yet: %s", e)

    # Per-family SDK work runs in parallel on the thread pool — was 24 sync
    # MLflow calls in series (4 families × 6 calls), now ~6 calls per family
    # in their own threads.
    def _family_block(fam: dict) -> dict:
        full_name = f"{catalog}.{schema}.{fam['key']}"
        champion_v = _get_alias_version(w, full_name, CHAMPION_ALIAS)
        previous_v = _get_alias_version(w, full_name, PREV_ALIAS)
        champ_info = _version_detail(w, full_name, champion_v)
        prev_info  = _version_detail(w, full_name, previous_v)
        fallback_latest = None
        if champ_info is None:
            try:
                versions = list(w.model_versions.list(full_name=full_name))
                if versions:
                    latest = max(versions, key=lambda x: int(x.version))
                    fallback_latest = _version_detail(w, full_name, str(latest.version))
            except Exception as e:
                logger.warning("fallback list for %s failed: %s", full_name, e)
        return {
            "family":            fam["key"],
            "label":             fam["label"],
            "uc_name":           full_name,
            "catalog_url":       f"{host}/explore/data/models/{catalog}/{schema}/{fam['key']}",
            "champ_info":        champ_info,
            "fallback_latest":   fallback_latest,
            "prev_info":         prev_info,
        }

    family_blocks = await asyncio.gather(*[
        asyncio.to_thread(_family_block, fam) for fam in PRODUCTION_FAMILIES
    ])

    out = []
    for fb in family_blocks:
        pack = packs_by_family.get(fb["family"])
        if require_pack and pack is None:
            continue
        out.append({
            "family":             fb["family"],
            "label":              fb["label"],
            "uc_name":            fb["uc_name"],
            "catalog_url":        fb["catalog_url"],
            "champion":           fb["champ_info"] or fb["fallback_latest"],
            "champion_is_alias":  fb["champ_info"] is not None,
            "previous_champion":  fb["prev_info"],
            "latest_pack":        {
                "pack_id":       pack["pack_id"],
                "pdf_path":      pack["pdf_path"],
                "generated_by":  pack["generated_by"],
                "generated_at":  str(pack["generated_at"]),
                "download_url":  f"/api/review/packs/{pack['pack_id']}/download",
            } if pack else None,
        })

    return {"families": out}


@router.get("/champions/{family}/history")
async def champion_history(family: str, limit: int = 10) -> dict:
    """Return the latest N promotion / rollback events for a family from
    audit_log — used by the expandable row on the Production Models tab."""
    limit = max(1, min(50, int(limit)))
    try:
        rows = await execute_query(f"""
            SELECT event_type, entity_version, user_id, timestamp, details
            FROM {fqn('audit_log')}
            WHERE entity_id = '{family}'
              AND event_type IN (
                'model_trained', 'governance_pack_generated',
                'model_promoted', 'model_rollback', 'model_rolled_back'
              )
            ORDER BY timestamp DESC
            LIMIT {limit}
        """)
    except Exception as e:
        logger.warning("history query failed for %s: %s", family, e)
        return {"family": family, "events": []}

    events = []
    for r in rows:
        details_raw = r.get("details") or "{}"
        try:
            det = json.loads(details_raw) if isinstance(details_raw, str) else (details_raw or {})
        except Exception:
            det = {}
        events.append({
            "event_type":     r["event_type"],
            "version":        r.get("entity_version"),
            "user":           r.get("user_id"),
            "timestamp":      str(r.get("timestamp", "")),
            "details":        det,
        })
    return {"family": family, "events": events}


# ---------------------------------------------------------------------------
# Inference-log backfill trigger — fire-and-forget when a champion changes
# ---------------------------------------------------------------------------

_BACKFILL_JOB_NAME = "v1 — Inference log backfill (score all UPT policies)"


def _trigger_inference_backfill(w) -> dict[str, Any]:
    """Kick off the inference_backfill job so `{fqn}.inference_logs` reflects
    the new champion set. Returns a dict with run_id + run_page_url if the
    job was found and submitted, or a `skipped` note otherwise. Never raises
    — champion promotion must succeed even if the backfill trigger fails."""
    try:
        # Bundle prefixes job names with "[dev <user>] "; use suffix match.
        job_id: int | None = None
        try:
            for j in w.jobs.list(name=_BACKFILL_JOB_NAME, limit=25):
                job_id = j.job_id
                break
        except Exception:
            pass
        if job_id is None:
            for j in w.jobs.list(limit=100):
                if (j.settings.name or "").endswith(_BACKFILL_JOB_NAME):
                    job_id = j.job_id
                    break
        if job_id is None:
            logger.warning("inference_backfill job not found — skipping trigger")
            return {"triggered": False, "reason": "job not found"}

        run = w.jobs.run_now(job_id=job_id, job_parameters={
            "catalog_name": get_catalog(),
            "schema_name":  get_schema(),
        })
        run_id = getattr(run, "run_id", None)
        host   = get_workspace_host()
        return {
            "triggered":    True,
            "job_id":       job_id,
            "run_id":       run_id,
            "run_page_url": f"{host}/jobs/{job_id}/runs/{run_id}" if host and run_id else None,
        }
    except Exception as e:
        logger.warning("inference_backfill trigger failed: %s", e)
        return {"triggered": False, "reason": str(e)}


@router.post("/inference-backfill/trigger")
async def trigger_inference_backfill() -> dict:
    """Manually fire the inference-backfill job. Useful after bulk operations,
    or when the auto-trigger on a promotion failed for any reason."""
    w = get_workspace_client()
    result = _trigger_inference_backfill(w)
    user = get_current_user()
    await log_audit_event(
        event_type="inference_backfill_triggered",
        entity_type="table",
        entity_id="inference_logs",
        user_id=user,
        details=result,
    )
    return result


# ---------------------------------------------------------------------------
# Rollback — swap champion alias back to previous_champion
# ---------------------------------------------------------------------------

class RollbackRequest(BaseModel):
    family: str
    note: str


@router.post("/rollback")
async def rollback_champion(req: RollbackRequest) -> dict:
    if not req.note or len(req.note.strip()) < 10:
        raise HTTPException(400, "A rollback justification of at least 10 characters is required.")
    if req.family not in {f["key"] for f in PRODUCTION_FAMILIES}:
        raise HTTPException(400, f"Unknown family {req.family}")

    w       = get_workspace_client()
    catalog = get_catalog()
    schema  = get_schema()
    full_name = f"{catalog}.{schema}.{req.family}"

    current_champion, previous = await asyncio.gather(
        asyncio.to_thread(_get_alias_version, w, full_name, CHAMPION_ALIAS),
        asyncio.to_thread(_get_alias_version, w, full_name, PREV_ALIAS),
    )
    if not previous:
        raise HTTPException(400,
            "No previous champion set — nothing to roll back to. "
            "The `previous_champion` alias is only populated by a successful promotion.")

    def _swap_aliases() -> None:
        w.registered_models.set_alias(full_name=full_name, alias=CHAMPION_ALIAS, version_num=int(previous))
        if current_champion:
            w.registered_models.set_alias(full_name=full_name, alias=PREV_ALIAS, version_num=int(current_champion))
        else:
            try:
                w.registered_models.delete_alias(full_name=full_name, alias=PREV_ALIAS)
            except Exception:
                pass
    try:
        await asyncio.to_thread(_swap_aliases)
    except Exception as e:
        raise HTTPException(500, f"Failed to swap aliases: {e}")

    # Bust pricing.py's alias cache so /pricing/status sees the new champion.
    try:
        from server.routes.pricing import _bust_alias_cache
        _bust_alias_cache()
    except Exception:
        pass

    user = get_current_user()
    backfill = await asyncio.to_thread(_trigger_inference_backfill, w)
    await log_audit_event(
        event_type="model_rollback",
        entity_type="model",
        entity_id=req.family,
        entity_version=str(previous),
        user_id=user,
        details={
            "from_version": current_champion,
            "to_version":   previous,
            "note":         req.note,
            "backfill":     backfill,
        },
    )

    return {
        "family":         req.family,
        "new_champion":   previous,
        "prior_champion": current_champion,
        "user":           user,
        "backfill":       backfill,
    }


@router.post("/champions/{family}/set")
async def set_champion(family: str, version: str) -> dict:
    """Directly set the champion alias to a version. Used during bootstrap
    when there's no previous_champion yet."""
    if family not in {f["key"] for f in PRODUCTION_FAMILIES}:
        raise HTTPException(400, f"Unknown family {family}")
    w = get_workspace_client()
    full_name = f"{get_catalog()}.{get_schema()}.{family}"
    current = await asyncio.to_thread(_get_alias_version, w, full_name, CHAMPION_ALIAS)

    def _set_aliases() -> None:
        if current and current != version:
            w.registered_models.set_alias(full_name=full_name, alias=PREV_ALIAS, version_num=int(current))
        w.registered_models.set_alias(full_name=full_name, alias=CHAMPION_ALIAS, version_num=int(version))
    try:
        await asyncio.to_thread(_set_aliases)
    except Exception as e:
        raise HTTPException(500, f"Alias set failed: {e}")

    try:
        from server.routes.pricing import _bust_alias_cache
        _bust_alias_cache()
    except Exception:
        pass

    user = get_current_user()
    backfill = await asyncio.to_thread(_trigger_inference_backfill, w)
    await log_audit_event(
        event_type="model_promoted",
        entity_type="model",
        entity_id=family,
        entity_version=str(version),
        user_id=user,
        details={"previous_champion": current, "new_champion": version, "backfill": backfill},
    )
    return {
        "family":   family,
        "champion": version,
        "previous": current,
        "backfill": backfill,
    }
