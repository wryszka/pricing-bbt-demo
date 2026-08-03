"""Review & Promote — list UC model versions, pull MLflow run detail + artifacts,
trigger the governance_pack_generation job, serve generated pack PDFs from the
UC volume.

This tab never mutates UC aliases. "Promote" == "generate governance pack".
Actual champion-alias rollovers live on the Model Deployment tab.
"""
from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from server.audit import log_audit_event
from server.config import (
    fqn, get_catalog, get_current_user, get_schema,
    get_workspace_client, get_workspace_host,
)
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/review", tags=["review"])

# The 4 production model families — the set the tab operates on.
FAMILIES: list[dict[str, Any]] = [
    {"key": "freq_glm",   "label": "Frequency (GLM)",  "type": "GLM",  "primary_metric": "gini"},
    {"key": "sev_glm",    "label": "Severity (GLM)",   "type": "GLM",  "primary_metric": "gini"},
    {"key": "demand_gbm", "label": "Demand (GBM)",     "type": "GBM",  "primary_metric": "auc"},
    {"key": "fraud_gbm",  "label": "Fraud (GBM)",      "type": "GBM",  "primary_metric": "auc"},
]
FAMILY_KEYS = {f["key"] for f in FAMILIES}
PACK_JOB_NAME = "v1 — Generate governance pack"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _family_meta(family: str) -> dict[str, Any]:
    for f in FAMILIES:
        if f["key"] == family:
            return f
    raise HTTPException(404, f"Unknown model family: {family}")


def _mlflow_client():
    # Late import — keeps mlflow out of cold-start for routes that don't need it.
    # Set BOTH tracking_uri and registry_uri so runs resolve in the workspace
    # and registered models resolve in Unity Catalog. Without tracking_uri,
    # get_run() falls back to the local file store and always returns
    # "Run not found" — which is what was leaving every version's
    # story/metrics empty on the Review & Promote tab.
    import mlflow
    from mlflow.tracking import MlflowClient
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    return MlflowClient()


def _fetch_run(client, run_id: str) -> dict:
    try:
        r = client.get_run(run_id)
    except Exception as e:
        logger.warning("get_run %s failed: %s", run_id, e)
        return {}
    return {
        "tags":          dict(r.data.tags or {}),
        "params":        dict(r.data.params or {}),
        "metrics":       dict(r.data.metrics or {}),
        "start_ms":      r.info.start_time or 0,
        "status":        r.info.status,
        "experiment_id": r.info.experiment_id,
    }


def _iso_from_ms(ms: int) -> str | None:
    if not ms:
        return None
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def _find_pack_job_id(w) -> int | None:
    # Exact match first, then suffix match — the dev bundle target prefixes
    # job names with "[dev <whoami>] ", so an exact lookup on the bare name
    # misses on dev and the promote button 500s with "Job not found".
    try:
        for j in w.jobs.list(name=PACK_JOB_NAME, limit=25):
            return j.job_id
    except Exception as e:
        logger.warning("jobs.list(name=%s) failed: %s", PACK_JOB_NAME, e)
    try:
        for j in w.jobs.list():
            settings = getattr(j, "settings", None)
            jname = getattr(settings, "name", None) if settings else None
            if jname and jname.endswith(PACK_JOB_NAME):
                return j.job_id
    except Exception as e:
        logger.warning("jobs.list() iter for %s failed: %s", PACK_JOB_NAME, e)
    return None


# ---------------------------------------------------------------------------
# 1. List families (with version counts)
# ---------------------------------------------------------------------------

@router.get("/families")
async def list_families() -> dict:
    client = _mlflow_client()
    catalog = get_catalog()
    schema  = get_schema()
    out = []
    for f in FAMILIES:
        uc_name = f"{catalog}.{schema}.{f['key']}"
        try:
            versions = client.search_model_versions(f"name='{uc_name}'")
        except Exception as e:
            logger.warning("search_model_versions failed for %s: %s", uc_name, e)
            versions = []
        latest = max((int(v.version) for v in versions), default=None)
        out.append({
            **f,
            "uc_name":        uc_name,
            "version_count":  len(versions),
            "latest_version": latest,
        })
    return {"families": out}


# ---------------------------------------------------------------------------
# 2. List versions for a family
# ---------------------------------------------------------------------------

_VERSIONS_CACHE: dict[str, dict] = {}
_VERSIONS_TTL_S = 30.0
_VERSIONS_LIMIT = 20  # cap to the 20 most-recent versions (was: all → 60+ → 30s)


@router.get("/families/{family}/versions")
async def list_versions(family: str) -> dict:
    """Return up to _VERSIONS_LIMIT most-recent versions for the family.
    MLflow run details fetched in parallel via asyncio + thread pool. Result
    cached for _VERSIONS_TTL_S so repeat opens are instant. The compare/test
    UI never needs >20 versions in the picker — older ones are still in the
    governance pack history if a regulator asks."""
    import asyncio, time as _time

    now = _time.time()
    entry = _VERSIONS_CACHE.get(family)
    if entry and now < entry["expires_at"]:
        return entry["value"]

    meta = _family_meta(family)
    client = _mlflow_client()
    uc_name = f"{get_catalog()}.{get_schema()}.{family}"
    try:
        all_versions = client.search_model_versions(f"name='{uc_name}'")
    except Exception as e:
        raise HTTPException(500, f"Could not list versions: {e}")

    # Cap to most-recent N before fetching run details — single biggest win
    sorted_versions = sorted(all_versions, key=lambda v: int(v.version), reverse=True)[:_VERSIONS_LIMIT]
    total_count = len(all_versions)

    host = get_workspace_host()

    # Fire all get_run calls in parallel — was the dominant 30s cost
    runs = await asyncio.gather(*[
        asyncio.to_thread(_fetch_run, client, v.run_id) for v in sorted_versions
    ])

    rows = []
    for v, run in zip(sorted_versions, runs):
        primary = run.get("metrics", {}).get(meta["primary_metric"])
        exp_id = run.get("experiment_id") or ""
        rows.append({
            "version":          int(v.version),
            "run_id":           v.run_id,
            "uc_name":          uc_name,
            "story":            run.get("tags", {}).get("story"),
            "story_text":       run.get("tags", {}).get("story_text"),
            "simulated":        run.get("tags", {}).get("simulated", "false") == "true",
            "simulation_date":  run.get("tags", {}).get("simulation_date"),
            "trained_by":       run.get("tags", {}).get("mlflow.user"),
            "trained_at":       _iso_from_ms(run.get("start_ms", 0)),
            "status":           str(v.status).split(".")[-1] if v.status else None,
            "primary_metric":   meta["primary_metric"],
            "primary_value":    primary,
            "metrics":          run.get("metrics", {}),
            "mlflow_url":       f"{host}/ml/experiments/{exp_id}/runs/{v.run_id}" if host else None,
        })

    payload = {
        "family":              family,
        "meta":                meta,
        "uc_name":             uc_name,
        "versions":            rows,
        "latest_version":      rows[0]["version"] if rows else None,
        "total_version_count": total_count,
        "shown":               len(rows),
    }
    _VERSIONS_CACHE[family] = {"value": payload, "expires_at": now + _VERSIONS_TTL_S}
    return payload


def _experiment_for_run(client, run_id: str) -> str | None:
    try:
        r = client.get_run(run_id)
        return r.info.experiment_id
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3. Version detail (metrics + params + artifacts)
# ---------------------------------------------------------------------------

@router.get("/families/{family}/versions/{version}")
async def version_detail(family: str, version: str) -> dict:
    meta    = _family_meta(family)
    client  = _mlflow_client()
    uc_name = f"{get_catalog()}.{get_schema()}.{family}"

    # Three MLflow REST calls — fan them out via the thread pool. They're
    # otherwise sequential and block the event loop.
    def _pull() -> tuple:
        try:
            mv_local = client.get_model_version(uc_name, version)
        except Exception as e:
            raise HTTPException(404, f"Version {version} not found for {uc_name}: {e}")
        run_local = _fetch_run(client, mv_local.run_id)
        try:
            arts = [
                {"path": a.path, "is_dir": a.is_dir, "file_size": a.file_size}
                for a in client.list_artifacts(mv_local.run_id)
            ]
        except Exception as e:
            logger.warning("list_artifacts(%s) failed: %s", mv_local.run_id, e)
            arts = []
        return mv_local, run_local, arts

    mv, run, artifacts = await asyncio.to_thread(_pull)
    host = get_workspace_host()

    # Classify the notable artifacts so the UI doesn't have to sniff filenames.
    def _find(tag: str) -> str | None:
        for a in artifacts:
            if a["path"].endswith(tag):
                return a["path"]
        return None

    notable = {
        "relativities_csv":    _find("relativities.csv"),
        "importance_csv":      _find("importance.csv"),
        "shap_importance_csv": _find("shap_importance.csv"),
        "shap_summary_png":    _find("shap_summary.png"),
    }

    # Lineage — the model was trained against this feature table
    feature_table = run.get("tags", {}).get("feature_table")

    return {
        "family":          family,
        "meta":            meta,
        "uc_name":         uc_name,
        "version":         int(version),
        "run_id":          mv.run_id,
        "run_name":        run.get("tags", {}).get("mlflow.runName"),
        "status":          str(mv.status).split(".")[-1] if mv.status else None,
        "trained_by":      run.get("tags", {}).get("mlflow.user"),
        "trained_at":      _iso_from_ms(run.get("start_ms", 0)),
        "tags":            run.get("tags", {}),
        "params":          run.get("params", {}),
        "metrics":         run.get("metrics", {}),
        "primary_metric":  meta["primary_metric"],
        "primary_value":   run.get("metrics", {}).get(meta["primary_metric"]),
        "artifacts":       artifacts,
        "notable":         notable,
        "feature_table":   feature_table,
        "mlflow_url":      (f"{host}/ml/experiments/{_experiment_for_run(client, mv.run_id) or ''}/runs/{mv.run_id}"
                            if host else None),
        "story":           run.get("tags", {}).get("story"),
        "story_text":      run.get("tags", {}).get("story_text"),
        "simulated":       run.get("tags", {}).get("simulated", "false") == "true",
        "simulation_date": run.get("tags", {}).get("simulation_date"),
    }


# ---------------------------------------------------------------------------
# 4. Artifact passthrough — stream an image or CSV from the MLflow run
# ---------------------------------------------------------------------------

@router.get("/families/{family}/versions/{version}/artifact")
async def get_artifact(family: str, version: str, path: str) -> Response:
    """Download an artifact from the model version's MLflow run. `path` is the
    artifact path as listed by `list_artifacts` (e.g. `fraud_shap_summary.png`)."""
    _family_meta(family)
    client = _mlflow_client()
    uc_name = f"{get_catalog()}.{get_schema()}.{family}"

    # Only allow files at the artifact root — no traversal.
    if "/" in path or ".." in path:
        raise HTTPException(400, "path may not contain '/' or '..'")

    def _fetch_artifact() -> bytes:
        try:
            mv = client.get_model_version(uc_name, version)
        except Exception as e:
            raise HTTPException(404, str(e))
        with tempfile.TemporaryDirectory() as tmp:
            try:
                local = client.download_artifacts(mv.run_id, path, dst_path=tmp)
            except Exception as e:
                raise HTTPException(404, f"artifact {path} not found: {e}")
            return Path(local).read_bytes()

    data = await asyncio.to_thread(_fetch_artifact)
    mime = "image/png" if path.endswith(".png") else (
           "text/csv"   if path.endswith(".csv") else "application/octet-stream")
    return Response(content=data, media_type=mime,
                    headers={"Cache-Control": "private, max-age=300"})


# ---------------------------------------------------------------------------
# 5. Relativities / importance / SHAP as structured JSON
# ---------------------------------------------------------------------------

@router.get("/families/{family}/versions/{version}/explainability")
async def version_explainability(family: str, version: str) -> dict:
    """Return the coefficient/importance/SHAP CSVs as structured JSON so the UI
    doesn't have to parse CSV in the browser."""
    import csv
    meta = _family_meta(family)
    client = _mlflow_client()
    uc_name = f"{get_catalog()}.{get_schema()}.{family}"
    try:
        mv = client.get_model_version(uc_name, version)
    except Exception as e:
        raise HTTPException(404, str(e))

    def _read(path: str | None):
        if not path:
            return None
        with tempfile.TemporaryDirectory() as tmp:
            try:
                local = client.download_artifacts(mv.run_id, path, dst_path=tmp)
            except Exception:
                return None
            with open(local, newline="") as fh:
                reader = csv.DictReader(fh)
                return [dict(r) for r in reader]

    try:
        artifacts = [a.path for a in client.list_artifacts(mv.run_id)]
    except Exception:
        artifacts = []

    def find(tag: str) -> str | None:
        for a in artifacts:
            if a.endswith(tag):
                return a
        return None

    return {
        "family": family,
        "version": int(version),
        "relativities":    _read(find("relativities.csv")),
        "importance":      _read(find("importance.csv")),
        "shap_importance": _read(find("shap_importance.csv")),
        "has_shap_plot":   find("shap_summary.png") is not None,
        "shap_plot_path":  find("shap_summary.png"),
    }


# ---------------------------------------------------------------------------
# 6. Trigger pack generation job
# ---------------------------------------------------------------------------

class GeneratePackRequest(BaseModel):
    family: str
    version: str | int


@router.post("/packs/generate")
async def generate_pack(req: GeneratePackRequest) -> dict:
    if req.family not in FAMILY_KEYS:
        raise HTTPException(400, f"family must be one of {sorted(FAMILY_KEYS)}")
    user = get_current_user()
    w = get_workspace_client()
    job_id = await asyncio.to_thread(_find_pack_job_id, w)
    if not job_id:
        raise HTTPException(500,
            f"Job '{PACK_JOB_NAME}' not found. Deploy the bundle with `databricks bundle deploy`.")

    try:
        run = await asyncio.to_thread(
            w.jobs.run_now,
            job_id=job_id,
            job_parameters={
                "catalog_name":  get_catalog(),
                "schema_name":   get_schema(),
                "model_family":  req.family,
                "model_version": str(req.version),
                "requested_by":  user,
            },
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to trigger pack job: {e}")

    run_id = run.run_id if hasattr(run, "run_id") else (run.get("run_id") if isinstance(run, dict) else None)
    host = get_workspace_host()

    await log_audit_event(
        event_type  = "governance_pack_requested",
        entity_type = "model",
        entity_id   = req.family,
        entity_version = str(req.version),
        user_id     = user,
        details     = {"job_id": job_id, "job_run_id": run_id, "requested_by": user},
    )

    return {
        "job_id":      job_id,
        "job_run_id":  run_id,
        "run_page_url": f"{host}/jobs/{job_id}/runs/{run_id}" if host and run_id else None,
        "requested_by": user,
        "family":      req.family,
        "version":     str(req.version),
    }


# ---------------------------------------------------------------------------
# 7. Poll pack job status + surface resulting pack_id once done
# ---------------------------------------------------------------------------

@router.get("/packs/runs/{run_id}")
async def pack_run_status(run_id: int) -> dict:
    w = get_workspace_client()
    try:
        run = await asyncio.to_thread(w.jobs.get_run, run_id=run_id)
    except Exception as e:
        raise HTTPException(500, f"Could not fetch run {run_id}: {e}")

    state = run.state
    life  = str(state.life_cycle_state).split(".")[-1] if state else None
    result = str(state.result_state).split(".")[-1] if state and state.result_state else None

    # Try to extract the notebook's dbutils.notebook.exit(...) payload
    pack_payload = None
    try:
        task_runs = run.tasks or []
        for t in task_runs:
            if t.task_key == "generate" and t.run_id:
                out = await asyncio.to_thread(w.jobs.get_run_output, run_id=t.run_id)
                if out.notebook_output and out.notebook_output.result:
                    try:
                        pack_payload = json.loads(out.notebook_output.result)
                    except Exception:
                        pack_payload = {"raw": out.notebook_output.result}
    except Exception as e:
        logger.warning("extract run %s output failed: %s", run_id, e)

    return {
        "run_id":        run_id,
        "life_cycle":    life,
        "result":        result,
        "state_message": state.state_message if state else None,
        "pack":          pack_payload,
    }


# ---------------------------------------------------------------------------
# 8. List previously-generated packs
# ---------------------------------------------------------------------------

@router.get("/packs")
async def list_packs(family: str | None = None, limit: int = 25) -> dict:
    limit = max(1, min(100, int(limit)))
    where = ""
    if family:
        if family not in FAMILY_KEYS:
            raise HTTPException(400, f"family must be one of {sorted(FAMILY_KEYS)}")
        where = f"WHERE model_family = '{family}'"
    try:
        rows = await execute_query(f"""
            SELECT pack_id, model_family, model_version, model_uc_name, mlflow_run_id,
                   story, simulated, primary_metric, primary_value, pdf_path, size_bytes,
                   generated_by, generated_at
            FROM {fqn('governance_packs_index')}
            {where}
            ORDER BY generated_at DESC
            LIMIT {limit}
        """)
    except Exception as e:
        # Table may not exist yet — no packs have been generated.
        logger.info("governance_packs_index not queryable yet: %s", e)
        return {"packs": [], "note": "No packs generated yet."}
    return {"packs": rows}


# ---------------------------------------------------------------------------
# 9. Download a generated pack PDF
# ---------------------------------------------------------------------------

@router.get("/packs/{pack_id}/download")
async def download_pack(pack_id: str) -> StreamingResponse:
    try:
        rows = await execute_query(f"""
            SELECT pdf_path, model_family, model_version
            FROM {fqn('governance_packs_index')}
            WHERE pack_id = '{pack_id}'
            LIMIT 1
        """)
    except Exception as e:
        raise HTTPException(500, f"Packs index unavailable: {e}")
    if not rows:
        raise HTTPException(404, f"Pack {pack_id} not found")

    path = rows[0]["pdf_path"]
    w = get_workspace_client()
    def _download() -> bytes:
        resp = w.files.download(file_path=path)
        return resp.contents.read() if hasattr(resp.contents, "read") else resp.contents
    try:
        data = await asyncio.to_thread(_download)
    except Exception as e:
        raise HTTPException(500, f"Could not download {path}: {e}")

    await log_audit_event(
        event_type="governance_pack_downloaded",
        entity_type="model",
        entity_id=rows[0]["model_family"],
        entity_version=str(rows[0]["model_version"]),
        details={"pack_id": pack_id, "pdf_path": path},
    )

    filename = path.rsplit("/", 1)[-1]
    return StreamingResponse(
        io.BytesIO(data),
        media_type="application/pdf",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
