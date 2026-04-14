"""Model Deployment routes — registered models, serving endpoints, metrics, and live scoring."""

import logging
import time
from datetime import datetime

import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from server.config import fqn, get_workspace_client, get_workspace_host, get_catalog, get_schema
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/deployment", tags=["deployment"])


@router.get("/models")
async def list_registered_models():
    """List all models registered in UC for this schema."""
    host = get_workspace_host()
    catalog = get_catalog()
    schema = get_schema()

    # Try SDK first, fall back to SQL
    results = []
    try:
        w = get_workspace_client()
        models_list = list(w.registered_models.list(
            catalog_name=catalog, schema_name=schema,
        ))

        for m in models_list:
            full_name = f"{catalog}.{schema}.{m.name}"
            versions = []
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

            results.append({
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


@router.get("/endpoints")
async def list_serving_endpoints():
    """List custom serving endpoints (not foundation model endpoints)."""
    host = get_workspace_host()
    try:
        w = get_workspace_client()
        all_eps = list(w.serving_endpoints.list())
        custom_eps = [e for e in all_eps if not e.name.startswith("databricks-")]
    except Exception as e:
        logger.warning("Failed to list endpoints: %s", e)
        return []

    results = []
    for ep in custom_eps:
        entities = []
        traffic = []
        try:
            full_ep = w.serving_endpoints.get(ep.name)
            cfg = full_ep.config
            if cfg and hasattr(cfg, 'served_entities') and cfg.served_entities:
                for e in cfg.served_entities:
                    entities.append({
                        "name": e.name,
                        "model": e.entity_name,
                        "version": e.entity_version,
                        "workload_size": e.workload_size,
                        "scale_to_zero": e.scale_to_zero_enabled,
                    })
            if cfg and hasattr(cfg, 'traffic_config') and cfg.traffic_config:
                for r in (cfg.traffic_config.routes or []):
                    traffic.append({
                        "model": r.served_model_name,
                        "traffic_pct": r.traffic_percentage,
                    })
        except Exception as ex:
            logger.warning("Failed to get endpoint details for %s: %s", ep.name, ex)

        results.append({
            "name": ep.name,
            "state": str(ep.state.ready).split(".")[-1] if ep.state else "UNKNOWN",
            "config_state": str(ep.state.config_update).split(".")[-1] if ep.state else "?",
            "creator": ep.creator,
            "creation_timestamp": ep.creation_timestamp,
            "entities": entities,
            "traffic": traffic,
            "url": f"{host}/ml/endpoints/{ep.name}",
        })

    return results


@router.get("/latency")
async def get_endpoint_latency():
    """Get latency metrics from test results."""
    try:
        rows = await execute_query(
            f"SELECT metric, value FROM {fqn('endpoint_latency')}"
        )
        return {r["metric"]: float(r["value"]) for r in rows}
    except Exception:
        return {}


class ScoreRequest(BaseModel):
    endpoint_name: str = "pricing-frequency-endpoint"
    features: dict = {}


@router.post("/score")
async def score_model(req: ScoreRequest):
    """Send a scoring request to a serving endpoint and return the prediction + latency."""
    try:
        w = get_workspace_client()
        host = w.config.host.rstrip("/")

        # Build auth header — in Databricks App context, use the SDK's authenticator
        headers = {"Content-Type": "application/json"}
        try:
            auth_headers = w.config._header_factory()
            headers.update(auth_headers)
        except Exception:
            # Fallback: try to get token from the API client
            headers["Authorization"] = f"Bearer {w.config.token}"

        payload = {"dataframe_split": {
            "columns": list(req.features.keys()),
            "data": [list(req.features.values())],
        }}

        start = time.time()
        resp = requests.post(
            f"{host}/serving-endpoints/{req.endpoint_name}/invocations",
            headers=headers,
            json=payload,
            timeout=120,
        )
        latency_ms = round((time.time() - start) * 1000)

        if resp.status_code != 200:
            return {
                "success": False,
                "error": resp.text[:300],
                "status_code": resp.status_code,
                "latency_ms": latency_ms,
            }

        data = resp.json()
        return {
            "success": True,
            "predictions": data.get("predictions"),
            "latency_ms": latency_ms,
            "endpoint": req.endpoint_name,
            "input_features": req.features,
        }
    except Exception as e:
        return {"success": False, "error": str(e)[:300]}
