import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

# AXA edition: data prep + modelling mart + model development only.
# Serving, deployment, governance, pricing-engine, factory, pricing-AI and
# quote-review routes are excluded from this build.
from server.routes import datasets, agent, features, genie, development, admin
import os
from server.config import get_workspace_host

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

FRONTEND_DIR = Path(__file__).parent / "frontend" / "dist"


@asynccontextmanager
async def lifespan(application: FastAPI):
    import asyncio
    logger.info("Starting Pricing Workbench")
    try:
        await datasets.ensure_approvals_table()
        logger.info("Approvals table ready")
    except Exception:
        logger.exception("Failed to ensure tables — will retry on first request")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Pricing Workbench",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(datasets.router)
app.include_router(agent.router)
app.include_router(features.router)
app.include_router(genie.router)
app.include_router(development.router)
app.include_router(admin.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


# IDs for Genie spaces / the mart dashboard normally come from env (set in
# app.yaml). On a fresh deploy they are blank until the post-deploy
# create_genie_and_dashboard job creates those objects and writes their ids to
# the `app_config` UC table. We read that table as a fallback so the panels
# light up without an app redeploy. Cached + fully fault-tolerant: any error
# (table missing, no warehouse) just yields blanks and the panels stay hidden.
_APP_CONFIG_CACHE: dict = {"value": None, "ts": 0.0}
_APP_CONFIG_TTL_S = 60.0


async def _app_config() -> dict:
    import time as _time
    from server.config import fqn
    from server.sql import execute_query
    now = _time.time()
    if _APP_CONFIG_CACHE["value"] is not None and now - _APP_CONFIG_CACHE["ts"] < _APP_CONFIG_TTL_S:
        return _APP_CONFIG_CACHE["value"]
    cfg: dict = {}
    try:
        rows = await execute_query(f"SELECT key, value FROM {fqn('app_config')}")
        cfg = {r["key"]: r["value"] for r in rows if r.get("value")}
    except Exception:
        cfg = {}
    _APP_CONFIG_CACHE["value"] = cfg
    _APP_CONFIG_CACHE["ts"] = now
    return cfg


@app.get("/api/config")
async def config():
    host = get_workspace_host()
    cfg = await _app_config()
    genie_id = os.getenv("GENIE_SPACE_ID", "") or cfg.get("genie_space_id", "")
    genie_quote_id = os.getenv("GENIE_QUOTE_SPACE_ID", "") or cfg.get("genie_quote_space_id", "")
    mart_dashboard_id = os.getenv("MART_DASHBOARD_ID", "") or cfg.get("mart_dashboard_id", "")
    # Derive the new_data_impact workspace folder from the notebooks base
    # (.../files/src/04_models -> .../files/src/new_data_impact) so the frontend
    # deep-links resolve in whatever workspace deployed the bundle.
    nb_base = os.getenv("BUNDLE_NOTEBOOKS_BASE", "")
    new_data_impact_base = (nb_base.rsplit("/", 1)[0] + "/new_data_impact") if nb_base else ""
    return {
        "workspace_host": host,
        "new_data_impact_base": new_data_impact_base,
        "genie_space_id": genie_id,
        "genie_url": f"{host}/genie/rooms/{genie_id}" if genie_id else None,
        "genie_embed_url": f"{host}/embed/genie/rooms/{genie_id}" if genie_id else None,
        "genie_quote_space_id": genie_quote_id,
        "genie_quote_url": f"{host}/genie/rooms/{genie_quote_id}" if genie_quote_id else None,
        "genie_quote_embed_url": f"{host}/embed/genie/rooms/{genie_quote_id}" if genie_quote_id else None,
        "mart_dashboard_id": mart_dashboard_id,
        "mart_dashboard_url": f"{host}/dashboardsv3/{mart_dashboard_id}"            if mart_dashboard_id else None,
        "mart_dashboard_embed_url": f"{host}/embed/dashboardsv3/{mart_dashboard_id}" if mart_dashboard_id else None,
    }


if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
