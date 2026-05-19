import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routes import datasets, agent, features, deployment, governance, quote_stream, genie, development, review, compare, factory, factory_real, pricing, admin, supervisor, live_pricing
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
        await factory.ensure_factory_tables()
        logger.info("Approvals and factory tables ready")
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
app.include_router(deployment.router)
app.include_router(governance.router)
app.include_router(quote_stream.router)
app.include_router(genie.router)
app.include_router(development.router)
app.include_router(review.router)
app.include_router(compare.router)
app.include_router(factory.router)
app.include_router(factory_real.router)
app.include_router(pricing.router)
app.include_router(admin.router)
app.include_router(supervisor.router)
app.include_router(live_pricing.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def config():
    host = get_workspace_host()
    genie_id = os.getenv("GENIE_SPACE_ID", "")
    genie_quote_id = os.getenv("GENIE_QUOTE_SPACE_ID", "")
    mart_dashboard_id = os.getenv("MART_DASHBOARD_ID", "")
    return {
        "workspace_host": host,
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
