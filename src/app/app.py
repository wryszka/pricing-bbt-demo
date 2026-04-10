import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routes import datasets, models, agent, features, governance
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
    logger.info("Starting Pricing UPT Ingestion Review App")
    try:
        await datasets.ensure_approvals_table()
        await models.ensure_model_factory_tables()
        logger.info("Approvals and model factory tables ready")
    except Exception:
        logger.exception("Failed to ensure tables — will retry on first request")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="Pricing Data Ingestion Review",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(datasets.router)
app.include_router(models.router)
app.include_router(agent.router)
app.include_router(features.router)
app.include_router(governance.router)


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.get("/api/config")
async def config():
    host = get_workspace_host()
    genie_id = os.getenv("GENIE_SPACE_ID", "")
    return {
        "workspace_host": host,
        "genie_space_id": genie_id,
        "genie_url": f"{host}/genie/rooms/{genie_id}" if genie_id else None,
        "genie_embed_url": f"{host}/embed/genie/rooms/{genie_id}" if genie_id else None,
    }


if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
