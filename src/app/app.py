import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from server.routes import datasets, models, agent
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


@app.get("/api/health")
async def health():
    return {"status": "ok"}


if FRONTEND_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=FRONTEND_DIR / "assets"), name="assets")

    @app.get("/{full_path:path}")
    async def serve_spa(full_path: str):
        file_path = FRONTEND_DIR / full_path
        if file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(FRONTEND_DIR / "index.html")
