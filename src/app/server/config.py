import os
import logging

from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)
_workspace_client: WorkspaceClient | None = None


def is_databricks_app() -> bool:
    return os.getenv("DATABRICKS_APP_NAME") is not None


def get_workspace_client() -> WorkspaceClient:
    global _workspace_client
    if _workspace_client is None:
        if is_databricks_app():
            _workspace_client = WorkspaceClient()
        else:
            profile = os.getenv("DATABRICKS_PROFILE", "DEFAULT")
            _workspace_client = WorkspaceClient(profile=profile)
    return _workspace_client


# Runtime config comes from env vars set in src/app/app.yaml (swapped per target
# by deploy.sh). The fallbacks below are generic — a real deployment always sets
# the env, so a fallback firing means the app.yaml env block is missing.
def get_catalog() -> str:
    return os.getenv("CATALOG_NAME", "pricing_workbench")


def get_schema() -> str:
    return os.getenv("SCHEMA_NAME", "pricing_upt")


def get_warehouse_id() -> str:
    return os.getenv("WAREHOUSE_ID", "")


def fqn(table: str) -> str:
    return f"{get_catalog()}.{get_schema()}.{table}"


def find_job_id(w, base_name: str):
    """Find a job id by base name, tolerant of DAB development-mode job name
    prefixes ('[dev <user>] '). The Jobs `name=` filter is an exact match, which
    misses dev-prefixed names, so we list and substring-match. Returns the first
    match or None."""
    try:
        for j in w.jobs.list():
            if j.settings and base_name in (j.settings.name or ""):
                return j.job_id
    except Exception:
        return None
    return None


def get_workspace_host() -> str:
    host = os.getenv("DATABRICKS_HOST", "")
    if not host:
        try:
            host = get_workspace_client().config.host
        except Exception:
            host = ""  # Could not resolve — set DATABRICKS_HOST env var
    host = host.rstrip("/")
    if host and not host.startswith("http"):
        host = f"https://{host}"
    return host


def get_current_user() -> str:
    try:
        me = get_workspace_client().current_user.me()
        return me.user_name or me.display_name or "unknown"
    except Exception:
        return os.getenv("USER", "demo-user")
