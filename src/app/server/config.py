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


def reset_workspace_client() -> None:
    """Drop the cached WorkspaceClient so the next call builds a fresh one.

    The route-optimized scorer is queried via the SDK's data-plane token source,
    which lives on the WorkspaceClient. When the endpoint is recreated
    (deactivate→activate mints a new data-plane host), the long-running app's
    cached client can't mint a scoped token for the new endpoint and every query
    fails with `invalid_authorization_details` until the app restarts. Rebuilding
    the client in-process re-mints correctly — same effect as a restart, no
    redeploy needed."""
    global _workspace_client
    _workspace_client = None


def get_catalog() -> str:
    return os.getenv("CATALOG_NAME", "lr_serverless_aws_us_catalog")


def get_schema() -> str:
    return os.getenv("SCHEMA_NAME", "pricing_upt")


def get_warehouse_id() -> str:
    return os.getenv("WAREHOUSE_ID", "ab79eced8207d29b")


def fqn(table: str) -> str:
    return f"{get_catalog()}.{get_schema()}.{table}"


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
