import os
import logging
import threading
import contextvars

from databricks.sdk import WorkspaceClient

logger = logging.getLogger(__name__)
_workspace_client: WorkspaceClient | None = None
# Guards the singleton so concurrent requests don't race to build/reset it
# (multi-user: two users hitting a cold app both saw None and both built one).
_client_lock = threading.Lock()

# Request-scoped end user, set by middleware from the Databricks Apps
# forwarded-identity header. Lets audit attribute actions to the real person
# instead of the app service principal when many people use the app at once.
_current_user: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "current_user", default=None)


def is_databricks_app() -> bool:
    return os.getenv("DATABRICKS_APP_NAME") is not None


def get_workspace_client() -> WorkspaceClient:
    global _workspace_client
    if _workspace_client is None:
        with _client_lock:                       # double-checked lock
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
    with _client_lock:
        _workspace_client = None


def set_current_user(user: str | None) -> None:
    """Set the request's end user (called by middleware from the forwarded
    identity header). Cleared per request via the contextvar default."""
    _current_user.set(user or None)


def get_catalog() -> str:
    return os.getenv("CATALOG_NAME", "lr_serverless_aws_us_catalog")


def get_schema() -> str:
    return os.getenv("SCHEMA_NAME", "pricing_upt")


def get_warehouse_id() -> str:
    return os.getenv("WAREHOUSE_ID", "")


def get_bundle_files_base() -> str:
    """Root of the deployed bundle's source tree, i.e. the `.../files/src`
    directory. Set as an env var per target (app.<target>.yaml) so the app can
    build workspace links to notebooks without hardcoding a deployer's home
    path. Falls back to the org-shared production convention (username-free) so
    a mis-configured deploy still points somewhere plausible rather than at a
    specific person's home directory."""
    base = os.getenv("BUNDLE_FILES_BASE", "").rstrip("/")
    if base:
        return base
    bundle = os.getenv("BUNDLE_NAME", "pricing-workbench")
    target = os.getenv("BUNDLE_TARGET", "prod")
    return f"/Workspace/Shared/.bundle/{bundle}/{target}/files/src"


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


_asset_cache: dict[str, str] = {}


def resolve_genie_space_by_title(title: str) -> str:
    """Look up a Genie space id by title (cached). Lets the app self-configure on
    a fresh deploy where GENIE_SPACE_ID isn't wired: the create_ai_assets job
    makes spaces with known titles and the app finds them. Env var still wins."""
    if not title:
        return ""
    if title in _asset_cache:
        return _asset_cache[title]
    try:
        resp = get_workspace_client().api_client.do("GET", "/api/2.0/genie/spaces")
        for sp in (resp.get("spaces") or []):
            if sp.get("title") == title:
                _asset_cache[title] = sp.get("space_id") or ""
                return _asset_cache[title]
    except Exception as e:
        logger.info("genie title resolve failed for %r: %s", title, e)
    return ""


def resolve_dashboard_by_title(name: str) -> str:
    """Look up a Lakeview dashboard id by display_name (cached)."""
    if not name:
        return ""
    key = f"dash::{name}"
    if key in _asset_cache:
        return _asset_cache[key]
    try:
        resp = get_workspace_client().api_client.do("GET", "/api/2.0/lakeview/dashboards")
        for d in (resp.get("dashboards") or []):
            if d.get("display_name") == name:
                _asset_cache[key] = d.get("dashboard_id") or ""
                return _asset_cache[key]
    except Exception as e:
        logger.info("dashboard title resolve failed for %r: %s", name, e)
    return ""


def get_current_user() -> str:
    # Prefer the real end user (set per request from the forwarded header) so
    # audit attributes to the person, not the app service principal.
    u = _current_user.get()
    if u:
        return u
    try:
        me = get_workspace_client().current_user.me()
        return me.user_name or me.display_name or "unknown"
    except Exception:
        return os.getenv("USER", "demo-user")
