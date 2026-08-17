# Databricks notebook source
# MAGIC %md
# MAGIC # Create AI/BI assets (Genie spaces + Modelling Mart dashboard)
# MAGIC
# MAGIC Genie spaces and Lakeview dashboards aren't DAB resources, but they ARE
# MAGIC scriptable via the official APIs (`genie create-space`, `lakeview`
# MAGIC dashboards). This job imports the serialized definitions shipped in
# MAGIC `resources/ai_assets/` — retargeting the catalog/schema to this workspace —
# MAGIC so a script deploy produces them with no dependency on a source workspace.
# MAGIC
# MAGIC Idempotent: if a space/dashboard with the same title already exists it's
# MAGIC left as-is. Grants the app service principal so the app can query them.
# MAGIC The app resolves ids by title at runtime, so no id-wiring step is needed.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_pricing_v2_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_workbench")
dbutils.widgets.text("warehouse_id", "")
dbutils.widgets.text("app_service_principal_id", "")
dbutils.widgets.text("assets_path", "")   # DAB passes ${workspace.file_path}/resources/ai_assets
dbutils.widgets.text("parent_path", "/Workspace/Shared")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
wh = dbutils.widgets.get("warehouse_id").strip()
app_sp = dbutils.widgets.get("app_service_principal_id").strip()
assets = dbutils.widgets.get("assets_path").strip()
parent = dbutils.widgets.get("parent_path").strip() or "/Workspace/Shared"

import json
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

if not wh:
    running = [x for x in w.warehouses.list() if str(x.state) == "WarehouseState.RUNNING"]
    wh = (running or list(w.warehouses.list()))[0].id
    print(f"warehouse auto-selected: {wh}")

def _read(fname: str) -> dict:
    path = f"{assets}/{fname}"
    try:
        with open(path) as fh:            # serverless FUSE-mounts workspace files
            return json.load(fh)
    except Exception:
        raw = w.workspace.download(path).read()   # fallback: workspace files API
        return json.loads(raw)

def _retarget(s: str) -> str:
    return s.replace("__PWCATALOG__", catalog).replace("__PWSCHEMA__", schema)

# COMMAND ----------
# MAGIC %md ## Genie spaces

# COMMAND ----------

genie_files = {
    "genie_modelling_mart.json": "CAN_RUN",
    "genie_commercial_quote_review.json": "CAN_RUN",
}
existing_spaces = {}
try:
    resp = w.api_client.do("GET", "/api/2.0/genie/spaces")
    for sp in (resp.get("spaces") or []):
        existing_spaces[sp.get("title")] = sp.get("space_id")
except Exception as e:
    print(f"list spaces: {e}")

genie_ids = {}
for fname in genie_files:
    d = _read(fname)
    title = d["title"]
    if title in existing_spaces:
        genie_ids[title] = existing_spaces[title]
        print(f"✓ Genie '{title}' already exists ({existing_spaces[title]}) — skip")
        continue
    body = {
        "title": title,
        "description": d.get("description", ""),
        "warehouse_id": wh,
        "parent_path": parent,
        "serialized_space": _retarget(d["serialized_space"]),
    }
    created = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)
    sid = created.get("space_id")
    genie_ids[title] = sid
    print(f"✓ created Genie '{title}' → {sid}")
    if app_sp and sid:
        try:
            w.api_client.do("PATCH", f"/api/2.0/permissions/genie/{sid}",
                            body={"access_control_list": [
                                {"service_principal_name": app_sp, "permission_level": "CAN_RUN"}]})
            print(f"  granted {app_sp} CAN_RUN")
        except Exception as e:
            print(f"  grant failed (non-fatal): {str(e)[:100]}")

# COMMAND ----------
# MAGIC %md ## Modelling Mart dashboard

# COMMAND ----------

d = _read("dashboard_mart.json")
dname = d["display_name"]
dash_id = None
try:
    resp = w.api_client.do("GET", "/api/2.0/lakeview/dashboards")
    for dd in (resp.get("dashboards") or []):
        if dd.get("display_name") == dname:
            dash_id = dd.get("dashboard_id"); break
except Exception as e:
    print(f"list dashboards: {e}")

if dash_id:
    print(f"✓ dashboard '{dname}' already exists ({dash_id}) — skip")
else:
    created = w.api_client.do("POST", "/api/2.0/lakeview/dashboards", body={
        "display_name": dname, "warehouse_id": wh, "parent_path": parent,
        "serialized_dashboard": _retarget(d["serialized_dashboard"]),
    })
    dash_id = created.get("dashboard_id")
    print(f"✓ created dashboard '{dname}' → {dash_id}")
    try:
        w.api_client.do("POST", f"/api/2.0/lakeview/dashboards/{dash_id}/published",
                        body={"warehouse_id": wh, "embed_credentials": True})
        print("  published with embed credentials")
    except Exception as e:
        print(f"  publish failed (non-fatal): {str(e)[:100]}")
    if app_sp:
        try:
            w.api_client.do("PATCH", f"/api/2.0/permissions/dashboards/{dash_id}",
                            body={"access_control_list": [
                                {"service_principal_name": app_sp, "permission_level": "CAN_READ"}]})
            print(f"  granted {app_sp} CAN_READ")
        except Exception as e:
            print(f"  grant failed (non-fatal): {str(e)[:100]}")

# COMMAND ----------
print("\nAI assets ready:")
for t, i in genie_ids.items():
    print(f"  Genie: {t} = {i}")
print(f"  Dashboard: {dname} = {dash_id}")
print("\nThe app resolves these by title at runtime — no id-wiring needed.")
