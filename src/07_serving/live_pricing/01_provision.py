# Databricks notebook source
# MAGIC %md
# MAGIC # Live Pricing System — provision
# MAGIC
# MAGIC One-shot bring-up of the live pricing demo:
# MAGIC  1. Lakebase online store (CU_2 — smallest performant tier)
# MAGIC  2. Continuous publish of `unified_pricing_table_live` → online store
# MAGIC  3. `pricing_scorer` champion logged + deployed to a route-optimised
# MAGIC     Model Serving endpoint with `scale_to_zero=True`
# MAGIC  4. 5-request warm-up so the first demo quote is sub-second
# MAGIC  5. `live_pricing_metrics` table for the load-test chart
# MAGIC
# MAGIC Idempotent — every step skips work that's already done.
# MAGIC
# MAGIC Notebook exits with JSON describing the live state. Tear-down is the
# MAGIC inverse: `02_teardown.py`.

# COMMAND ----------

dbutils.widgets.text("catalog_name",      "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",       "pricing_upt")
dbutils.widgets.text("online_store_name", "pricing-upt-online-store-live")
dbutils.widgets.text("endpoint_name",     "pricing_scorer")
dbutils.widgets.text("online_store_capacity", "CU_2")
dbutils.widgets.text("app_service_principal_id", "")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-feature-engineering databricks-sdk \
# MAGIC   statsmodels lightgbm scikit-learn --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json, os, time
catalog        = dbutils.widgets.get("catalog_name")
schema         = dbutils.widgets.get("schema_name")
online_store   = dbutils.widgets.get("online_store_name")
endpoint_name  = dbutils.widgets.get("endpoint_name")
capacity       = dbutils.widgets.get("online_store_capacity")
app_sp_id      = dbutils.widgets.get("app_service_principal_id")
fqn            = f"{catalog}.{schema}"
upt_table      = f"{fqn}.unified_pricing_table_live"
scorer_uc_name = f"{fqn}.pricing_scorer"
metrics_table  = f"{fqn}.live_pricing_metrics"

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

# MAGIC %run ../../utils/audit

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, ServedEntityInput,
)
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
w  = WorkspaceClient()
mc = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lakebase online store + SNAPSHOT publish
# MAGIC
# MAGIC SNAPSHOT mode (not CONTINUOUS) — CONTINUOUS publish requires DLT
# MAGIC streaming pipeline support which isn't enabled on the dev workspace
# MAGIC tier. The claim-filing endpoint refreshes the snapshot on demand by
# MAGIC triggering an update on the backing DLT pipeline (id captured below).
# MAGIC Same architecture works on prod — flip publish_mode to CONTINUOUS
# MAGIC there once the workspace tier supports it.

# COMMAND ----------

from databricks.sdk.service.ml import (
    OnlineStore, PublishSpec, PublishSpecPublishMode,
)
from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# 1a. Provision Lakebase store
try:
    store = w.feature_store.get_online_store(online_store)
    print(f"online store exists: {store.name} (state={store.state})")
except Exception:
    print(f"creating online store {online_store} at {capacity}…")
    w.feature_store.create_online_store(
        online_store=OnlineStore(name=online_store, capacity=capacity)
    )
for i in range(60):
    store = w.feature_store.get_online_store(online_store)
    if str(store.state).endswith("AVAILABLE"):
        print(f"online store AVAILABLE after {i*5}s")
        break
    time.sleep(5)
else:
    raise RuntimeError(f"online store {online_store} not AVAILABLE in 5 min")

# 1b. Register UPT as FE feature table (idempotent — the diag run confirmed
# UPT is already registered in dev; on a fresh catalog this would be the
# first-time call)
try:
    fe.get_table(name=upt_table)
    print(f"FE table already registered: {upt_table}")
except Exception:
    print(f"registering {upt_table} as FE table…")
    fe.create_table(name=upt_table, primary_keys="policy_id",
                    df=spark.table(upt_table),
                    description="UPT for live pricing FeatureLookup.")

# 1c. Publish UPT to Lakebase, SNAPSHOT mode.
# Important: online_table_name MUST differ from the source table name —
# publish_table rejects "table X is not a valid online feature table" when
# both names match (an undocumented constraint we surfaced via the diag run).
# Use the same schema with an `_online` suffix.
online_table_name = f"{upt_table}_online"
publish_pipeline_id = None
try:
    res = w.feature_store.publish_table(
        source_table_name = upt_table,
        publish_spec      = PublishSpec(
            online_store      = online_store,
            online_table_name = online_table_name,
            publish_mode      = PublishSpecPublishMode.SNAPSHOT,
        ),
    )
    publish_pipeline_id = getattr(res, "pipeline_id", None)
    print(f"publish_table OK (SNAPSHOT) → {online_table_name}  pipeline_id={publish_pipeline_id}")
except Exception as e:
    if "already published" in str(e).lower() or "already exists" in str(e).lower():
        print("already published — fetching existing pipeline_id")
        for p in w.pipelines.list_pipelines():
            if p.name and online_table_name in p.name:
                publish_pipeline_id = p.pipeline_id
                break
        print(f"  resolved pipeline_id={publish_pipeline_id}")
    else:
        raise

# 1d. Persist the pipeline_id so the FastAPI /claim endpoint can trigger
# refreshes without having to re-derive it on every call.
spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {fqn}.live_pricing_runtime_state (
        key   STRING,
        value STRING,
        ts    TIMESTAMP
    ) USING DELTA
""")
spark.sql(f"""
    MERGE INTO {fqn}.live_pricing_runtime_state t
    USING (SELECT 'publish_pipeline_id' AS key,
                  '{publish_pipeline_id or ""}' AS value,
                  current_timestamp() AS ts) s
    ON t.key = s.key
    WHEN MATCHED THEN UPDATE SET value = s.value, ts = s.ts
    WHEN NOT MATCHED THEN INSERT (key, value, ts) VALUES (s.key, s.value, s.ts)
""")
print(f"persisted publish_pipeline_id={publish_pipeline_id} to live_pricing_runtime_state")

# 1e. Grant the app SP CAN_MANAGE on the pipeline so /api/live-pricing/claim
# can fire start_update from the FastAPI route. publish_table creates the
# pipeline owned by whoever ran the notebook — the app SP needs an explicit
# grant. Note: Lakebase synced-table pipelines reject CAN_RUN on
# start_update; only CAN_MANAGE works. Idempotent.
if publish_pipeline_id and app_sp_id:
    try:
        from databricks.sdk.service.iam import (
            AccessControlRequest, PermissionLevel,
        )
        w.permissions.update(
            request_object_type = "pipelines",
            request_object_id   = publish_pipeline_id,
            access_control_list = [AccessControlRequest(
                service_principal_name = app_sp_id,
                permission_level       = PermissionLevel.CAN_MANAGE,
            )],
        )
        print(f"granted CAN_MANAGE to {app_sp_id} on pipeline {publish_pipeline_id}")
    except Exception as e:
        print(f"pipeline ACL grant failed (continuing): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. pricing_scorer champion → endpoint
# MAGIC
# MAGIC If no `pricing_scorer` model exists yet (first-run on this catalog),
# MAGIC trigger the production scorer notebook to log + register one. Then
# MAGIC deploy the latest version to a route-optimised endpoint.

# COMMAND ----------

def _latest_version(name: str) -> str | None:
    versions = list(mc.search_model_versions(f"name='{name}'"))
    if not versions:
        return None
    return str(max(int(v.version) for v in versions))

scorer_version = _latest_version(scorer_uc_name)
if scorer_version is None:
    print(f"no version of {scorer_uc_name} found — running 04_models/production/pricing_scorer.py")
    dbutils.notebook.run(
        "../../04_models/production/pricing_scorer",
        timeout_seconds=1800,
        arguments={
            "catalog_name":  catalog,
            "schema_name":   schema,
            "endpoint_name": endpoint_name,
        },
    )
    scorer_version = _latest_version(scorer_uc_name)
    if scorer_version is None:
        raise RuntimeError(f"failed to log {scorer_uc_name}")

print(f"deploying {scorer_uc_name} v{scorer_version} → endpoint {endpoint_name}")

served = [ServedEntityInput(
    entity_name           = scorer_uc_name,
    entity_version        = str(scorer_version),
    scale_to_zero_enabled = True,
    workload_size         = "Large",
)]

# Reconcile the endpoint state before issuing an update_config:
#  - If endpoint absent → create
#  - If endpoint already serving the target version → skip
#  - If a pending config update is already targeting that version (e.g.
#    from a parallel pricing_scorer_deploy run) → skip and let it land
#  - Otherwise → update_config
existing = None
try:
    existing = w.serving_endpoints.get(endpoint_name)
except Exception:
    pass

def _versions(cfg):
    return {(e.entity_version) for e in (getattr(cfg, "served_entities", []) or [])}

if existing is None:
    w.serving_endpoints.create(
        name             = endpoint_name,
        config           = EndpointCoreConfigInput(name=endpoint_name, served_entities=served),
        route_optimized  = True,
    )
    print("created endpoint (route_optimized)")
else:
    served_versions  = _versions(getattr(existing, "config", None))
    pending_versions = _versions(getattr(existing, "pending_config", None))
    target = str(scorer_version)
    if target in served_versions and not pending_versions:
        print(f"endpoint already serving v{target} — skip update")
    elif target in pending_versions:
        print(f"endpoint pending update to v{target} — skip; the update will land")
    else:
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=served)
        print(f"updated existing endpoint to v{target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Wait for endpoint READY + warm-up

# COMMAND ----------

for i in range(180):  # up to 15 min — first deploy can be slow
    ep = w.serving_endpoints.get(endpoint_name)
    state = getattr(ep.state, "ready", None)
    config_state = getattr(ep.state, "config_update", None)
    print(f"  endpoint state ready={state} config_update={config_state}")
    if str(state).endswith("READY") and not str(config_state).endswith("IN_PROGRESS"):
        break
    time.sleep(5)
else:
    raise RuntimeError(f"endpoint {endpoint_name} not READY in 15 min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics table for load-test chart
# MAGIC
# MAGIC Created BEFORE warm-up so a warm-up failure (auth quirks in the
# MAGIC notebook context) doesn't leave the system without a metrics sink.

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {metrics_table} (
        ts             TIMESTAMP,
        source         STRING,
        policy_id      STRING,
        latency_ms     DOUBLE,
        final_premium  DOUBLE,
        status_code    INT,
        run_id         STRING
    ) USING DELTA
""")
print(f"metrics table ready: {metrics_table}")

# COMMAND ----------

# Warm up — issue 5 sequential quotes against random policy_ids; discard the
# first 2 latencies (cold path / lazy import). Non-fatal: notebook auth into
# the serving endpoint is finicky on dev — log the failure but don't fail
# the provision job, since the FastAPI app warm-paths the endpoint naturally
# on the first user click.
import requests as _rq

warm_latencies_ms: list[float] = []
warm_p50: float | None = None
try:
    sample_pids = [r["policy_id"] for r in spark.sql(
        f"SELECT policy_id FROM {upt_table} LIMIT 5"
    ).collect()]
    print(f"warm-up policy ids: {sample_pids}")

    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()
    for pid in sample_pids:
        t0 = time.perf_counter()
        resp = _rq.post(
            f"{host}/serving-endpoints/{endpoint_name}/invocations",
            headers={**token, "Content-Type": "application/json"},
            json={"dataframe_records": [{"policy_id": pid}]},
            timeout=120,
        )
        dt = (time.perf_counter() - t0) * 1000
        if resp.status_code == 200:
            warm_latencies_ms.append(round(dt, 1))
            print(f"  {pid} → {dt:.0f} ms")
        else:
            print(f"  {pid} → HTTP {resp.status_code}: {resp.text[:200]}")

    warm_after_initial = warm_latencies_ms[2:]
    warm_p50 = sorted(warm_after_initial)[len(warm_after_initial) // 2] if warm_after_initial else None
    print(f"warm latencies: {warm_latencies_ms}  warm-after-initial p50: {warm_p50} ms")
except Exception as e:
    print(f"warm-up failed (non-fatal): {type(e).__name__}: {str(e)[:200]}")

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type    = "live_pricing_started",
    entity_type   = "endpoint",
    entity_id     = endpoint_name,
    entity_version= str(scorer_version),
    user_id       = user,
    details={
        "online_store":         online_store,
        "online_table":         upt_table,
        "scorer_version":       scorer_version,
        "warm_latencies_ms":    warm_latencies_ms,
        "warm_p50_ms":          warm_p50,
        "publish_mode":         "SNAPSHOT",
        "publish_pipeline_id":  publish_pipeline_id,
    },
    source="notebook",
)

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "online_store_name":   online_store,
    "endpoint_name":       endpoint_name,
    "scorer_version":      scorer_version,
    "warm_p50_ms":         warm_p50,
    "warm_latencies_ms":   warm_latencies_ms,
    "metrics_table":       metrics_table,
    "publish_pipeline_id": publish_pipeline_id,
}))
