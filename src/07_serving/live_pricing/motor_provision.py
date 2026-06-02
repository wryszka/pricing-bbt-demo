# Databricks notebook source
# MAGIC %md
# MAGIC # Motor live serving — provision
# MAGIC
# MAGIC One-shot bring-up of the motor live-serving stack:
# MAGIC   1. Lakebase online store
# MAGIC   2. UPT (unified_motor_table_live) registered as FE table + SNAPSHOT
# MAGIC      published to the online store
# MAGIC   3. Endpoint `motor_pricing_scorer` re-deployed to pick up online
# MAGIC      feature metadata (auto-resolves now that publish exists)
# MAGIC   4. App SP granted CAN_MANAGE on the publish pipeline so the FastAPI
# MAGIC      claim/event endpoint can fire SNAPSHOT refreshes
# MAGIC   5. live_motor_metrics table for the load-test chart
# MAGIC   6. live_motor_runtime_state stores the publish pipeline_id
# MAGIC
# MAGIC Idempotent. Re-run any time the motor scorer needs a fresh online sync.

# COMMAND ----------

dbutils.widgets.text("catalog_name",            "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",             "pricing_upt")
dbutils.widgets.text("online_store_name",       "motor-pricing-online-store")
dbutils.widgets.text("online_store_capacity",   "CU_4")
dbutils.widgets.text("endpoint_name",           "motor_pricing_scorer")
dbutils.widgets.text("app_service_principal_id","")

# COMMAND ----------

# MAGIC %pip install mlflow databricks-feature-engineering databricks-sdk --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import json, time
catalog       = dbutils.widgets.get("catalog_name")
schema        = dbutils.widgets.get("schema_name")
online_store  = dbutils.widgets.get("online_store_name")
capacity      = dbutils.widgets.get("online_store_capacity")
endpoint_name = dbutils.widgets.get("endpoint_name")
app_sp_id     = dbutils.widgets.get("app_service_principal_id")
fqn           = f"{catalog}.{schema}"
upt_table     = f"{fqn}.unified_motor_table_live"
online_table  = f"{upt_table}_online"      # MUST differ from source name
scorer_uc     = f"{fqn}.motor_pricing_scorer"
metrics_table = f"{fqn}.live_motor_metrics"
state_table   = f"{fqn}.live_motor_runtime_state"

user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get()

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore, PublishSpec, PublishSpecPublishMode
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
from databricks.feature_engineering import FeatureEngineeringClient
from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
w  = WorkspaceClient()
mc = MlflowClient()
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Lakebase online store

# COMMAND ----------

import requests as _rq, json as _json
_host = w.config.host.rstrip("/")
_hdrs = lambda: {**w.config._header_factory(), "Content-Type": "application/json"}

try:
    store = w.feature_store.get_online_store(online_store)
    print(f"store exists: {store.name} state={store.state} cap={store.capacity}")
except Exception:
    print(f"creating store {online_store} at {capacity}…")
    w.feature_store.create_online_store(
        online_store=OnlineStore(name=online_store, capacity=capacity))

# The online store is backed by a Lakebase instance the app STOPS on
# deactivate. Resume it here (and retry through transitional states) before
# waiting for AVAILABLE — otherwise this loop spins forever on a stopped store,
# which is exactly the "Activate hangs" failure. Resuming inside the job (not
# fire-and-forget from the app) makes activate reliable.
def _instance():
    r = _rq.get(f"{_host}/api/2.0/database/instances/{online_store}", headers=_hdrs(), timeout=30)
    return r.json() if r.status_code == 200 else {}

for i in range(72):   # ~6 min
    inst = _instance()
    state = (inst.get("state") or "").upper()
    if state == "AVAILABLE" and not inst.get("effective_stopped"):
        print(f"store AVAILABLE after ~{i*5}s")
        break
    # Issue a resume whenever it's stopped/available-but-flagged and not already
    # mid-transition; STARTING/UPDATING just need to be waited out.
    if (state in ("STOPPED", "AVAILABLE")) and inst.get("effective_stopped", state == "STOPPED"):
        rr = _rq.patch(
            f"{_host}/api/2.0/database/instances/{online_store}?update_mask=stopped",
            headers=_hdrs(), data=_json.dumps({"stopped": False}), timeout=30)
        print(f"  resume PATCH (state={state}) -> {rr.status_code}")
    time.sleep(5)
else:
    raise RuntimeError(f"online store {online_store} not AVAILABLE in 6 min")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Register UPT as FE table + publish SNAPSHOT

# COMMAND ----------

# Enable CDF on UPT (idempotent — needed if we ever flip to CONTINUOUS)
spark.sql(f"ALTER TABLE {upt_table} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

try:
    fe.get_table(name=upt_table)
    print(f"FE table already registered: {upt_table}")
except Exception:
    fe.create_table(name=upt_table, primary_keys="policy_id",
                    df=spark.table(upt_table),
                    description="Motor UPT for live-serving FeatureLookup.")
    print(f"registered: {upt_table}")

publish_pipeline_id = None
try:
    res = w.feature_store.publish_table(
        source_table_name = upt_table,
        publish_spec      = PublishSpec(
            online_store      = online_store,
            online_table_name = online_table,
            publish_mode      = PublishSpecPublishMode.SNAPSHOT,
        ),
    )
    publish_pipeline_id = getattr(res, "pipeline_id", None)
    print(f"publish OK → {online_table}  pipeline={publish_pipeline_id}")
except Exception as e:
    if "already published" in str(e).lower() or "already exists" in str(e).lower():
        for p in w.pipelines.list_pipelines():
            if p.name and online_table in p.name:
                publish_pipeline_id = p.pipeline_id; break
        print(f"already published; pipeline={publish_pipeline_id}")
    else:
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Persist pipeline_id + grant app SP CAN_MANAGE on pipeline

# COMMAND ----------

spark.sql(f"""
    CREATE TABLE IF NOT EXISTS {state_table} (
        key STRING, value STRING, ts TIMESTAMP
    ) USING DELTA
""")
spark.sql(f"""
    MERGE INTO {state_table} t
    USING (SELECT 'publish_pipeline_id' AS key,
                  '{publish_pipeline_id or ""}' AS value,
                  current_timestamp() AS ts) s
    ON t.key = s.key
    WHEN MATCHED THEN UPDATE SET value = s.value, ts = s.ts
    WHEN NOT MATCHED THEN INSERT (key, value, ts) VALUES (s.key, s.value, s.ts)
""")
print(f"persisted publish_pipeline_id={publish_pipeline_id} to {state_table}")

if publish_pipeline_id and app_sp_id:
    try:
        from databricks.sdk.service.iam import AccessControlRequest, PermissionLevel
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
        print(f"grant failed (continuing): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Reconcile motor_pricing_scorer endpoint
# MAGIC
# MAGIC With the Lakebase publish in place, the FE wrapper can now resolve
# MAGIC online metadata. Trigger a config update so the endpoint moves out of
# MAGIC the UPDATE_FAILED state.

# COMMAND ----------

scorer_version = None
versions = list(mc.search_model_versions(f"name='{scorer_uc}'"))
if versions:
    scorer_version = str(max(int(v.version) for v in versions))
    print(f"scorer latest: v{scorer_version}")

if scorer_version:
    # Provision the live endpoint with explicit 4-64 concurrency and
    # scale_to_zero DISABLED — the endpoint stays warm for the whole time the
    # system is "on" and only comes down when the teardown button removes it
    # (so it rises/falls with the system, not on its own idle timer). Set via
    # REST so the min/max provisioned-concurrency fields apply regardless of
    # the installed databricks-sdk version.
    import requests as _rq, json as _json
    _served_entity = {
        "entity_name": scorer_uc,
        "entity_version": scorer_version,
        "scale_to_zero_enabled": False,
        "min_provisioned_concurrency": 4,
        "max_provisioned_concurrency": 64,
        "workload_type": "CPU",
    }
    _host  = w.config.host.rstrip("/")
    _hdrs  = {**w.config._header_factory(), "Content-Type": "application/json"}
    try:
        existing = w.serving_endpoints.get(endpoint_name)
    except Exception:
        existing = None

    # Endpoint is route-optimized (direct data-plane path for low-latency
    # serving). Route optimization is create-time only, so the Start button's
    # recreate must always set it — and if a non-route-optimized endpoint is
    # found (e.g. a legacy one), delete and recreate it. Queried via endpoint_url
    # with OAuth (the app resolves the host and uses the app SP's OAuth token).
    _create_payload = {"name": endpoint_name, "route_optimized": True,
                       "config": {"served_entities": [_served_entity]}}
    _is_ro = bool(getattr(existing, "route_optimized", False)) if existing else False

    if existing is None:
        _r = _rq.post(f"{_host}/api/2.0/serving-endpoints", headers=_hdrs,
                      data=_json.dumps(_create_payload), timeout=60)
        print(f"route-optimized endpoint create -> {_r.status_code}: {_r.text[:200]}")
    elif not _is_ro:
        print(f"endpoint {endpoint_name} exists but is NOT route-optimized — deleting to recreate…")
        _rq.delete(f"{_host}/api/2.0/serving-endpoints/{endpoint_name}", headers=_hdrs, timeout=60)
        for _ in range(60):
            try:
                w.serving_endpoints.get(endpoint_name); time.sleep(5)
            except Exception:
                break
        _r = _rq.post(f"{_host}/api/2.0/serving-endpoints", headers=_hdrs,
                      data=_json.dumps(_create_payload), timeout=60)
        print(f"route-optimized endpoint recreate -> {_r.status_code}: {_r.text[:200]}")
    else:
        _r = _rq.put(f"{_host}/api/2.0/serving-endpoints/{endpoint_name}/config", headers=_hdrs,
                     data=_json.dumps({"served_entities": [_served_entity]}), timeout=60)
        print(f"route-optimized endpoint reconcile -> {_r.status_code}: {_r.text[:200]}")

    # Grant the app SP CAN_QUERY on the endpoint. A route-optimized endpoint
    # mints its scoped query token from the caller's endpoint permission, and a
    # teardown→recreate resets the ACL to creator-only — so without this the app
    # would 401 after every Start. PATCH merges, so it's safe to re-run.
    if app_sp_id:
        try:
            _eid = w.serving_endpoints.get(endpoint_name).id
            _pr = _rq.patch(
                f"{_host}/api/2.0/permissions/serving-endpoints/{_eid}", headers=_hdrs,
                data=_json.dumps({"access_control_list": [
                    {"service_principal_name": app_sp_id, "permission_level": "CAN_QUERY"}]}),
                timeout=30)
            print(f"app SP CAN_QUERY grant on endpoint {_eid} -> {_pr.status_code}")
        except Exception as e:
            print(f"endpoint query grant failed (continuing): {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Metrics table

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

dbutils.notebook.exit(json.dumps({
    "online_store":         online_store,
    "online_table":         online_table,
    "endpoint":             endpoint_name,
    "scorer_version":       scorer_version,
    "publish_pipeline_id":  publish_pipeline_id,
    "metrics_table":        metrics_table,
    "state_table":          state_table,
}))
