# Databricks notebook source
# MAGIC %md
# MAGIC # Deploy Model to Serving Endpoint
# MAGIC
# MAGIC Deploys the GLM frequency model to a Mosaic AI Model Serving endpoint.
# MAGIC
# MAGIC **The key point:** Because the model was logged with `fe.log_model()` and
# MAGIC `FeatureLookup`, the endpoint automatically fetches features from the
# MAGIC Online Feature Store at inference time. **No custom integration code needed.**
# MAGIC
# MAGIC You send just a `policy_id` → the endpoint looks up all features → returns
# MAGIC the prediction. This is the "one platform" story.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("endpoint_name", "pricing-frequency-endpoint")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")

model_name = f"{catalog}.{schema}.glm_frequency_model"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Find the latest model version

# COMMAND ----------

import mlflow
from databricks.sdk import WorkspaceClient

mlflow.set_registry_uri("databricks-uc")
client = mlflow.MlflowClient()
w = WorkspaceClient()

# Get latest version of the registered model
versions = client.search_model_versions(f"name='{model_name}'")
if not versions:
    raise ValueError(f"No versions found for model {model_name}. Run model training first.")

latest = sorted(versions, key=lambda v: int(v.version), reverse=True)[0]
champion_version = latest.version

print(f"Model: {model_name}")
print(f"Latest version: {champion_version}")
print(f"Run ID: {latest.run_id}")
print(f"Status: {latest.status}")

# If there's a previous version, use it as challenger
challenger_version = None
if len(versions) >= 2:
    challenger = sorted(versions, key=lambda v: int(v.version), reverse=True)[1]
    challenger_version = challenger.version
    print(f"Challenger version: {challenger_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Deploy to Serving Endpoint
# MAGIC
# MAGIC This endpoint automatically looks up features from the Online Feature Store
# MAGIC because the model was logged with FeatureLookup. No custom integration code
# MAGIC needed — the platform handles feature resolution at serving time.

# COMMAND ----------

from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedEntityInput,
    TrafficConfig,
    Route,
)

# Build served entities — champion + optional challenger
served_entities = [
    ServedEntityInput(
        entity_name=model_name,
        entity_version=champion_version,
        name="champion",
        workload_size="Small",
        scale_to_zero_enabled=True,  # Cost: scale to zero when not in use
    )
]

# Traffic routing
routes = [Route(served_model_name="champion", traffic_percentage=100)]

if challenger_version:
    served_entities.append(
        ServedEntityInput(
            entity_name=model_name,
            entity_version=challenger_version,
            name="challenger",
            workload_size="Small",
            scale_to_zero_enabled=True,
        )
    )
    # 90/10 traffic split for safe rollout
    routes = [
        Route(served_model_name="champion", traffic_percentage=90),
        Route(served_model_name="challenger", traffic_percentage=10),
    ]

config = EndpointCoreConfigInput(
    served_entities=served_entities,
    traffic_config=TrafficConfig(routes=routes),
)

# Create or update the endpoint
try:
    existing = w.serving_endpoints.get(endpoint_name)
    print(f"Updating existing endpoint: {endpoint_name}")
    w.serving_endpoints.update_config(name=endpoint_name, served_entities=served_entities,
                                       traffic_config=TrafficConfig(routes=routes))
except Exception:
    print(f"Creating new endpoint: {endpoint_name}")
    w.serving_endpoints.create(
        name=endpoint_name,
        config=config,
    )

print(f"✓ Endpoint '{endpoint_name}' deployment initiated")
print(f"  Champion: v{champion_version} (90% traffic)")
if challenger_version:
    print(f"  Challenger: v{challenger_version} (10% traffic)")

# FUTURE: Mock Earnix/Radar integration — external system calls this endpoint
# for feature enrichment or model scoring during its own pricing run.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Wait for endpoint to be ready

# COMMAND ----------

import time

print("Waiting for endpoint to become ready...")
for i in range(60):
    ep = w.serving_endpoints.get(endpoint_name)
    state = ep.state.ready if ep.state else "UNKNOWN"
    if str(state) == "EndpointStateReady.READY":
        print(f"✓ Endpoint is READY (took ~{i * 10}s)")
        break
    print(f"  State: {state} ({i * 10}s)")
    time.sleep(10)
else:
    print("⚠ Endpoint not ready after 10 minutes — check the Serving UI")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Log to audit trail

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="model_deployed",
    entity_type="endpoint",
    entity_id=endpoint_name,
    entity_version=f"champion=v{champion_version}",
    user_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    details={
        "model_name": model_name,
        "champion_version": champion_version,
        "challenger_version": challenger_version,
        "traffic_split": "90/10" if challenger_version else "100/0",
        "endpoint_name": endpoint_name,
    },
    source="notebook",
)
print("✓ Deployment logged to audit_log")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture
# MAGIC
# MAGIC ```
# MAGIC   Client sends: { "policy_id": "POL-100042" }
# MAGIC                        │
# MAGIC                        ▼
# MAGIC        ┌────────────────────────────────┐
# MAGIC        │   Model Serving Endpoint        │
# MAGIC        │   (pricing-frequency-endpoint)  │
# MAGIC        │                                 │
# MAGIC        │  1. Receive policy_id           │
# MAGIC        │  2. Auto-lookup features ──────►│──► Online Store (Lakebase)
# MAGIC        │     from FeatureLookup spec     │       │
# MAGIC        │  3. Run model.predict()   ◄─────│◄──── features returned
# MAGIC        │  4. Return prediction           │
# MAGIC        └────────────────────────────────┘
# MAGIC                        │
# MAGIC                        ▼
# MAGIC   Response: { "prediction": 0.42 }
# MAGIC ```
# MAGIC
# MAGIC **No custom code needed.** The model knows which features to look up because
# MAGIC lineage was captured at training time via `FeatureLookup` and Unity Catalog.
