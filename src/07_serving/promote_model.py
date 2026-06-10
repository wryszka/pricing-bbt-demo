# Databricks notebook source
# MAGIC %md
# MAGIC # Promote Challenger to Champion
# MAGIC
# MAGIC Promotes the challenger model to champion by updating the traffic routing
# MAGIC on the serving endpoint. Logs the promotion event to the audit trail.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("endpoint_name", "pricing-frequency-endpoint")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
endpoint_name = dbutils.widgets.get("endpoint_name")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import TrafficConfig, Route

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Before: Current state

# COMMAND ----------

ep = w.serving_endpoints.get(endpoint_name)
print(f"Endpoint: {endpoint_name}")
print(f"State: {ep.state.ready if ep.state else 'UNKNOWN'}")

if ep.config and ep.config.served_entities:
    for entity in ep.config.served_entities:
        print(f"  {entity.name}: v{entity.entity_version}")

if ep.config and ep.config.traffic_config and ep.config.traffic_config.routes:
    print("Traffic:")
    for route in ep.config.traffic_config.routes:
        print(f"  {route.served_model_name}: {route.traffic_percentage}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote: Route 100% traffic to challenger

# COMMAND ----------

# Find current champion and challenger
entities = ep.config.served_entities if ep.config else []
champion = next((e for e in entities if e.name == "champion"), None)
challenger = next((e for e in entities if e.name == "challenger"), None)

if not challenger:
    print("No challenger deployed — nothing to promote")
    dbutils.notebook.exit("NO_CHALLENGER")

old_champion_version = champion.entity_version if champion else "?"
new_champion_version = challenger.entity_version

print(f"Promoting: v{new_champion_version} (challenger) → champion")
print(f"Demoting: v{old_champion_version} (champion) → removed")

# Update traffic to 100% challenger
w.serving_endpoints.update_config(
    name=endpoint_name,
    traffic_config=TrafficConfig(routes=[
        Route(served_model_name="challenger", traffic_percentage=100),
    ]),
)

print(f"✓ Traffic routed 100% to v{new_champion_version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## After: New state

# COMMAND ----------

import time
time.sleep(5)

ep = w.serving_endpoints.get(endpoint_name)
if ep.config and ep.config.traffic_config and ep.config.traffic_config.routes:
    print("Updated traffic:")
    for route in ep.config.traffic_config.routes:
        print(f"  {route.served_model_name}: {route.traffic_percentage}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log promotion to audit trail

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="model_promoted",
    entity_type="endpoint",
    entity_id=endpoint_name,
    entity_version=f"v{new_champion_version}",
    user_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    details={
        "old_champion": old_champion_version,
        "new_champion": new_champion_version,
        "endpoint": endpoint_name,
    },
    source="notebook",
)
print("✓ Promotion logged to audit_log")
