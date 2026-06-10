# Databricks notebook source
# MAGIC %md
# MAGIC # Online Feature Store Setup
# MAGIC
# MAGIC Publishes the Unified Pricing Table (UPT) to a Databricks Online Feature Store
# MAGIC backed by **Lakebase** — providing millisecond-latency key-value lookups for
# MAGIC real-time pricing.
# MAGIC
# MAGIC ## What is an Online Feature Store?
# MAGIC
# MAGIC | | Offline (Delta Lake) | Online (Lakebase) |
# MAGIC |---|---|---|
# MAGIC | **Use case** | Model training, batch scoring | Real-time model serving |
# MAGIC | **Latency** | Seconds–minutes | Sub-10ms |
# MAGIC | **Access pattern** | Full table scans, SQL | Key-value lookups by primary key |
# MAGIC | **Storage** | Delta Lake (object storage) | Lakebase (managed PostgreSQL) |
# MAGIC | **Update** | Batch writes, streaming | Synced from offline automatically |
# MAGIC
# MAGIC ## Why this matters for pricing
# MAGIC
# MAGIC A commercial insurance quote engine needs to look up 100+ features for a single
# MAGIC policy in under 30ms. The offline Delta table is great for training but too slow
# MAGIC for live quoting. The online store solves this — same features, millisecond access.
# MAGIC
# MAGIC ## Prerequisites
# MAGIC - Unity Catalog with a feature table (PK constraint on the source table)
# MAGIC - `databricks-feature-engineering` package
# MAGIC - Workspace with Online Feature Store / Lakebase enabled

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("online_store_name", "pricing-upt-online-store")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
online_store_name = dbutils.widgets.get("online_store_name")

upt_table = f"{catalog}.{schema}.unified_pricing_table_live"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 1: Verify the feature table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

fe = FeatureEngineeringClient()

# The UPT has a PRIMARY KEY constraint on policy_id (set in build_upt.py)
# This makes it automatically visible as a feature table in UC
upt_df = spark.table(upt_table)
row_count = upt_df.count()
col_count = len(upt_df.columns)

print(f"Feature table: {upt_table}")
print(f"  Rows: {row_count:,}")
print(f"  Columns: {col_count}")
print(f"  Primary key: policy_id")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 2: Create the Online Store (if it doesn't exist)
# MAGIC
# MAGIC The online store is a managed Lakebase instance that holds a copy of the
# MAGIC feature table optimised for key-value lookups. You create it once, then
# MAGIC publish tables to it.

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import OnlineStore

w = WorkspaceClient()

# Check if store already exists
try:
    store = w.feature_store.get_online_store(online_store_name)
    print(f"✓ Online store exists: {store.name} (state: {store.state})")
except Exception:
    # Create the store — CU_1 is the smallest capacity (1 compute unit)
    print(f"Creating online store: {online_store_name}...")
    store = w.feature_store.create_online_store(
        online_store=OnlineStore(
            name=online_store_name,
            capacity="CU_1",  # Scale up to CU_2, CU_4, CU_8 for higher throughput
        )
    )
    print(f"✓ Created: {store.name} (state: {store.state})")

# Wait for the store to be available
import time
for i in range(24):
    store = w.feature_store.get_online_store(online_store_name)
    if str(store.state).endswith("AVAILABLE"):
        print(f"✓ Store is AVAILABLE")
        break
    print(f"  Waiting... state={store.state}")
    time.sleep(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 3: Publish the UPT to the Online Store
# MAGIC
# MAGIC `publish_table()` copies the feature table data from Delta Lake into the
# MAGIC Lakebase online store. Modes:
# MAGIC - **SNAPSHOT**: Full copy (used here for initial load)
# MAGIC - **INCREMENTAL**: Only changed rows (for subsequent updates)
# MAGIC
# MAGIC After publishing, any model served via Model Serving that was logged with
# MAGIC `FeatureLookup` will automatically resolve features from this online store —
# MAGIC zero additional wiring needed.

# COMMAND ----------

from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

print(f"Publishing {upt_table} → {online_store_name}...")

try:
    result = w.feature_store.publish_table(
        source_table_name=upt_table,
        publish_spec=PublishSpec(
            online_store=online_store_name,
            online_table_name=upt_table,
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        ),
    )
    print(f"✓ Published! Status: {result.status if hasattr(result, 'status') else result}")
except Exception as e:
    err = str(e)
    if "already published" in err.lower() or "already exists" in err.lower():
        print(f"✓ Table already published to {online_store_name}")
    else:
        print(f"Publish error: {err}")
        print("\nNote: If publish_table fails, the table may need to be explicitly")
        print("registered via fe.create_table() first. Run this on the feature table:")
        print(f"  fe.create_table(name='{upt_table}', primary_keys=['policy_id'], df=spark.table('{upt_table}'))")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 4: Verify the Online Store
# MAGIC
# MAGIC Check that the data is synced and accessible.

# COMMAND ----------

# Check store status
store = w.feature_store.get_online_store(online_store_name)
print(f"Online Store: {store.name}")
print(f"  State: {store.state}")
print(f"  Capacity: {store.capacity}")
print(f"  Created: {store.creation_time}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Step 5: Log to audit trail

# COMMAND ----------

# MAGIC %run ../utils/audit

# COMMAND ----------

log_event(
    spark, catalog, schema,
    event_type="feature_version_created",
    entity_type="feature",
    entity_id="unified_pricing_table_live",
    entity_version=f"online_store:{online_store_name}",
    user_id=dbutils.notebook.entry_point.getDbutils().notebook().getContext().userName().get(),
    details={
        "online_store": online_store_name,
        "source_table": upt_table,
        "row_count": row_count,
        "column_count": col_count,
        "publish_mode": "SNAPSHOT",
    },
    source="notebook",
)
print("✓ Audit event logged")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Summary
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────────────┐
# MAGIC │                     Unified Pricing Table (UPT)                    │
# MAGIC │                        Delta Lake (Gold)                           │
# MAGIC │                    Primary Key: policy_id                          │
# MAGIC ├──────────────────────────────┬──────────────────────────────────────┤
# MAGIC │         Offline Store        │          Online Store               │
# MAGIC │        (Delta Lake)          │         (Lakebase)                  │
# MAGIC │                              │                                     │
# MAGIC │  • Model training            │  • Real-time feature lookup         │
# MAGIC │  • Batch scoring             │  • Sub-10ms latency                 │
# MAGIC │  • Ad-hoc analysis           │  • Key-value by policy_id           │
# MAGIC │  • Full SQL access           │  • Auto-synced from offline         │
# MAGIC │  • Time Travel               │  • Auto-resolved by Model Serving   │
# MAGIC └──────────────────────────────┴──────────────────────────────────────┘
# MAGIC
# MAGIC Models logged with FeatureLookup automatically resolve features from the
# MAGIC online store at serving time — zero additional integration needed.
# MAGIC ```
