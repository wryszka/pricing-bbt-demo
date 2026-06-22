# Databricks notebook source
# MAGIC %md
# MAGIC # Post-deploy bootstrap — grant the App service principal + set initial champions
# MAGIC
# MAGIC A Databricks App runs as its own auto-minted service principal (SP). On a
# MAGIC fresh deploy that SP has **no** Unity Catalog or MLflow access, so every
# MAGIC data-backed tab errors (approve/reject 500s, "0 versions", blank metrics).
# MAGIC
# MAGIC This notebook is the single post-deploy step that makes a fresh clone-and-deploy
# MAGIC actually work. It is **idempotent** — safe to re-run. It does three things:
# MAGIC
# MAGIC 1. **UC grants** — `USE CATALOG`, and on the schema: `USE SCHEMA`, `SELECT`,
# MAGIC    `MODIFY`, `CREATE TABLE`, `EXECUTE` (models/functions), `READ VOLUME`.
# MAGIC 2. **Experiment read** — `CAN_READ` on every MLflow experiment backing a
# MAGIC    registered model version, so the Promote tab shows metrics / story / gini
# MAGIC    instead of blank `—`.
# MAGIC 3. **Initial champions** — alias `champion` to the real (non-simulated)
# MAGIC    version of each production model family, so the Deployment tab is not
# MAGIC    empty on first load.
# MAGIC
# MAGIC The App SP is auto-minted only when the app is created, so this cannot live
# MAGIC in the bundle's `grants` blocks (the principal does not exist at
# MAGIC `bundle deploy` time). Running it as a post-deploy job — after the app
# MAGIC resource exists — resolves the chicken-and-egg: we look the SP up by name.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "pricing_workbench")
dbutils.widgets.text("schema_name", "pricing_upt")
dbutils.widgets.text("volume_name", "external_landing")
dbutils.widgets.text("app_name", "pricing-workbench")

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
volume = dbutils.widgets.get("volume_name")
app_name = dbutils.widgets.get("app_name")
fqn = f"{catalog}.{schema}"

# Production model families whose champion alias the Deployment tab reads.
PRODUCTION_FAMILIES = ["freq_glm", "sev_glm", "demand_gbm", "fraud_gbm"]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Resolve the App service principal
# MAGIC The app exposes its SP application-id once it has been created. If the app
# MAGIC is not deployed yet this raises — deploy the app before running this job.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
app = w.apps.get(name=app_name)
sp = app.service_principal_client_id
if not sp:
    raise RuntimeError(
        f"App '{app_name}' has no service_principal_client_id yet. "
        "Deploy the app (databricks apps deploy) before running this bootstrap."
    )
print(f"App '{app_name}' service principal: {sp}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Unity Catalog grants
# MAGIC Schema-level grants cascade to the tables, models and functions inside the
# MAGIC schema, so we do not have to enumerate 20+ tables individually.

# COMMAND ----------

# Backtick-quote the SP so the application-id (with hyphens) parses as a principal.
GRANTS = [
    f"GRANT USE CATALOG ON CATALOG `{catalog}` TO `{sp}`",
    f"GRANT USE SCHEMA ON SCHEMA {fqn} TO `{sp}`",
    f"GRANT SELECT ON SCHEMA {fqn} TO `{sp}`",
    f"GRANT MODIFY ON SCHEMA {fqn} TO `{sp}`",
    f"GRANT CREATE TABLE ON SCHEMA {fqn} TO `{sp}`",
    f"GRANT EXECUTE ON SCHEMA {fqn} TO `{sp}`",
    f"GRANT READ VOLUME ON SCHEMA {fqn} TO `{sp}`",
]

for stmt in GRANTS:
    try:
        spark.sql(stmt)
        print(f"OK   {stmt}")
    except Exception as e:
        print(f"FAIL {stmt}\n       {e}")
        raise

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Experiment read access
# MAGIC The Promote/Compare tabs read run metrics and tags (gini, story, trained-by,
# MAGIC the `simulated` flag) from the MLflow experiment that owns each model
# MAGIC version's run. Those experiments live under the deployer's workspace folder,
# MAGIC which the App SP cannot read by default — hence the blank `—` columns.
# MAGIC
# MAGIC We resolve the experiments dynamically (version -> run -> experiment_id) so
# MAGIC this stays correct regardless of experiment naming.

# COMMAND ----------

# Resolve experiment ids via the MLflow REST run-get (version -> run -> experiment).
# We deliberately use the REST permissions API (PATCH merges into the ACL) rather
# than the typed SDK permission helpers, which proved brittle here.
experiment_ids = set()
for family in PRODUCTION_FAMILIES:
    full_name = f"{catalog}.{schema}.{family}"
    try:
        versions = list(w.model_versions.list(full_name=full_name))
    except Exception as e:
        print(f"skip {family}: cannot list versions ({e})")
        continue
    for v in versions:
        if not v.run_id:
            continue
        try:
            resp = w.api_client.do("GET", "/api/2.0/mlflow/runs/get", query={"run_id": v.run_id})
            exp_id = resp.get("run", {}).get("info", {}).get("experiment_id")
            if exp_id:
                experiment_ids.add(str(exp_id))
        except Exception as e:
            print(f"  could not resolve experiment for run {v.run_id}: {e}")

print(f"Found {len(experiment_ids)} experiment(s) backing model versions: {sorted(experiment_ids)}")

for exp_id in sorted(experiment_ids):
    try:
        # PATCH merges (adds) the SP into the existing ACL without clobbering owners.
        w.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/experiments/{exp_id}",
            body={"access_control_list": [
                {"service_principal_name": sp, "permission_level": "CAN_READ"}
            ]},
        )
        print(f"OK   CAN_READ on experiment {exp_id}")
    except Exception as e:
        print(f"FAIL experiment {exp_id}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Initial champion aliases
# MAGIC A fresh training run registers versions but sets no `champion` alias, so the
# MAGIC Deployment tab is empty on first load. Promote the real (non-simulated)
# MAGIC version of each family to `champion`. We pick the lowest-numbered
# MAGIC non-simulated version (the base trained model); fall back to v1.

# COMMAND ----------

from mlflow.tracking import MlflowClient
import mlflow

mlflow.set_registry_uri("databricks-uc")
mc = MlflowClient(registry_uri="databricks-uc")

for family in PRODUCTION_FAMILIES:
    full_name = f"{catalog}.{schema}.{family}"
    try:
        versions = list(w.model_versions.list(full_name=full_name))
    except Exception as e:
        print(f"skip {family}: {e}")
        continue
    if not versions:
        print(f"skip {family}: no versions")
        continue

    # If a champion already exists, leave it (idempotent — do not override a
    # deliberate promotion).
    try:
        existing = mc.get_model_version_by_alias(full_name, "champion")
        print(f"{family}: champion already set -> v{existing.version}, leaving as-is")
        continue
    except Exception:
        pass

    # Choose the real (non-simulated) base version; fall back to lowest version.
    real = []
    for v in versions:
        sim = "false"
        if v.run_id:
            try:
                run = w.experiments.get_run(run_id=v.run_id)
                tags = {t.key: t.value for t in (run.run.data.tags or [])} if run and run.run else {}
                sim = tags.get("simulated", "false")
            except Exception:
                pass
        if sim != "true":
            real.append(int(v.version))
    target = str(min(real)) if real else str(min(int(v.version) for v in versions))

    mc.set_registered_model_alias(full_name, "champion", target)
    print(f"{family}: champion -> v{target}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. CAN_MANAGE_RUN on app-triggered jobs
# MAGIC Several app actions (Promote, Compare & Test, Demo reset, live pricing,
# MAGIC backfills, feature-table rebuild) trigger bundle jobs via `jobs.run_now` as
# MAGIC the app SP. The SP is auto-minted and is not an owner of those jobs, so it
# MAGIC needs `CAN_MANAGE_RUN`. We grant it by base name (tolerant of the DAB
# MAGIC development-mode `[dev <user>] ` prefix).

# COMMAND ----------

APP_TRIGGERED_JOBS = [
    "Generate governance pack",
    "Inference log backfill",
    "Compare & test models",
    "Historical quote score",
    "Factory training (real)",
    "Demo reset",
    "Live pricing: load test",
    "Motor live serving: provision",
    "Motor live serving: teardown",
    "Build Unified Pricing Table",
]

granted = 0
try:
    all_jobs = list(w.jobs.list())
except Exception as e:
    all_jobs = []
    print(f"could not list jobs: {e}")

for j in all_jobs:
    name = (j.settings.name if j.settings else "") or ""
    if not any(frag in name for frag in APP_TRIGGERED_JOBS):
        continue
    try:
        # PATCH merges the SP into the job ACL without clobbering the owner.
        w.api_client.do(
            "PATCH",
            f"/api/2.0/permissions/jobs/{j.job_id}",
            body={"access_control_list": [
                {"service_principal_name": sp, "permission_level": "CAN_MANAGE_RUN"}
            ]},
        )
        granted += 1
        print(f"OK   CAN_MANAGE_RUN on job {j.job_id} ({name})")
    except Exception as e:
        print(f"FAIL job {j.job_id} ({name}): {e}")

print(f"Granted CAN_MANAGE_RUN on {granted} app-triggered job(s).")

# COMMAND ----------

print("Post-deploy bootstrap complete.")
print(f"  App SP:      {sp}")
print(f"  Catalog:     {catalog}")
print(f"  Schema:      {schema}")
print(f"  Families:    {', '.join(PRODUCTION_FAMILIES)}")
