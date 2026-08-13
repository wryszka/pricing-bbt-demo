# Databricks notebook source
# MAGIC %md
# MAGIC # Governance Agent — Databricks Agent Framework
# MAGIC
# MAGIC Deploys a **real** Agent Framework model-serving endpoint that the Model
# MAGIC Governance tab's chat panel calls.
# MAGIC
# MAGIC The agent can call three tools on-demand:
# MAGIC
# MAGIC | Tool                    | Source                                                 |
# MAGIC |---|---|
# MAGIC | `query_pack_index`      | Delta table `governance_packs_index` (catalog metadata) |
# MAGIC | `read_pack_artefact`    | Volume sidecars: `model_card.md`, `metrics.json`, `importance.parquet`, `shap.parquet`, `fairness.md`, `lineage.json`, `approvals.json` |
# MAGIC | `query_audit_log`       | Delta table `audit_log` (events filtered by entity)     |
# MAGIC
# MAGIC The underlying LLM is `databricks-claude-sonnet-4-6` (Foundation Model API).
# MAGIC Tool-use happens via Anthropic's native tool-calling protocol.
# MAGIC
# MAGIC The agent's system prompt strictly bounds it to pack data + cites the
# MAGIC source of every claim. Declines on information that isn't documented.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_serverless_aws_us_catalog")
dbutils.widgets.text("schema_name",  "pricing_upt")
dbutils.widgets.text("endpoint_name","pricing_governance_agent")
dbutils.widgets.text("fm_endpoint",  "databricks-claude-sonnet-4-6")
dbutils.widgets.text("warehouse_id", "")  # SQL warehouse the agent tools query; defaults to first running serverless warehouse

# COMMAND ----------

# MAGIC %pip install mlflow databricks-agents databricks-sdk pandas pyarrow --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog         = dbutils.widgets.get("catalog_name")
schema          = dbutils.widgets.get("schema_name")
endpoint_name   = dbutils.widgets.get("endpoint_name")
fm_endpoint     = dbutils.widgets.get("fm_endpoint")
warehouse_id    = dbutils.widgets.get("warehouse_id")

if not warehouse_id:
    from databricks.sdk import WorkspaceClient as _W
    _w = _W()
    _running = [wh for wh in _w.warehouses.list() if str(wh.state) == "WarehouseStatusState.RUNNING"]
    _all = [wh for wh in _w.warehouses.list()]
    warehouse_id = (_running or _all)[0].id
    print(f"warehouse_id auto-selected: {warehouse_id}")

fqn             = f"{catalog}.{schema}"
agent_uc_name   = f"{fqn}.governance_agent"
volume_path_base = f"/Volumes/{catalog}/{schema}/governance_packs"
sidecars_base    = f"{volume_path_base}/sidecars"

import json, os
import mlflow
from mlflow.pyfunc import PythonModel
from mlflow.models import ModelSignature
from mlflow.types.schema import Schema, ColSpec

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Agent definition — tool-use loop over Claude

# COMMAND ----------

SYSTEM_PROMPT = """You are the Bricksurance SE model-governance assistant.
A compliance officer, senior actuary, or regulator is asking questions about a
specific production pricing model. You answer by looking up the model's
governance pack artefacts through the tools provided — never from your prior
knowledge.

Rules you MUST follow:
 * For every factual claim, cite either the pack section (e.g. "Section 4 — Model specification")
   OR the artefact name (e.g. "from model_card.md", "from metrics.json").
 * Always call a tool before answering — even if you think you know. You do NOT know unless the
   tools return it.
 * If the tools don't surface the needed information, reply exactly:
   "The governance pack does not document this — further investigation required."
 * Never speculate about fairness, bias, or model behaviour beyond what is documented.
 * When drafting regulator / customer responses, stay bounded by what the pack says.
 * Keep answers concise (4-8 sentences unless the user asks for more detail).

Available artefact filenames (use with `read_pack_artefact`):
  - `model_card.md`     — purpose, intended use, owner, risk profile
  - `metrics.json`      — all training metrics + params
  - `importance.parquet`— top features (relativities for GLMs, gain for GBMs)
  - `shap.parquet`      — mean-absolute SHAP per feature (GBMs only)
  - `fairness.md`       — fairness, bias, and ethical considerations section
  - `lineage.json`      — feature table, mlflow run, upstream data
  - `approvals.json`    — audit-log events that touched this model
"""


TOOL_DEFS = [
    {
        "name": "query_pack_index",
        "description": (
            "Look up rows in the governance_packs_index Delta table. "
            "Use this to find a pack by family, find the latest pack for a family, "
            "or list packs generated on a specific date."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pack_id":      {"type": "string", "description": "exact pack_id if known"},
                "model_family": {"type": "string", "description": "e.g. freq_glm, sev_glm, demand_gbm, fraud_gbm"},
                "latest_only":  {"type": "boolean", "description": "return only the newest pack per family"},
                "limit":        {"type": "integer", "description": "max rows", "default": 10},
            },
        },
    },
    {
        "name": "read_pack_artefact",
        "description": (
            "Fetch a single sidecar file from a pack's sidecars directory in the UC volume. "
            "Use this when you need the actual content of a model card, metrics, importance, "
            "shap, fairness, lineage, or approvals artefact."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "pack_id":  {"type": "string", "description": "pack_id (required)"},
                "filename": {
                    "type": "string",
                    "description": "one of: model_card.md, metrics.json, importance.parquet, shap.parquet, fairness.md, lineage.json, approvals.json",
                },
            },
            "required": ["pack_id", "filename"],
        },
    },
    {
        "name": "query_audit_log",
        "description": (
            "Query the audit_log Delta table for events touching a specific model family. "
            "Use this for approval history, pack generations, promotions, rollbacks, prior chats."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "description": "usually model family (freq_glm, etc.)"},
                "event_types": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "filter to these event types (optional)",
                },
                "limit": {"type": "integer", "default": 25},
            },
            "required": ["entity_id"],
        },
    },
]

# COMMAND ----------

# MAGIC %md
# MAGIC ## Tool implementations — run inside the serving endpoint

# COMMAND ----------

class GovernanceAgent(PythonModel):
    """MLflow pyfunc agent with tool-calling via Databricks FM API.

    Call shape:
        model_input = {"messages": [{"role": "user", "content": "..."}]}
    Returns:
        {"messages": [{"role": "assistant", "content": "..."}], "trace": [...], "model": "..."}
    """

    def load_context(self, context):
        # Values baked at log time so the agent knows its catalog / schema / FM endpoint
        import os, json
        cfg_path = context.artifacts.get("config")
        if cfg_path:
            with open(cfg_path) as fh:
                cfg = json.load(fh)
            self.catalog       = cfg["catalog"]
            self.schema        = cfg["schema"]
            self.fm_endpoint   = cfg["fm_endpoint"]
        else:
            self.catalog       = os.environ.get("AGENT_CATALOG", "lr_serverless_aws_us_catalog")
            self.schema        = os.environ.get("AGENT_SCHEMA",  "pricing_upt")
            self.fm_endpoint   = os.environ.get("AGENT_FM_ENDPOINT", "databricks-claude-sonnet-4-6")
        self.sidecars_base = f"/Volumes/{self.catalog}/{self.schema}/governance_packs/sidecars"

    # -- Tool implementations -------------------------------------------------

    def _tool_query_pack_index(self, args):
        import json
        from databricks.sdk import WorkspaceClient
        where = []
        if args.get("pack_id"):
            where.append(f"pack_id = '{args['pack_id']}'")
        if args.get("model_family"):
            where.append(f"model_family = '{args['model_family']}'")
        where_sql = f"WHERE {' AND '.join(where)}" if where else ""
        limit = max(1, min(25, int(args.get("limit", 10))))
        if args.get("latest_only"):
            sql = f"""
                SELECT pack_id, model_family, model_version, story, primary_metric,
                       primary_value, generated_by, cast(generated_at as string) as generated_at
                FROM (
                  SELECT *, row_number() OVER (PARTITION BY model_family ORDER BY generated_at DESC) AS rn
                  FROM {self.catalog}.{self.schema}.governance_packs_index
                  {where_sql}
                )
                WHERE rn = 1
                ORDER BY model_family
            """
        else:
            sql = f"""
                SELECT pack_id, model_family, model_version, story, primary_metric,
                       primary_value, generated_by, cast(generated_at as string) as generated_at
                FROM {self.catalog}.{self.schema}.governance_packs_index
                {where_sql}
                ORDER BY generated_at DESC
                LIMIT {limit}
            """
        rows = _run_sql(sql)
        return {"rows": rows, "count": len(rows)}

    def _tool_read_pack_artefact(self, args):
        """Read a text sidecar (model_card.md, metrics.json, fairness.md,
        lineage.json, approvals.json) from the governance_pack_sidecars Delta
        table. Parquet sidecars (importance, shap) aren't served through this
        tool — the committee reviews those via the PDF and dashboard UI."""
        pack_id  = args["pack_id"]
        filename = args["filename"]
        if filename.endswith(".parquet"):
            return {
                "error": f"{filename} is a parquet artefact, not retrievable by the agent. "
                         f"The committee reviews it via the pack PDF / dashboard UI.",
            }
        rows = _run_sql(f"""
            SELECT content, content_type
            FROM {self.catalog}.{self.schema}.governance_pack_sidecars
            WHERE pack_id = '{pack_id}' AND filename = '{filename}'
            LIMIT 1
        """)
        if not rows:
            return {"error": f"Artefact not found: {filename} for pack {pack_id}. "
                             f"Either the pack pre-dates sidecar generation or the "
                             f"artefact name is misspelled."}
        content = rows[0].get("content") or ""
        ctype   = rows[0].get("content_type") or ""
        if filename.endswith(".json"):
            try:
                return {"filename": filename, "content": json.loads(content)}
            except Exception:
                return {"filename": filename, "content": content[:8000]}
        return {"filename": filename, "content": content[:8000], "content_type": ctype}

    def _tool_query_audit_log(self, args):
        entity_id = args["entity_id"]
        event_types = args.get("event_types") or []
        limit = max(1, min(50, int(args.get("limit", 25))))
        filters = [f"entity_id = '{entity_id}'"]
        if event_types:
            quoted = ",".join("'" + e + "'" for e in event_types)
            filters.append(f"event_type IN ({quoted})")
        where = " AND ".join(filters)
        sql = f"""
            SELECT event_type, entity_version, user_id, cast(timestamp as string) as timestamp, source,
                   substr(details, 1, 400) as details_preview
            FROM {self.catalog}.{self.schema}.audit_log
            WHERE {where}
            ORDER BY timestamp DESC
            LIMIT {limit}
        """
        rows = _run_sql(sql)
        return {"rows": rows, "count": len(rows)}

    def _exec_tool(self, name, args):
        if name == "query_pack_index":   return self._tool_query_pack_index(args or {})
        if name == "read_pack_artefact": return self._tool_read_pack_artefact(args or {})
        if name == "query_audit_log":    return self._tool_query_audit_log(args or {})
        return {"error": f"Unknown tool: {name}"}

    # -- Main predict loop ----------------------------------------------------

    def predict(self, context, model_input, params=None):
        # Accept either a pandas.DataFrame (single row with "messages" col) or a dict
        if hasattr(model_input, "to_dict"):
            if len(model_input) == 0:
                return {"messages": [{"role": "assistant", "content": ""}], "trace": []}
            rec = model_input.iloc[0].to_dict()
        elif isinstance(model_input, list):
            rec = model_input[0] if model_input else {}
        else:
            rec = dict(model_input) if model_input else {}

        messages = rec.get("messages", [])
        if isinstance(messages, str):
            try:
                messages = json.loads(messages)
            except Exception:
                messages = [{"role": "user", "content": messages}]

        context_info = rec.get("custom_inputs") or {}
        if isinstance(context_info, str):
            try:
                context_info = json.loads(context_info)
            except Exception:
                context_info = {}
        pack_id   = context_info.get("pack_id") if isinstance(context_info, dict) else None
        policy_id = context_info.get("policy_id") if isinstance(context_info, dict) else None

        user_hint = ""
        if pack_id:
            user_hint += f"\nContext: the user is viewing pack_id='{pack_id}'. Use this when calling `read_pack_artefact`."
        if policy_id:
            user_hint += f"\nContext: the question concerns policy_id='{policy_id}'."

        full_messages = [{"role": "system", "content": SYSTEM_PROMPT + user_hint}]
        for m in messages:
            if isinstance(m, dict):
                full_messages.append(m)

        trace = []
        final_text = ""
        total_input_tokens = 0
        total_output_tokens = 0

        for hop in range(6):    # safety cap on tool-use loop
            resp = _call_fm(self.fm_endpoint, full_messages, TOOL_DEFS)
            usage = resp.get("usage") or {}
            total_input_tokens  += int(usage.get("prompt_tokens") or 0)
            total_output_tokens += int(usage.get("completion_tokens") or 0)

            choices = resp.get("choices") or []
            if not choices:
                break
            msg = choices[0].get("message") or {}
            content = msg.get("content")

            # Detect tool calls
            tool_calls = msg.get("tool_calls") or []

            if tool_calls:
                # Append the assistant message with tool_calls to conversation
                full_messages.append({
                    "role": "assistant",
                    "content": content or "",
                    "tool_calls": tool_calls,
                })
                # Execute each tool call
                for tc in tool_calls:
                    tool_name = (tc.get("function") or {}).get("name") or tc.get("name")
                    raw_args  = (tc.get("function") or {}).get("arguments") or tc.get("input") or "{}"
                    if isinstance(raw_args, str):
                        try:
                            args = json.loads(raw_args)
                        except Exception:
                            args = {}
                    else:
                        args = raw_args or {}
                    result = self._exec_tool(tool_name, args)
                    trace.append({
                        "hop": hop,
                        "tool": tool_name,
                        "arguments": args,
                        "result_summary": _summarise_result(result),
                    })
                    # Feed the tool result back
                    full_messages.append({
                        "role": "tool",
                        "tool_call_id": tc.get("id") or tc.get("tool_use_id") or tool_name,
                        "content": json.dumps(result)[:12000],
                    })
                continue

            # No tool calls — this is the final assistant text
            final_text = content or ""
            break

        return {
            "messages": [{"role": "assistant", "content": final_text}],
            "trace":    trace,
            "model":    self.fm_endpoint,
            "usage":    {
                "prompt_tokens":     total_input_tokens,
                "completion_tokens": total_output_tokens,
                "total_tokens":      total_input_tokens + total_output_tokens,
            },
        }


def _summarise_result(result) -> str:
    try:
        if isinstance(result, dict):
            if "error" in result:
                return f"error: {result['error']}"
            if "rows" in result:
                return f"{result.get('count', len(result.get('rows', [])))} rows"
            if "content" in result:
                c = result["content"]
                if isinstance(c, str):
                    return f"{len(c)} chars"
                if isinstance(c, (dict, list)):
                    return f"structured content ({len(str(c))} chars)"
        return "ok"
    except Exception:
        return "ok"


def _run_sql(sql: str):
    """Execute SQL via the Databricks SDK's statement execution API."""
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.sql import StatementState
    import os as _os
    tok  = _os.environ.get("AGENT_TOKEN")
    host = _os.environ.get("AGENT_HOST") or _os.environ.get("DATABRICKS_HOST")
    w = WorkspaceClient(host=host, token=tok) if tok and host else WorkspaceClient()
    warehouse_id = _os.environ.get("AGENT_WAREHOUSE_ID", "")
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=warehouse_id, wait_timeout="30s",
    )
    if resp.status and resp.status.state == StatementState.FAILED:
        err = resp.status.error.message if resp.status.error else "unknown"
        raise RuntimeError(f"SQL failed: {err}")
    if not resp.manifest or not resp.manifest.schema or not resp.manifest.schema.columns:
        return []
    cols = [c.name for c in resp.manifest.schema.columns]
    out = []
    if resp.result and resp.result.data_array:
        for row in resp.result.data_array:
            out.append(dict(zip(cols, row)))
    return out


def _call_fm(endpoint: str, messages: list, tools: list):
    """Call Databricks Foundation Model API with tool definitions. Returns the
    raw response dict so we can inspect tool_calls."""
    from databricks.sdk import WorkspaceClient
    import requests as _r
    w = WorkspaceClient()
    host  = w.config.host.rstrip("/")
    token = w.config._header_factory()
    openai_tools = [{
        "type": "function",
        "function": {
            "name": t["name"],
            "description": t["description"],
            "parameters": t["input_schema"],
        },
    } for t in tools]
    resp = _r.post(
        f"{host}/serving-endpoints/{endpoint}/invocations",
        headers={**token, "Content-Type": "application/json"},
        json={
            "messages":    messages,
            "tools":       openai_tools,
            "tool_choice": "auto",
            "max_tokens":  900,
            "temperature": 0.1,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the agent model to UC

# COMMAND ----------

import tempfile
cfg_path = f"{tempfile.mkdtemp()}/agent_config.json"
with open(cfg_path, "w") as fh:
    json.dump({"catalog": catalog, "schema": schema, "fm_endpoint": fm_endpoint}, fh)

from mlflow.models.resources import DatabricksServingEndpoint, DatabricksTable, DatabricksFunction
try:
    from mlflow.models.resources import DatabricksUCVolume as _UCVolumeResource
except ImportError:
    try:
        from mlflow.models.resources import DatabricksUCConnection as _UCVolumeResource
    except ImportError:
        _UCVolumeResource = None

input_example = {
    "messages":      json.dumps([{"role": "user", "content": "What fraud tier does this model target?"}]),
    "custom_inputs": json.dumps({"pack_id": "GP-20260423112519-fraud_gbm-v41"}),
}

signature = ModelSignature(
    inputs=Schema([
        ColSpec("string", "messages"),
        ColSpec("string", "custom_inputs"),
    ]),
    outputs=Schema([
        ColSpec("string", "messages"),
        ColSpec("string", "model"),
        ColSpec("string", "trace"),
        ColSpec("string", "usage"),
    ]),
)

resources_list = [
    DatabricksServingEndpoint(endpoint_name=fm_endpoint),
    DatabricksTable(table_name=f"{fqn}.governance_packs_index"),
    DatabricksTable(table_name=f"{fqn}.governance_pack_sidecars"),
    DatabricksTable(table_name=f"{fqn}.audit_log"),
]
if _UCVolumeResource is not None:
    # Tell the endpoint it needs READ on the governance_packs volume so that
    # embedded-credentials serving has the right scope to read sidecar files.
    try:
        resources_list.append(_UCVolumeResource(volume_name=f"{fqn}.governance_packs"))
    except TypeError:
        # Some older signatures use `name=` or `connection_name=` instead of volume_name
        try:
            resources_list.append(_UCVolumeResource(name=f"{fqn}.governance_packs"))
        except Exception:
            print("  (could not attach UC Volume resource — perms may need manual grant)")

with mlflow.start_run(run_name="governance_agent_deploy"):
    # Log
    mi = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=GovernanceAgent(),
        artifacts={"config": cfg_path},
        resources=resources_list,
        signature=signature,
        input_example=input_example,
        registered_model_name=agent_uc_name,
        pip_requirements=[
            "mlflow>=2.12",
            "databricks-sdk>=0.30.0",
            "pandas",
            "pyarrow",
            "requests",
        ],
    )
    print(f"Logged {mi.model_uri}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Deploy as a Model Serving endpoint

# COMMAND ----------

from mlflow.tracking import MlflowClient
client = MlflowClient()
latest = max(
    [int(v.version) for v in client.search_model_versions(f"name='{agent_uc_name}'")],
    default=None,
)
print(f"Deploying {agent_uc_name} v{latest} → endpoint '{endpoint_name}'")

env_vars = {"AGENT_WAREHOUSE_ID": warehouse_id}

# agents.deploy() wipes env_vars set out-of-band. Snapshot so we can merge.
from databricks.sdk import WorkspaceClient as _W
_w_pre = _W()
try:
    _existing = (_w_pre.serving_endpoints.get(endpoint_name)
                 .config.served_entities[0].environment_vars or {})
    for k, v in _existing.items():
        env_vars.setdefault(k, v)
    print(f"Preserving existing env vars: {sorted(_existing.keys())}")
except Exception as _e:
    print(f"No existing endpoint to inherit env_vars from: {_e}")

# Self-provision the SQL OBO token if none is inherited/injected — the System SP
# can't be granted UC perms, so the agent's tools need a token. This notebook
# runs as the deploying identity; mint a PAT for it so a fresh-workspace deploy
# is hands-off. If PATs are disabled, inject AGENT_TOKEN/AGENT_HOST out-of-band.
if not env_vars.get("AGENT_TOKEN"):
    try:
        _tok = _w_pre.tokens.create(
            comment="pricing-workbench governance agent SQL OBO (auto)",
            lifetime_seconds=7776000,
        )
        env_vars["AGENT_TOKEN"] = _tok.token_value
        env_vars["AGENT_HOST"] = _w_pre.config.host
        print("Minted AGENT_TOKEN for SQL OBO (value not shown).")
    except Exception as _te:
        print(f"⚠ Could not mint AGENT_TOKEN ({str(_te)[:100]}). "
              f"Agent SQL tools will fail until AGENT_TOKEN/AGENT_HOST are set.")

# Use databricks-agents if present, else fall back to serving_endpoints
try:
    from databricks import agents
    deployment = agents.deploy(
        model_name=agent_uc_name,
        model_version=latest,
        scale_to_zero=True,  # keep warm for demo cadence
        environment_vars=env_vars,
        tags={"project": "pricing_workbench", "purpose": "governance_agent"},
    )
    print(f"databricks-agents deploy kicked off: {deployment}")
except Exception as e:
    print(f"databricks-agents.deploy failed, falling back to serving_endpoints: {e}")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import (
        EndpointCoreConfigInput, ServedEntityInput, AutoCaptureConfigInput,
    )
    w = WorkspaceClient()
    served = [ServedEntityInput(
        entity_name=agent_uc_name,
        entity_version=str(latest),
        scale_to_zero_enabled = True,  # keep warm for demo cadence
        workload_size="Small",
        environment_vars=env_vars,
    )]
    cfg = EndpointCoreConfigInput(
        name=endpoint_name,
        served_entities=served,
    )
    try:
        w.serving_endpoints.get(endpoint_name)
        w.serving_endpoints.update_config(name=endpoint_name, served_entities=served)
        print("Updated existing endpoint.")
    except Exception:
        w.serving_endpoints.create(name=endpoint_name, config=cfg)
        print("Created new endpoint.")

# COMMAND ----------

# Re-assert env_vars after agents.deploy (which strips them on each update).
import time as _t
from databricks.sdk import WorkspaceClient as _W2
from databricks.sdk.service.serving import ServedEntityInput as _SEI
_w2 = _W2()
for _attempt in range(60):
    _ep = _w2.serving_endpoints.get(endpoint_name)
    _cur_v = _ep.config.served_entities[0].entity_version if _ep.config else None
    _upd = str(_ep.state.config_update) if _ep.state else ""
    if _cur_v == str(latest) and "NOT_UPDATING" in _upd:
        break
    _t.sleep(15)
_existing_env = (_ep.config.served_entities[0].environment_vars or {}) if _ep.config else {}
_merged = {**env_vars, **_existing_env, "AGENT_WAREHOUSE_ID": warehouse_id}
if set(_merged.keys()) != set(_existing_env.keys()) or any(_merged[k] != _existing_env.get(k) for k in _merged):
    print(f"Re-asserting env_vars: {sorted(_merged.keys())}")
    _w2.serving_endpoints.update_config(
        name=endpoint_name,
        served_entities=[_SEI(
            entity_name=agent_uc_name,
            entity_version=str(latest),
            scale_to_zero_enabled = True,
            workload_size="Small",
            environment_vars=_merged,
        )],
    )

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "agent_uc_name":   agent_uc_name,
    "model_version":   latest,
    "endpoint_name":   endpoint_name,
}))
