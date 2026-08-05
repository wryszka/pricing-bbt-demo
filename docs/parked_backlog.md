# Parked backlog — 2026-08-05

Parked deliberately. The app is in shared use (org-wide `account users` access on
dev), so changes carry blast radius beyond a single demo. Nothing below has been
started; the workbench is left exactly as found.

Ordered as agreed: section C (fixes and housekeeping) first, then section A (the
agent-channel governance rework).

---

## C — broken now, or drifted

### C1. Pricing Engine "Run quote" is permanently disabled

The `pricing_scorer` serving endpoint does not exist on dev (verified: zero
matches in `GET /api/2.0/serving-endpoints`). `/api/pricing/status` therefore
returns `ready: false`, and `PricingEngine.tsx:314` gates the Run button on that
flag — so the button reads "Warming endpoint…" forever and the page polls
`/api/pricing/status` every 4s indefinitely.

The fallback already works: `_score_via_inference_logs` in `routes/pricing.py`
serves predictions from `inference_logs` (50,000 rows, last scored
2026-05-07), and `/api/pricing/mta/simulate` returns 200 through it. Only the
button gating was never relaxed to match. Historical releases are unaffected —
they run through a batch job.

Two ways out: relax the gate so "current release" uses the fallback and labels
itself honestly, or deploy `pricing_scorer` (job `469650315334087`,
`scale_to_zero=True`, workload `Large`). The first is cheaper and matches the
scale-to-zero preference.

### C2. `pricing_engine_releases` pins model versions that don't exist

| Release | freq | sev | demand | fraud |
|---|---|---|---|---|
| apr_2026 (champion) | 53 | 48 | 51 | 53 |
| mar_2026 | 52 | 47 | 50 | 52 |
| feb_2026 / jan_2026 / dec_2025 | 62–64 | 57–59 | 60–62 | 62–64 |

Actual highest versions in UC: **freq 14, sev 14, demand 12, fraud 14**.
Current champion aliases: **freq v12, sev v14, demand v12, fraud v14**.

This was remapped to real versions once before and has since regressed — worth
finding what re-seeds it, not just remapping again.

### C3. `demo_reset.py` hardcodes the same phantom versions

`src/04_models/production/demo_reset.py:40` — `CHAMPION_VERSIONS = {freq 53,
sev 48, demand 51, fraud 53}`. Every alias set fails, caught non-fatally, so the
reset prints failures nobody reads. Should resolve the real champion per family
(or be given the true v12/v14/v12/v14).

### C4. Live motor system has been running since 4 Aug 09:25

- Lakebase `motor-pricing-online-store` — `AVAILABLE`, `CU_4`, not stopped
- `motor_pricing_scorer` — min 4 / max 64 provisioned, `scale_to_zero: false`

Against the standing scale-to-zero preference. Teardown job
`1052938101499033` takes ~43s; re-activation ~3m10s. **Decision not taken** —
left running in case a demo depends on instant pricing.

### C5. AI cache is nearly empty while mode is `cached`

`/api/admin/ai-mode` → `mode: cached, entries: 3`. Three keys only (two
`explain` personas, one `bias_investigator`). Every other AI panel misses and
falls through to live, so "consistent and fast" is not yet true in practice.
Needs a warm-up pass over the panels that matter.

### C6. Zero UC column tags in the schema

`system.information_schema.column_tags` for `lr_dev_aws_us_catalog.pricing_upt`
returns 0. The PII-shaped columns identified earlier (`prior_convictions`,
`ethnicity_proxy`, and the demographics set) carry no tags and no masks.

### C7. Motor governance packs are thin

`governance_packs_index`: commercial families have 3–5 packs each
(`freq_glm` 5, `sev_glm` / `demand_gbm` / `fraud_gbm` 3). The four `*_motor`
families have **1 each** — so the motor story, which is the one the live demo
tells, has the weakest pack coverage.

### C8. Repo hygiene

- `01f1c08` ("Agentic MCP sales in Live Pricing demo pages") is committed but
  **unpushed**; `dist/` carries an uncommitted rebuild alongside it.
- `dev` is **104 commits ahead of `main`**.

---

## A — agent-channel governance rework

Planned 2026-08-04, deferred by the user the same day ("we will redo the whole
thing but later"). Five pieces. One hard constraint established by probing the
AI Gateway API on our own endpoints:

| Gateway feature | Status in this workspace |
|---|---|
| `inference_table_config` | ✅ accepted |
| `rate_limits` | ✅ accepted |
| `usage_tracking_config` | ✅ accepted |
| **`guardrails`** | ❌ *"not currently supported for this endpoint type in this workspace"* |

So there is **no gateway-level PII blocking available**. The honest control story
is "log every interaction to UC and mask on read", not "we block PII at the
gateway". State it that way.

### A1. A gateway endpoint we own

`broker.py:35` still points at `databricks-claude-sonnet-4-6` — a shared
foundation-model endpoint (`foundation_model=True`, no creator), so it cannot be
configured for us without affecting every other consumer in the workspace.
Create a separate endpoint for the agentic channel with `inference_table_config`
(the transcript, for free), `rate_limits` (abuse control on a publicly reachable
MCP) and `usage_tracking`. Keep the current path behind an env flag so it can be
flipped back.

### A2. `agent_interactions` — the conversation record

The inference table gives raw LLM traffic; it knows nothing about insurance. One
table joining the two, keyed by `session_id`: `turn_no`, `surface`, `agent_id`,
`caller_identity`, `user_message`, `assistant_reply`, `tools_called`,
`tool_args_json`, `answers_snapshot`, `quoted_premium`, `engine`,
`model_version`, `rating_engine_version`, `provenance_json`, `guardrail_flags`,
and **`inference_request_id`** — the stitch to the gateway's raw record.

### A3. Governance → "Agent channel" tab

Fifth tab alongside Monitor / Search / Agent / What's-collected
(`Governance.tsx:25`). Four blocks: **controls in force** (live read of
`ai_gateway` per endpoint, rendered as a control register — honest by
construction, if guardrails are off the page says so), **conversation search**
(turn-by-turn with tools and premium at each step), **reproduce a quote** (replay
the stored feature vector against the named model version — same pattern as
Quote Review and the ifrs17 auditor beat), **exceptions** (guardrail hits,
incomplete-attempt rate, engine failures, rate-limit rejections).

### A4. Extend the pack + audit taxonomy

`/api/governance/data-summary` declares 7 input tables; neither `mcp_tool_calls`
(53 rows, 22 sessions) nor `agent_interactions` is among them, so the governance
narrative does not cover the channel at all. Add both, plus event types
`agent_session_started`, `agent_guardrail_triggered`, `agent_quote_reproduced`,
`agent_rate_limited`. Today the audit log carries only `agent_quote` (8) and
`agent_recommendation` (43) for this channel.

### A5. Caller identity for MCP

`agent_id` is the User-Agent string — `curl/8.7.1`. Unauthenticated and trivially
spoofed. A per-partner token or OAuth client-id mapped to a registered-partner
table turns the telemetry from a chart into a control.

### Open design question — not answered

Customers type free text into the chat, so transcripts raise a real retention
question even on synthetic data. Three options: store verbatim; store with
masking applied; or store verbatim in a restricted schema with UC column masks
and show masked in the UI. **Recommendation: the third** — for a demo about
control, it demonstrates UC masking rather than describing it.

---

## Loose ends I own from the 4 Aug probing

Both still present, both verified 2026-08-05:

1. **`lr_dev_aws_us_catalog.pricing_upt.agent_gw_probe_payload`** — created by my
   gateway probe. Exists, 0 rows, inert. Clutter in the demo schema.
2. **`pricing_chat_agent.ai_gateway`** — now
   `{inference_table: disabled (prefix agent_gw_probe), usage_tracking: enabled}`
   where it was `None` before I probed it. Functionally equivalent and verified
   working, but not bit-identical to how I found it.

Each gateway PUT **replaces** the whole config rather than merging — which is how
this happened. Probe a throwaway endpoint next time.

---

## Verified healthy at park time

All app routes 200 across a 21-endpoint sweep: health, config, datasets,
mart-profile, data-summary, factory runs, review packs, distribution telemetry,
supervisor agents, ai-mode, live-pricing status (`state: on`), MTA simulate.
`motor_pricing_scorer`, `motor_pricing_scorer_direct`, `pricing_chat_agent` and
`pricing_governance_agent` all READY. App SP `63890164-…-59c5` holds `CAN_USE` on
warehouse `a3b61648ea4809e3` (14-entry ACL intact).

The one user-visible defect is **C1**.
