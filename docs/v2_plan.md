# Pricing Workbench v2 — rebuild plan

**Status:** planning (approved concept, 2026-08-11). Branch `v2` off `dev`.
**Why:** the current demo is a months-long vibe-coded build that only deploys
cleanly as `laurence.ryszka` on the dev workspace. v2 is a clean, tiered,
multi-user-safe, clearly-named version that deploys to any workspace — the
first target being a fresh FEVM, then the AXA sandbox.

**Ground rules**
- Do **not** touch the dev workspace or its assets. v2 deploys elsewhere.
- Keep **full functionality**; the live feature-table serving is the only thing
  excluded by default (kept in-codebase, deployable via a flag).
- Everything scale-to-zero. No always-on Lakebase in the default profile.

---

## 1. Deploy tiers (one bundle variable, `deploy_profile`)

Everything stays in the codebase; a flag decides what is actually created.

| Tier | Contents | Default |
|---|---|---|
| **Core** | Commercial data + ingestion + silver DLT, UPT/mart, 4 commercial champions + versions, **commercial rating-engine endpoint**, rating config + releases, inference/compare, governance packs + bias, both agents (governance + chat), Genie ×2, mart dashboard, the app — **plus the MCP server + agentic buyer journey** (motor `_direct` scorer + reduced motor dataset + motor champions) | ✅ on |
| **Live serving** | Lakebase online store, route-optimized scorer, QPS load-tester, realtime feature refresh (commercial + motor) | ⬜ off |

Decisions taken 2026-08-11:
- **Agentic distribution (MCP + buyer chat) moves into Core.** It rides on
  `motor_pricing_scorer_direct` (a plain scale-to-zero endpoint), *not* the
  Lakebase online store — so it survives the "no live feature table" cut. This
  pulls **motor champions + a motor feature table** into Core.
- **Scale the motor dataset right down.** The 1M rows existed only for the QPS
  speed demo; the direct scorer and the book-mean provenance need only a modest
  table. Keeps Core cheap and fast.
- The online store / route-optimized scorer / QPS tester stay in the **Live
  serving** tier, off by default.

The commercial rating engine is deployed as a **raw-vector, scale-to-zero
endpoint** (like the motor `_direct` one), *not* a FeatureLookup-on-online-store
model. This:
- fixes the permanently-dead "Run quote" button (it's dead precisely because
  `pricing_scorer` doesn't exist on dev), and
- makes a real what-if possible later (raw-vector endpoint is exactly what the
  candidate-mart impact analysis needs) — see backlog.

---

## 2. Workstream 1 — deploy-clean (runs on any workspace)

The fresh-deploy blockers are all "only works as laurence.ryszka":
- **Hardcoded home-dir paths** — `NewDataImpact.tsx:10`, `app.yaml:24`,
  `app.dev.yaml:24`, `development.py:40`. → derive from bundle root.
- **Hardcoded `[dev laurence_ryszka]` job-name lookup** — `features.py:174`.
  → resolve by suffix / bundle-aware name.
- **Hardcoded experiment path + audit username** —
  `new_data_impact/01_build_all_models.py:82`, `demo_reset.py:314`.
  → derive from current user.
- **Phantom model versions** — `demo_reset.py:40-45` (v53/48/51/53) and the
  release seed pin versions that don't exist on a fresh deploy. → resolve
  `@champion` at runtime; seed releases dynamically.
- **Committed junk** — strip `src/app/.venv-local/` and `src/app/frontend/dist/`;
  build at deploy, add to `.gitignore`.

## 3. Workstream 2 — bundle tiers + naming + tags

**Naming (decided):**
- Own one clearly-named schema — rename `pricing_upt` → **`pricing_workbench`**.
  Everything lives under a single schema, so generic table names
  (`audit_log`, `inference_logs`) need **no prefix** — schema isolation is enough.
- **Prefix only workspace-global assets** so they stand out among other assets:
  serving endpoints, jobs, app, online stores → `pw_` / "Pricing Workbench — …".
  Today inconsistent (`pricing_scorer` vs `motor_pricing_scorer` vs
  `pricing_chat_agent` vs `pricing_governance_agent`).
- **Fix model-name confusion** — one name per model (`freq_glm`, drop the
  parallel `pricing_frequency_glm`); resolve `fraud_gbm` vs "risk_uplift"
  mislabel; register the two supplementary models that currently live only in
  notebooks (`fraud_propensity`, `retention_churn`).
- **Rename bundle** `pricing-upt-demo` → `pricing-workbench` to match the app.

**Tags (decided — apply to everything):**
- Today: partial and inconsistent (motor/factory/governance/use-case tables and
  all models/endpoints untagged; **0 column tags exist**).
- Extend `apply_metadata.py` to apply a uniform tag set to every table, model,
  endpoint, job and volume: `project=pricing_workbench`, `owner`,
  `environment=demo`, `managed_by=dab`, `tier=core|optional`, `contains_pii`.
  This is what lets the whole thing be filtered out of "a sea of other assets".

## 4. Workstream 3 — multi-user hardening

Most of the worst concurrency bugs live in the live-serving/QPS code, which is
optional — so they largely evaporate. What remains for the always-on Core app:
- **Thread-safe WorkspaceClient** (`config.py`) — currently a bare module global
  with a reset-on-401 that thrashes other users' tokens. Highest priority.
  → lock + generation counter; drop the global reset.
- **AI mode/cache is a global flip-for-everyone** — one viewer switching
  live↔cached changes it for all, and the volume-file write races.
  → admin-only (or per-session) toggle; default cached.
- **`_ALIAS_CACHE`** — add lock + TTL, tolerate bust races.
- **Real end-user identity** — everything is attributed to the app service
  principal, so the audit trail is meaningless with many users. → read the
  Databricks Apps forwarded-user header; attribute audit rows to the person.
- **Gate the kill-switches** (`reset-demo`, `sleep-all`, `clear-cache`) behind an
  admin allowlist so a viewer can't wipe a live demo for everyone.

## 5. Workstream 4 — gaps, bugs, cut-corners

- Replace the **6 silent `except: pass`** sites that return 200 with a hidden
  failure (`admin.py:23,28`, `pricing.py:206,211,415,418`, `live_pricing.py:961`).
- Remove dead `Monitoring.tsx`; resolve the `06_*` folder numbering gap.
- **Impact analysis stays as-is (decided).** The proxy shadow-pricing formula
  and the static New Data Impact metrics are kept — the demo needs to be clear,
  concise and fast, and the real scorer wouldn't change much because we bring no
  real data here anyway. See backlog for the future "drop a dataset" trigger.
- The two dev probe artifacts (`agent_gw_probe_payload`, the `pricing_chat_agent`
  gateway config) simply won't exist on a clean workspace.

## 6. Workstream 5 — interface

Needs its own design pass with Laurence (don't guess what to change). Anchor
principles: unify the design language across pages; apply the standing
page-explainer pattern (unfoldable "what am I seeing" + per-datapoint "why");
a proper landing page; move the AI live/cached control out of shared chrome
(ties to the multi-user fix). Parallel track once the backend is deploy-clean.

---

## 7. Sequencing & workspace

1. Branch `v2` off `dev` (good code; `main` is 104 behind). ✅ done.
2. WS1 (deploy-clean) → WS2 (tiers/naming/tags) → deploy **Core-only** to the
   fresh FEVM → smoke test.
3. WS3 + WS4 hardening → WS5 interface, iterating on the live workspace.
4. Replicate to the AXA sandbox once Core is proven clean.

**Workspace:** a **fresh FEVM** (AWS Stable Serverless) — a clean workspace is
the honest test of "deploys anywhere" and the best backdrop for the stand-out
naming. Requested 2026-08-11.

**Docs for Cedric/AXA:** compile the light component-overview doc *later*, once
Laurence is happy with how v2 looks.

---

## WS7 — Price Optimisation demo (accepted 2026-08-13)

The Earnix-displacement ask. Full spec: `docs/optimisation_demo_spec.md` (see its
§0 for the decided scope). **This is a demo OF optimisation, not an optimiser** —
smallest credible example, Core tier, new-business profit first.

Decided scope (trimmed MVP):
- **WS7.1 — minimal elasticity DGP.** Add a simple calibrated price→conversion
  logit to the quote generator so the demand curve slopes and there's a visible
  optimum (today conversion is flat ~0.62–0.68 across all prices → no optimum).
  Regenerate quotes + retrain `demand_gbm`; validate the curve slopes down. This
  also fixes the flat demand-curve views in the base demo. Foundational — do
  first; it changes shared data.
- **WS7.2 — optimisation engine notebook** — per-segment profit-max, grid search
  over a price multiplier, one rate-change cap + margin floor; portfolio impact
  (book vs optimised) + a simple volume↔profit sweep. MLflow-tracked.
- **WS7.3 — one "Price Optimisation" app page** — demand curve + cost line, p*,
  portfolio impact, what-if lever, and a light guardrail panel (rate cap + a
  one-line fair-value nod — NOT the full FCA framework).
- Renewal/`retention_churn` and the governed multi-objective + full FCA framing
  are future (spec §6–7), not the first cut.

Sequencing note: WS7.1 regenerates v2 data, so run it before/with WS2b polish,
not after. Overlaps the backlog "drop a new dataset" item (same raw-vector need).

---

## 8. Backlog (not in v2 scope)

- **"Drop a new dataset" what-if trigger.** A simulated pull-from-the-internet
  of a new dataset that flows through ingestion → candidate mart → portfolio
  what-if (fixed model version, current vs candidate mart) → present. This is
  the real version of what the proxy shadow-pricing fakes today; the deployed
  raw-vector commercial scorer is the piece that makes it feasible. Concept
  validated 2026-08-11 (dataset-level diff = gate 1, portfolio what-if = gate 2).
