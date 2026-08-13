# Price Optimisation demo — specification & gap analysis

**Status:** specification only — nothing built. 2026-08-13.
**Target:** Pricing Workbench **v2** (`fevm-lr-pricing-v2-aws-us`, catalog
`lr_pricing_v2_aws_us_catalog`, schema `pricing_workbench`, warehouse
`f738fde9a1197aeb`). Dev is explicitly out of scope.
**Trigger:** a GI client wants to move off **Earnix**; price optimisation is
their main concern right now. This spec scopes what it would take to show a
credible optimisation *example* on the workbench — not to deliver a production
optimiser.

---

## 0. Decisions taken (2026-08-13) — scope this to a DEMO

Accepted into v2. Explicit steer from Laurence: **this is a demo OF optimisation,
not an implementation of an optimiser.** The goal is to make an evaluator who is
shopping for optimisation see that the pattern is available on Databricks — open,
governed, their models. So build the *smallest credible* example, not a
framework.

- **Lead with new business** (conversion/profit). Renewal/retention is beneficial
  and comes second — do NOT build `retention_churn` for the first cut.
- **Jurisdiction: not important** — a UKI/EMEA insurer. Keep regulatory framing to
  a **light, one-line nod** (a fair-value / no-price-walking guardrail shown as an
  enforced constraint), NOT the full FCA framework of §6. §6 stays as optional
  depth for a UK-specific conversation.
- **Scope = MVP, trimmed.** One objective (profit), one or two constraints
  (rate-change cap + margin floor), one app page, one segment view + portfolio
  impact. Grid-search solve for transparency. No multi-objective frontier
  machinery beyond a simple volume-vs-profit sweep if it's cheap.
- **Elasticity DGP fix is still required but kept MINIMAL** — a simple calibrated
  price→conversion logit so the demand curve actually slopes and there is a
  visible optimum. Enough to be credible, not an elaborate segment-varying model.
- **Tier: Core.** No live-serving/online-store dependency.

Everything below is the fuller specification; the first build is the trimmed MVP
above. §7 "Full" and §6 UK-specific depth are future, not now.

---

## 1. Positioning (what we are and are not showing)

Databricks does not sell a pricing optimiser and does not deliver models. What
this demo shows is the **platform pattern** for price optimisation:

- the client's **own** cost and demand models, governed side by side in Unity
  Catalog;
- a **reference optimisation routine** (a worked example the client's actuaries
  own and can rewrite) that turns those models into a price recommendation;
- every recommendation **explainable, versioned and audited**, with regulatory
  constraints enforced as first-class inputs.

This is the wedge against Earnix: **open, transparent, your models, no black
box, no per-seat lock-in** — the optimisation logic is readable code and its
config is governed data, not a vendor's closed engine. We win on governance and
transparency, not on "our optimiser is better."

> Keep the standard demo disclaimer: Bricksurance SE is fictional; all data
> synthetic; the optimisation method is illustrative, not a certified pricing
> model.

---

## 2. What price optimisation is (the decision we are demonstrating)

For each risk (or segment), given:

- **expected cost** `c(x)` = frequency × severity (the technical/risk price), and
- **demand** `d(p | x)` = probability of conversion (new business) or renewal
  (retention) as a function of the offered price `p`,

choose the price `p*` that maximises a **business objective** subject to
**constraints**:

| Objective | Maximise |
|---|---|
| Profit | `d(p) · (p − c)` |
| Volume | `d(p)` s.t. margin floor `p − c ≥ m` |
| Lifetime value | multi-year `d`/retention × margin |
| Blended | `α·profit + β·volume` on an efficient frontier |

**Constraints** (all first-class, all audited):
- rate-change caps (e.g. ±15% vs current book);
- minimum margin / maximum loss ratio;
- competitive guardrails (stay within *k* of market/competitor);
- **regulatory** — see §6. For UK GI this is not optional.

Two demand regimes, both relevant to an Earnix displacement:
- **New business** — conversion elasticity (the `demand_gbm` model).
- **Renewal** — retention/churn elasticity (the `retention_churn` model, not yet
  built). Renewal price optimisation is Earnix's core use case.

The showable artefacts are the **demand curve + cost line per segment**, the
**optimal price**, the **efficient frontier** (volume vs profit), and the
**portfolio impact** of moving from the current rate book to the optimised one.

---

## 3. Target-state architecture in v2

How it plugs into the existing flow (additions in **bold**):

```
UPT (gold) ──> Cost model  (freq_glm × sev_glm)              ─┐
           └─> Demand model (demand_gbm: conversion|price)    │
Quote stream ─> Retention model (retention_churn: renewal|price) ─┤
                                                               ▼
                              **Optimisation engine**  (notebook + util)
                         objective + constraints  ──>  p* per segment
                                                               │
                    **Optimisation config** (governed table, versioned)
                                                               ▼
              **App page "Price Optimisation"**  ── efficient frontier,
              demand/cost curves, p* per segment, portfolio impact vs book,
              what-if levers, **guardrail + regulatory panel**
                                                               │
                           Governance packs · audit_log · shadow_pricing_impact
                              (reuse existing governance & A/B rails)
```

Nothing here requires the live-serving tier. It is a batch/interactive
optimisation over governed tables + scale-to-zero model endpoints — fits the
**Core** profile philosophy.

---

## 4. Current v2 state (verified 2026-08-13)

| Building block | v2 state | Verdict |
|---|---|---|
| Cost model | `freq_glm`, `sev_glm` registered; `pricing_scorer` endpoint READY | ✅ present |
| Demand/elasticity model | `demand_gbm` registered (target `converted`; price features `log_premium`, `quote_to_market_ratio`, `competitor_a_min_premium`) | ⚠️ present but see §5.1 |
| Retention model | **absent** (planned in v2_plan WS2 as `retention_churn`, notebook only) | ❌ missing |
| Quote stream w/ conversion | `quotes` (conversion outcomes), 3 payload tables | ✅ present |
| Rate book / rating config | `rating_engine_config`, `pricing_engine_releases` | ✅ present |
| A/B & impact framework | `shadow_pricing_impact` (proxy formula — kept as-is per v2 plan) | ✅ reusable |
| Governance / audit | governance packs, `audit_log`, bias | ✅ reusable |
| App shell, monitoring, agents | `pricing_chat_agent`, `pricing_governance_agent` READY; mature React app | ✅ reusable |
| **Optimisation engine** | **absent** | ❌ core gap |
| **Constraint / guardrail framework** | **absent** | ❌ missing |
| **Optimisation objective/config governance** | **absent** | ❌ missing |
| **"Price Optimisation" app page** | **absent** | ❌ missing |
| **Efficient frontier / portfolio impact view** | **absent** | ❌ missing |

v2 runs the "core" profile (35 tables) — no live-serving/MCP/OTel assets, which
is fine: optimisation does not need them.

---

## 5. Gap analysis — what is missing

### 5.1 The elasticity signal in the data (foundational — the long pole)

**Verified on v2:** conversion rate is *flat* at ~0.62–0.68 across the entire
price range (0.1× to 3.3× of market rate). The synthetic `converted` flag was
generated **independently of price**, so there is **no price→demand
relationship** to optimise against.

Consequence: the demand model can be accurate on risk/location/credit features
yet carry **near-zero real price elasticity**. An optimiser on this data finds no
interior optimum — with demand insensitive to price, "optimal" price runs to the
constraint ceiling, which makes the demo tell the wrong story.

**What's needed (spec, not build):** the quote-generation process must embed a
**calibrated price-elasticity data-generating process** — conversion as a logit
declining in price-relative-to-market and to competitor, with the slope
(elasticity) varying by segment (SIC, size band, region, risk tier). Same
treatment for the renewal/retention series. This is the single most important
missing piece; without it, everything downstream is unconvincing. Best fixed in
the v2 data generator so both commercial and motor lines inherit it.

### 5.2 Demand & retention models (re-spec, not just retrain)

- **Demand model** — after 5.1, re-train and **validate the price response**:
  the modelled demand curve must be monotonically decreasing in price and produce
  sensible per-segment elasticities. Add an elasticity-by-segment diagnostic. The
  current model has no such validation.
- **Retention model** — does not exist as a deployed asset. Needs building and
  registering (`retention_churn`, already flagged in v2_plan WS2) for renewal
  optimisation. Requires a renewal dataset with retention outcomes and price
  variation (tie to 5.1).

### 5.3 Optimisation engine (the core net-new artefact)

Absent entirely. Specification:
- A **worked-example notebook** + a small reusable util that, per segment (and
  optionally per policy), takes `c(x)` and `d(p|x)` and solves for `p*` under a
  chosen objective and constraint set. Start with a **grid search** over a price
  multiplier for transparency; offer a `scipy.optimize` constrained solve as the
  "proper" version. Portfolio-level constraints (e.g. hold overall volume while
  lifting profit) via a Lagrange multiplier / dual sweep.
- Emit an **efficient frontier** (volume vs profit as the objective weight
  sweeps) and a **portfolio impact** table (current book vs optimised: GWP,
  expected profit, conversion, loss ratio).
- MLflow-tracked runs so each optimisation is reproducible and comparable.

### 5.4 Constraint & guardrail framework

Absent. Needs a declarative, **governed constraint config** (a versioned table):
rate-change caps, margin floors, competitive bounds, and the regulatory rules of
§6 — each applied in the engine and surfaced in the app with a pass/fail and the
binding constraint per segment.

### 5.5 Objective/config governance

The optimisation objective + constraints + model versions used should be a
**versioned, audited config** (same pattern as `rating_engine_config` /
`pricing_engine_releases`). This is a headline differentiator vs a black-box
optimiser: the "why" of every price is a governed, diffable artefact.

### 5.6 App page + narrative

Absent. New **"Price Optimisation"** page: objective + constraint selector;
demand curve + cost line + `p*` per segment; efficient frontier; portfolio impact
vs current book; what-if levers; and a **guardrail/regulatory panel** explaining
every constraint that bound. Reuse the existing page-explainer pattern and the
governance/audit chrome. Add a talk-track section.

### 5.7 Monitoring of optimisation outcomes

Partial. `shadow_pricing_impact` (kept as a proxy per v2 plan) can host a
before/after A-B of book vs optimised. A monitoring view of realised vs predicted
conversion/margin after a price move would complete the loop (can be phase 2).

---

## 6. Regulatory framing (must be explicit for a UK GI client)

If the client is **UK GI**, price optimisation is *regulated* and this must be
built into the demo, not bolted on:

- **FCA GI pricing rules (PS21/5, in force Jan 2022):** the renewal price for an
  existing customer must be **no higher** than the equivalent new-business price
  (the ban on "price walking"). A naïve retention-elasticity optimiser will *want*
  to walk prices — so the demo must show the **constraint enforced** and the
  optimiser respecting it.
- **Fair Value (PROD 4 / Consumer Duty):** price must reflect fair value; the
  governed objective/constraint config + audit trail is exactly the evidence a
  fair-value assessment needs.

This is a **strength**, not a caveat: the demo's punchline is that the platform
makes the optimiser *provably compliant* — every price move traceable to a
governed rule — which a closed vendor engine struggles to evidence. Confirm the
client's jurisdiction before committing to the FCA framing; the constraint
pattern generalises to other regulators.

---

## 7. Scope options & indicative effort (no build implied)

Effort is engineering-days for a demo on the v2 base; ranges, not commitments.

### MVP — new-business profit optimisation (~4–6 days)
1. Elasticity DGP in the v2 data generator + regenerate quotes (§5.1) — 1.5–2d.
2. Re-train + validate `demand_gbm` price response; segment elasticity diagnostic
   (§5.2) — 0.5–1d.
3. Optimisation engine notebook: segment-level profit-max, grid solve, one
   constraint (rate-change cap); efficient frontier (§5.3) — 1.5d.
4. One app page: demand/cost curves, `p*`, frontier, portfolio impact (§5.6) — 1.5d.

### Full — multi-objective + renewal + regulatory governance (~9–13 days)
Adds: `retention_churn` model + renewal price×retention optimisation (§5.2);
multi-objective with portfolio constraints (§5.3); the governed constraint/config
framework incl. FCA no-price-walking + fair-value (§5.4–5.5, §6); guardrail panel
+ monitoring A-B (§5.6–5.7); talk-track + docs.

The mature v2 app, governance rails, A/B framework and existing cost/demand
models are what keep this small — the real work is **the elasticity data fix + an
optimisation layer + one app page + the regulatory constraint framing**, not a
ground-up build.

---

## 8. Alignment with v2 conventions

- **Tier:** fits **Core** (batch/interactive, scale-to-zero, no online store).
  Could alternatively be gated behind a new `deploy_profile` value if we want it
  optional — decision for Laurence.
- **Naming:** single schema `pricing_workbench`, generic table names, no prefix;
  any workspace-global asset (a `pw_optimiser` job/endpoint) takes the `pw_`
  prefix per v2 plan §3.
- **Tags:** apply the uniform set (`project=pricing_workbench`, `owner`,
  `environment=demo`, `managed_by=dab`, `tier`, `contains_pii`) to every new
  asset.
- **Adjacency:** overlaps the backlog "drop a new dataset → portfolio what-if"
  item (v2_plan §8) — the raw-vector scorer both need is the same enabler; worth
  sequencing together.

---

## 9. Open decisions for Laurence

1. **Primary objective to lead with** — new-business conversion, or renewal
   retention? (Steers whether we build `retention_churn` first.)
2. **Jurisdiction** — is the client UK GI (→ make FCA no-price-walking + fair
   value a headline), or another market (→ generalise the constraint framing)?
3. **Core vs optional tier** for the optimisation assets.
4. **Fix the elasticity DGP globally?** It currently makes *every* demand-curve
   view flat, so fixing it improves the base demo too — but it touches the shared
   data generator. In-scope for this, or a separate data workstream?
5. **MVP first or straight to full** — do we want a fast, single-objective
   proof to put in front of the client, or the governed multi-objective version?

---

## 10. Risks & dependencies

- **Data realism (highest):** the whole demo lives or dies on a believable,
  segment-varying elasticity. Getting the DGP calibrated (sensible elasticities,
  not degenerate) is the main technical risk.
- **Regulatory correctness:** the FCA framing must be accurate; validate the
  no-price-walking constraint logic with an SME before it goes client-facing.
- **Scope creep into "we deliver models":** hold the line — this is a reference
  pattern the client owns, demonstrated on synthetic data.
- **Depends on** the v2 base being stable post-WS2 (naming/tags) and, for
  renewal, on a renewal/retention dataset that does not yet exist.
```
