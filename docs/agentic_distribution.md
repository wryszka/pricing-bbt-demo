# Agentic Distribution — add-on

Being present and priceable when the customer arrives through an AI agent rather
than a website. Three channels, one pricing engine — the same deployed motor
models the rest of the workbench trains, promotes and governs.

## Why this framing

An aggregator feed publishes **raw data**: an API takes a payload and returns a
number. Anyone can be a row in that comparison. What a carrier can do that an
aggregator cannot is publish **services** — discover what we need to price you,
get a real price, *ask why it is that price*, read the terms. The explanation is
the moat: it comes from the models that set the premium.

## The three surfaces

| Surface | Route | What it is |
|---|---|---|
| Direct / broker chat | `/quote-chat` | Customer buys by conversation. Claude runs the journey. |
| MCP server | `POST /api/mcp` | The same tools for outside agents (partner assistants, general agents). |
| Channel telemetry | `/add-ons/agentic-distribution` | Who called us, where journeys stalled, what converted. |

The chat page is deliberately chrome-less (like `/quote`) so it reads as the
insurer's own front end, not a page inside a Databricks app.

## Tools published

| Tool | Backed by |
|---|---|
| `get_quote_requirements` | the question contract in `server/mcp_tools.py` |
| `check_answers` | validation + next-best-question (no engine call) |
| `price_motor_risk` | `motor_pricing_scorer_direct` serving endpoint |
| `explain_price` | `pricing_chat_agent` Agent Framework endpoint |
| `policy_terms` | cover / excess options |

The chat surface adds `record_answers`, so answers accumulate turn by turn
instead of only at pricing time (this is what drives the progress bar).

## The rule that makes it credible

**The LLM never produces a premium.** Claude gathers answers and decides what to
ask next; the price comes from the serving endpoint. The system prompt forbids
inventing one, and the UI shows every tool call with its latency and the endpoint
name so an audience can see the number was computed, not generated. When the
engine is unavailable the assistant says so rather than substituting a figure —
verified behaviour, not just an instruction.

## Which engine, and why

`price_motor_risk` calls **`motor_pricing_scorer_direct`**, the plain pyfunc
endpoint that accepts a full 28-feature vector.

It does *not* call the route-optimized `motor_pricing_scorer`, even though that
one is faster and is what the Live Pricing System and load tester use. That
endpoint is a **FeatureLookup** model: you pass a `policy_id` and it hydrates
features from the Lakebase online store. Ideal for repricing the existing book at
100 QPS — but an agent-channel prospect has no policy with us yet, so there is
nothing to look up, and passing a raw feature vector fails schema enforcement
(correctly). Serving a brand-new risk means the direct endpoint.

Consequence for a live demo: `_direct` is scale-to-zero. The first call after an
idle period pays a cold start; after that it settles around 200–550ms. **Warm it
before you present** (see checklist).

## Answering with data a new customer cannot have

A real quote journey asks 15 questions. The models want 28 features. The gap is
split three ways, and every response reports which is which:

- **`customer`** — they told us.
- **`default`** — journey default or derived (e.g. `at_fault_count_5y` follows
  from declared claims).
- **`book_mean`** — genuinely unknowable for a new customer: telematics and
  driving-behaviour history, plus `current_premium` (no prior premium with us).
  Measured over `unified_motor_table_live` (1,000,000 policies, 2026-08-04).

This is a talking point, not a fudge: a new customer has no driving record, so
those inputs start at the book mean and tighten once real telematics arrive.
Radar Live behaves the same way. The provenance panel in the chat UI shows the
split live.

## Telemetry

Every tool call on both surfaces writes to `mcp_tool_calls` (Delta, created on
first use). The telemetry view reads it for the funnel
(sessions → discovered → priced → asked-why), latency per tool, premium spread,
and which agents are calling. Telemetry failures are logged and swallowed — a
warehouse blip must never break a live demo.

## Pre-demo checklist

1. **Warm the engine** — open `/quote-chat` and run one throwaway quote, or
   `POST /api/mcp` with `price_motor_risk`. Removes the cold-start pause.
2. Check `pricing_chat_agent` is READY if you plan to show `explain_price`.
3. Have the MCP URL ready to paste (`<app-url>/api/mcp`) if connecting an
   external client live.

## If the engine misbehaves

`scripts/restore_motor_scorer.py` rolls either serving endpoint back to a
known-good model version without re-running the training notebook:

```bash
python3 scripts/restore_motor_scorer.py --show               # current state
python3 scripts/restore_motor_scorer.py --restore --dry-run   # payloads only
python3 scripts/restore_motor_scorer.py --restore             # roll back
```

It only ever PUTs a new config — never delete+recreate — so the route-optimized
endpoint keeps its data-plane host and ACL. The app self-heals its cached client
after a version roll (see `reset_workspace_client`).

## Suggested run of play

1. **Open the tool manifest** — this is all an outside agent gets. No insurance
   knowledge assumed.
2. **Run the chat journey** — four turns to a real price. Point at the tool-call
   panel: `record_answers` … `price_motor_risk` … 250ms … the endpoint name.
3. **Ask "why is it that much?"** — `explain_price` answers from the rating
   breakdown. This is the beat an aggregator cannot do.
4. **Point an external MCP client at the URL** — same tools, no UI, the third
   modality with nothing extra built.
5. **Close on the telemetry page** — the channel is now measurable and tunable.

## Known limitations (state them, don't hide them)

- **Motor only.** The live scorer and online store are motor. Home and medical
  would be mock, so they are not included.
- **Premium level.** Fixed and live on dev 2026-08-04 as scorer **v6** /
  direct **v3** (rating engine `motor_v1.3`); verify with
  `scripts/restore_motor_scorer.py --probe`. The frequency GLM is trained on
  `claim_count_5y`, a five-year count, and the scorer had been treating it as
  annual — so every quote was ~5x too high. It now divides by
  `freq_exposure_years` before multiplying by per-claim severity. The reference
  risk moved **£4,679.80 → £935.96** (£78.00/month) against a book average of
  £523.40, and the existing `/quote` page came down with it (~£1,006). A
  residual gap remains — that is severity-model calibration, not a defect;
  premiums are now in a plausible band but are still not a market-realistic
  rate card. `annual_freq` is returned alongside `freq_pred` so the breakdown
  reconciles by hand.
- **No bind.** The journey quotes and explains; it does not issue a policy.
- `explain_price` output is salvaged from a truncated agent response (the
  endpoint's token cap cuts the JSON envelope). Prose is recovered rather than
  leaking raw JSON into the conversation.

## About this demo

Bricksurance SE is a fictional insurer. The models, serving endpoints,
governance and Unity Catalog assets are real Databricks components; the
portfolio is synthetic. Nothing here reflects any real insurer's rates or book.
