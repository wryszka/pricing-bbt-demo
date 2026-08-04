"""Agentic-distribution tool layer — the carrier's *services*, not its raw data.

This module is the single definition of what an outside agent can do with the
motor pricing engine. Both surfaces sit on top of it:

  server/routes/mcp.py    JSON-RPC MCP server (external agents, Claude Desktop)
  server/routes/broker.py conversational quote page (Claude drives these tools)

Design rule: the LLM never invents a premium. Every price returned here comes
from the `motor_pricing_scorer` Model Serving endpoint — the same route-optimized
endpoint the Live Pricing System and the load tester hit. The engine is
data-in / price-out; these tools only decide *what data* goes in.

The feature contract has three tiers:

  ASKED      what a real quote journey asks the customer (15 fields)
  DERIVED    computed from the asked answers (no customer input)
  BOOK_MEAN  genuinely unknowable for a new customer — telematics and
             behaviour history. A new customer has no driving record with us,
             so these start at the book mean and only tighten once real
             telematics arrive. Radar Live behaves the same way.

That last tier is a talking point, not a fudge: `price_motor_risk` reports which
inputs were customer-supplied and which fell back to book mean, so the
provenance of every number is visible in the response.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature contract
# ---------------------------------------------------------------------------
# The 28 features the motor models consume. Must stay in step with
# WHATIF_FEATURES in routes/live_pricing.py and UNION_FEATURES in the scorer
# notebook (src/07_serving/live_pricing/motor_pricing_scorer.py).
MOTOR_FEATURES = [
    "annual_mileage", "at_fault_count_5y", "avg_speed_mph", "behaviour_score",
    "business_use", "claim_count_5y", "current_premium", "distinct_perils",
    "driver_age", "fuel_type", "gender", "hours_driven_30d", "license_years_held",
    "marital_status", "night_driving_pct", "no_claims_years", "occupation_class",
    "open_claims_count", "parking_overnight", "prior_accidents_5y",
    "prior_convictions", "recent_curfew_breaches", "recent_harsh_braking_30d",
    "recent_speeding_events", "telematics_recent_event_count", "vehicle_age",
    "vehicle_group", "vehicle_value",
]

# Book means over unified_motor_table_live (1,000,000 policies, measured
# 2026-08-04). These cover the telematics/behaviour block a brand-new customer
# cannot answer, plus current_premium (no prior premium with us) and
# distinct_perils (no claim history with us).
BOOK_MEANS: dict[str, Any] = {
    "behaviour_score":               75.39,
    "avg_speed_mph":                 35.49,
    "night_driving_pct":             18.6567,
    "hours_driven_30d":              39.97,
    "recent_harsh_braking_30d":      2.03,
    "recent_speeding_events":        0.53,
    "recent_curfew_breaches":        0.23,
    "telematics_recent_event_count": 2.79,
    "current_premium":               523.40,
    "distinct_perils":               0.44,
}

# What a real motor quote journey asks. `required` fields must be present before
# we will price; the rest materially move the price but have a sane default.
# `domain` values are the real value sets in the UPT — an agent that reads this
# schema learns the carrier's accepted vocabulary, which is the whole point.
QUOTE_QUESTIONS: list[dict[str, Any]] = [
    # --- driver ---
    {"field": "driver_age", "label": "Driver age", "type": "integer",
     "required": True, "min": 17, "max": 99,
     "why": "Age is one of the strongest frequency signals in motor."},
    {"field": "license_years_held", "label": "Years licence held", "type": "integer",
     "required": True, "min": 0, "max": 80,
     "why": "Experience separates young drivers from newly-licensed older ones."},
    {"field": "no_claims_years", "label": "No-claims years", "type": "integer",
     "required": True, "min": 0, "max": 25,
     "why": "Drives the NCD discount and is a strong proxy for past risk."},
    {"field": "occupation_class", "label": "Occupation", "type": "enum",
     "required": True,
     "domain": ["Office", "Skilled Manual", "Professional", "Service",
                "Student", "Self-Employed"],
     "why": "Correlates with mileage pattern and time-of-day exposure."},
    {"field": "marital_status", "label": "Marital status", "type": "enum",
     "required": False, "default": "Single",
     "domain": ["Single", "Married", "Divorced"]},
    {"field": "gender", "label": "Gender", "type": "enum",
     "required": False, "default": "M", "domain": ["M", "F"],
     "why": "Held for statistical reporting; EU/UK pricing must not rate on it."},

    # --- vehicle ---
    {"field": "vehicle_value", "label": "Vehicle value (GBP)", "type": "number",
     "required": True, "min": 500, "max": 250000,
     "why": "Primary severity driver — sets the size of a total-loss claim."},
    {"field": "vehicle_age", "label": "Vehicle age (years)", "type": "integer",
     "required": True, "min": 0, "max": 40},
    {"field": "vehicle_group", "label": "ABI vehicle group", "type": "integer",
     "required": False, "default": 8, "min": 1, "max": 50,
     "why": "Industry vehicle-rating group; derived from make/model in a real journey."},
    {"field": "fuel_type", "label": "Fuel type", "type": "enum",
     "required": False, "default": "Petrol",
     "domain": ["Petrol", "Diesel", "Hybrid", "Electric"],
     "why": "EV repair costs run materially higher — a live severity issue."},

    # --- use ---
    {"field": "annual_mileage", "label": "Annual mileage", "type": "integer",
     "required": True, "min": 500, "max": 60000,
     "why": "Exposure: more miles, more chance of a claim."},
    {"field": "parking_overnight", "label": "Overnight parking", "type": "enum",
     "required": True, "domain": ["Garage", "Driveway", "Street"],
     "why": "Theft and vandalism frequency differ sharply by where the car sleeps."},
    {"field": "business_use", "label": "Business use", "type": "enum",
     "required": False, "default": "N", "domain": ["Y", "N"]},

    # --- history ---
    {"field": "claim_count_5y", "label": "Claims in last 5 years", "type": "integer",
     "required": True, "min": 0, "max": 10},
    {"field": "prior_convictions", "label": "Motoring convictions", "type": "integer",
     "required": False, "default": 0, "min": 0, "max": 10},
]

QUESTION_INDEX = {q["field"]: q for q in QUOTE_QUESTIONS}
REQUIRED_FIELDS = [q["field"] for q in QUOTE_QUESTIONS if q["required"]]

# Whole-number vs continuous features. `motor_pricing_scorer_direct` is a plain
# pyfunc and tolerates floats throughout, but sending a clean 42 rather than
# 42.0 keeps the payload honest and stays correct if a signature is ever
# enforced on it (the route-optimized sibling does enforce one).
INT_FEATURES = frozenset({
    "annual_mileage", "at_fault_count_5y", "behaviour_score", "claim_count_5y",
    "distinct_perils", "driver_age", "license_years_held", "no_claims_years",
    "open_claims_count", "prior_accidents_5y", "prior_convictions",
    "recent_curfew_breaches", "recent_harsh_braking_30d",
    "recent_speeding_events", "telematics_recent_event_count", "vehicle_age",
    "vehicle_group",
})
FLOAT_FEATURES = frozenset({
    "avg_speed_mph", "current_premium", "hours_driven_30d",
    "night_driving_pct", "vehicle_value",
})
CATEGORICAL_FEATURES = frozenset({
    "business_use", "fuel_type", "gender", "marital_status",
    "occupation_class", "parking_overnight",
})

# Cover options shown to a customer. These are presentation-level (the engine
# prices the risk; cover choice scales what the customer buys).
COVER_OPTIONS = [
    {"cover": "Comprehensive", "excess_options": [250, 500, 750],
     "includes": ["Accidental damage", "Fire & theft", "Third-party liability",
                  "Windscreen cover", "Personal belongings"]},
    {"cover": "Third Party, Fire & Theft", "excess_options": [250, 500],
     "includes": ["Fire & theft", "Third-party liability"]},
]


def derived_features(answers: dict[str, Any]) -> dict[str, Any]:
    """Fields a real journey computes rather than asks.

    `at_fault_count_5y` and `open_claims_count` follow from the declared claim
    count; `prior_accidents_5y` mirrors it. A production integration would pull
    these from the claims system / MID rather than infer them.
    """
    claims = int(answers.get("claim_count_5y") or 0)
    return {
        "at_fault_count_5y":  max(0, claims - 1) if claims > 1 else claims,
        "prior_accidents_5y": claims,
        "open_claims_count":  0,
    }


def build_feature_vector(answers: dict[str, Any]) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Turn customer answers into the full 28-feature vector the engine wants.

    Returns (features, provenance) where provenance splits every field into
    'customer_supplied' (they told us), 'journey_default' (journey default for a
    non-required question, or derived) and 'book_mean_fallback' (unknowable for a
    new customer). The caller surfaces this so nobody has to wonder where a
    number came from.

    These key names are the wire contract — both the MCP response and the chat
    response pass this dict straight through, and BrokerChat.tsx reads these
    names. Renaming a key here silently blanks the "where the inputs came from"
    panel, so keep them in step with the UI.
    """
    features: dict[str, Any] = {}
    provenance: dict[str, list[str]] = {
        "customer_supplied": [], "journey_default": [], "book_mean_fallback": []}

    for q in QUOTE_QUESTIONS:
        f = q["field"]
        v = answers.get(f)
        if v is not None and str(v) != "":
            features[f] = v
            provenance["customer_supplied"].append(f)
        else:
            features[f] = q.get("default")
            provenance["journey_default"].append(f)

    for f, v in derived_features(answers).items():
        features[f] = v
        provenance["journey_default"].append(f)

    for f, v in BOOK_MEANS.items():
        if f not in features or features[f] is None:
            features[f] = v
            provenance["book_mean_fallback"].append(f)

    # Coerce to the dtypes the served model's signature enforces. An agent may
    # hand us strings, and MLflow schema enforcement rejects a float where the
    # signature says integer/long — so this split matters.
    for f in INT_FEATURES:
        if features.get(f) is not None:
            try:
                features[f] = int(round(float(features[f])))
            except (TypeError, ValueError):
                features[f] = 0
    for f in FLOAT_FEATURES:
        if features.get(f) is not None:
            try:
                features[f] = float(features[f])
            except (TypeError, ValueError):
                features[f] = 0.0

    return features, provenance


def validate(answers: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Check answers against the published contract.

    Returns (missing_required, errors). Kept separate from pricing so an agent
    can ask "am I ready to quote?" without burning an engine call.
    """
    missing = [f for f in REQUIRED_FIELDS
               if answers.get(f) is None or str(answers.get(f)) == ""]
    errors: list[str] = []

    for f, v in (answers or {}).items():
        q = QUESTION_INDEX.get(f)
        if not q or v is None or str(v) == "":
            continue
        if q["type"] == "enum":
            if str(v) not in q["domain"]:
                errors.append(
                    f"{f}: '{v}' is not accepted — must be one of {q['domain']}")
        else:
            try:
                n = float(v)
            except (TypeError, ValueError):
                errors.append(f"{f}: '{v}' is not a number")
                continue
            if "min" in q and n < q["min"]:
                errors.append(f"{f}: {v} is below the minimum {q['min']}")
            if "max" in q and n > q["max"]:
                errors.append(f"{f}: {v} is above the maximum {q['max']}")

    return missing, errors


def next_questions(answers: dict[str, Any], limit: int = 3) -> list[dict[str, Any]]:
    """The most useful things still to ask — required fields first, then the
    optional ones that move the price most. Lets an agent run a short journey
    instead of dumping a 15-field form on the customer.
    """
    out = [QUESTION_INDEX[f] for f in REQUIRED_FIELDS
           if answers.get(f) is None or str(answers.get(f)) == ""]
    if len(out) < limit:
        for q in QUOTE_QUESTIONS:
            if q["required"]:
                continue
            if answers.get(q["field"]) in (None, "") and q.get("why"):
                out.append(q)
    return out[:limit]
