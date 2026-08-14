"""Price Optimisation — serves the worked-example optimiser output.

Reads the governed tables written by src/04_models/production/price_optimiser.py
(optimisation_summary, optimisation_curve, optimisation_config). This is a demo
OF optimisation: the app renders the per-segment demand curve + cost line, the
profit-optimal price, the volume/profit frontier, and the governed constraints —
every number traceable to readable code, the wedge against a black-box optimiser.
"""
import logging

from fastapi import APIRouter

from server.config import fqn
from server.sql import execute_query

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/optimisation", tags=["optimisation"])


async def _safe(sql: str):
    try:
        return await execute_query(sql)
    except Exception as e:
        logger.warning("optimisation query failed: %s", str(e)[:160])
        return None


# The SQL Statement API returns every value as a string; the UI does maths on
# these (.toFixed, comparisons), so coerce numeric columns to real numbers.
_NUM_COLS = {
    "n_quotes", "elasticity", "market_ref", "cost_line", "current_multiplier",
    "current_conversion", "current_profit_per_quote", "optimal_multiplier",
    "optimal_conversion", "optimal_profit_per_quote", "profit_uplift_per_quote",
    "profit_uplift_pct", "price_multiplier", "expected_conversion", "price",
    "expected_profit_per_quote", "rate_change_cap", "target_loss_ratio",
    "margin_floor",
}
_BOOL_COLS = {"within_rate_cap"}


def _coerce(rows):
    if not rows:
        return rows
    for r in rows:
        for k, v in list(r.items()):
            if v is None:
                continue
            if k in _NUM_COLS:
                try: r[k] = float(v)
                except (TypeError, ValueError): pass
            elif k in _BOOL_COLS:
                r[k] = str(v).lower() in ("true", "1", "t")
    return rows


@router.get("/summary")
async def optimisation_summary():
    """Per-segment current-vs-optimal, the price/demand/profit curve for the
    frontier, and the governed objective/constraint config."""
    summary = await _safe(f"""
        SELECT segment, n_quotes, elasticity, market_ref, cost_line,
               current_multiplier, current_conversion, current_profit_per_quote,
               optimal_multiplier, optimal_conversion, optimal_profit_per_quote,
               profit_uplift_per_quote, profit_uplift_pct, binding_constraint
        FROM {fqn('optimisation_summary')} ORDER BY segment
    """)
    curve = await _safe(f"""
        SELECT segment, price_multiplier, expected_conversion,
               price, expected_profit_per_quote, within_rate_cap
        FROM {fqn('optimisation_curve')} ORDER BY segment, price_multiplier
    """)
    config = await _safe(f"""
        SELECT version, objective, rate_change_cap, target_loss_ratio,
               margin_floor, demand_source, cost_source, created_at
        FROM {fqn('optimisation_config')} ORDER BY created_at DESC LIMIT 1
    """)

    if summary is None:
        return {
            "available": False,
            "message": "Optimisation not run yet on this workspace. Run the "
                       "'Price optimisation — worked example' job (bundle key "
                       "price_optimiser).",
            "segments": [], "curve": [], "config": None,
        }

    return {
        "available": True,
        "segments": _coerce(summary) or [],
        "curve": _coerce(curve) or [],
        "config": (_coerce(config) or [None])[0],
    }
