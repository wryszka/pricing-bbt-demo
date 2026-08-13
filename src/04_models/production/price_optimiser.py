# Databricks notebook source
# MAGIC %md
# MAGIC # Price Optimisation — worked example (new-business profit)
# MAGIC
# MAGIC A **demo of optimisation, not a production optimiser.** For each segment it
# MAGIC fits a transparent price-elasticity demand curve from the quote data,
# MAGIC lays a cost line against it, and grid-searches the price multiplier that
# MAGIC maximises expected profit `d(p)·(p − c)` subject to a rate-change cap and a
# MAGIC margin floor. Every number is readable code over governed tables — the
# MAGIC wedge against a black-box optimiser.
# MAGIC
# MAGIC Writes two governed tables:
# MAGIC  * `optimisation_curve`   — per segment × price multiplier: demand, profit
# MAGIC  * `optimisation_summary` — per segment: current vs optimal, uplift, binding constraint
# MAGIC
# MAGIC Objective/constraints are recorded as a versioned config row so the "why"
# MAGIC of every recommended price is auditable.

# COMMAND ----------

dbutils.widgets.text("catalog_name", "lr_pricing_v2_aws_us_catalog")
dbutils.widgets.text("schema_name", "pricing_workbench")
dbutils.widgets.text("rate_change_cap", "0.15")     # ±15% vs current book
dbutils.widgets.text("target_loss_ratio", "0.62")   # cost line = LR × market (illustrative)
dbutils.widgets.text("margin_floor", "0.05")        # min (p−c)/p

catalog = dbutils.widgets.get("catalog_name")
schema = dbutils.widgets.get("schema_name")
RATE_CAP = float(dbutils.widgets.get("rate_change_cap"))
TARGET_LR = float(dbutils.widgets.get("target_loss_ratio"))
MARGIN_FLOOR = float(dbutils.widgets.get("margin_floor"))
fqn = f"{catalog}.{schema}"

import numpy as np
import pandas as pd
import mlflow
from datetime import datetime

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Load quotes with the price-elasticity signal
# MAGIC Segment by account size (the DGP makes larger accounts more price-sensitive,
# MAGIC so segments show genuinely different elasticities → different optimal prices).

# COMMAND ----------

q = spark.sql(f"""
    SELECT gross_premium, market_premium, vs_market_rate,
           CASE WHEN converted IN ('Y','1','true','True') THEN 1 ELSE 0 END AS converted,
           CASE
             WHEN gross_premium < 10000  THEN '1 Micro (<£10k)'
             WHEN gross_premium < 50000  THEN '2 SME (£10–50k)'
             WHEN gross_premium < 250000 THEN '3 Mid (£50–250k)'
             ELSE '4 Large (£250k+)'
           END AS segment
    FROM {fqn}.quotes
    WHERE gross_premium IS NOT NULL AND vs_market_rate IS NOT NULL
      AND vs_market_rate BETWEEN 0.5 AND 2.0 AND is_outlier = false
""").toPandas()
print(f"{len(q):,} priced quotes across {q['segment'].nunique()} segments")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Fit a transparent demand curve per segment
# MAGIC Logistic conversion ~ price-to-market. Slope `b` is the segment elasticity.

# COMMAND ----------

from sklearn.linear_model import LogisticRegression

MULT_GRID = np.round(np.arange(0.80, 1.301, 0.02), 3)  # price vs market

# Expected loss ratio varies by segment — larger commercial accounts typically
# run hotter. Illustrative (disclaimed), but it's what makes optimisation find
# real, DIFFERENTIATED moves: a segment priced at market but running a high loss
# ratio is underpriced and should move up; a low-LR segment can trade margin for
# volume. Flat TARGET_LR is the fallback for any unmapped segment.
SEGMENT_LR = {
    "1 Micro (<£10k)":   0.54,
    "2 SME (£10–50k)":   0.63,
    "3 Mid (£50–250k)":  0.72,
    "4 Large (£250k+)":  0.80,
}
curve_rows, summary_rows = [], []

for seg, g in q.groupby("segment"):
    if len(g) < 200 or g["converted"].nunique() < 2:
        continue
    X = g[["vs_market_rate"]].values
    y = g["converted"].values
    lr = LogisticRegression().fit(X, y)
    b = float(lr.coef_[0][0])           # < 0 : demand falls as price rises
    demand = lr.predict_proba(MULT_GRID.reshape(-1, 1))[:, 1]

    market_ref = float(g["market_premium"].median())
    seg_lr = SEGMENT_LR.get(seg, TARGET_LR)
    cost = seg_lr * market_ref          # illustrative expected-claims cost line
    price = MULT_GRID * market_ref
    margin = price - cost
    profit = demand * margin            # expected profit per quote

    # Current position: where the book sits today (median offer vs market).
    cur_mult = float(np.clip(g["vs_market_rate"].median(), MULT_GRID.min(), MULT_GRID.max()))
    cur_i = int(np.argmin(np.abs(MULT_GRID - cur_mult)))

    # Constraints: rate-change cap around current, and a margin floor.
    allowed = (np.abs(MULT_GRID - cur_mult) <= RATE_CAP) & ((margin / np.maximum(price, 1)) >= MARGIN_FLOOR)
    if not allowed.any():
        allowed = np.abs(MULT_GRID - cur_mult) <= RATE_CAP  # fall back to rate cap only
    prof_masked = np.where(allowed, profit, -np.inf)
    opt_i = int(np.argmax(prof_masked))

    # What would the *unconstrained* optimum be? (to show what the cap costs)
    unc_i = int(np.argmax(profit))
    binding = "rate-change cap" if unc_i != opt_i and abs(MULT_GRID[unc_i] - cur_mult) > RATE_CAP \
              else ("margin floor" if unc_i != opt_i else "none")

    for m, d, pr, pf in zip(MULT_GRID, demand, price, profit):
        curve_rows.append((seg, float(m), float(d), float(pr), float(pf),
                           bool(np.abs(m - cur_mult) <= RATE_CAP)))

    summary_rows.append((
        seg, int(len(g)), round(b, 3), round(market_ref, 2), round(cost, 2),
        round(cur_mult, 3), round(float(demand[cur_i]), 4), round(float(profit[cur_i]), 2),
        round(float(MULT_GRID[opt_i]), 3), round(float(demand[opt_i]), 4), round(float(profit[opt_i]), 2),
        round(float(profit[opt_i] - profit[cur_i]), 2),
        round(100 * (profit[opt_i] - profit[cur_i]) / max(abs(profit[cur_i]), 1e-9), 1),
        binding,
    ))
    print(f"{seg}: elasticity b={b:.2f}  current mult={cur_mult:.2f}→ optimal {MULT_GRID[opt_i]:.2f}  "
          f"profit/quote {profit[cur_i]:.0f}→{profit[opt_i]:.0f}  binding={binding}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Write governed result tables + config

# COMMAND ----------

curve_df = spark.createDataFrame(
    pd.DataFrame(curve_rows, columns=[
        "segment", "price_multiplier", "expected_conversion",
        "price", "expected_profit_per_quote", "within_rate_cap"]))
curve_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fqn}.optimisation_curve")

summary_df = spark.createDataFrame(
    pd.DataFrame(summary_rows, columns=[
        "segment", "n_quotes", "elasticity", "market_ref", "cost_line",
        "current_multiplier", "current_conversion", "current_profit_per_quote",
        "optimal_multiplier", "optimal_conversion", "optimal_profit_per_quote",
        "profit_uplift_per_quote", "profit_uplift_pct", "binding_constraint"]))
summary_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fqn}.optimisation_summary")

# Versioned, audited objective/constraint config — the "why" of every price.
cfg = spark.createDataFrame(pd.DataFrame([{
    "version": datetime.utcnow().strftime("opt_%Y%m%d_%H%M%S"),
    "objective": "maximise expected profit d(p)*(p-c)",
    "rate_change_cap": RATE_CAP,
    "target_loss_ratio": TARGET_LR,
    "margin_floor": MARGIN_FLOOR,
    "demand_source": "per-segment logistic on vs_market_rate (quote data)",
    "cost_source": "illustrative: target_loss_ratio * market_premium",
    "created_at": datetime.utcnow().isoformat(),
}]))
cfg.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{fqn}.optimisation_config")

with mlflow.start_run(run_name="price_optimisation"):
    mlflow.log_params({"rate_change_cap": RATE_CAP, "target_loss_ratio": TARGET_LR,
                       "margin_floor": MARGIN_FLOOR, "segments": len(summary_rows)})
    tot_uplift = float(sum(r[11] for r in summary_rows))
    mlflow.log_metric("total_profit_uplift_per_quote", tot_uplift)

print(f"\n✓ optimisation_curve, optimisation_summary, optimisation_config written")
print(f"  segments optimised: {len(summary_rows)}")
