#!/usr/bin/env python3
"""Roll the motor scorer endpoints back to a known-good model version.

Written alongside the 2026-08-04 frequency-annualisation fix (rating engine
motor_v1.2 -> motor_v1.3). If a re-log leaves either endpoint serving a bad
version, this puts them back without re-running the training notebook.

Restore points on fevm-lr-dev-aws-us (both route-optimized flags preserved):
    v4 / v1  2026-08-04 10:09 BST — last motor_v1.2 build before the fix
    v5 / v2  2026-08-04 10:30 BST — ALSO motor_v1.2 (ran a stale notebook copy)
    v6 / v3  motor_v1.3, the annualisation fix — the intended good state

Defaults below point at the fix. To go back to pre-fix behaviour deliberately:
    --scorer-version 5 --direct-version 2

Quick check of which code a live endpoint is running: a motor_v1.3 response
includes an `annual_freq` field; motor_v1.2 does not.

Usage:
    python3 scripts/restore_motor_scorer.py --show
    python3 scripts/restore_motor_scorer.py --restore
    python3 scripts/restore_motor_scorer.py --restore --scorer-version 4 --direct-version 1

Note on the route-optimized endpoint: `route_optimized` is create-time only, so
this script only ever PUTs a new config (which preserves the flag). It never
deletes the endpoint — a delete/recreate rotates the data-plane host and resets
the ACL, which is exactly what you do not want mid-incident.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

os.environ.setdefault("DATABRICKS_PROFILE", "DEV")

from databricks.sdk import WorkspaceClient  # noqa: E402

CATALOG = os.getenv("CATALOG_NAME", "lr_dev_aws_us_catalog")
SCHEMA = os.getenv("SCHEMA_NAME", "pricing_upt")

SCORER = "motor_pricing_scorer"
DIRECT = "motor_pricing_scorer_direct"

# Known-good versions — the motor_v1.3 build with the frequency fix.
GOOD_SCORER_VERSION = os.getenv("GOOD_SCORER_VERSION", "6")
GOOD_DIRECT_VERSION = os.getenv("GOOD_DIRECT_VERSION", "3")

# Last pre-fix build, if you need to demonstrate the old behaviour.
PREFIX_SCORER_VERSION = "5"
PREFIX_DIRECT_VERSION = "2"


def _client() -> WorkspaceClient:
    return WorkspaceClient(profile=os.environ["DATABRICKS_PROFILE"])


def show(w: WorkspaceClient) -> None:
    for name in (SCORER, DIRECT):
        try:
            ep = w.serving_endpoints.get(name)
        except Exception as e:
            print(f"{name}: NOT FOUND ({str(e)[:120]})")
            continue
        print(f"\n{name}")
        print(f"  route_optimized : {getattr(ep, 'route_optimized', None)}")
        print(f"  ready           : {ep.state.ready} / {ep.state.config_update}")
        for e in (ep.config.served_entities or []) if ep.config else []:
            print(f"  served          : {e.entity_name} v{e.entity_version}")
        for e in (ep.pending_config.served_entities or []) if ep.pending_config else []:
            print(f"  PENDING         : {e.entity_name} v{e.entity_version}")


def probe(w: WorkspaceClient) -> None:
    """Price a fixed risk against the direct endpoint and report which rating
    engine answered. motor_v1.3 returns `annual_freq`; motor_v1.2 does not."""
    risk = {
        "annual_mileage": 9000, "at_fault_count_5y": 0, "avg_speed_mph": 35.49,
        "behaviour_score": 75, "business_use": "N", "claim_count_5y": 0,
        "current_premium": 523.4, "distinct_perils": 0, "driver_age": 42,
        "fuel_type": "Petrol", "gender": "M", "hours_driven_30d": 39.97,
        "license_years_held": 20, "marital_status": "Single",
        "night_driving_pct": 18.6567, "no_claims_years": 8,
        "occupation_class": "Office", "open_claims_count": 0,
        "parking_overnight": "Driveway", "prior_accidents_5y": 0,
        "prior_convictions": 0, "recent_curfew_breaches": 0,
        "recent_harsh_braking_30d": 2, "recent_speeding_events": 0,
        "telematics_recent_event_count": 3, "vehicle_age": 3,
        "vehicle_group": 8, "vehicle_value": 18000.0,
    }
    print("\n==> probing motor_pricing_scorer_direct with the reference risk")
    try:
        resp = w.api_client.do(
            "POST", f"/serving-endpoints/{DIRECT}/invocations",
            body={"dataframe_records": [risk]})
    except Exception as e:
        print(f"    FAILED: {str(e)[:200]}")
        return
    preds = (resp or {}).get("predictions") or []
    if not preds:
        print(f"    unexpected response: {json.dumps(resp)[:200]}")
        return
    row = preds[0]
    ver = row.get("rating_engine_version", "?")
    has_annual = "annual_freq" in row
    print(f"    rating_engine_version : {ver}")
    print(f"    annual_freq present   : {has_annual}  "
          f"({'motor_v1.3 fix IS live' if has_annual else 'still pre-fix code'})")
    print(f"    freq_pred (5-year)    : {row.get('freq_pred')}")
    if has_annual:
        print(f"    annual_freq           : {row.get('annual_freq')}")
    print(f"    technical_premium     : GBP {row.get('technical_premium')}")
    print(f"    final_premium         : GBP {row.get('final_premium')}")
    print("    book reference        : avg GBP 523.40 / p50 GBP 406.00")


def restore_one(w: WorkspaceClient, endpoint: str, model: str, version: str,
                *, scale_to_zero: bool, min_conc: int | None,
                max_conc: int | None, dry_run: bool) -> None:
    entity: dict = {
        "entity_name": model,
        "entity_version": version,
        "scale_to_zero_enabled": scale_to_zero,
    }
    if min_conc is not None:
        entity["min_provisioned_concurrency"] = min_conc
    if max_conc is not None:
        entity["max_provisioned_concurrency"] = max_conc

    body = {"served_entities": [entity]}
    print(f"\n==> {endpoint}: PUT config -> {model} v{version}")
    print(json.dumps(body, indent=2))
    if dry_run:
        print("    (dry run — nothing sent)")
        return
    w.api_client.do("PUT", f"/api/2.0/serving-endpoints/{endpoint}/config", body=body)
    print("    submitted; endpoint will roll to the restored version")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--show", action="store_true", help="print current state only")
    ap.add_argument("--probe", action="store_true",
                    help="price a reference risk and report which rating engine answered")
    ap.add_argument("--restore", action="store_true", help="apply the rollback")
    ap.add_argument("--dry-run", action="store_true", help="show the payloads only")
    ap.add_argument("--scorer-version", default=GOOD_SCORER_VERSION)
    ap.add_argument("--direct-version", default=GOOD_DIRECT_VERSION)
    ap.add_argument("--only", choices=["scorer", "direct"],
                    help="restore just one endpoint")
    args = ap.parse_args()

    w = _client()
    print(f"workspace: {w.config.host}")

    if args.show or args.probe or not args.restore:
        show(w)
        if args.probe:
            probe(w)
        if not args.restore:
            print("\n(no changes made — pass --restore to roll back)")
            return 0

    if args.only in (None, "scorer"):
        restore_one(w, SCORER, f"{CATALOG}.{SCHEMA}.{SCORER}", args.scorer_version,
                    scale_to_zero=False, min_conc=4, max_conc=64,
                    dry_run=args.dry_run)
    if args.only in (None, "direct"):
        restore_one(w, DIRECT, f"{CATALOG}.{SCHEMA}.{DIRECT}", args.direct_version,
                    scale_to_zero=False, min_conc=None, max_conc=None,
                    dry_run=args.dry_run)

    print("\nDone. Re-check with --show; the app self-heals its cached client.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
