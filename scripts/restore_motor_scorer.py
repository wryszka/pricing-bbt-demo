#!/usr/bin/env python3
"""Roll the motor scorer endpoints back to a known-good model version.

Written alongside the 2026-08-04 frequency-annualisation fix (rating engine
motor_v1.2 -> motor_v1.3). If a re-log leaves either endpoint serving a bad
version, this puts them back without re-running the training notebook.

Restore point captured 2026-08-04 10:09 BST on fevm-lr-dev-aws-us:
    motor_pricing_scorer         v4   route-optimized, min 4 / max 64, no s2z
    motor_pricing_scorer_direct  v1   plain, scale-to-zero

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

# Known-good versions at the restore point above.
GOOD_SCORER_VERSION = "4"
GOOD_DIRECT_VERSION = "1"


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
    ap.add_argument("--restore", action="store_true", help="apply the rollback")
    ap.add_argument("--dry-run", action="store_true", help="show the payloads only")
    ap.add_argument("--scorer-version", default=GOOD_SCORER_VERSION)
    ap.add_argument("--direct-version", default=GOOD_DIRECT_VERSION)
    ap.add_argument("--only", choices=["scorer", "direct"],
                    help="restore just one endpoint")
    args = ap.parse_args()

    w = _client()
    print(f"workspace: {w.config.host}")

    if args.show or not args.restore:
        show(w)
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
