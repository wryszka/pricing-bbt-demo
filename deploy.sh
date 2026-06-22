#!/usr/bin/env bash
# Pricing Workbench deploy helper.
#
#   ./deploy.sh <target> [profile]
#
#   ./deploy.sh dev               — maintainer target (DEV profile)
#   ./deploy.sh prod              — maintainer target (DEFAULT profile)
#   ./deploy.sh pinchu            — maintainer target (fevm-pinchudemo profile)
#   ./deploy.sh mytarget myprofile — any target; pass your CLI profile explicitly
#
# Copies the right app.yaml template into place before bundle deploy so the
# app picks up the target's catalog / warehouse / Genie space ids, then runs the
# post-deploy bootstrap (grants the app SP + sets initial champions) so a fresh
# deploy actually works end to end.
#
# NOTE: run the data pipeline (setup_demo → … → production_training) BEFORE this
# script — the grant job sets champions and grants experiment read, which need the
# schema and trained models to already exist.
set -euo pipefail

TARGET=${1:-}
PROFILE=${2:-}
# Known maintainer targets map to a default profile; any other target requires
# the profile to be passed explicitly so the template needs no edits here.
if [ -z "$PROFILE" ]; then
  case "$TARGET" in
    dev)    PROFILE=DEV ;;
    prod)   PROFILE=DEFAULT ;;
    pinchu) PROFILE=fevm-pinchudemo ;;
    *)
      echo "Usage: $0 <target> [profile]" >&2
      echo "  built-in targets dev|prod|pinchu have default profiles;" >&2
      echo "  for any other target pass your CLI profile as the 2nd arg." >&2
      exit 1
      ;;
  esac
fi

cd "$(dirname "$0")"

APP_NAME="pricing-workbench"

echo "==> [$TARGET] swapping src/app/app.yaml to app.${TARGET}.yaml"
cp "src/app/app.${TARGET}.yaml" "src/app/app.yaml"

echo "==> [$TARGET] building frontend"
( cd src/app/frontend && npm run build )

echo "==> [$TARGET] deploying bundle"
databricks bundle deploy --target "$TARGET" --profile "$PROFILE"

# Derive the bundle root from the deployer's own identity rather than hardcoding
# a username — matches databricks.yml root_path (${workspace.current_user.userName}).
DEPLOYER=$(databricks current-user me --profile "$PROFILE" -o json | python3 -c "import sys,json; print(json.load(sys.stdin)['userName'])")
APP_PATH="/Workspace/Users/${DEPLOYER}/.bundle/pricing-upt-demo/${TARGET}/files/src/app"

# The app must be RUNNING before `apps deploy`; on first create it is STOPPED.
# Start is a no-op error if already running, so guard it.
echo "==> [$TARGET] ensuring app '$APP_NAME' is running"
databricks apps start "$APP_NAME" --profile "$PROFILE" >/dev/null 2>&1 || true

echo "==> [$TARGET] deploying app source from $APP_PATH"
databricks apps deploy "$APP_NAME" \
    --source-code-path "$APP_PATH" --profile "$PROFILE"

# Post-deploy bootstrap: the app SP is auto-minted on app creation, so its UC /
# experiment grants and the initial champion aliases can only be set now that the
# app exists. Idempotent — safe to re-run.
echo "==> [$TARGET] running post-deploy bootstrap (grant app SP + set champions)"
databricks bundle run grant_app_permissions --target "$TARGET" --profile "$PROFILE"

echo "==> [$TARGET] creating Genie spaces + wiring app_config"
databricks bundle run create_genie_spaces --target "$TARGET" --profile "$PROFILE"

echo "==> [$TARGET] done."
