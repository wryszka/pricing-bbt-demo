#!/usr/bin/env bash
# Pricing Workbench deploy helper.
#
#   ./deploy.sh v2    — deploy bundle + app to fevm-lr-pricing-v2-aws-us (PRICING_V2)
#   ./deploy.sh dev   — deploy bundle + app to fevm-lr-dev-aws-us        (DEV)
#   ./deploy.sh prod  — deploy bundle + app to fevm-lr-serverless-aws-us (DEFAULT)
#
# Copies the right app.<target>.yaml into place before bundle deploy so the app
# picks up the target's catalog / warehouse / Genie ids and bundle path.
#
# NOTE — first deploy to a fresh workspace is two-phase (see README): the app's
# service principal doesn't exist until the app is created, but jobs grant it
# CAN_MANAGE_RUN. Create the app, capture its SP, set app_service_principal_id,
# then run this. On a workspace where the app already exists this is a no-op.
set -euo pipefail

TARGET=${1:-}
case "$TARGET" in
  v2)   PROFILE=PRICING_V2 ;;
  dev)  PROFILE=DEV ;;
  prod) PROFILE=DEFAULT ;;
  *)    echo "Usage: $0 v2|dev|prod" >&2; exit 1 ;;
esac

cd "$(dirname "$0")"
BUNDLE=pricing-workbench

echo "==> [$TARGET] swapping src/app/app.yaml to app.${TARGET}.yaml"
cp "src/app/app.${TARGET}.yaml" "src/app/app.yaml"

echo "==> [$TARGET] building frontend"
( cd src/app/frontend && npm run build )

echo "==> [$TARGET] deploying bundle"
databricks bundle deploy --target "$TARGET" --profile "$PROFILE"

# Per-target bundle root: v2/prod deploy under /Workspace/Shared (org-shared);
# dev under the deployer's home (development mode).
case "$TARGET" in
  v2)   APP_PATH="/Workspace/Shared/.bundle/${BUNDLE}/v2/files/src/app" ;;
  prod) APP_PATH="/Workspace/Shared/.bundle/pricing-upt-demo/prod/files/src/app" ;;
  dev)  APP_PATH="/Workspace/Users/$(databricks --profile "$PROFILE" current-user me | python3 -c 'import json,sys;print(json.load(sys.stdin)["userName"])')/.bundle/${BUNDLE}/dev/files/src/app" ;;
esac

echo "==> [$TARGET] deploying app source from $APP_PATH"
databricks apps deploy pricing-workbench \
    --source-code-path "$APP_PATH" --profile "$PROFILE"

echo "==> [$TARGET] done."
