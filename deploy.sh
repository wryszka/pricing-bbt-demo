#!/usr/bin/env bash
# Pricing Workbench deploy helper.
#
#   ./deploy.sh dev   — deploy bundle + app to fevm-lr-dev-aws-us  (DEV profile)
#   ./deploy.sh prod  — deploy bundle + app to fevm-lr-serverless-aws-us (DEFAULT profile)
#
# Copies the right app.yaml template into place before bundle deploy so the
# app picks up the target's catalog / warehouse / Genie space ids.
set -euo pipefail

TARGET=${1:-}
case "$TARGET" in
  dev)
    PROFILE=DEV
    ;;
  prod)
    PROFILE=DEFAULT
    ;;
  *)
    echo "Usage: $0 dev|prod" >&2
    exit 1
    ;;
esac

cd "$(dirname "$0")"

echo "==> [$TARGET] swapping src/app/app.yaml to app.${TARGET}.yaml"
cp "src/app/app.${TARGET}.yaml" "src/app/app.yaml"

echo "==> [$TARGET] building frontend"
( cd src/app/frontend && npm run build )

echo "==> [$TARGET] deploying bundle"
databricks bundle deploy --target "$TARGET" --profile "$PROFILE"

APP_PATH="/Workspace/Users/laurence.ryszka@databricks.com/.bundle/pricing-upt-demo/${TARGET}/files/src/app"
echo "==> [$TARGET] deploying app source from $APP_PATH"
databricks apps deploy pricing-workbench \
    --source-code-path "$APP_PATH" --profile "$PROFILE"

echo "==> [$TARGET] done."
