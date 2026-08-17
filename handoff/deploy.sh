#!/usr/bin/env bash
# Pricing Workbench deploy helper (sandbox target).
#
#   ./deploy.sh          — deploy bundle + app to the workspace in databricks.yml
#
# Uses your default Databricks CLI auth. See README.md for the full three-command
# deploy including the app service-principal bootstrap.
set -euo pipefail
cd "$(dirname "$0")"
TARGET=sandbox
BUNDLE=pricing-workbench

echo "==> [$TARGET] app.yaml <- app.${TARGET}.yaml"
cp "src/app/app.${TARGET}.yaml" "src/app/app.yaml"

echo "==> [$TARGET] building frontend"
( cd src/app/frontend && npm install --no-audit --no-fund && npm run build )

echo "==> [$TARGET] deploying bundle"
databricks bundle deploy --target "$TARGET"

APP_PATH="/Workspace/Shared/.bundle/${BUNDLE}/${TARGET}/files/src/app"
echo "==> [$TARGET] deploying app source from $APP_PATH"
databricks apps deploy pricing-workbench --source-code-path "$APP_PATH"

echo "==> [$TARGET] done. Now populate: databricks bundle run full_build --target $TARGET"
