#!/usr/bin/env bash
# Build the AXA hand-off tree from the current HEAD: a clean, generic export with
# dev/prod/v2 specifics stripped and internal docs/scripts removed. Idempotent.
# Usage: scripts/make_handoff.sh [dest]   (default /tmp/bricksurance-pricing-workbench)
set -euo pipefail
cd "$(dirname "$0")/.."
DEST="${1:-/tmp/bricksurance-pricing-workbench}"

echo "==> exporting HEAD to $DEST"
rm -rf "$DEST"; mkdir -p "$DEST"
git archive --format=tar HEAD | tar -x -C "$DEST"

echo "==> swapping in generic target/app/deploy templates"
rm -f "$DEST"/src/app/app.dev.yaml "$DEST"/src/app/app.prod.yaml \
      "$DEST"/src/app/app.v2.yaml "$DEST"/src/app/app.yaml
cp handoff/databricks.yml   "$DEST"/databricks.yml
cp handoff/deploy.sh        "$DEST"/deploy.sh
cp handoff/app.sandbox.yaml "$DEST"/src/app/app.sandbox.yaml
chmod +x "$DEST"/deploy.sh "$DEST"/scripts/*.sh 2>/dev/null || true

echo "==> removing internal-only docs / scripts / handoff dir"
rm -rf "$DEST"/handoff
rm -f "$DEST"/docs/v2_plan.md "$DEST"/docs/parked_backlog.md \
      "$DEST"/docs/optimisation_demo_spec.md "$DEST"/docs/DEPLOY.md \
      "$DEST"/scripts/patch_mart_dashboard.py "$DEST"/scripts/restore_motor_scorer.py \
      "$DEST"/scripts/make_handoff.sh

echo "==> scrubbing internal references"
# generic catalog/schema defaults in widget defaults + docs
find "$DEST" -type f \( -name "*.py" -o -name "*.md" -o -name "*.sql" -o -name "*.json" \) \
   ! -path "*/node_modules/*" ! -path "*/.venv*" -print0 \
 | xargs -0 sed -i '' \
   -e 's/lr_serverless_aws_us_catalog/main/g' \
   -e 's/lr_dev_aws_us_catalog/main/g' \
   -e 's/lr_pricing_v2_aws_us_catalog/main/g' \
   -e 's/"pricing_upt"/"pricing_workbench"/g' \
   -e 's#https://fevm-lr-serverless-aws-us.cloud.databricks.com#https://YOUR-WORKSPACE.cloud.databricks.com#g' \
   -e 's#https://fevm-lr-dev-aws-us.cloud.databricks.com#https://YOUR-WORKSPACE.cloud.databricks.com#g' \
   -e 's#https://fevm-lr-pricing-v2-aws-us.cloud.databricks.com#https://YOUR-WORKSPACE.cloud.databricks.com#g'
# README deploy commands -> sandbox target, no profile
sed -i '' \
  -e '/docs.google.com\/document\/d\/1VHV/d' \
  -e 's/--target v2 --profile <PROFILE>/--target sandbox/g' \
  -e 's/--profile <PROFILE> --auto-approve/--auto-approve/g' \
  -e 's/databricks apps create pricing-workbench --profile <PROFILE>/databricks apps create pricing-workbench/g' \
  -e 's#`./deploy.sh v2`#`./deploy.sh`#g; s#./deploy.sh v2 #./deploy.sh #g' \
  -e 's/ --profile <PROFILE>//g' -e 's/targets.v2/targets.sandbox/g' \
  "$DEST"/README.md
# app DemoDocCard link
sed -i '' 's#https://docs.google.com/document/d/1VHVMrbwo1D2Gfl2NKnKJzosBlS-hltcFZ9guvBejUkM/edit#https://example.com/your-demo-runbook#g' \
  "$DEST"/src/app/frontend/src/App.tsx 2>/dev/null || true

echo "==> verifying no internal refs remain"
if grep -rlE "laurence|lr_pricing_v2|lr_serverless_aws|lr_dev_aws|fevm-lr-|PRICING_V2|7474655676955816|acd2d10b-1d14|docs.google.com/document/d/1VHV" "$DEST" 2>/dev/null | grep -v node_modules | grep -v "/.venv"; then
  echo "!! internal refs still present — review above"; exit 1
else
  echo "CLEAN — no internal/workspace refs"
fi
echo "==> hand-off tree ready at $DEST"
