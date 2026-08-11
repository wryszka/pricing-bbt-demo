-- Seed historical governance pack rows so the Governance "by date" tab shows
-- a meaningful timeline (Feb / Mar baselines + the live April champions).
--
-- Idempotent: deletes any row with story prefix "[seed-history]" before
-- re-inserting, so the demo_reset job can call this without piling rows up.
-- The pdf_path on the synthetic rows points at the live April PDF — a March
-- "Open pack" link still resolves; the demo point is the version timeline,
-- not a unique PDF per row.
--
-- Replace {catalog} / {schema} placeholders before executing.

DELETE FROM {catalog}.{schema}.governance_packs_index
WHERE story LIKE '[seed-history]%';

INSERT INTO {catalog}.{schema}.governance_packs_index
  (pack_id, model_family, model_version, model_uc_name, mlflow_run_id,
   story, simulated, primary_metric, primary_value, pdf_path, size_bytes,
   generated_by, generated_at)
VALUES
  -- ---- February 2026: initial production baseline ----
  ('GP-20260219094501-freq_glm-v25',   'freq_glm',   '25',
   '{catalog}.{schema}.freq_glm', NULL,
   '[seed-history] Feb baseline — initial production cut',
   false, 'gini', 0.1342,
   '/Volumes/{catalog}/{schema}/governance_packs/freq_glm_v41_20260423_162042.pdf',
   180000, current_user(), TIMESTAMP'2026-02-19 09:45:01'),
  ('GP-20260219102230-sev_glm-v22',    'sev_glm',    '22',
   '{catalog}.{schema}.sev_glm', NULL,
   '[seed-history] Feb baseline — initial production cut',
   false, 'gini', 0.0188,
   '/Volumes/{catalog}/{schema}/governance_packs/sev_glm_v36_20260423_170833.pdf',
   180000, current_user(), TIMESTAMP'2026-02-19 10:22:30'),
  ('GP-20260219104812-demand_gbm-v25', 'demand_gbm', '25',
   '{catalog}.{schema}.demand_gbm', NULL,
   '[seed-history] Feb baseline — initial production cut',
   false, 'auc', 0.5102,
   '/Volumes/{catalog}/{schema}/governance_packs/demand_gbm_v39_20260423_170833.pdf',
   180000, current_user(), TIMESTAMP'2026-02-19 10:48:12'),
  ('GP-20260219111545-fraud_gbm-v25',  'fraud_gbm',  '25',
   '{catalog}.{schema}.fraud_gbm', NULL,
   '[seed-history] Feb baseline — initial production cut',
   false, 'auc', 0.6841,
   '/Volumes/{catalog}/{schema}/governance_packs/fraud_gbm_v41_20260423_171945.pdf',
   180000, current_user(), TIMESTAMP'2026-02-19 11:15:45'),

  -- ---- March 2026: mid-cycle refit (pre-vendor-refresh) ----
  ('GP-20260317143205-freq_glm-v33',   'freq_glm',   '33',
   '{catalog}.{schema}.freq_glm', NULL,
   '[seed-history] Mar refit — credit bureau feature added',
   false, 'gini', 0.1410,
   '/Volumes/{catalog}/{schema}/governance_packs/freq_glm_v41_20260423_162042.pdf',
   180000, current_user(), TIMESTAMP'2026-03-17 14:32:05'),
  ('GP-20260317150022-sev_glm-v28',    'sev_glm',    '28',
   '{catalog}.{schema}.sev_glm', NULL,
   '[seed-history] Mar refit — IBNR floor adjustment',
   false, 'gini', 0.0214,
   '/Volumes/{catalog}/{schema}/governance_packs/sev_glm_v36_20260423_170833.pdf',
   180000, current_user(), TIMESTAMP'2026-03-17 15:00:22'),
  ('GP-20260317152715-demand_gbm-v32', 'demand_gbm', '32',
   '{catalog}.{schema}.demand_gbm', NULL,
   '[seed-history] Mar refit — broker-channel features',
   false, 'auc', 0.5193,
   '/Volumes/{catalog}/{schema}/governance_packs/demand_gbm_v39_20260423_170833.pdf',
   180000, current_user(), TIMESTAMP'2026-03-17 15:27:15'),
  ('GP-20260317155930-fraud_gbm-v34',  'fraud_gbm',  '34',
   '{catalog}.{schema}.fraud_gbm', NULL,
   '[seed-history] Mar refit — SIU referral threshold tuning',
   false, 'auc', 0.7012,
   '/Volumes/{catalog}/{schema}/governance_packs/fraud_gbm_v41_20260423_171945.pdf',
   180000, current_user(), TIMESTAMP'2026-03-17 15:59:30');
