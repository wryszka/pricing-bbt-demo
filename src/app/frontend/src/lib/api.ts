const BASE = '/api';

async function fetchJson<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });
  if (!res.ok) throw new Error(`API error: ${res.status} ${res.statusText}`);
  return res.json();
}

export const api = {
  // App config
  getConfig: () => fetchJson<any>('/config'),

  // Admin
  resetDemo: () => fetchJson<any>('/admin/reset-demo', { method: 'POST', body: JSON.stringify({}) }),

  // Model Supervisor — single chat surface fronting all sub-agents
  getSupervisorAgents: () => fetchJson<any>('/supervisor/agents'),
  askSupervisor: (body: { question: string; sub_agent?: string; pack_id?: string; run_id?: string; family?: string }) =>
    fetchJson<any>('/supervisor/ask', {
      method: 'POST',
      body: JSON.stringify(body),
    }),

  // Dataset routes
  getDatasets:       () => fetchJson<any[]>('/datasets'),
  getDatasetsMeta:   () => fetchJson<any[]>('/datasets/meta'),
  getDatasetDiff: (id: string) => fetchJson<any>(`/datasets/${id}/diff`),
  getDatasetImpact: (id: string) => fetchJson<any>(`/datasets/${id}/impact`),
  getDatasetQuality: (id: string) => fetchJson<any>(`/datasets/${id}/quality`),
  approveDataset: (id: string, decision: string, notes: string) =>
    fetchJson<any>(`/datasets/${id}/approve`, {
      method: 'POST',
      body: JSON.stringify({ decision, reviewer_notes: notes }),
    }),
  getApprovalHistory: (id: string) => fetchJson<any[]>(`/datasets/${id}/approvals`),

  // Download
  downloadDataset: (id: string, layer: string = 'silver') =>
    `${BASE}/datasets/${id}/download?layer=${layer}`,
  downloadImpactReport: (id: string) =>
    `${BASE}/datasets/${id}/impact/download`,

  // Upload
  validateUpload: async (id: string, file: File) => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/datasets/${id}/upload/validate`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Validation error: ${res.status}`);
    return res.json();
  },
  confirmUpload: async (id: string, file: File, mode: string = 'replace') => {
    const form = new FormData();
    form.append('file', file);
    const res = await fetch(`${BASE}/datasets/${id}/upload/confirm?mode=${mode}`, { method: 'POST', body: form });
    if (!res.ok) throw new Error(`Upload error: ${res.status}`);
    return res.json();
  },
  getUploadHistory: (id: string) => fetchJson<any[]>(`/datasets/${id}/uploads`),

  // Agent (plain-English explainability for dataset diffs)
  runExplainability: (question: string) =>
    fetchJson<any>('/agent/explain', {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),

  // Model Factory (new)
  factoryPropose: (family: string) =>
    fetchJson<any>('/factory/plan', {
      method: 'POST', body: JSON.stringify({ family }),
    }),
  factoryApprove: (family: string, plan: any[], narrative: string) =>
    fetchJson<any>('/factory/approve', {
      method: 'POST', body: JSON.stringify({ family, plan, narrative }),
    }),
  factoryGetRun: (runId: string) =>
    fetchJson<any>(`/factory/runs/${encodeURIComponent(runId)}`),
  factoryLeaderboard: (runId: string) =>
    fetchJson<any>(`/factory/runs/${encodeURIComponent(runId)}/leaderboard`),
  factoryShortlist: (runId: string) =>
    fetchJson<any>(`/factory/runs/${encodeURIComponent(runId)}/shortlist`),
  factoryPortfolio: (runId: string) =>
    fetchJson<any>(`/factory/runs/${encodeURIComponent(runId)}/portfolio`),
  factoryChat: (runId: string, question: string) =>
    fetchJson<any>('/factory/chat', {
      method: 'POST', body: JSON.stringify({ run_id: runId, question }),
    }),
  factoryPromoteVariant: (runId: string, variantId: string) =>
    fetchJson<any>(`/factory/runs/${encodeURIComponent(runId)}/variants/${encodeURIComponent(variantId)}/pack`, {
      method: 'POST', body: JSON.stringify({}),
    }),
  factoryRecentRuns: (limit = 5) =>
    fetchJson<any>(`/factory/runs?limit=${limit}`),

  // Model Factory — Real (second tab)
  factoryRealPropose: (family: string, maxVariants?: number) =>
    fetchJson<any>('/factory-real/plan', {
      method: 'POST', body: JSON.stringify({ family, max_variants: maxVariants }),
    }),
  factoryRealApprove: (family: string, plan: any[], narrative: string) =>
    fetchJson<any>('/factory-real/approve', {
      method: 'POST', body: JSON.stringify({ family, plan, narrative }),
    }),
  factoryRealGetRun: (runId: string) =>
    fetchJson<any>(`/factory-real/runs/${encodeURIComponent(runId)}`),
  factoryRealLeaderboard: (runId: string) =>
    fetchJson<any>(`/factory-real/runs/${encodeURIComponent(runId)}/leaderboard`),
  factoryRealShortlist: (runId: string) =>
    fetchJson<any>(`/factory-real/runs/${encodeURIComponent(runId)}/shortlist`),
  factoryRealChat: (runId: string, question: string) =>
    fetchJson<any>('/factory-real/chat', {
      method: 'POST', body: JSON.stringify({ run_id: runId, question }),
    }),
  factoryRealPromoteVariant: (runId: string, variantId: string) =>
    fetchJson<any>(`/factory-real/runs/${encodeURIComponent(runId)}/variants/${encodeURIComponent(variantId)}/pack`, {
      method: 'POST', body: JSON.stringify({}),
    }),

  // Model Development
  getDevelopmentNotebooks: () => fetchJson<any>('/development/notebooks'),
  getRecentMlflowRuns:     (limit = 10) => fetchJson<any>(`/development/recent-runs?limit=${limit}`),
  openNotebook:            (notebookId: string) => fetchJson<any>('/development/open-notebook', {
    method: 'POST', body: JSON.stringify({ notebook_id: notebookId }),
  }),

  // Review & Promote
  getReviewFamilies:       () => fetchJson<any>('/review/families'),
  getReviewVersions:       (family: string) =>
    fetchJson<any>(`/review/families/${family}/versions`),
  getReviewVersionDetail:  (family: string, version: number | string) =>
    fetchJson<any>(`/review/families/${family}/versions/${version}`),
  getReviewExplainability: (family: string, version: number | string) =>
    fetchJson<any>(`/review/families/${family}/versions/${version}/explainability`),
  getReviewArtifactUrl:    (family: string, version: number | string, path: string) =>
    `${BASE}/review/families/${family}/versions/${version}/artifact?path=${encodeURIComponent(path)}`,
  generateGovernancePack:  (family: string, version: number | string) =>
    fetchJson<any>('/review/packs/generate', {
      method: 'POST',
      body: JSON.stringify({ family, version: String(version) }),
    }),
  getPackRunStatus:        (runId: number | string) =>
    fetchJson<any>(`/review/packs/runs/${runId}`),
  listGovernancePacks:     (family?: string, limit = 25) =>
    fetchJson<any>(`/review/packs?limit=${limit}${family ? `&family=${family}` : ''}`),
  downloadPackUrl:         (packId: string) =>
    `${BASE}/review/packs/${encodeURIComponent(packId)}/download`,

  // Compare & Test
  listCompareScenarios:    (family?: string) =>
    fetchJson<any>(`/compare/scenarios${family ? `?family=${family}` : ''}`),
  triggerCompareRun:       (body: { family: string; versions: (string | number)[]; portfolio_size: number; scenario_id: string }) =>
    fetchJson<any>('/compare/run', { method: 'POST', body: JSON.stringify({
      ...body, versions: body.versions.map(String),
    }) }),
  getCompareRunStatus:     (runId: number | string) =>
    fetchJson<any>(`/compare/runs/${runId}`),
  getCompareCache:         (cacheKey: string) =>
    fetchJson<any>(`/compare/cache/${encodeURIComponent(cacheKey)}`),
  getCompareHistory:       (limit = 10) =>
    fetchJson<any>(`/compare/history?limit=${limit}`),

  // Modelling Mart
  getFeatureStoreStatus: () => fetchJson<any>('/features/status'),
  getFeatureSources:     () => fetchJson<any>('/features/sources'),
  getFeatureCatalog:     () => fetchJson<any>('/features/catalog'),
  getMartProfile:        () => fetchJson<any>('/features/mart-profile'),
  rebuildFeatureTable:   () => fetchJson<any>('/features/rebuild',        { method: 'POST', body: JSON.stringify({}) }),
  promoteOnline:         () => fetchJson<any>('/features/online/promote', { method: 'POST', body: JSON.stringify({}) }),
  pauseOnline:           () => fetchJson<any>('/features/online/pause',   { method: 'POST', body: JSON.stringify({}) }),

  // Deployment
  getRegisteredModels: () => fetchJson<any[]>('/deployment/models'),
  getChampions: () => fetchJson<any>('/deployment/champions?require_pack=false'),
  getChampionHistory: (family: string, limit = 10) =>
    fetchJson<any>(`/deployment/champions/${family}/history?limit=${limit}`),
  rollbackChampion: (family: string, note: string) =>
    fetchJson<any>('/deployment/rollback', {
      method: 'POST', body: JSON.stringify({ family, note }),
    }),

  // Governance
  getGovernanceSummary: () => fetchJson<any>('/governance/summary'),
  listAllPacks:         () => fetchJson<any>('/governance/packs'),
  getPacksOnDate:       (date: string) => fetchJson<any>(`/governance/packs/by-date?date=${date}`),
  getPackDetail:        (packId: string) => fetchJson<any>(`/governance/packs/${encodeURIComponent(packId)}`),
  packPdfUrl:           (packId: string) => `${BASE}/governance/packs/${encodeURIComponent(packId)}/pdf`,
  getPackText:          (packId: string) => fetchJson<any>(`/governance/packs/${encodeURIComponent(packId)}/text`),
  getPolicyScoring:     (policyId: string) => fetchJson<any>(`/governance/policy/${encodeURIComponent(policyId)}/scoring`),
  chatWithPack:         (packId: string, question: string, policyId?: string) =>
    fetchJson<any>('/governance/chat', {
      method: 'POST',
      body: JSON.stringify({ pack_id: packId, question, policy_id: policyId }),
    }),
  askGovernanceAgent: (question: string) =>
    fetchJson<any>('/governance/ask', {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),
  getBiasMonitor: (protectedAttribute: string = 'director_gender', family?: string) =>
    fetchJson<any>(`/governance/bias-monitor?protected_attribute=${encodeURIComponent(protectedAttribute)}${family ? `&family=${family}` : ''}`),
  biasInvestigate: (question: string, protectedAttribute: string = 'director_gender', family?: string) =>
    fetchJson<any>('/governance/bias-investigate', {
      method: 'POST',
      body: JSON.stringify({ question, protected_attribute: protectedAttribute, family }),
    }),
  getPremiumAdequacy: (cohortDimension: string = 'industry_risk_tier') =>
    fetchJson<any>(`/governance/premium-adequacy?cohort_dimension=${encodeURIComponent(cohortDimension)}`),
  getGovernanceDataSummary: () => fetchJson<any>('/governance/data-summary'),
  adequacyInvestigate: (question: string, cohortDimension: string = 'industry_risk_tier') =>
    fetchJson<any>('/governance/adequacy-investigate', {
      method: 'POST',
      body: JSON.stringify({ question, cohort_dimension: cohortDimension }),
    }),
  biasReviewCandidate: (family: string, version: string, protectedAttribute: string = 'director_gender', question?: string) =>
    fetchJson<any>('/governance/bias-review-candidate', {
      method: 'POST',
      body: JSON.stringify({ family, version, protected_attribute: protectedAttribute, question }),
    }),

  // Pricing Engine
  getPricingStatus:        () => fetchJson<any>('/pricing/status'),
  getRatingConfig:         () => fetchJson<any>('/pricing/rating-config/current'),
  getRatingConfigHistory:  () => fetchJson<any>('/pricing/rating-config/history'),
  getPricingModelVersions: () => fetchJson<any>('/pricing/model-versions'),
  listReleases:            () => fetchJson<any>('/pricing/releases'),
  getCurrentRelease:       () => fetchJson<any>('/pricing/releases/current'),
  getRelease:              (id: string) => fetchJson<any>(`/pricing/releases/${id}`),
  compareReleases:         (releaseId: string, portfolioSize = 2000) =>
    fetchJson<any>('/pricing/compare-release', {
      method: 'POST',
      body: JSON.stringify({ release_id: releaseId, portfolio_size: portfolioSize }),
    }),
  scoreOnRelease:          (releaseId: string, features: any, label?: string) =>
    fetchJson<any>(`/pricing/releases/${releaseId}/score-quote`, {
      method: 'POST',
      body: JSON.stringify({ features, label }),
    }),
  getHistoricalScoreStatus: (runId: number | string) =>
    fetchJson<any>(`/pricing/historical-score/${runId}`),
  runQuote:                (body: any) =>
    fetchJson<any>('/pricing/quote/run', { method: 'POST', body: JSON.stringify(body) }),
  simulateMta:             (body: any) =>
    fetchJson<any>('/pricing/mta/simulate', { method: 'POST', body: JSON.stringify(body) }),
  getPolicyContext:        (policyId: string) =>
    fetchJson<any>(`/pricing/policy-context/${encodeURIComponent(policyId)}`),

  // Quote Stream
  getQuoteStreamRecent: (limit: number = 50) =>
    fetchJson<any[]>(`/quote-stream/recent?limit=${limit}`),
  getQuoteStreamTransaction: (txId: string) =>
    fetchJson<any>(`/quote-stream/${encodeURIComponent(txId)}`),
  replayQuote: (txId: string) =>
    fetchJson<any>(`/quote-stream/${encodeURIComponent(txId)}/replay`, {
      method: 'POST',
      body: JSON.stringify({}),
    }),
  saveQuotePayload: (txId: string, kind: string, payload: any) =>
    fetchJson<any>(`/quote-stream/${encodeURIComponent(txId)}/save`, {
      method: 'POST',
      body: JSON.stringify({ payload, kind }),
    }),
  getQuoteStreamSummary: () => fetchJson<any>('/quote-stream/analytics/summary'),
  getQuoteStreamOutliers: () => fetchJson<any[]>('/quote-stream/analytics/outliers'),
  getQuoteStreamFunnel: () => fetchJson<any[]>('/quote-stream/analytics/funnel'),
  getQuoteStreamDistribution: () => fetchJson<any[]>('/quote-stream/analytics/distribution'),

  // Live Pricing System
  livePricingStatus: () => fetchJson<any>('/live-pricing/status'),
  livePricingStart:  () => fetchJson<any>('/live-pricing/start',  { method: 'POST', body: JSON.stringify({}) }),
  livePricingStop:   () => fetchJson<any>('/live-pricing/stop',   { method: 'POST', body: JSON.stringify({}) }),
  livePricingQuote: (policyId: string) =>
    fetchJson<any>('/live-pricing/quote', { method: 'POST', body: JSON.stringify({ policy_id: policyId }) }),
  livePricingPolicy: (policyId: string) =>
    fetchJson<any>(`/live-pricing/policy/${encodeURIComponent(policyId)}`),
  livePricingQuoteWhatIf: (policyId: string, overrides: Record<string, any>) =>
    fetchJson<any>('/live-pricing/quote-whatif', { method: 'POST', body: JSON.stringify({ policy_id: policyId, overrides }) }),
  livePricingClaim: (body: { policy_id: string; claim_amount: number; claim_type?: string }) =>
    fetchJson<any>('/live-pricing/claim', { method: 'POST', body: JSON.stringify(body) }),
  livePricingTelematicsEvent: (body: { policy_id: string; speeding_event?: boolean; curfew_breach?: boolean; behaviour_score_delta?: number; harsh_braking_delta?: number }) =>
    fetchJson<any>('/live-pricing/telematics-event', { method: 'POST', body: JSON.stringify(body) }),
  livePricingClaimStatus: (runId: number | string) =>
    fetchJson<any>(`/live-pricing/claim/${runId}`),
  livePricingStreamStart: (targetQps: number) =>
    fetchJson<any>('/live-pricing/stream/start', { method: 'POST', body: JSON.stringify({ target_qps: targetQps }) }),
  livePricingStreamStop: () =>
    fetchJson<any>('/live-pricing/stream/stop', { method: 'POST', body: JSON.stringify({}) }),
  livePricingStreamMetrics: () => fetchJson<any>('/live-pricing/stream/metrics'),
  livePricingEndpointScale: () => fetchJson<any>('/live-pricing/endpoint-scale'),
  livePricingLoadTestStart: (body: { target_qps?: number; duration_seconds?: number; concurrency?: number }) =>
    fetchJson<any>('/live-pricing/load-test/start', { method: 'POST', body: JSON.stringify(body) }),
  livePricingLoadTestStop: (runId: number | string) =>
    fetchJson<any>(`/live-pricing/load-test/stop?run_id=${encodeURIComponent(String(runId))}`,
      { method: 'POST', body: JSON.stringify({}) }),
  livePricingLoadTestMetrics: (params: { since?: string; run_id?: string } = {}) => {
    const q = new URLSearchParams();
    if (params.since)  q.set('since',  params.since);
    if (params.run_id) q.set('run_id', params.run_id);
    const qs = q.toString();
    return fetchJson<any>(`/live-pricing/load-test/metrics${qs ? `?${qs}` : ''}`);
  },

  // Agentic distribution — broker/direct chat, MCP surface, channel telemetry
  brokerChat: (body: { message: string; history?: any[]; answers?: Record<string, any>; session_id?: string | null; breakdown?: any }) =>
    fetchJson<any>('/broker/chat', { method: 'POST', body: JSON.stringify(body) }),
  brokerTools: () => fetchJson<any>('/broker/tools'),
  mcpManifest: () => fetchJson<any>('/mcp/manifest'),
  distributionTelemetry: (hours = 24) =>
    fetchJson<any>(`/distribution/telemetry?hours=${encodeURIComponent(hours)}`),

  // Price optimisation (worked example)
  optimisationSummary: () => fetchJson<any>('/optimisation/summary'),
};
