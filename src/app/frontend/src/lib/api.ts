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

  // Dataset routes
  getDatasets: () => fetchJson<any[]>('/datasets'),
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

  // Model Factory routes
  getFactoryRuns: () => fetchJson<any[]>('/models/runs'),
  getLeaderboard: (runId: string) => fetchJson<any[]>(`/models/runs/${runId}/leaderboard`),
  getModelDetail: (runId: string, configId: string) =>
    fetchJson<any>(`/models/runs/${runId}/models/${configId}`),
  decideModel: (runId: string, configId: string, decision: string, notes: string, conditions: string) =>
    fetchJson<any>(`/models/runs/${runId}/models/${configId}/decide`, {
      method: 'POST',
      body: JSON.stringify({ decision, reviewer_notes: notes, conditions }),
    }),
  getAuditTrail: (runId: string) => fetchJson<any[]>(`/models/runs/${runId}/audit`),
  getRunLog: (runId: string) => fetchJson<any>(`/models/runs/${runId}/log`),
  downloadRunLogReport: (runId: string) =>
    `${BASE}/models/runs/${runId}/log/export`,
  downloadModelReport: (runId: string, configId: string) =>
    `${BASE}/models/runs/${runId}/models/${configId}/report`,
  getFeatureProfile: (runId: string) => fetchJson<any[]>(`/models/runs/${runId}/features`),
  getChallengerComparison: () => fetchJson<any>('/models/challenger'),

  // Agents (optional AI assistants)
  getAgentStatus: () => fetchJson<any>('/agent/status'),
  runAgentAnalysis: () => fetchJson<any>('/agent/analyze', { method: 'POST' }),
  runDqMonitor: () => fetchJson<any>('/agent/dq-monitor', { method: 'POST' }),
  runExplainability: (question: string) =>
    fetchJson<any>('/agent/explain', {
      method: 'POST',
      body: JSON.stringify({ question }),
    }),

  // Agentic Model Factory planner
  analyseFeatures: () => fetchJson<any>('/agent/analyse-features'),
  proposePlan: (intent: {
    target: string; model_family: string; feature_scope: string;
    sweep_size: number; focus: string; note?: string;
  }) => fetchJson<any>('/agent/propose-plan', {
    method: 'POST', body: JSON.stringify(intent),
  }),
  submitPlan: (payload: {
    intent: any; plan_summary?: string; configs: any[]; feature_analysis_text?: string;
  }) => fetchJson<any>('/agent/submit-plan', {
    method: 'POST', body: JSON.stringify(payload),
  }),

  // Model Development
  getDevelopmentNotebooks: () => fetchJson<any>('/development/notebooks'),
  getRecentMlflowRuns:     (limit = 10) => fetchJson<any>(`/development/recent-runs?limit=${limit}`),
  openNotebook:            (notebookId: string) => fetchJson<any>('/development/open-notebook', {
    method: 'POST', body: JSON.stringify({ notebook_id: notebookId }),
  }),

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
  getServingEndpoints: () => fetchJson<any[]>('/deployment/endpoints'),
  getEndpointLatency: () => fetchJson<any>('/deployment/latency'),
  scoreModel: (endpointName: string, features: Record<string, number>) =>
    fetchJson<any>('/deployment/score', {
      method: 'POST',
      body: JSON.stringify({ endpoint_name: endpointName, features }),
    }),

  // Governance
  getGovernanceSummary: () => fetchJson<any>('/governance/summary'),

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
};
