import { useEffect, useMemo, useState } from 'react';
import {
  ExternalLink, FileCheck2, Loader2, ShieldCheck,
  ChevronDown, ChevronRight, Check, X, GitCompare, Bot,
} from 'lucide-react';
import { api } from '../lib/api';

type Family = {
  key: string; label: string; type: string; primary_metric: string;
  uc_name?: string; version_count?: number; latest_version?: number | null;
};

type Version = {
  version: number;
  run_id: string;
  uc_name: string;
  story?: string;
  story_text?: string;
  simulated: boolean;
  simulation_date?: string;
  trained_by?: string;
  trained_at?: string;
  status?: string;
  primary_metric: string;
  primary_value: number | null;
  metrics: Record<string, number>;
  mlflow_url?: string;
};

const STATUS_COLOR: Record<string, string> = {
  READY:    'bg-emerald-100 text-emerald-700',
  PENDING_REGISTRATION: 'bg-amber-100 text-amber-700',
  FAILED_REGISTRATION:  'bg-red-100 text-red-700',
};

export default function ReviewPromote() {
  const [families, setFamilies]       = useState<Family[]>([]);
  const [activeFamily, setActiveFamily] = useState<string>('freq_glm');
  const [versions, setVersions]       = useState<Version[]>([]);
  const [loadingVersions, setLoadingVersions] = useState(false);
  const [selectedVersion, setSelectedVersion] = useState<number | null>(null);
  const [filter, setFilter]           = useState<string>('all');  // all | champion | simulated
  const [compare, setCompare]         = useState<Set<number>>(new Set());
  const [genRuns, setGenRuns]         = useState<Record<string, { runId: number; phase: string; packId?: string; packPath?: string; error?: string }>>({});
  const [toast, setToast]             = useState<string | null>(null);

  // Load family list on mount
  useEffect(() => {
    api.getReviewFamilies().then(d => setFamilies(d.families || [])).catch(() => setFamilies([]));
  }, []);

  // Load versions when family changes
  useEffect(() => {
    if (!activeFamily) return;
    setLoadingVersions(true);
    setSelectedVersion(null);
    setCompare(new Set());
    api.getReviewVersions(activeFamily)
      .then(d => setVersions(d.versions || []))
      .catch(() => setVersions([]))
      .finally(() => setLoadingVersions(false));
  }, [activeFamily]);

  // Poll any in-flight generation runs
  useEffect(() => {
    const inFlight = Object.entries(genRuns).filter(([, r]) => r.phase !== 'SUCCESS' && r.phase !== 'FAILED');
    if (inFlight.length === 0) return;
    const t = setInterval(() => {
      inFlight.forEach(([key, r]) => {
        api.getPackRunStatus(r.runId).then((s: any) => {
          const phase =
            s.result === 'SUCCESS' ? 'SUCCESS' :
            s.result === 'FAILED' ? 'FAILED' :
            s.life_cycle || 'RUNNING';
          setGenRuns(cur => ({
            ...cur,
            [key]: {
              ...cur[key],
              phase,
              packId:   s.pack?.pack_id || cur[key]?.packId,
              packPath: s.pack?.pdf_path || cur[key]?.packPath,
              error:    s.result === 'FAILED' ? (s.state_message || 'job failed') : undefined,
            },
          }));
          if (phase === 'SUCCESS') {
            setToast(`${key} promoted`);
          }
        }).catch(() => {});
      });
    }, 5000);
    return () => clearInterval(t);
  }, [genRuns]);

  const filtered = useMemo(() => {
    if (filter === 'champion')  return versions.filter(v => !v.simulated);
    if (filter === 'simulated') return versions.filter(v => v.simulated);
    return versions;
  }, [versions, filter]);

  const toggleCompare = (v: number) => {
    setCompare(cur => {
      const next = new Set(cur);
      if (next.has(v)) next.delete(v);
      else if (next.size < 5) next.add(v);
      return next;
    });
  };

  const generatePack = async (family: string, version: number) => {
    const key = `${family}_v${version}`;
    setGenRuns(cur => ({ ...cur, [key]: { runId: 0, phase: 'STARTING' } }));
    try {
      const r = await api.generateGovernancePack(family, version);
      setGenRuns(cur => ({ ...cur, [key]: { runId: r.job_run_id, phase: 'RUNNING' } }));
      setToast(`Promotion started — run ${r.job_run_id}`);
    } catch (e: any) {
      setGenRuns(cur => ({ ...cur, [key]: { runId: 0, phase: 'FAILED', error: e.message } }));
      setToast(`Failed to start promotion: ${e.message}`);
    }
  };

  const activeMeta = families.find(f => f.key === activeFamily);

  return (
    <div>
      {/* Header */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <h3 className="font-semibold text-blue-800 mb-1 text-sm">Promote</h3>
        <p className="text-sm text-blue-700">
          Review trained models and promote them by generating the governance pack. Click a model version to see metrics,
          explanations, and lineage. Clicking <em>Promote</em> produces a comprehensive PDF for CDO / CRO / CFO / Chief Actuary review and
          makes this version available in the Deployment and Governance tabs.
        </p>
        <div className="flex flex-wrap gap-1.5 mt-2.5">
          {['MLflow model registry', 'UC artifact lineage', 'Governance pack pipeline (Databricks Jobs)',
            'SHAP explainability', 'Unified audit trail'].map(f => (
            <span key={f} className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-100 text-blue-700">{f}</span>
          ))}
        </div>
      </div>

      {/* Family picker */}
      <div className="bg-white rounded-lg border border-gray-200 p-3 mb-4 flex flex-wrap gap-2">
        {families.map(f => (
          <button key={f.key} onClick={() => setActiveFamily(f.key)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition ${
                    activeFamily === f.key
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }`}>
            {f.label}
            <span className={`ml-2 text-xs ${activeFamily === f.key ? 'text-blue-100' : 'text-gray-500'}`}>
              {f.version_count ?? '—'} versions
            </span>
          </button>
        ))}
      </div>

      {/* Filter + compare bar */}
      <div className="flex items-center justify-between mb-2">
        <div className="flex items-center gap-2 text-xs">
          <span className="text-gray-500">Filter:</span>
          {[
            ['all', 'All versions'],
            ['champion', 'Champion only'],
            ['simulated', 'Replays only'],
          ].map(([v, label]) => (
            <button key={v} onClick={() => setFilter(v)}
                    className={`px-2 py-1 rounded ${
                      filter === v ? 'bg-gray-800 text-white' : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}>
              {label}
            </button>
          ))}
        </div>
        {compare.size >= 2 && (
          <div className="text-xs text-gray-700 flex items-center gap-2">
            <GitCompare className="w-3 h-3" />
            Comparing {compare.size} versions
            <button onClick={() => setCompare(new Set())}
                    className="underline text-gray-500 hover:text-gray-700">clear</button>
          </div>
        )}
      </div>

      {/* Versions table */}
      <div className="bg-white rounded-lg border border-gray-200 mb-6 overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">
            {activeMeta?.label || activeFamily} · versions
          </h3>
          <span className="text-xs text-gray-500 font-mono">
            {activeMeta?.uc_name}
          </span>
        </div>
        {loadingVersions ? (
          <div className="py-8 text-center text-sm text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin inline mr-1" /> Loading versions…
          </div>
        ) : filtered.length === 0 ? (
          <div className="py-8 text-center text-sm text-gray-500 italic">No versions match this filter.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b bg-gray-50">
                <th className="text-left px-3 py-2 font-medium w-8"><GitCompare className="w-3 h-3 inline" /></th>
                <th className="text-left px-3 py-2 font-medium">Version</th>
                <th className="text-left px-3 py-2 font-medium">Story</th>
                <th className="text-left px-3 py-2 font-medium">Trained</th>
                <th className="text-left px-3 py-2 font-medium">By</th>
                <th className="text-right px-3 py-2 font-medium">{filtered[0]?.primary_metric}</th>
                <th className="text-left px-3 py-2 font-medium">Status</th>
                <th className="text-right px-3 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map(v => {
                const key = `${activeFamily}_v${v.version}`;
                const gen = genRuns[key];
                const isOpen = selectedVersion === v.version;
                const isChampion = !v.simulated;
                const metric = v.primary_value;
                return (
                  <>
                    <tr key={`${v.version}-row`}
                        className={`border-b last:border-0 hover:bg-gray-50 ${isOpen ? 'bg-blue-50' : ''}`}>
                      <td className="px-3 py-2">
                        <input type="checkbox" checked={compare.has(v.version)}
                               onChange={(e) => { e.stopPropagation(); toggleCompare(v.version); }} />
                      </td>
                      <td className="px-3 py-2">
                        <button onClick={() => setSelectedVersion(isOpen ? null : v.version)}
                                className="inline-flex items-center gap-1 font-medium text-gray-900 hover:text-blue-600">
                          {isOpen ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
                          v{v.version}
                          {isChampion && (
                            <span className="ml-2 px-1.5 py-0.5 text-[10px] rounded bg-emerald-100 text-emerald-700 font-medium">
                              champion
                            </span>
                          )}
                        </button>
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-700 max-w-xs truncate" title={v.story_text}>
                        {v.story || '—'}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-600">
                        {v.simulation_date || formatDateShort(v.trained_at)}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-600">
                        {(v.trained_by || '').split('@')[0] || '—'}
                      </td>
                      <td className="px-3 py-2 text-right text-xs font-mono text-gray-900">
                        {metric !== null ? metric.toFixed(4) : '—'}
                      </td>
                      <td className="px-3 py-2">
                        <span className={`text-[10px] px-2 py-0.5 rounded font-medium ${STATUS_COLOR[v.status || ''] || 'bg-gray-100 text-gray-700'}`}>
                          {v.status || '—'}
                        </span>
                      </td>
                      <td className="px-3 py-2 text-right text-xs">
                        <div className="inline-flex items-center gap-2">
                          {v.mlflow_url && (
                            <a href={v.mlflow_url} target="_blank" rel="noopener noreferrer"
                               className="text-gray-500 hover:text-blue-600 inline-flex items-center gap-0.5"
                               title="Open MLflow run">
                              MLflow <ExternalLink className="w-3 h-3" />
                            </a>
                          )}
                          {gen ? (
                            <GenStatus gen={gen} />
                          ) : (
                            <button onClick={() => generatePack(activeFamily, v.version)}
                                    className="inline-flex items-center gap-1 px-2.5 py-1 rounded bg-blue-600 text-white hover:bg-blue-700 text-[11px] font-medium">
                              <FileCheck2 className="w-3 h-3" /> Promote
                            </button>
                          )}
                        </div>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr key={`${v.version}-detail`} className="bg-blue-50/30 border-b">
                        <td colSpan={8} className="px-4 py-4">
                          <VersionDetail family={activeFamily} version={v.version} />
                        </td>
                      </tr>
                    )}
                  </>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {/* Compare panel */}
      {compare.size >= 2 && (
        <ComparePanel family={activeFamily}
                      versionNumbers={Array.from(compare).sort((a, b) => b - a)}
                      versions={versions} />
      )}

      {toast && (
        <div className="fixed bottom-4 right-4 bg-gray-900 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50"
             onClick={() => setToast(null)}>
          {toast}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Version detail panel — inlined under the selected row
// ---------------------------------------------------------------------------

function VersionDetail({ family, version }: { family: string; version: number }) {
  const [detail, setDetail]   = useState<any>(null);
  const [explain, setExplain] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  // Bias pre-promotion review (agent-driven)
  const [biasAttr, setBiasAttr]         = useState<'director_gender' | 'postcode_demographic'>('director_gender');
  const [biasRunning, setBiasRunning]   = useState(false);
  const [biasReview, setBiasReview]     = useState<any>(null);

  const runBiasReview = async () => {
    setBiasRunning(true); setBiasReview(null);
    try {
      const r = await api.biasReviewCandidate(family, String(version), biasAttr);
      setBiasReview(r);
    } catch (e: any) {
      setBiasReview({ ok: false, error: e.message || String(e) });
    } finally {
      setBiasRunning(false);
    }
  };

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getReviewVersionDetail(family, version),
      api.getReviewExplainability(family, version).catch(() => null),
    ]).then(([d, e]) => { setDetail(d); setExplain(e); })
      .finally(() => setLoading(false));
  }, [family, version]);

  if (loading) {
    return (
      <div className="text-sm text-gray-500 italic">
        <Loader2 className="w-4 h-4 inline animate-spin mr-1" /> Loading detail…
      </div>
    );
  }
  if (!detail) return <div className="text-sm text-red-600">Could not load detail.</div>;

  const isGLM = family.endsWith('_glm');
  const isGBM = family.endsWith('_gbm');
  const shapUrl = detail.notable?.shap_summary_png
    ? api.getReviewArtifactUrl(family, version, detail.notable.shap_summary_png)
    : null;

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
      {/* Overview + lineage + config */}
      <div className="space-y-3">
        <DetailCard title="Overview">
          <KVRow label="MLflow run" value={<code className="text-xs">{detail.run_id}</code>} />
          <KVRow label="Trained by" value={(detail.trained_by || '').split('@')[0] || '—'} />
          <KVRow label="Trained at" value={formatDateShort(detail.trained_at)} />
          <KVRow label="Status"     value={detail.status} />
          <KVRow label="Simulation" value={detail.simulated ? `yes (${detail.simulation_date})` : 'no — current champion'} />
          {detail.story_text && (
            <div className="mt-2 text-xs text-gray-600 italic border-l-2 border-blue-200 pl-2">
              {detail.story_text}
            </div>
          )}
          {detail.mlflow_url && (
            <a href={detail.mlflow_url} target="_blank" rel="noopener noreferrer"
               className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 mt-2">
              Open in MLflow <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </DetailCard>
        <DetailCard title="Lineage">
          <KVRow label="Feature table" value={<code className="text-[11px]">{detail.feature_table}</code>} />
          <KVRow label="Feature lookup" value="Offline FE (FeatureLookup)" />
          <KVRow label="Training split" value="Deterministic 80/20" />
        </DetailCard>
        <DetailCard title="Training config">
          <div className="text-xs">
            {Object.entries(detail.params || {}).slice(0, 10).map(([k, v]) => (
              <KVRow key={k} label={k} value={String(v)} />
            ))}
            {Object.keys(detail.params || {}).length === 0 && (
              <div className="text-gray-500 italic">(no params logged)</div>
            )}
          </div>
        </DetailCard>
      </div>

      {/* Metrics */}
      <div className="space-y-3">
        <DetailCard title="Performance">
          <table className="w-full text-xs">
            <tbody>
              {Object.entries(detail.metrics || {}).sort().map(([k, v]) => (
                <tr key={k} className="border-b last:border-0">
                  <td className="py-1 text-gray-600">{k}</td>
                  <td className="py-1 text-right font-mono text-gray-900">
                    {typeof v === 'number' ? (Math.abs(v) < 1000 ? v.toFixed(4) : v.toLocaleString()) : String(v)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </DetailCard>
        {isGBM && explain?.shap_importance?.length > 0 && (
          <DetailCard title="SHAP importance (top 10)">
            <ShapBars rows={explain.shap_importance} />
          </DetailCard>
        )}
      </div>

      {/* Relativities / importance / SHAP plot */}
      <div className="space-y-3">
        {isGLM && explain?.relativities?.length > 0 && (
          <DetailCard title="Coefficient relativities (top 12)">
            <RelativityTable rows={explain.relativities} />
          </DetailCard>
        )}
        {isGBM && explain?.importance?.length > 0 && (
          <DetailCard title="Gain importance (top 12)">
            <ImportanceBars rows={explain.importance} />
          </DetailCard>
        )}
        {isGBM && shapUrl && (
          <DetailCard title="SHAP summary plot">
            <img src={shapUrl} alt="SHAP summary" className="w-full rounded border border-gray-200" />
          </DetailCard>
        )}
      </div>

      {/* Bias pre-promotion review — spans full width */}
      <div className="lg:col-span-3 mt-2 rounded-lg border border-indigo-200 bg-indigo-50/40">
        <div className="px-4 py-3 border-b border-indigo-200 flex items-center justify-between bg-indigo-50">
          <div>
            <div className="text-sm font-semibold text-indigo-900">
              Pre-promotion bias review
            </div>
            <div className="text-xs text-indigo-700 mt-0.5">
              Audit this candidate for disparate impact on a protected attribute before you promote.
              The agent returns a DETECTION / DIAGNOSIS / JUSTIFICATION / EVIDENCE / MITIGATION / CONCLUSION report.
            </div>
          </div>
          <div className="flex items-center gap-2">
            <div className="inline-flex rounded-md border border-indigo-300 bg-white overflow-hidden text-xs">
              {([
                { k: 'director_gender',      label: 'Gender' },
                { k: 'postcode_demographic', label: 'Postcode' },
              ] as const).map(opt => (
                <button key={opt.k}
                        onClick={() => setBiasAttr(opt.k)}
                        className={`px-2.5 py-1 font-medium transition ${
                          biasAttr === opt.k ? 'bg-indigo-600 text-white' : 'text-indigo-800 hover:bg-indigo-50'
                        }`}>
                  {opt.label}
                </button>
              ))}
            </div>
            <button onClick={runBiasReview}
                    disabled={biasRunning}
                    className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-indigo-600 text-white text-xs font-medium hover:bg-indigo-700 disabled:opacity-50">
              {biasRunning ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Bot className="w-3.5 h-3.5" />}
              {biasRunning ? 'Agent reviewing…' : 'Run bias review'}
            </button>
          </div>
        </div>
        <div className="px-4 py-3">
          {!biasReview && !biasRunning && (
            <div className="text-xs text-indigo-900/70 italic">
              Click <span className="font-semibold">Run bias review</span> to see whether promoting v{version} would introduce (or inherit) a defensible gap.
              Also auto-runs inside every generated governance pack.
            </div>
          )}
          {biasReview?.ok === false && biasReview.error && (
            <div className="text-sm text-red-700">Agent failed: {biasReview.error}</div>
          )}
          {biasReview?.ok && (
            <>
              <div className="flex flex-wrap gap-1 mb-2.5">
                {(biasReview.trace || []).map((t: any, i: number) => (
                  <span key={i}
                        className="text-[10px] px-1.5 py-0.5 rounded border border-indigo-300 bg-white text-indigo-800">
                    {t.tool}{String(t.result_summary || '').startsWith('error') ? ' ⚠' : ''}
                  </span>
                ))}
              </div>
              <div className="rounded bg-white border border-indigo-200 p-3 text-xs whitespace-pre-wrap leading-relaxed">
                {biasReview.answer}
              </div>
              <div className="mt-1.5 text-[11px] text-indigo-800/70">
                model: {biasReview.model} · {(biasReview.trace || []).length} tool calls ·
                {biasReview.usage?.total_tokens?.toLocaleString() || '?'} tokens ·
                logged as a <code>bias_pre_promotion_review</code> event.
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Compare panel
// ---------------------------------------------------------------------------

function ComparePanel({ family, versionNumbers, versions }:
  { family: string; versionNumbers: number[]; versions: Version[] }) {

  const selected = versionNumbers
    .map(n => versions.find(v => v.version === n))
    .filter(Boolean) as Version[];

  if (selected.length < 2) return null;
  const allMetricKeys = Array.from(new Set(selected.flatMap(v => Object.keys(v.metrics || {})))).sort();

  return (
    <section className="bg-white rounded-lg border border-gray-200 mb-6 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
          <GitCompare className="w-4 h-4" /> Comparison — {selected.length} versions of {family}
        </h3>
      </div>
      <div className="p-4 overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b text-xs text-gray-500">
              <th className="text-left py-1.5 pr-3 font-medium">Metric</th>
              {selected.map(v => (
                <th key={v.version} className="text-right py-1.5 pr-3 font-medium">
                  v{v.version}
                  {!v.simulated && <span className="ml-1 text-emerald-600">★</span>}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <CompareRow label="Story" cells={selected.map(v => v.story || '—')} />
            <CompareRow label="Simulation date" cells={selected.map(v => v.simulation_date || 'champion')} />
            <CompareRow label="Trained" cells={selected.map(v => formatDateShort(v.trained_at))} />
            {allMetricKeys.map(k => {
              const vals = selected.map(v => v.metrics[k] as number | undefined);
              const nums = vals.filter((x): x is number => typeof x === 'number');
              const hi = Math.max(...nums);
              return (
                <CompareRow
                  key={k}
                  label={k}
                  cells={vals.map(v => typeof v === 'number'
                    ? <span className={v === hi ? 'font-bold text-emerald-700' : 'font-mono'}>{v.toFixed(4)}</span>
                    : '—'
                  )}
                />
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function CompareRow({ label, cells }: { label: string; cells: any[] }) {
  return (
    <tr className="border-b last:border-0">
      <td className="py-1.5 pr-3 text-xs text-gray-600">{label}</td>
      {cells.map((c, i) => (
        <td key={i} className="py-1.5 pr-3 text-xs text-right text-gray-800">{c}</td>
      ))}
    </tr>
  );
}

// ---------------------------------------------------------------------------
// Generation status chip
// ---------------------------------------------------------------------------

function GenStatus({ gen }: { gen: { runId: number; phase: string; packId?: string; packPath?: string; error?: string } }) {
  if (gen.phase === 'SUCCESS') {
    return (
      <span className="inline-flex items-center gap-1 text-emerald-700 text-[11px] font-medium">
        <Check className="w-3 h-3" /> Promoted
        {gen.packId && (
          <a href={api.downloadPackUrl(gen.packId)} target="_blank" rel="noopener noreferrer"
             className="ml-1 underline">pack</a>
        )}
      </span>
    );
  }
  if (gen.phase === 'FAILED') {
    return (
      <span className="inline-flex items-center gap-1 text-red-700 text-[11px]" title={gen.error}>
        <X className="w-3 h-3" /> Failed
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 text-blue-700 text-[11px]">
      <Loader2 className="w-3 h-3 animate-spin" /> Promoting…
    </span>
  );
}

// ---------------------------------------------------------------------------
// Small UI primitives
// ---------------------------------------------------------------------------

function DetailCard({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <h4 className="text-xs font-semibold text-gray-800 uppercase tracking-wide mb-2 flex items-center gap-1.5">
        <ShieldCheck className="w-3 h-3 text-gray-400" /> {title}
      </h4>
      {children}
    </div>
  );
}

function KVRow({ label, value }: { label: string; value: any }) {
  return (
    <div className="flex items-start justify-between gap-2 text-xs py-0.5">
      <span className="text-gray-500 shrink-0">{label}</span>
      <span className="text-gray-900 text-right break-all">{value}</span>
    </div>
  );
}

function RelativityTable({ rows }: { rows: any[] }) {
  const sorted = [...rows]
    .sort((a, b) => Math.abs(Number(b.coefficient || 0)) - Math.abs(Number(a.coefficient || 0)))
    .slice(0, 12);
  return (
    <table className="w-full text-xs">
      <thead>
        <tr className="border-b text-gray-500">
          <th className="text-left py-1 font-medium">Feature</th>
          <th className="text-right py-1 font-medium">Coef</th>
          <th className="text-right py-1 font-medium">Relativity</th>
          <th className="text-right py-1 font-medium">p</th>
        </tr>
      </thead>
      <tbody>
        {sorted.map((r, i) => (
          <tr key={i} className="border-b last:border-0">
            <td className="py-1 text-gray-800 truncate max-w-[140px]" title={r.feature}>{r.feature}</td>
            <td className="py-1 text-right font-mono">{Number(r.coefficient || 0).toFixed(3)}</td>
            <td className="py-1 text-right font-mono">{Number(r.relativity || 0).toFixed(3)}</td>
            <td className="py-1 text-right font-mono text-gray-500">{Number(r.p_value || 0).toExponential(1)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ImportanceBars({ rows }: { rows: any[] }) {
  const sorted = [...rows]
    .map(r => ({ ...r, gain: Number(r.gain || 0) }))
    .sort((a, b) => b.gain - a.gain)
    .slice(0, 12);
  const max = Math.max(1, ...sorted.map(r => r.gain));
  return (
    <div className="space-y-0.5">
      {sorted.map(r => (
        <div key={r.feature} className="flex items-center gap-2 text-xs">
          <div className="w-28 truncate text-gray-800" title={r.feature}>{r.feature}</div>
          <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
            <div className="bg-blue-500 h-full" style={{ width: `${(r.gain / max) * 100}%` }} />
          </div>
          <div className="w-16 text-right text-gray-600 font-mono">{r.gain.toFixed(0)}</div>
        </div>
      ))}
    </div>
  );
}

function ShapBars({ rows }: { rows: any[] }) {
  const sorted = [...rows]
    .map(r => ({ ...r, v: Number(r.mean_abs_shap || 0) }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 10);
  const max = Math.max(1e-9, ...sorted.map(r => r.v));
  return (
    <div className="space-y-0.5">
      {sorted.map(r => (
        <div key={r.feature} className="flex items-center gap-2 text-xs">
          <div className="w-28 truncate text-gray-800" title={r.feature}>{r.feature}</div>
          <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
            <div className="bg-emerald-500 h-full" style={{ width: `${(r.v / max) * 100}%` }} />
          </div>
          <div className="w-16 text-right text-gray-600 font-mono">{r.v.toFixed(3)}</div>
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDateShort(iso?: string): string {
  if (!iso) return '—';
  const t = new Date(iso);
  if (isNaN(t.getTime())) return iso.substring(0, 10);
  return t.toISOString().substring(0, 10);
}

