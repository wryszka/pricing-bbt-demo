import { useEffect, useMemo, useState } from 'react';
import {
  GitCompare, Loader2, Play, Sparkles, AlertTriangle, TrendingUp, TrendingDown,
  CheckCircle2, XCircle, HelpCircle, Beaker, ExternalLink,
} from 'lucide-react';
import { api } from '../lib/api';

type Scenario = { id: string; label: string; description: string; applies_to: string[] };
type Family = { key: string; label: string; type: string; primary_metric: string;
                uc_name?: string; version_count?: number; latest_version?: number | null };
type Version = {
  version: number; run_id: string; story?: string; story_text?: string;
  simulated: boolean; simulation_date?: string; trained_at?: string;
  primary_metric: string; primary_value: number | null;
};

const REC_STYLE: Record<string, { icon: any; cls: string; label: string }> = {
  PROMOTE:     { icon: CheckCircle2, cls: 'bg-emerald-100 text-emerald-800 border-emerald-200', label: 'Promote' },
  INVESTIGATE: { icon: HelpCircle,   cls: 'bg-amber-100 text-amber-800 border-amber-200',       label: 'Investigate' },
  REJECT:      { icon: XCircle,      cls: 'bg-red-100 text-red-800 border-red-200',             label: 'Reject' },
};

export default function CompareTest() {
  const [families, setFamilies]         = useState<Family[]>([]);
  const [family, setFamily]             = useState<string>('freq_glm');
  const [versions, setVersions]         = useState<Version[]>([]);
  const [selected, setSelected]         = useState<number[]>([]);  // ordered A, B, [C, D, E]
  const [portfolioSize, setPortfolioSize] = useState(5000);
  const [scenarios, setScenarios]       = useState<Scenario[]>([]);
  const [scenarioId, setScenarioId]     = useState('none');
  const [running, setRunning]           = useState<{ runId: number; phase: string; cacheKey?: string; error?: string } | null>(null);
  const [result, setResult]             = useState<any>(null);
  const [loadingResult, setLoadingResult] = useState(false);
  const [toast, setToast]               = useState<string | null>(null);

  useEffect(() => {
    api.getReviewFamilies().then(d => setFamilies(d.families || []));
  }, []);

  useEffect(() => {
    if (!family) return;
    setVersions([]);
    setSelected([]);
    setResult(null);
    api.getReviewVersions(family).then(d => {
      const vs: Version[] = d.versions || [];
      setVersions(vs);
      // Default selection: pick the two versions with the largest primary-metric
      // gap among those that have a value. Maximises visible difference in the
      // run-comparison pane — instead of comparing two near-identical runs.
      const scored = vs.filter(v => v.primary_value != null);
      const pick: number[] = [];
      if (scored.length >= 2) {
        const sorted = [...scored].sort((a, b) => (b.primary_value as number) - (a.primary_value as number));
        const best = sorted[0];
        const worst = sorted[sorted.length - 1];
        pick.push(best.version);
        pick.push(worst.version);
      } else {
        const champion = vs.find(v => !v.simulated);
        const nextCandidate = vs.find(v => v.simulated);
        if (champion) pick.push(champion.version);
        if (nextCandidate && nextCandidate.version !== champion?.version) pick.push(nextCandidate.version);
      }
      setSelected(pick);

      // Auto-hydrate the most recent cached run for this family so the demo
      // landing state shows a populated comparison without needing a fresh job.
      api.getCompareHistory(20).then((h: any) => {
        const matching = (h.runs || []).find((r: any) => r.family === family && r.cache_key);
        if (!matching) return;
        // Sync selected versions to whatever the cached run used (so the chips
        // light up consistent with the displayed result).
        try {
          const cachedVersions = String(matching.versions || '').split(',').map((s: string) => Number(s.trim())).filter(n => !isNaN(n));
          if (cachedVersions.length >= 2) setSelected(cachedVersions);
        } catch { /* fall back to best/worst */ }
        setLoadingResult(true);
        api.getCompareCache(matching.cache_key).then(setResult)
          .catch(() => { /* silent — page still works */ })
          .finally(() => setLoadingResult(false));
      }).catch(() => { /* history endpoint not seeded; default UX still works */ });
    });
    api.listCompareScenarios(family).then(d => {
      setScenarios(d.scenarios || []);
      setScenarioId('none');
    });
  }, [family]);

  // Poll run
  useEffect(() => {
    if (!running || ['SUCCESS', 'FAILED', 'CACHED'].includes(running.phase)) return;
    const t = setInterval(() => {
      api.getCompareRunStatus(running.runId).then((s: any) => {
        const phase =
          s.result === 'SUCCESS' ? 'SUCCESS' :
          s.result === 'FAILED' ? 'FAILED' :
          s.life_cycle || 'RUNNING';
        setRunning(cur => cur ? { ...cur, phase, cacheKey: s.summary?.cache_key } : cur);
        if (phase === 'SUCCESS' && s.summary?.cache_key) {
          hydrateCache(s.summary.cache_key);
        }
        if (phase === 'FAILED') {
          setToast(`Compare run failed: ${s.state_message || 'see Databricks run page'}`);
        }
      }).catch(() => {});
    }, 4000);
    return () => clearInterval(t);
  }, [running]);

  const hydrateCache = (cacheKey: string) => {
    setLoadingResult(true);
    api.getCompareCache(cacheKey).then(setResult)
      .catch((e) => setToast(`Could not load result: ${e.message}`))
      .finally(() => setLoadingResult(false));
  };

  const toggleVersion = (v: number) => {
    setSelected(cur => {
      if (cur.includes(v)) return cur.filter(x => x !== v);
      if (cur.length >= 5) return cur;
      return [...cur, v];
    });
  };

  const runCompare = async () => {
    if (selected.length < 2) { setToast('Select 2-5 versions to compare.'); return; }
    setResult(null);
    try {
      const r: any = await api.triggerCompareRun({
        family, versions: selected, portfolio_size: portfolioSize, scenario_id: scenarioId,
      });
      // Cached mode: server hands back the prior cache_key directly — skip
      // the job-polling phase and hydrate the result immediately.
      if (r.cached && r.cache_key) {
        setRunning({ runId: r.job_run_id ?? -1, phase: 'CACHED', cacheKey: r.cache_key });
        api.getCompareCache(r.cache_key).then(setResult).catch(() => undefined);
        setToast('Served from cache — repeat of a prior identical run.');
        return;
      }
      setRunning({ runId: r.job_run_id, phase: 'RUNNING' });
      setToast(`Run ${r.job_run_id} started — this takes ~30-60s`);
    } catch (e: any) {
      setToast(`Could not start run: ${e.message}`);
    }
  };

  const selectedVersions = useMemo(() =>
    selected.map(v => versions.find(x => x.version === v)).filter(Boolean) as Version[],
    [selected, versions]);

  const activeScenario = scenarios.find(s => s.id === scenarioId);

  return (
    <div>
      {/* Header */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <h3 className="font-semibold text-blue-800 mb-1 text-sm flex items-center gap-1.5">
          <GitCompare className="w-4 h-4" /> Compare &amp; Test
        </h3>
        <p className="text-sm text-blue-700">
          Evaluate candidate models before promoting. Pick 2-5 versions of the same family and optionally apply a what-if scenario —
          the job batch-scores them on the same stratified portfolio sample for an apples-to-apples view.
        </p>
        <div className="flex flex-wrap gap-1.5 mt-2.5">
          {['MLflow pyfunc loading', 'UC model versioning', 'Databricks Jobs batch scoring',
            'Delta cache for re-runs', 'Stratified portfolio sampling', 'Deterministic holdout'].map(f => (
            <span key={f} className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-100 text-blue-700">{f}</span>
          ))}
        </div>
      </div>

      {/* Selector */}
      <section className="bg-white border border-gray-200 rounded-lg p-4 mb-4">
        <div className="grid grid-cols-1 md:grid-cols-4 gap-3">
          <div>
            <label className="text-xs text-gray-500 font-medium block mb-1">Model family</label>
            <select value={family} onChange={e => setFamily(e.target.value)}
                    className="w-full text-sm border border-gray-300 rounded px-2 py-1.5">
              {families.map(f => (
                <option key={f.key} value={f.key}>{f.label} · {f.version_count} versions</option>
              ))}
            </select>
          </div>
          <div>
            <label className="text-xs text-gray-500 font-medium block mb-1">Portfolio sample size</label>
            <input type="number" min={500} max={50000} step={500}
                   value={portfolioSize}
                   onChange={e => setPortfolioSize(Math.max(500, Math.min(50000, Number(e.target.value) || 5000)))}
                   className="w-full text-sm border border-gray-300 rounded px-2 py-1.5" />
          </div>
          <div className="md:col-span-2">
            <label className="text-xs text-gray-500 font-medium block mb-1">What-if scenario</label>
            <select value={scenarioId} onChange={e => setScenarioId(e.target.value)}
                    className="w-full text-sm border border-gray-300 rounded px-2 py-1.5">
              {scenarios.map(s => (
                <option key={s.id} value={s.id}>{s.label}</option>
              ))}
            </select>
            {activeScenario && activeScenario.id !== 'none' && (
              <p className="text-[11px] text-gray-500 italic mt-1">{activeScenario.description}</p>
            )}
          </div>
        </div>

        {/* Version picker */}
        <div className="mt-4">
          <div className="flex items-center justify-between">
            <label className="text-xs text-gray-500 font-medium">
              Pick 2-5 versions — order matters (A = baseline, B = candidate, …)
            </label>
            <div className="text-xs text-gray-500">
              Selected: {selected.length > 0 ? selected.map((v, i) => `${String.fromCharCode(65 + i)}=v${v}`).join(' · ') : 'none'}
            </div>
          </div>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {versions.map(v => {
              const pickedAt = selected.indexOf(v.version);
              const picked = pickedAt >= 0;
              const letter = picked ? String.fromCharCode(65 + pickedAt) : null;
              const isChampion = !v.simulated;
              return (
                <button key={v.version} onClick={() => toggleVersion(v.version)}
                        className={`px-2 py-1 rounded text-[11px] font-medium border transition ${
                          picked
                            ? 'bg-blue-600 text-white border-blue-600'
                            : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
                        }`}>
                  {letter && <span className="mr-1 font-bold">{letter}:</span>}
                  v{v.version}
                  {isChampion && <span className={`ml-1 ${picked ? 'text-blue-100' : 'text-emerald-600'}`}>★</span>}
                  {v.story && <span className={`ml-1 ${picked ? 'text-blue-100' : 'text-gray-500'}`}>· {v.story}</span>}
                </button>
              );
            })}
          </div>
        </div>

        {/* Run */}
        <div className="mt-4 flex items-center justify-between gap-3">
          <div className="text-xs text-gray-500">
            Models load via <code>mlflow.pyfunc.load_model</code> and score the same stratified sample.
            Results cache by (family, versions, scenario, size).
          </div>
          <button onClick={runCompare}
                  disabled={selected.length < 2 || Boolean(running && !['SUCCESS', 'FAILED', 'CACHED'].includes(running.phase))}
                  className="inline-flex items-center gap-2 px-4 py-2 rounded bg-blue-600 text-white hover:bg-blue-700 disabled:opacity-50 text-sm font-medium">
            {running && !['SUCCESS', 'FAILED', 'CACHED'].includes(running.phase) ? (
              <><Loader2 className="w-4 h-4 animate-spin" /> Running…</>
            ) : (
              <><Play className="w-4 h-4" /> Run comparison</>
            )}
          </button>
        </div>
        {running && (
          <div className="mt-3 text-xs text-gray-600 flex items-center gap-2">
            <span>{running.phase === 'CACHED' ? 'Cached result' : `Job run ${running.runId}`}:</span>
            <span className={`px-1.5 py-0.5 rounded font-medium ${
              running.phase === 'SUCCESS' ? 'bg-emerald-100 text-emerald-700' :
              running.phase === 'CACHED' ? 'bg-amber-100 text-amber-700' :
              running.phase === 'FAILED' ? 'bg-red-100 text-red-700' :
              'bg-blue-100 text-blue-700'
            }`}>{running.phase}</span>
            {running.cacheKey && <span className="font-mono text-[10px]">cache={running.cacheKey}</span>}
          </div>
        )}
      </section>

      {/* Note on fair comparison */}
      <div className="bg-amber-50 border border-amber-200 rounded p-3 mb-4 text-xs text-amber-900 flex gap-2">
        <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-700" />
        <div>
          All versions are scored on the current Modelling Mart snapshot. Simulated historical replays share
          bytes with the champion (no independent training), so differences here arise only from the applied
          scenario or from genuinely distinct versions once rolling retrains are wired up.
        </div>
      </div>

      {/* Loading / results */}
      {loadingResult && (
        <div className="py-10 text-center text-sm text-gray-500">
          <Loader2 className="w-4 h-4 inline animate-spin mr-1" /> Loading results…
        </div>
      )}

      {result && <ResultView result={result} selectedVersions={selectedVersions} />}

      {toast && (
        <div onClick={() => setToast(null)}
             className="fixed bottom-4 right-4 bg-gray-900 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50 cursor-pointer">
          {toast}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Result view
// ---------------------------------------------------------------------------

function ResultView({ result, selectedVersions }: { result: any; selectedVersions: Version[] }) {
  const versions: number[] = (result.score_summary || []).map((s: any) => s.version);
  const a = versions[0];
  const b = versions[versions.length - 1];
  const pairShift = (result.pair_shifts || []).find((p: any) => p.a_version === a && p.b_version === b) || {};

  const Rec = REC_STYLE[result.review?.recommendation || 'INVESTIGATE'] || REC_STYLE.INVESTIGATE;

  return (
    <div className="space-y-4">
      {/* Agent review banner */}
      <section className={`border rounded-lg p-4 flex items-start gap-3 ${Rec.cls}`}>
        <Rec.icon className="w-5 h-5 shrink-0 mt-0.5" />
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-sm">Recommendation: {Rec.label}</h4>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-white/60 font-medium">
              {result.review?.agent_type || 'rule-based'}
            </span>
          </div>
          <ul className="text-sm mt-2 space-y-0.5 list-disc list-inside">
            {(result.review?.findings || []).map((f: string, i: number) => <li key={i}>{f}</li>)}
          </ul>
        </div>
      </section>

      {/* Headline metrics */}
      <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800">Headline metrics (fresh holdout)</h3>
        </div>
        <div className="p-4 overflow-x-auto">
          <HoldoutTable holdout={result.holdout_metrics || []} versions={versions} scoreSummary={result.score_summary || []} />
        </div>
      </section>

      {/* Shift summary */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <ShiftCard label="Mean |shift|" value={fmt(pairShift.mean_abs_shift)} />
        <ShiftCard label={`Policies shifted >10%`} value={(pairShift.n_shift_gt_10pct ?? 0).toLocaleString()}
                   tone={pairShift.n_shift_gt_10pct > result.portfolio_size * 0.1 ? 'warn' : 'normal'} />
        <ShiftCard label={`Policies shifted >25%`} value={(pairShift.n_shift_gt_25pct ?? 0).toLocaleString()}
                   tone={pairShift.n_shift_gt_25pct > 0 ? 'warn' : 'normal'} />
      </div>

      {/* Distribution histogram (shift buckets) */}
      <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800">
            Relative shift distribution (v{a} → v{b})
          </h3>
        </div>
        <div className="p-4">
          <ShiftHistogram buckets={pairShift.histogram_buckets || []} />
        </div>
      </section>

      {/* Score quantiles */}
      <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800">Score distribution per version</h3>
        </div>
        <div className="p-4 overflow-x-auto">
          <ScoreQuantiles rows={result.score_summary || []} />
        </div>
      </section>

      {/* Segment breakdown */}
      <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800">
            Segments most affected (v{a} → v{b})
          </h3>
        </div>
        <div className="p-4 overflow-x-auto">
          <SegmentTable rows={result.segment_rows || []} />
        </div>
      </section>

      {/* Outliers */}
      <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800">Top policy-level shifts</h3>
        </div>
        <div className="p-4 overflow-x-auto">
          <OutlierTable rows={result.outlier_rows || []} />
        </div>
      </section>

      {/* Explainability diff */}
      <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800">
            {result.explain_diff?.type === 'glm' ? 'Coefficient shifts' : 'Feature importance shifts'}
            {' · '}
            <span className="text-xs text-gray-500 font-normal">
              v{result.explain_diff?.a_version} → v{result.explain_diff?.b_version}
            </span>
          </h3>
        </div>
        <div className="p-4 overflow-x-auto">
          <ExplainDiff ed={result.explain_diff} />
        </div>
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small viz components
// ---------------------------------------------------------------------------

function ShiftCard({ label, value, tone }: { label: string; value: string; tone?: 'warn' | 'normal' }) {
  const cls = tone === 'warn' ? 'text-amber-700' : 'text-gray-900';
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${cls}`}>{value}</div>
    </div>
  );
}

function HoldoutTable({ holdout, versions, scoreSummary }:
  { holdout: any[]; versions: number[]; scoreSummary: any[] }) {
  // Group holdout metrics by metric-name → per-version value
  const byMetric: Record<string, Record<number, number>> = {};
  for (const h of holdout) {
    byMetric[h.metric] = byMetric[h.metric] || {};
    byMetric[h.metric][h.version] = h.value;
  }
  const metricKeys = Object.keys(byMetric);
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-gray-500 border-b">
          <th className="text-left py-1.5 pr-3 font-medium">Metric</th>
          {versions.map((v, i) => (
            <th key={v} className="text-right py-1.5 pr-3 font-medium">
              {String.fromCharCode(65 + i)} · v{v}
            </th>
          ))}
        </tr>
      </thead>
      <tbody>
        {metricKeys.length === 0 ? (
          <tr><td colSpan={versions.length + 1} className="py-4 text-gray-500 italic text-center">
            Holdout metric not available for this family.
          </td></tr>
        ) : metricKeys.map(m => {
          const vals = versions.map(v => byMetric[m]?.[v] as number | undefined);
          const nums = vals.filter((x): x is number => typeof x === 'number');
          const hi = Math.max(...nums);
          return (
            <tr key={m} className="border-b last:border-0">
              <td className="py-1.5 pr-3 font-medium text-gray-700">{m}</td>
              {vals.map((x, i) => (
                <td key={i} className="py-1.5 pr-3 text-right font-mono text-xs">
                  {typeof x === 'number' ? (
                    <span className={x === hi ? 'font-bold text-emerald-700' : 'text-gray-900'}>
                      {m === 'mae_gbp' ? x.toLocaleString(undefined, { maximumFractionDigits: 0 }) : x.toFixed(4)}
                    </span>
                  ) : '—'}
                </td>
              ))}
            </tr>
          );
        })}
        <tr className="border-t-2 border-gray-200">
          <td className="py-1.5 pr-3 text-[11px] text-gray-500 italic">train-time (from MLflow)</td>
          {versions.map(v => {
            const s = scoreSummary.find((x: any) => x.version === v);
            const m = s?.training_metrics || {};
            const primary = m.gini ?? m.auc;
            return (
              <td key={v} className="py-1.5 pr-3 text-right text-[11px] text-gray-500 font-mono">
                {typeof primary === 'number' ? primary.toFixed(4) : '—'}
              </td>
            );
          })}
        </tr>
      </tbody>
    </table>
  );
}

function ShiftHistogram({ buckets }: { buckets: any[] }) {
  if (!buckets || buckets.length === 0) return <div className="text-xs text-gray-500 italic">No histogram data.</div>;
  const max = Math.max(1, ...buckets.map((b: any) => b.count));
  const label = (b: any) => {
    if (b.lo == null) return `< ${(b.hi * 100).toFixed(0)}%`;
    if (b.hi == null) return `> ${(b.lo * 100).toFixed(0)}%`;
    return `${(b.lo * 100).toFixed(0)}..${(b.hi * 100).toFixed(0)}%`;
  };
  return (
    <div className="space-y-1">
      {buckets.map((b: any, i: number) => {
        const pct = (b.count / max) * 100;
        const isBigShift = Math.abs(b.lo ?? 0) >= 0.25 || Math.abs(b.hi ?? 0) >= 0.25;
        const isZero = (b.lo ?? 1) <= 0 && (b.hi ?? -1) >= 0;
        const color = isBigShift ? 'bg-amber-500' : (isZero ? 'bg-emerald-500' : 'bg-blue-400');
        return (
          <div key={i} className="flex items-center gap-2 text-xs">
            <div className="w-24 text-gray-700 text-right">{label(b)}</div>
            <div className="flex-1 bg-gray-100 h-4 rounded overflow-hidden">
              <div className={`${color} h-full`} style={{ width: `${pct}%` }} />
            </div>
            <div className="w-16 text-gray-600 font-mono">{b.count.toLocaleString()}</div>
          </div>
        );
      })}
    </div>
  );
}

function ScoreQuantiles({ rows }: { rows: any[] }) {
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-gray-500 border-b">
          <th className="text-left py-1.5 pr-3 font-medium">Version</th>
          <th className="text-left py-1.5 pr-3 font-medium">Story</th>
          <th className="text-right py-1.5 pr-3 font-medium">Mean</th>
          <th className="text-right py-1.5 pr-3 font-medium">p25</th>
          <th className="text-right py-1.5 pr-3 font-medium">p50</th>
          <th className="text-right py-1.5 pr-3 font-medium">p75</th>
          <th className="text-right py-1.5 pr-3 font-medium">p95</th>
          <th className="text-right py-1.5 pr-3 font-medium">p99</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r: any) => (
          <tr key={r.version} className="border-b last:border-0">
            <td className="py-1.5 pr-3 font-medium">
              v{r.version}{!r.simulated && <span className="ml-1 text-emerald-600">★</span>}
            </td>
            <td className="py-1.5 pr-3 text-xs text-gray-600">{r.story || '—'}</td>
            {['mean', 'p25', 'p50', 'p75', 'p95', 'p99'].map(k => (
              <td key={k} className="py-1.5 pr-3 text-right font-mono text-xs">
                {fmt(r[k])}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function SegmentTable({ rows }: { rows: any[] }) {
  if (rows.length === 0) return <div className="text-xs text-gray-500 italic">No segments recorded.</div>;
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-gray-500 border-b">
          <th className="text-left py-1.5 pr-3 font-medium">Segment type</th>
          <th className="text-left py-1.5 pr-3 font-medium">Value</th>
          <th className="text-right py-1.5 pr-3 font-medium">n</th>
          <th className="text-right py-1.5 pr-3 font-medium">A mean</th>
          <th className="text-right py-1.5 pr-3 font-medium">B mean</th>
          <th className="text-right py-1.5 pr-3 font-medium">Rel. shift</th>
        </tr>
      </thead>
      <tbody>
        {rows.map((r: any, i: number) => {
          const pct = (r.rel_shift || 0) * 100;
          const warn = Math.abs(pct) >= 25;
          const up = pct > 0;
          return (
            <tr key={i} className="border-b last:border-0 hover:bg-gray-50">
              <td className="py-1.5 pr-3 text-xs text-gray-600">{r.segment_type}</td>
              <td className="py-1.5 pr-3 text-xs text-gray-800">{r.segment}</td>
              <td className="py-1.5 pr-3 text-right text-xs text-gray-600 font-mono">{r.n}</td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono">{fmt(r.a_mean)}</td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono">{fmt(r.b_mean)}</td>
              <td className={`py-1.5 pr-3 text-right text-xs font-mono font-medium inline-flex items-center gap-0.5 ${
                warn ? 'text-amber-700' : (up ? 'text-blue-700' : 'text-gray-600')
              }`}>
                {up ? <TrendingUp className="w-3 h-3" /> : <TrendingDown className="w-3 h-3" />}
                {pct >= 0 ? '+' : ''}{pct.toFixed(1)}%
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function OutlierTable({ rows }: { rows: any[] }) {
  if (rows.length === 0) return <div className="text-xs text-gray-500 italic">No outliers sampled.</div>;
  const cols = Object.keys(rows[0]);
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-gray-500 border-b">
          {cols.map(c => <th key={c} className="text-left py-1.5 pr-3 font-medium">{c}</th>)}
        </tr>
      </thead>
      <tbody>
        {rows.map((r, i) => (
          <tr key={i} className="border-b last:border-0 hover:bg-gray-50">
            {cols.map(c => (
              <td key={c} className="py-1.5 pr-3 text-xs font-mono text-gray-800">
                {typeof r[c] === 'number' ? fmt(r[c]) : String(r[c] ?? '—')}
              </td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function ExplainDiff({ ed }: { ed: any }) {
  if (!ed || !ed.rows || ed.rows.length === 0) return <div className="text-xs text-gray-500 italic">No explainability artefacts available.</div>;
  if (ed.type === 'glm') {
    return (
      <table className="w-full text-sm">
        <thead>
          <tr className="text-xs text-gray-500 border-b">
            <th className="text-left py-1.5 pr-3 font-medium">Feature</th>
            <th className="text-right py-1.5 pr-3 font-medium">A coef</th>
            <th className="text-right py-1.5 pr-3 font-medium">B coef</th>
            <th className="text-right py-1.5 pr-3 font-medium">Δ coef</th>
            <th className="text-right py-1.5 pr-3 font-medium">A rel.</th>
            <th className="text-right py-1.5 pr-3 font-medium">B rel.</th>
          </tr>
        </thead>
        <tbody>
          {ed.rows.map((r: any, i: number) => (
            <tr key={i} className="border-b last:border-0">
              <td className="py-1.5 pr-3 text-xs text-gray-800 truncate max-w-[260px]" title={r.feature}>{r.feature}</td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono">{fmt(r.a_coef)}</td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono">{fmt(r.b_coef)}</td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono font-semibold">
                {r.delta_coef >= 0 ? '+' : ''}{fmt(r.delta_coef)}
              </td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono text-gray-500">{fmt(r.a_relativity)}</td>
              <td className="py-1.5 pr-3 text-right text-xs font-mono text-gray-500">{fmt(r.b_relativity)}</td>
            </tr>
          ))}
        </tbody>
      </table>
    );
  }
  // GBM
  return (
    <table className="w-full text-sm">
      <thead>
        <tr className="text-xs text-gray-500 border-b">
          <th className="text-left py-1.5 pr-3 font-medium">Feature</th>
          <th className="text-right py-1.5 pr-3 font-medium">A gain</th>
          <th className="text-right py-1.5 pr-3 font-medium">B gain</th>
          <th className="text-right py-1.5 pr-3 font-medium">Δ gain</th>
        </tr>
      </thead>
      <tbody>
        {ed.rows.map((r: any, i: number) => (
          <tr key={i} className="border-b last:border-0">
            <td className="py-1.5 pr-3 text-xs text-gray-800 truncate max-w-[260px]" title={r.feature}>{r.feature}</td>
            <td className="py-1.5 pr-3 text-right text-xs font-mono">{Number(r.a_gain || 0).toFixed(0)}</td>
            <td className="py-1.5 pr-3 text-right text-xs font-mono">{Number(r.b_gain || 0).toFixed(0)}</td>
            <td className="py-1.5 pr-3 text-right text-xs font-mono font-semibold">
              {(r.delta_gain || 0) >= 0 ? '+' : ''}{Number(r.delta_gain || 0).toFixed(0)}
            </td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function fmt(x: any): string {
  if (x == null || Number.isNaN(x)) return '—';
  if (typeof x !== 'number') return String(x);
  if (Math.abs(x) >= 1000) return x.toLocaleString(undefined, { maximumFractionDigits: 0 });
  if (Math.abs(x) >= 1) return x.toFixed(3);
  return x.toFixed(4);
}
