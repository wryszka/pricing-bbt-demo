import { useEffect, useMemo, useState } from 'react';
import {
  Calculator, Loader2, PlayCircle, History, Sparkles, ShieldCheck, ExternalLink,
  ChevronDown, ChevronRight, GitCompareArrows, FileText, Calendar, RefreshCw, Tag,
} from 'lucide-react';
import { api } from '../lib/api';

// ---------------------------------------------------------------------------
// Pricing Engine — organised around MONTHLY releases. One release bundles
// all 4 model versions + rating-engine config + narrative + approval.
// ---------------------------------------------------------------------------

type Release = {
  release_id: string;
  display_name: string;
  effective_date: string;
  status: 'champion' | 'previous_champion' | 'archived' | string;
  freq_glm_version: string;
  sev_glm_version: string;
  demand_gbm_version: string;
  fraud_gbm_version: string;
  rating_engine_version: string;
  approved_by?: string;
  narrative?: string;
};

type Section = 'quote' | 'mta' | 'history';

export default function PricingEngine() {
  const [section, setSection] = useState<Section>('quote');
  const [current, setCurrent] = useState<Release | null>(null);
  const [releases, setReleases] = useState<Release[]>([]);
  const [loading, setLoading]   = useState(true);

  useEffect(() => {
    setLoading(true);
    Promise.all([
      api.getCurrentRelease().catch(() => null),
      api.listReleases().then(d => d?.releases || []).catch(() => []),
    ]).then(([c, all]) => { setCurrent(c); setReleases(all); })
      .finally(() => setLoading(false));
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Calculator className="w-6 h-6 text-teal-600" /> Pricing Engine
        </h2>
        <p className="text-gray-500 mt-1">
          Our rate book. Ships as a monthly <em>release</em> — one object, committee-approved,
          bundling every model version and the rating-engine config that priced a quote on that date.
        </p>
      </div>

      {/* Current release headline */}
      {current && <CurrentReleaseCard release={current} />}
      {loading && <div className="h-32 rounded-lg bg-gray-100 animate-pulse" />}

      {/* Sub-tabs */}
      <div className="flex gap-1 border-b border-gray-200 mt-6 mb-5">
        <Tab active={section === 'quote'}   onClick={() => setSection('quote')}
             icon={<PlayCircle className="w-4 h-4" />}  label="Quote Runner" />
        <Tab active={section === 'mta'}     onClick={() => setSection('mta')}
             icon={<Sparkles className="w-4 h-4" />}    label="Mid-term Adjustment" />
        <Tab active={section === 'history'} onClick={() => setSection('history')}
             icon={<History className="w-4 h-4" />}     label="Release History" />
      </div>

      {section === 'quote'   && <QuoteRunner   current={current} />}
      {section === 'mta'     && <MtaSimulator  current={current} />}
      {section === 'history' && <ReleaseHistory releases={releases} />}
    </div>
  );
}

function Tab({ active, onClick, icon, label }:
  { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button onClick={onClick}
            className={`inline-flex items-center gap-1.5 px-4 py-2 text-sm font-medium border-b-2 transition ${
              active ? 'border-teal-600 text-teal-700' : 'border-transparent text-gray-600 hover:text-gray-900'
            }`}>
      {icon} {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Current release — the big "April 2026 rate book" card
// ---------------------------------------------------------------------------

function CurrentReleaseCard({ release }: { release: Release }) {
  return (
    <div className="rounded-xl border border-teal-300 bg-gradient-to-r from-teal-50 to-white p-5">
      <div className="flex items-start justify-between mb-2">
        <div>
          <div className="text-[11px] font-semibold uppercase tracking-wider text-teal-700">
            Current release · effective {release.effective_date}
          </div>
          <div className="text-2xl font-bold text-gray-900 mt-0.5 flex items-center gap-2">
            {release.display_name} rate book
            <ShieldCheck className="w-5 h-5 text-teal-600" />
          </div>
          {release.approved_by && (
            <div className="text-xs text-gray-500 mt-0.5">
              Approved by {release.approved_by.split('@')[0]}
            </div>
          )}
        </div>
        <span className="text-[10px] font-semibold px-2 py-0.5 rounded-full bg-teal-600 text-white">
          CHAMPION
        </span>
      </div>

      {release.narrative && (
        <div className="mt-2 text-sm text-gray-700 leading-relaxed max-w-3xl">
          {release.narrative}
        </div>
      )}

      <div className="mt-4 grid grid-cols-2 md:grid-cols-5 gap-2 text-sm">
        <VersionChip label="Frequency GLM"   v={release.freq_glm_version} />
        <VersionChip label="Severity GLM"    v={release.sev_glm_version} />
        <VersionChip label="Demand GBM"      v={release.demand_gbm_version} />
        <VersionChip label="Fraud GBM"       v={release.fraud_gbm_version} />
        <VersionChip label="Rating engine"   v={release.rating_engine_version} highlight />
      </div>
    </div>
  );
}

function VersionChip({ label, v, highlight }: { label: string; v: string; highlight?: boolean }) {
  return (
    <div className={`rounded-md border px-2.5 py-1.5 ${
      highlight ? 'border-teal-300 bg-teal-100/60' : 'border-gray-200 bg-white'
    }`}>
      <div className="text-[10px] text-gray-500 uppercase tracking-wider">{label}</div>
      <div className="font-mono text-gray-900 text-sm tabular-nums">{v}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Quote Runner — simplified: always runs on the current release
// ---------------------------------------------------------------------------

const DEFAULT_FEATURES: Record<string, any> = {
  sum_insured:                  2_500_000,
  annual_turnover:              850_000,
  current_premium:              1_200,
  industry_risk_tier:           'Medium',
  construction_type:            'Non-Combustible',
  region:                       'London',
  postcode_sector:              'EC1A',
  credit_score:                 620,
  ccj_count:                    1,
  years_trading:                12,
  flood_zone_rating:            3,
  proximity_to_fire_station_km: 2.4,
  crime_theft_index:            55,
  subsidence_risk:              2,
  composite_location_risk:      55,
  urban_score:                  0.9,
  is_coastal:                   0,
  population_density_per_km2:   12000,
  elevation_metres:             25,
  annual_rainfall_mm:           620,
  director_stability_score:     0.8,
  employee_count_est:           18,
  distance_to_coast_km:         12,
  neighbourhood_claim_frequency: 0.88,
  year_built:                   1985,
};

function QuoteRunner({ current }: { current: Release | null }) {
  const [features, setFeatures] = useState<Record<string, any>>(DEFAULT_FEATURES);
  const [status, setStatus]     = useState<any>(null);
  const [running, setRunning]   = useState(false);
  const [result, setResult]     = useState<any>(null);
  const [error, setError]       = useState<string | null>(null);
  const [releases, setReleases] = useState<Release[]>([]);
  const [releaseId, setReleaseId] = useState<string>('');      // '' = current
  const [batchState, setBatchState] = useState<{ runId?: number; url?: string; phase: string } | null>(null);

  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const s = await api.getPricingStatus();
        if (!alive) return;
        setStatus(s);
        if (!s.ready) setTimeout(tick, 4000);
      } catch { if (alive) setTimeout(tick, 6000); }
    };
    tick();
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    api.listReleases().then(d => setReleases(d?.releases || [])).catch(() => {});
  }, []);

  const isHistorical = releaseId && releaseId !== '' && releaseId !== (current?.release_id || '');

  const run = async () => {
    setRunning(true); setError(null); setResult(null); setBatchState(null);

    // Current release → live serving endpoint, instant
    if (!isHistorical) {
      try {
        const r = await api.runQuote({ features, label: 'ui' });
        setResult(r);
      } catch (e: any) {
        setError(e.message || String(e));
      } finally {
        setRunning(false);
      }
      return;
    }

    // Historical → batch job; poll for completion
    try {
      const trigger = await api.scoreOnRelease(releaseId, features, 'ui');
      setBatchState({ runId: trigger.run_id, url: trigger.run_page_url, phase: 'queued' });
      // Poll
      for (let i = 0; i < 40; i++) {
        await new Promise(res => setTimeout(res, 8000));
        const s = await api.getHistoricalScoreStatus(trigger.run_id);
        setBatchState({ runId: trigger.run_id, url: trigger.run_page_url, phase: s.state });
        if (s.state === 'TERMINATED' && s.result_state === 'SUCCESS' && s.result) {
          // Shape the result like /quote/run so <QuoteResult/> can render it
          const rel = releases.find(r => r.release_id === releaseId);
          setResult({
            rating_engine: { version: s.result.model_versions?.rating_engine },
            quotes: [{
              model_versions: {
                freq_glm:   s.result.model_versions?.freq_glm,
                sev_glm:    s.result.model_versions?.sev_glm,
                demand_gbm: s.result.model_versions?.demand_gbm,
                fraud_gbm:  s.result.model_versions?.fraud_gbm,
              },
              predictions:  s.result.predictions,
              price_buildup: s.result.price_buildup,
              source:       'historical_batch',
            }],
            scored_at: new Date().toISOString(),
            scoring_engine: `historical_batch (${rel?.display_name})`,
          });
          break;
        }
        if (s.state === 'TERMINATED' || s.result_state === 'FAILED' || s.state === 'INTERNAL_ERROR') {
          setError(`Historical score failed — state=${s.state} result_state=${s.result_state}. See ${trigger.run_page_url}`);
          break;
        }
      }
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setRunning(false);
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      <section className="lg:col-span-5 bg-white border border-gray-200 rounded-lg p-5">
        <h3 className="font-semibold text-gray-900 mb-3">Quote features</h3>
        <p className="text-xs text-gray-500 mb-3">
          Edit any field. These flow through {current?.display_name || 'the current'} release — the same rate
          book that prices real quotes on {current?.effective_date || 'today'}.
        </p>
        <div className="grid grid-cols-2 gap-3 max-h-[560px] overflow-y-auto pr-1">
          {Object.entries(features).map(([k, v]) => (
            <label key={k} className="block">
              <span className="text-[11px] text-gray-500">{k}</span>
              <input
                value={v ?? ''}
                onChange={e => {
                  const raw = e.target.value;
                  const num = raw !== '' && !isNaN(Number(raw)) ? Number(raw) : raw;
                  setFeatures(f => ({ ...f, [k]: num }));
                }}
                className="w-full mt-0.5 px-2 py-1 rounded border border-gray-200 text-sm focus:outline-none focus:ring-2 focus:ring-teal-400"
              />
            </label>
          ))}
        </div>
      </section>

      <section className="lg:col-span-7 space-y-4">
        <div className="bg-white border border-gray-200 rounded-lg p-5">
          <div className="flex items-center justify-between gap-4">
            <div className="flex-1">
              <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 mb-1">
                Score on release
              </div>
              <select value={releaseId}
                      onChange={e => setReleaseId(e.target.value)}
                      className="w-full px-3 py-1.5 rounded border border-gray-300 text-sm bg-white">
                <option value="">{current?.display_name || 'Current release'} (live)</option>
                {releases.filter(r => r.release_id !== current?.release_id).map(r => (
                  <option key={r.release_id} value={r.release_id}>
                    {r.display_name} — {r.status === 'previous_champion' ? 'previous champion' : 'archived'}
                  </option>
                ))}
              </select>
              <div className="text-[11px] text-gray-500 mt-1.5">
                {isHistorical
                  ? <>Historical release — scored via a short batch job that loads the pinned UC model versions on demand. ~1-2 min.</>
                  : <>Current release — scored live through the <code>pricing_scorer</code> serving endpoint. Sub-second.</>}
              </div>
            </div>
            <button onClick={run}
                    disabled={running || (!isHistorical && status && !status.ready)}
                    title={!isHistorical && status && !status.ready ? 'Endpoint warming up…' : ''}
                    className="shrink-0 inline-flex items-center gap-1.5 px-4 py-2 rounded bg-teal-600 text-white text-sm font-medium hover:bg-teal-700 disabled:opacity-50">
              {running ? <Loader2 className="w-4 h-4 animate-spin" /> :
               (!isHistorical && status && !status.ready) ? <Loader2 className="w-4 h-4 animate-spin" /> :
               <PlayCircle className="w-4 h-4" />}
              {running ? 'Scoring…' :
               (!isHistorical && status && !status.ready) ? 'Warming endpoint…' :
               'Run quote'}
            </button>
          </div>

          {batchState && running && (
            <div className="mt-3 pt-3 border-t border-gray-200 text-[11px] text-gray-600 flex items-center justify-between">
              <span>
                <Loader2 className="w-3 h-3 animate-spin inline mr-1" />
                Batch job <code>{batchState.runId}</code> — state: <span className="font-medium">{batchState.phase}</span>
              </span>
              {batchState.url && (
                <a href={batchState.url} target="_blank" rel="noreferrer"
                   className="text-teal-600 hover:underline inline-flex items-center gap-1">
                  open run <ExternalLink className="w-3 h-3" />
                </a>
              )}
            </div>
          )}
        </div>

        {error && <div className="bg-red-50 border border-red-200 rounded-lg p-3 text-sm text-red-700">{error}</div>}
        {result && <QuoteResult result={result} release={
          releases.find(r => r.release_id === releaseId) || current
        } />}
      </section>
    </div>
  );
}

function QuoteResult({ result, release }: { result: any; release: Release | null }) {
  const q = (result.quotes || [])[0];
  if (!q) return null;
  const p = q.price_buildup || {};
  const rows = [
    { label: 'Base technical (freq × sev)',       amount: p.base_premium },
    { label: 'Fraud-risk loading',                 amount: p.fraud_loading },
    { label: 'Demand-elasticity adjustment',       amount: p.demand_adj },
    { label: 'Technical premium', amount: p.technical_premium, bold: true },
    { label: 'Expense loading',                    amount: p.expense_loading },
    { label: 'Broker commission',                  amount: p.commission },
    { label: 'Gross premium (to customer)',        amount: p.gross_premium, bold: true, highlight: true },
  ];

  return (
    <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="px-5 py-3 border-b border-gray-200 bg-teal-50/60 flex items-center justify-between">
        <div className="text-sm">
          <span className="font-semibold text-gray-900">
            £{p.gross_premium?.toLocaleString(undefined, { minimumFractionDigits: 2 })}
          </span>
          <span className="ml-2 text-gray-600">gross premium · {release?.display_name}</span>
        </div>
        <div className="text-[11px] text-gray-500">
          Scored {new Date(result.scored_at).toLocaleTimeString()} · {result.scoring_engine}
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-4 gap-3 p-5 border-b border-gray-200">
        <Tile label="Freq pred"   value={q.predictions?.freq_pred?.toFixed?.(4) ?? '—'} />
        <Tile label="Sev pred"    value={q.predictions?.sev_pred != null ? `£${q.predictions.sev_pred.toLocaleString()}` : '—'} />
        <Tile label="Demand pred" value={q.predictions?.demand_pred?.toFixed?.(3) ?? '—'} />
        <Tile label="Fraud pred"  value={q.predictions?.fraud_pred?.toFixed?.(3) ?? '—'} />
      </div>

      <div className="p-5">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 mb-2">Price build-up</div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-y-0.5 text-sm">
          {rows.map((r, i) => (
            <div key={i} className={`flex justify-between px-3 py-1 rounded ${r.highlight ? 'bg-teal-100/60 text-teal-900' : ''}`}>
              <span className={r.bold ? 'font-semibold' : 'text-gray-700'}>{r.label}</span>
              <span className={`tabular-nums ${r.bold ? 'font-semibold' : ''}`}>
                £{(r.amount ?? 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}
              </span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function Tile({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded bg-gray-50 border border-gray-200 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-gray-500">{label}</div>
      <div className="text-base font-semibold tabular-nums text-gray-900">{value}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// MTA Simulator — unchanged flow, but references current release in copy
// ---------------------------------------------------------------------------

// MTA flow — policy-context-aware. Provide a policy ID, the app reads its
// inception date and auto-selects the release-of-record. We re-price on that
// release (the MTA invoice's consistent answer) AND on today's live release
// (for reference). The "simulate" wording is gone — this IS the MTA reprice;
// the audit-log step is de-emphasised so the demo flow stays clean.

function MtaSimulator({ current: _current }: { current: Release | null }) {
  const [policyId, setPolicyId]   = useState('POL-100009');
  const [change, setChange]       = useState<{ field: string; value: string }>({ field: 'annual_turnover', value: '1500000' });
  const [effective, setEffective] = useState(new Date().toISOString().slice(0,10));
  const [reason, setReason]       = useState('Post-acquisition turnover bump');
  const [running, setRunning]     = useState(false);
  const [result, setResult]       = useState<any>(null);
  const [error, setError]         = useState<string | null>(null);

  // Policy preview — fetched whenever policyId changes (debounced).
  const [ctx, setCtx]             = useState<any>(null);
  const [ctxErr, setCtxErr]       = useState<string | null>(null);
  const [ctxLoading, setCtxLoad]  = useState(false);

  useEffect(() => {
    const id = policyId.trim().toUpperCase();
    if (!id) { setCtx(null); setCtxErr(null); return; }
    let cancelled = false;
    setCtxLoad(true); setCtxErr(null);
    const t = setTimeout(async () => {
      try {
        const c = await api.getPolicyContext(id);
        if (!cancelled) setCtx(c);
      } catch (e: any) {
        if (!cancelled) { setCtx(null); setCtxErr(e.message || String(e)); }
      } finally {
        if (!cancelled) setCtxLoad(false);
      }
    }, 300);
    return () => { cancelled = true; clearTimeout(t); };
  }, [policyId]);

  const reprice = async () => {
    setRunning(true); setError(null); setResult(null);
    try {
      const parsed = Number(change.value);
      const v = !isNaN(parsed) && change.value !== '' ? parsed : change.value;
      const r = await api.simulateMta({
        policy_id: policyId,
        changes: { [change.field]: v },
        effective_date: effective,
        reason,
      });
      setResult(r);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setRunning(false);
    }
  };

  // Demo policies — each in a different release window so the auto-selection
  // is visibly different.
  const demoPolicies: { id: string; label: string }[] = [
    { id: 'POL-100001', label: 'Dec 2025 inception' },
    { id: 'POL-100009', label: 'Feb 2026 inception' },
    { id: 'POL-100011', label: 'Apr 2026 inception (live)' },
  ];

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
      <section className="lg:col-span-4 bg-white border border-gray-200 rounded-lg p-5 space-y-3">
        <h3 className="font-semibold text-gray-900">Mid-term adjustment</h3>
        <p className="text-xs text-gray-500">
          Re-prices the policy on the release that was in force at inception (release-of-record)
          and, for reference, on today's live release.
        </p>

        <label className="block text-sm">
          <span className="text-[11px] text-gray-500">Policy ID</span>
          <input value={policyId} onChange={e => setPolicyId(e.target.value.toUpperCase())}
                 className="w-full mt-0.5 px-3 py-1.5 rounded border border-gray-300 font-mono focus:outline-none focus:ring-2 focus:ring-teal-400" />
        </label>
        <div className="flex flex-wrap gap-1.5">
          {demoPolicies.map(d => (
            <button key={d.id}
                    onClick={() => setPolicyId(d.id)}
                    className={`text-[10px] px-2 py-0.5 rounded-full border transition ${
                      policyId === d.id
                        ? 'bg-teal-600 text-white border-teal-600'
                        : 'bg-white text-gray-700 border-gray-300 hover:border-teal-400 hover:bg-teal-50'
                    }`}>
              {d.id} · {d.label}
            </button>
          ))}
        </div>

        <PolicyContextCard ctx={ctx} loading={ctxLoading} err={ctxErr} />

        <label className="block text-sm">
          <span className="text-[11px] text-gray-500">Field to change</span>
          <select value={change.field} onChange={e => setChange(c => ({ ...c, field: e.target.value }))}
                  className="w-full mt-0.5 px-3 py-1.5 rounded border border-gray-300">
            {['annual_turnover', 'sum_insured', 'current_premium', 'industry_risk_tier',
              'construction_type', 'ccj_count', 'flood_zone_rating', 'employee_count_est'].map(f => (
              <option key={f} value={f}>{f}</option>
            ))}
          </select>
        </label>
        <label className="block text-sm">
          <span className="text-[11px] text-gray-500">New value</span>
          <input value={change.value} onChange={e => setChange(c => ({ ...c, value: e.target.value }))}
                 className="w-full mt-0.5 px-3 py-1.5 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-teal-400" />
        </label>
        <label className="block text-sm">
          <span className="text-[11px] text-gray-500">Effective date</span>
          <input type="date" value={effective} onChange={e => setEffective(e.target.value)}
                 className="w-full mt-0.5 px-3 py-1.5 rounded border border-gray-300" />
        </label>
        <label className="block text-sm">
          <span className="text-[11px] text-gray-500">Change reason</span>
          <input value={reason} onChange={e => setReason(e.target.value)}
                 className="w-full mt-0.5 px-3 py-1.5 rounded border border-gray-300" />
        </label>
        <button onClick={reprice}
                disabled={running || !ctx}
                className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded bg-teal-600 text-white text-sm font-medium hover:bg-teal-700 disabled:opacity-50">
          {running ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
          {running ? 'Re-pricing…' : 'Re-price policy'}
        </button>
      </section>

      <section className="lg:col-span-8 space-y-3">
        {error && <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-700">{error}</div>}
        {!result && !running && (
          <div className="bg-gray-50 border border-gray-200 rounded p-6 text-sm text-gray-500 italic">
            {ctx
              ? <>Ready to re-price <code>{ctx.policy_id}</code> on <strong>{ctx.inception_release?.display_name}</strong> (release-of-record).
                  Today's live release ({ctx.current_release?.display_name}) shown for reference.</>
              : <>Pick a policy ID — the system will auto-select the release the policy was bound on.</>}
          </div>
        )}
        {result && <MtaResult r={result} />}
      </section>
    </div>
  );
}

function PolicyContextCard({ ctx, loading, err }: { ctx: any; loading: boolean; err: string | null }) {
  if (loading) return (
    <div className="rounded border border-gray-200 bg-gray-50 p-3 text-xs text-gray-500 flex items-center gap-1.5">
      <Loader2 className="w-3.5 h-3.5 animate-spin" /> Looking up policy…
    </div>
  );
  if (err) return (
    <div className="rounded border border-red-200 bg-red-50 p-3 text-xs text-red-700">{err}</div>
  );
  if (!ctx) return null;
  const ir = ctx.inception_release;
  const cr = ctx.current_release;
  const isLive = ir && cr && ir.release_id === cr.release_id;
  return (
    <div className="rounded border border-teal-200 bg-teal-50/50 p-3 space-y-2">
      <div className="text-[10px] uppercase tracking-wider font-bold text-teal-800 flex items-center gap-1">
        <FileText className="w-3 h-3" /> Policy context
      </div>
      <div className="text-xs text-gray-700 space-y-0.5">
        <div className="flex items-center gap-1.5">
          <Calendar className="w-3 h-3 text-teal-700" />
          Bound <span className="font-semibold">{ctx.inception_date}</span>
          {' '}· renews <span className="font-semibold">{ctx.renewal_date}</span>
        </div>
        <div className="flex items-center gap-1.5">
          <Tag className="w-3 h-3 text-teal-700" />
          Premium <span className="font-semibold">£{Number(ctx.current_premium || 0).toLocaleString()}</span>
          {' '}· {ctx.industry_risk_tier || '—'} tier
          {ctx.region ? ` · ${ctx.region}` : ''}
        </div>
      </div>
      {ir && (
        <div className="rounded bg-white border border-teal-200 p-2">
          <div className="text-[10px] uppercase tracking-wider font-semibold text-teal-700 mb-1">
            Auto-selected release-of-record
          </div>
          <div className="text-sm font-semibold text-gray-900">
            {ir.display_name}
            {isLive && (
              <span className="ml-2 text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                also live
              </span>
            )}
          </div>
          <div className="text-[10px] text-gray-500 mt-0.5">
            effective {ir.effective_date} · {ir.status}
          </div>
          <div className="flex flex-wrap gap-1 mt-1.5">
            {Object.entries(ir.model_versions || {}).filter(([_, v]) => v).map(([k, v]) => (
              <span key={k} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-700 font-mono">
                {k}: v{String(v)}
              </span>
            ))}
          </div>
        </div>
      )}
      {!isLive && cr && (
        <div className="text-[11px] text-gray-500">
          Live release <span className="font-semibold text-gray-700">{cr.display_name}</span> ({cr.effective_date})
          shown alongside for reference.
        </div>
      )}
    </div>
  );
}

function MtaResult({ r }: { r: any }) {
  const ir = r.inception_release;
  const cr = r.current_release;
  const onIncep = r.on_inception_release;
  const onCur   = r.on_current_release;
  const isLive  = ir && cr && ir.release_id === cr.release_id;

  // Cross-release delta: live engine vs release-of-record
  const incepGross = onIncep?.after?.price_buildup?.gross_premium;
  const curGross   = onCur?.after?.price_buildup?.gross_premium;
  const crossDelta = (typeof incepGross === 'number' && typeof curGross === 'number')
    ? curGross - incepGross : null;
  const crossPct   = (typeof incepGross === 'number' && incepGross > 0 && crossDelta != null)
    ? (crossDelta / incepGross) * 100 : null;

  return (
    <div className="space-y-3">
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-baseline justify-between flex-wrap gap-2 mb-3">
          <div className="text-sm font-semibold text-gray-900">
            {r.policy_id} · effective {r.effective_date}
          </div>
          <div className="text-[11px] text-gray-500">
            inception {r.inception_date} · {r.remaining_days} / {r.term_days} days remaining · prorating {Math.round(r.remaining_frac * 100)}%
          </div>
        </div>

        <div className={`grid grid-cols-1 ${isLive ? '' : 'md:grid-cols-2'} gap-3`}>
          <RepricedCard
            heading="Release-of-record"
            sub={ir ? `${ir.display_name} · effective ${ir.effective_date}` : ''}
            block={onIncep}
            versions={ir?.model_versions}
            primary
          />
          {!isLive && (
            <RepricedCard
              heading="Live release (reference)"
              sub={cr ? `${cr.display_name} · effective ${cr.effective_date}` : ''}
              block={onCur}
              versions={cr?.model_versions}
            />
          )}
        </div>

        {!isLive && crossDelta != null && (
          <div className="mt-3 rounded border border-gray-200 bg-gray-50 p-3 text-xs text-gray-700">
            <span className="font-semibold">Cross-release reconciliation:</span>{' '}
            today's live release prices this policy{' '}
            <span className={`font-semibold ${crossDelta < 0 ? 'text-emerald-700' : 'text-red-700'}`}>
              {crossDelta > 0 ? '+' : ''}£{Math.abs(crossDelta).toLocaleString(undefined, {maximumFractionDigits: 0})}
              {crossPct != null && <> ({crossPct > 0 ? '+' : ''}{crossPct.toFixed(1)}%)</>}
            </span>
            {' '}vs the release-of-record. The MTA invoice uses the release-of-record for consistency
            with the policyholder's bound contract.
          </div>
        )}
        {isLive && (
          <div className="mt-3 rounded border border-emerald-200 bg-emerald-50 p-3 text-xs text-emerald-900">
            <span className="font-semibold">Inception falls inside the live release window.</span>{' '}
            The live release IS the release-of-record for this policy — only one re-price is needed.
          </div>
        )}
      </div>

      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-gray-500 mb-2">Audit</div>
        <div className="text-[11px] text-gray-500">
          Recorded as an <code>mta_simulated</code> event with the version chain and calibration
          factor; the priced quote is also written to <code>inference_logs</code> with <code>is_mta=true</code>.
        </div>
      </div>
    </div>
  );
}

function RepricedCard({ heading, sub, block, versions, primary }:
  { heading: string; sub: string; block: any; versions: any; primary?: boolean }) {
  const before = block?.before?.price_buildup;
  const after  = block?.after?.price_buildup;
  const prorated = block?.prorated_delta;
  const full = block?.full_delta;
  return (
    <div className={`rounded-lg border p-3 ${primary ? 'border-teal-300 bg-teal-50/40' : 'border-gray-200 bg-gray-50/50'}`}>
      <div className="flex items-baseline justify-between mb-2 gap-2">
        <div>
          <div className={`text-[10px] uppercase tracking-wider font-bold ${primary ? 'text-teal-800' : 'text-gray-600'}`}>
            {heading}
          </div>
          <div className="text-sm font-semibold text-gray-900">{sub}</div>
        </div>
        {primary && (
          <span className="text-[9px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-teal-200 text-teal-900 font-bold">
            primary
          </span>
        )}
      </div>
      <div className="grid grid-cols-3 gap-2 mb-2">
        <MiniTile label="Before"   value={`£${(before?.gross_premium || 0).toLocaleString()}`} />
        <MiniTile label="After"    value={`£${(after?.gross_premium  || 0).toLocaleString()}`} />
        <MiniTile
          label="Prorated"
          value={`${(prorated || 0) >= 0 ? '+' : ''}£${(prorated || 0).toLocaleString()}`}
          tone={(prorated || 0) > 0 ? 'up' : (prorated || 0) < 0 ? 'down' : 'neutral'}
          sub={`Annual ${(full || 0) >= 0 ? '+' : ''}£${(full || 0).toLocaleString()}`}
        />
      </div>
      {versions && (
        <div className="flex flex-wrap gap-1">
          {Object.entries(versions).filter(([_, v]) => v).map(([k, v]) => (
            <span key={k} className="text-[10px] px-1.5 py-0.5 rounded bg-white border border-gray-200 text-gray-700 font-mono">
              {k}: v{String(v)}
            </span>
          ))}
        </div>
      )}
    </div>
  );
}

function MiniTile({ label, value, tone, sub }:
  { label: string; value: string; tone?: 'up' | 'down' | 'neutral'; sub?: string }) {
  const color = tone === 'up'   ? 'bg-red-50 text-red-800 border-red-200'
              : tone === 'down' ? 'bg-emerald-50 text-emerald-800 border-emerald-200'
              : 'bg-gray-50 text-gray-900 border-gray-200';
  return (
    <div className={`rounded border ${color} px-3 py-2`}>
      <div className="text-[11px] uppercase tracking-wide opacity-70">{label}</div>
      <div className="text-lg font-semibold tabular-nums">{value}</div>
      {sub && <div className="text-[11px] opacity-80 mt-0.5">{sub}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Release History — timeline of monthly releases
// ---------------------------------------------------------------------------

function ReleaseHistory({ releases }: { releases: Release[] }) {
  const [expanded, setExpanded] = useState<string | null>(null);
  const [comparing, setComparing] = useState<Record<string, any>>({});

  const triggerCompare = async (r: Release) => {
    setComparing(s => ({ ...s, [r.release_id]: { state: 'queuing' } }));
    try {
      const res = await api.compareReleases(r.release_id, 2000);
      setComparing(s => ({ ...s, [r.release_id]: { state: 'done', result: res } }));
    } catch (e: any) {
      setComparing(s => ({ ...s, [r.release_id]: { state: 'error', error: e.message || String(e) } }));
    }
  };

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">Releases</h3>
        <span className="text-xs text-gray-500">{releases.length} total</span>
      </div>
      <div>
        {releases.map(r => {
          const open = expanded === r.release_id;
          const comp = comparing[r.release_id];
          const isChampion = r.status === 'champion';
          const isPrevious = r.status === 'previous_champion';
          return (
            <div key={r.release_id} className="border-b last:border-b-0">
              <button onClick={() => setExpanded(open ? null : r.release_id)}
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50 text-left">
                <div className="flex items-center gap-3">
                  {open ? <ChevronDown className="w-4 h-4 text-gray-500" /> : <ChevronRight className="w-4 h-4 text-gray-500" />}
                  <div>
                    <div className="font-medium text-gray-900">{r.display_name}</div>
                    <div className="text-[11px] text-gray-500">
                      Effective {r.effective_date} · approved by {(r.approved_by || '-').split('@')[0]}
                    </div>
                  </div>
                </div>
                <span className={`text-[10px] font-semibold px-2 py-0.5 rounded ${
                  isChampion ? 'bg-emerald-100 text-emerald-800' :
                  isPrevious ? 'bg-blue-100 text-blue-700' :
                  'bg-gray-100 text-gray-600'}`}>
                  {r.status}
                </span>
              </button>
              {open && (
                <div className="px-4 pb-4 pt-1 bg-gray-50/40">
                  {r.narrative && (
                    <div className="mb-3 text-[13px] text-gray-700 italic leading-relaxed">{r.narrative}</div>
                  )}
                  <div className="grid grid-cols-2 md:grid-cols-5 gap-2">
                    <VersionChip label="Frequency GLM" v={r.freq_glm_version} />
                    <VersionChip label="Severity GLM"  v={r.sev_glm_version} />
                    <VersionChip label="Demand GBM"    v={r.demand_gbm_version} />
                    <VersionChip label="Fraud GBM"     v={r.fraud_gbm_version} />
                    <VersionChip label="Rating engine" v={r.rating_engine_version} highlight />
                  </div>

                  {!isChampion && (
                    <div className="mt-4">
                      {comp?.state === 'done' ? (
                        <div className="flex flex-col gap-1">
                          <div className="text-xs font-semibold text-gray-700">Batch comparison queued — 4 runs</div>
                          <div className="flex flex-wrap gap-1">
                            {Object.entries(comp.result.run_page_urls || {}).map(([fam, url]: [string, any]) => (
                              <a key={fam} href={url as string} target="_blank" rel="noreferrer"
                                 className="text-[11px] px-2 py-0.5 rounded border border-teal-300 bg-teal-50 text-teal-800 hover:bg-teal-100 inline-flex items-center gap-1">
                                {fam} <ExternalLink className="w-2.5 h-2.5" />
                              </a>
                            ))}
                          </div>
                          <div className="text-[11px] text-gray-500 mt-1">
                            Each run scores 2,000 policies. ~2 min per family. Results land in Compare & Test history.
                          </div>
                        </div>
                      ) : comp?.state === 'error' ? (
                        <div className="text-xs text-red-700">Failed: {comp.error}</div>
                      ) : (
                        <button onClick={() => triggerCompare(r)}
                                disabled={comp?.state === 'queuing'}
                                className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded border border-teal-300 bg-white text-teal-700 text-xs font-medium hover:bg-teal-50 disabled:opacity-50">
                          {comp?.state === 'queuing' ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <GitCompareArrows className="w-3.5 h-3.5" />}
                          {comp?.state === 'queuing' ? 'Queuing…' : `Compare ${r.display_name} vs current`}
                        </button>
                      )}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
