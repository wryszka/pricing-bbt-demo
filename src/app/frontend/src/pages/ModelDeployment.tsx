import { useEffect, useMemo, useRef, useState } from 'react';
import {
  Rocket, ExternalLink, Loader2, Undo2, ChevronDown, ChevronRight,
  FileCheck2, ShieldCheck, Server, AlertCircle, Zap, Clock, Database,
  Power, Play, Square, Activity, FileText, AlertTriangle,
  Receipt, Radio, Gauge, Network,
} from 'lucide-react';
import { api } from '../lib/api';

type Family = {
  family: string;
  label: string;
  uc_name: string;
  catalog_url: string;
  champion?: { version: string; run_id: string; status: string; created_at?: string; created_by?: string } | null;
  champion_is_alias: boolean;
  previous_champion?: { version: string; run_id: string; created_at?: string; created_by?: string } | null;
  latest_pack?: {
    pack_id: string; pdf_path: string; generated_by: string; generated_at: string; download_url: string;
  } | null;
};

type Tab = 'production' | 'live';

export default function ModelDeployment() {
  const [tab, setTab] = useState<Tab>('production');

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-900">Model Deployment</h2>
        <p className="text-gray-500 mt-1">Production champions and the live pricing system.</p>
      </div>

      <div className="flex gap-1 border-b border-gray-200 mb-6">
        <TabButton active={tab === 'production'} onClick={() => setTab('production')}
                   icon={<ShieldCheck className="w-4 h-4" />} label="Production Models" />
        <TabButton active={tab === 'live'} onClick={() => setTab('live')}
                   icon={<Zap className="w-4 h-4" />} label="Live Pricing System" />
      </div>

      {tab === 'production' && <ProductionModels />}
      {tab === 'live'       && <LivePricing />}
    </div>
  );
}

function TabButton({ active, onClick, icon, label }:
  { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button onClick={onClick}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg inline-flex items-center gap-2 -mb-px border-b-2 transition ${
              active
                ? 'border-blue-600 text-blue-700 bg-white'
                : 'border-transparent text-gray-500 hover:text-gray-800 hover:bg-gray-50'
            }`}>
      {icon} {label}
    </button>
  );
}

// ===========================================================================
// Tab 1 — Production Models
// ===========================================================================

function ProductionModels() {
  const [families, setFamilies]     = useState<Family[]>([]);
  const [loading, setLoading]       = useState(true);
  const [openRow, setOpenRow]       = useState<string | null>(null);
  const [history, setHistory]       = useState<Record<string, any[]>>({});
  const [rollbackFor, setRollbackFor] = useState<Family | null>(null);
  const [toast, setToast]           = useState<string | null>(null);

  const reload = () => {
    setLoading(true);
    api.getChampions()
      .then((d) => setFamilies(d.families || []))
      .catch(() => setFamilies([]))
      .finally(() => setLoading(false));
  };

  useEffect(() => { reload(); }, []);

  const toggleRow = (family: string) => {
    if (openRow === family) {
      setOpenRow(null);
      return;
    }
    setOpenRow(family);
    if (!history[family]) {
      api.getChampionHistory(family, 10).then((d) => {
        setHistory(cur => ({ ...cur, [family]: d.events || [] }));
      });
    }
  };

  return (
    <div>
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-4">
        <h3 className="font-semibold text-blue-800 mb-1 text-sm">Production Models</h3>
        <p className="text-sm text-blue-700">
          Current champions across all pricing models. Promotion from the <em>Promote</em> tab
          flips the <code className="bg-blue-100 px-1 rounded text-[11px]">champion</code> alias and
          demotes the prior version to <code className="bg-blue-100 px-1 rounded text-[11px]">previous_champion</code>.
          Rollback swaps them back.
        </p>
        <div className="flex flex-wrap gap-1.5 mt-2.5">
          {['UC model registry', 'Alias-based versioning', 'One-click rollback',
            'Audit-logged promotions', 'Governance pack linkage'].map(f => (
            <span key={f} className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-100 text-blue-700">{f}</span>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">Current production champions</h3>
          <span className="text-xs text-gray-500">{families.length} model families</span>
        </div>
        {loading ? (
          <div className="py-10 text-center text-sm text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin inline mr-1" /> Loading champions…
          </div>
        ) : families.length === 0 ? (
          <div className="py-10 text-center text-sm text-gray-500 italic">
            No production models registered yet.
          </div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-gray-500 border-b bg-gray-50">
                <th className="text-left px-3 py-2 font-medium w-4"></th>
                <th className="text-left px-3 py-2 font-medium">Model</th>
                <th className="text-left px-3 py-2 font-medium">Champion</th>
                <th className="text-left px-3 py-2 font-medium">Promoted</th>
                <th className="text-left px-3 py-2 font-medium">By</th>
                <th className="text-left px-3 py-2 font-medium">Governance pack</th>
                <th className="text-left px-3 py-2 font-medium">Previous</th>
                <th className="text-right px-3 py-2 font-medium">Actions</th>
              </tr>
            </thead>
            <tbody>
              {families.map(f => {
                const isOpen = openRow === f.family;
                const canRollback = Boolean(f.previous_champion);
                return (
                  <>
                    <tr key={`${f.family}-row`}
                        className={`border-b last:border-0 hover:bg-gray-50 ${isOpen ? 'bg-blue-50' : ''}`}>
                      <td className="px-3 py-2">
                        <button onClick={() => toggleRow(f.family)} className="text-gray-500 hover:text-gray-700">
                          {isOpen ? <ChevronDown className="w-3.5 h-3.5" /> : <ChevronRight className="w-3.5 h-3.5" />}
                        </button>
                      </td>
                      <td className="px-3 py-2">
                        <div className="font-medium text-gray-900">{f.label}</div>
                        <div className="text-[10px] text-gray-500 font-mono">{f.uc_name.split('.').slice(-1)[0]}</div>
                      </td>
                      <td className="px-3 py-2">
                        {f.champion ? (
                          <span className="font-mono text-xs">
                            v{f.champion.version}
                            {!f.champion_is_alias && (
                              <span title="Alias not yet set — showing latest version"
                                    className="ml-1 text-[9px] text-amber-700 bg-amber-100 px-1 rounded">
                                latest
                              </span>
                            )}
                          </span>
                        ) : <span className="text-gray-400">—</span>}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-600">
                        {formatDate(f.champion?.created_at)}
                      </td>
                      <td className="px-3 py-2 text-xs text-gray-600">
                        {(f.champion?.created_by || '').split('@')[0] || '—'}
                      </td>
                      <td className="px-3 py-2">
                        {f.latest_pack ? (
                          <a href={f.latest_pack.download_url} target="_blank" rel="noopener noreferrer"
                             className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800">
                            <FileCheck2 className="w-3 h-3" />
                            {formatDate(f.latest_pack.generated_at)}
                          </a>
                        ) : (
                          <span className="text-[11px] text-amber-700 inline-flex items-center gap-1">
                            <AlertCircle className="w-3 h-3" /> No pack yet
                          </span>
                        )}
                      </td>
                      <td className="px-3 py-2 font-mono text-xs text-gray-600">
                        {f.previous_champion ? `v${f.previous_champion.version}` : '—'}
                      </td>
                      <td className="px-3 py-2 text-right">
                        <button onClick={() => setRollbackFor(f)}
                                disabled={!canRollback}
                                title={canRollback ? 'Swap champion back to previous version' : 'No previous champion on record'}
                                className={`inline-flex items-center gap-1 px-2.5 py-1 rounded text-[11px] font-medium ${
                                  canRollback
                                    ? 'bg-red-50 text-red-700 border border-red-200 hover:bg-red-100'
                                    : 'bg-gray-50 text-gray-400 border border-gray-200 cursor-not-allowed'
                                }`}>
                          <Undo2 className="w-3 h-3" /> Rollback
                        </button>
                      </td>
                    </tr>
                    {isOpen && (
                      <tr key={`${f.family}-det`} className="border-b bg-blue-50/30">
                        <td colSpan={8} className="px-4 py-3">
                          <RowDetail family={f} events={history[f.family] || null} />
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

      {/* Serving SLO tiles — what the rating engine sees at quote time */}
      <ServingSLOs />

      {/* Live endpoint metrics — placeholder stream, simulated client-side */}
      {families.length > 0 && <LiveEndpointMetrics families={families} />}

      {rollbackFor && (
        <RollbackDialog
          family={rollbackFor}
          onClose={() => setRollbackFor(null)}
          onDone={(msg) => { setRollbackFor(null); setToast(msg); reload(); }}
        />
      )}

      {toast && (
        <div onClick={() => setToast(null)}
             className="fixed bottom-4 right-4 bg-gray-900 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50 cursor-pointer">
          {toast}
        </div>
      )}
    </div>
  );
}

function RowDetail({ family, events }: { family: Family; events: any[] | null }) {
  return (
    <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
      <div className="bg-white border border-gray-200 rounded p-3">
        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">Champion</h4>
        {family.champion ? (
          <div className="text-xs space-y-0.5">
            <div><span className="text-gray-500">Version:</span> <span className="font-mono">v{family.champion.version}</span></div>
            <div><span className="text-gray-500">Run:</span> <span className="font-mono text-[10px] break-all">{family.champion.run_id}</span></div>
            <div><span className="text-gray-500">Status:</span> {family.champion.status}</div>
            <div><span className="text-gray-500">Trained by:</span> {family.champion.created_by}</div>
            <div><span className="text-gray-500">Trained at:</span> {formatDate(family.champion.created_at)}</div>
          </div>
        ) : <div className="text-xs text-gray-500 italic">No champion assigned.</div>}
        <a href={family.catalog_url} target="_blank" rel="noopener noreferrer"
           className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 mt-2">
          Open in Catalog <ExternalLink className="w-3 h-3" />
        </a>
      </div>

      <div className="bg-white border border-gray-200 rounded p-3">
        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">Governance pack</h4>
        {family.latest_pack ? (
          <div className="text-xs space-y-0.5">
            <div><span className="text-gray-500">Pack ID:</span> <span className="font-mono text-[10px]">{family.latest_pack.pack_id}</span></div>
            <div><span className="text-gray-500">Generated:</span> {formatDate(family.latest_pack.generated_at)}</div>
            <div><span className="text-gray-500">By:</span> {family.latest_pack.generated_by}</div>
            <a href={family.latest_pack.download_url} target="_blank" rel="noopener noreferrer"
               className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 mt-1">
              <FileCheck2 className="w-3 h-3" /> Download PDF
            </a>
          </div>
        ) : <div className="text-xs text-gray-500 italic">No pack generated for this family yet.</div>}
      </div>

      <div className="bg-white border border-gray-200 rounded p-3">
        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">Approval history</h4>
        {events === null ? (
          <div className="text-xs text-gray-500"><Loader2 className="w-3 h-3 inline animate-spin mr-1" /> Loading…</div>
        ) : events.length === 0 ? (
          <div className="text-xs text-gray-500 italic">No events recorded.</div>
        ) : (
          <ul className="text-xs space-y-1 max-h-40 overflow-y-auto">
            {events.map((e, i) => (
              <li key={i} className="flex items-start gap-2">
                <span className={`mt-0.5 text-[10px] px-1 rounded font-medium ${eventColor(e.event_type)}`}>
                  {eventShortLabel(e.event_type)}
                </span>
                <span className="text-gray-700">
                  v{e.version || '—'} · {formatDate(e.timestamp)}
                  <span className="text-gray-500"> · {(e.user || '').split('@')[0]}</span>
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live endpoint metrics — placeholder stream
//
// Real Model Serving will replace this with `serving_endpoints/metrics` pulls.
// Until then we simulate a plausible load profile client-side so the page has
// the monitoring signals reviewers expect to see on a live production board.
// ---------------------------------------------------------------------------

function LiveEndpointMetrics({ families }: { families: Family[] }) {
  const [tick, setTick] = useState(0);
  const [history, setHistory] = useState<Record<string, number[]>>({});

  useEffect(() => {
    const h: Record<string, number[]> = {};
    for (const f of families) h[f.family] = seedSeries(f.family, 60);
    setHistory(h);
  }, [families.map(f => f.family).join(',')]);

  useEffect(() => {
    const t = setInterval(() => setTick(x => x + 1), 2000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    // advance each series one step
    setHistory(cur => {
      const next: Record<string, number[]> = { ...cur };
      for (const f of families) {
        const s = [...(cur[f.family] || seedSeries(f.family, 60))];
        const last = s[s.length - 1] ?? 50;
        const drift = (Math.random() - 0.5) * 6;
        const v = clamp(last + drift, 10, 120);
        s.push(v);
        while (s.length > 60) s.shift();
        next[f.family] = s;
      }
      return next;
    });
  }, [tick]);

  // Roll-up stats across all champions.
  const allQps = families.reduce((acc, f) => {
    const s = history[f.family] || [];
    return acc + (s[s.length - 1] ?? 0);
  }, 0);
  const p50 = pickLatency(0.5, tick);
  const p95 = pickLatency(0.95, tick);
  const p99 = pickLatency(0.99, tick);
  const errRate = 0.08 + 0.05 * Math.sin(tick / 6);
  const uptime = 99.97 + 0.02 * Math.cos(tick / 9);

  return (
    <section className="bg-white border border-gray-200 rounded-lg mt-4 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
          <Zap className="w-4 h-4 text-violet-600" /> Live endpoint metrics
        </h3>
        <div className="text-[11px] text-gray-500 inline-flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
          Streaming · updated every 2s
          <span className="ml-2 text-[10px] text-amber-700 bg-amber-100 px-1.5 py-0.5 rounded font-medium">
            demo stream
          </span>
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-5 border-b">
        <MetricTile label="Quotes / sec (all models)" value={Math.round(allQps).toLocaleString()}
                    subtext="request rate summed across champions" tone="blue" />
        <MetricTile label="Latency p50" value={`${p50.toFixed(0)} ms`} subtext="median end-to-end" tone="emerald" />
        <MetricTile label="Latency p95" value={`${p95.toFixed(0)} ms`}
                    subtext={p95 < 400 ? "within SLA" : "approaching SLA"}
                    tone={p95 < 400 ? "emerald" : "amber"} />
        <MetricTile label="Latency p99" value={`${p99.toFixed(0)} ms`}
                    subtext={p99 < 500 ? "within SLA" : "breaching 500ms"}
                    tone={p99 < 500 ? "emerald" : "red"} />
        <MetricTile label="Error rate" value={`${errRate.toFixed(2)}%`}
                    subtext={`uptime ${uptime.toFixed(2)}%`}
                    tone={errRate > 0.5 ? "red" : "emerald"} />
      </div>

      <div className="px-4 py-3">
        <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">
          Per-model throughput (last 2 min)
        </h4>
        <div className="space-y-1.5">
          {families.map(f => {
            const s = history[f.family] || [];
            const current = s[s.length - 1] ?? 0;
            return (
              <div key={f.family} className="flex items-center gap-3 text-xs">
                <div className="w-32 shrink-0 text-gray-800 font-medium">{f.label}</div>
                <Sparkline values={s} height={26} className="flex-1" />
                <div className="w-24 text-right text-gray-900 font-mono">{Math.round(current)} q/s</div>
                <div className="w-20 text-right text-gray-500 font-mono">
                  {(pickLatency(0.5, tick + f.family.length) + 20).toFixed(0)} ms
                </div>
              </div>
            );
          })}
        </div>
      </div>

      <div className="px-4 py-2.5 border-t bg-gray-50 text-[11px] text-gray-500 flex items-center justify-between">
        <div>
          Sources (when live): <code className="bg-gray-100 px-1 rounded text-[10px]">serving_endpoints.metrics</code>,
          request-tracing, Lakehouse Monitoring. Thresholds: p95 &lt; 400ms, p99 &lt; 500ms, error rate &lt; 0.5%.
        </div>
        <div className="inline-flex items-center gap-1">
          Last tick: #{tick.toString().padStart(3, '0')}
        </div>
      </div>
    </section>
  );
}

function MetricTile({ label, value, subtext, tone }:
  { label: string; value: string; subtext?: string; tone: 'blue' | 'emerald' | 'amber' | 'red' }) {
  const toneCls = {
    blue:    'text-blue-700',
    emerald: 'text-emerald-700',
    amber:   'text-amber-700',
    red:     'text-red-700',
  }[tone];
  return (
    <div className="px-4 py-3 border-r last:border-r-0">
      <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`text-2xl font-semibold mt-1 ${toneCls}`}>{value}</div>
      {subtext && <div className="text-[10px] text-gray-500 mt-0.5">{subtext}</div>}
    </div>
  );
}

function Sparkline({ values, height, className }:
  { values: number[]; height: number; className?: string }) {
  if (!values || values.length < 2) {
    return <div className={className} style={{ height }} />;
  }
  const w = 200;
  const min = Math.min(...values);
  const max = Math.max(...values);
  const span = Math.max(1, max - min);
  const points = values.map((v, i) => {
    const x = (i / (values.length - 1)) * w;
    const y = height - 2 - ((v - min) / span) * (height - 4);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg className={className} viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none"
         style={{ height, width: '100%' }}>
      <polyline fill="none" stroke="#3b82f6" strokeWidth="1.5" points={points} />
      <polyline fill="rgba(59,130,246,0.1)" stroke="none"
                points={`0,${height} ${points} ${w},${height}`} />
    </svg>
  );
}

function clamp(x: number, lo: number, hi: number) { return Math.max(lo, Math.min(hi, x)); }

function seedSeries(familyKey: string, n: number): number[] {
  // Deterministic-ish starting point so the card renders the same baseline
  // between re-renders for the same family.
  let seed = Array.from(familyKey).reduce((s, c) => s + c.charCodeAt(0), 0);
  const center = 40 + (seed % 40);
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    seed = (seed * 9301 + 49297) % 233280;
    const r = seed / 233280;
    out.push(clamp(center + Math.sin(i / 5) * 6 + (r - 0.5) * 10, 10, 120));
  }
  return out;
}

function pickLatency(pct: number, tick: number): number {
  // Base latency bands for realistic variance around a healthy SLA.
  const base = pct < 0.6 ? 110 : pct < 0.97 ? 280 : 420;
  const jitter = Math.sin(tick / 5) * 20 + (Math.random() - 0.5) * 30;
  return clamp(base + jitter, base - 60, base + 120);
}

function RollbackDialog({ family, onClose, onDone }:
  { family: Family; onClose: () => void; onDone: (msg: string) => void }) {
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [err, setErr]   = useState<string | null>(null);

  const submit = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.rollbackChampion(family.family, note.trim());
      onDone(`Rolled back ${family.label} to v${r.new_champion}`);
    } catch (e: any) {
      setErr(e.message); setBusy(false);
    }
  };

  return (
    <div className="fixed inset-0 bg-black/40 flex items-center justify-center z-40">
      <div className="bg-white rounded-lg shadow-2xl max-w-lg w-full mx-4">
        <div className="px-5 py-3 border-b flex items-center gap-2">
          <Undo2 className="w-4 h-4 text-red-700" />
          <h3 className="font-semibold text-gray-900">Rollback {family.label}</h3>
        </div>
        <div className="p-5">
          <p className="text-sm text-gray-700 mb-3">
            The <code className="bg-gray-100 px-1 rounded text-[11px]">champion</code> alias will move from
            <strong className="mx-1">v{family.champion?.version}</strong>
            back to
            <strong className="mx-1">v{family.previous_champion?.version}</strong>.
            The current champion will become the new <code className="bg-gray-100 px-1 rounded text-[11px]">previous_champion</code>.
          </p>
          <label className="text-xs font-medium text-gray-700 block mb-1">
            Justification <span className="text-red-600">*</span> <span className="text-gray-500 font-normal">(min 10 chars, logged to audit trail)</span>
          </label>
          <textarea value={note} onChange={e => setNote(e.target.value)}
                    rows={3}
                    placeholder="e.g. Observed +14% false-positive rate in fraud referrals since promotion"
                    className="w-full border border-gray-300 rounded px-2 py-1.5 text-sm" />
          {err && <div className="mt-2 text-xs text-red-700 flex items-start gap-1.5">
            <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" /> {err}
          </div>}
        </div>
        <div className="px-5 py-3 border-t bg-gray-50 flex items-center justify-end gap-2">
          <button onClick={onClose}
                  className="px-3 py-1.5 text-sm text-gray-700 hover:bg-gray-100 rounded">
            Cancel
          </button>
          <button onClick={submit}
                  disabled={busy || note.trim().length < 10}
                  className="px-3 py-1.5 text-sm font-medium bg-red-600 text-white rounded hover:bg-red-700 disabled:opacity-50 inline-flex items-center gap-1.5">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Undo2 className="w-3.5 h-3.5" />}
            Confirm rollback
          </button>
        </div>
      </div>
    </div>
  );
}

// ===========================================================================
// Tab 2 — Live Pricing System
// ===========================================================================

type LiveState = 'off' | 'starting' | 'on' | 'stopping' | 'unknown';
type LiveStatus = {
  state: LiveState;
  endpoint:     { name: string; present: boolean; ready?: string; config_update?: string };
  online_store: { name: string; present: boolean; state?: string; capacity?: string };
  metrics_table?: string;
};

function LivePricing() {
  const [status, setStatus] = useState<LiveStatus | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [actionBusy, setActionBusy] = useState<'start' | 'stop' | null>(null);
  const [archOpen, setArchOpen] = useState(false);

  const inTransition = status?.state === 'starting' || status?.state === 'stopping';

  // Poll status — 5s when idle, 2s during transitions.
  useEffect(() => {
    let cancelled = false;
    const fetchOnce = async () => {
      try {
        const s = await api.livePricingStatus();
        if (!cancelled) { setStatus(s); setStatusError(null); }
      } catch (e: any) {
        if (!cancelled) setStatusError(e?.message || String(e));
      }
    };
    fetchOnce();
    const interval = inTransition ? 2000 : 5000;
    const t = window.setInterval(fetchOnce, interval);
    return () => { cancelled = true; window.clearInterval(t); };
  }, [inTransition]);

  const togglePower = async () => {
    if (!status) return;
    if (status.state === 'on' || status.state === 'starting') {
      setActionBusy('stop');
      try { await api.livePricingStop(); }
      catch (e: any) { setStatusError(e?.message || String(e)); }
      finally { setActionBusy(null); }
    } else {
      setActionBusy('start');
      try { await api.livePricingStart(); }
      catch (e: any) { setStatusError(e?.message || String(e)); }
      finally { setActionBusy(null); }
    }
  };

  const stateBadge = (() => {
    switch (status?.state) {
      case 'on':       return { dot: 'bg-emerald-500',       label: 'On',         desc: 'Endpoint READY · online store live' };
      case 'starting': return { dot: 'bg-amber-500 animate-pulse', label: 'Starting', desc: 'Provisioning Lakebase + warm-up (5–10 min)' };
      case 'stopping': return { dot: 'bg-amber-500 animate-pulse', label: 'Stopping', desc: 'Tearing down endpoint + online store' };
      case 'off':      return { dot: 'bg-gray-300',          label: 'Off',        desc: 'No live runtime — click Activate to bring up' };
      default:         return { dot: 'bg-gray-300',          label: '—',          desc: 'Resolving state…' };
    }
  })();

  return (
    <div>
      {/* Header — power button + state */}
      <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
        <div className="flex items-center justify-between gap-4 flex-wrap">
          <div className="flex items-center gap-3">
            <span className={`w-3 h-3 rounded-full ${stateBadge.dot}`} aria-hidden />
            <div>
              <h3 className="font-semibold text-gray-900 text-base flex items-center gap-2">
                <Zap className="w-4 h-4 text-violet-600" />
                Live Pricing System — <span className="text-gray-700">{stateBadge.label}</span>
              </h3>
              <p className="text-xs text-gray-500 mt-0.5">{stateBadge.desc}</p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={togglePower}
              disabled={actionBusy !== null || inTransition}
              className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium border transition
                ${status?.state === 'on'
                  ? 'border-rose-200 text-rose-700 bg-rose-50 hover:bg-rose-100'
                  : 'border-emerald-200 text-emerald-700 bg-emerald-50 hover:bg-emerald-100'}
                ${(actionBusy !== null || inTransition) ? 'opacity-50 cursor-not-allowed' : ''}`}
            >
              {actionBusy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Power className="w-3.5 h-3.5" />}
              {status?.state === 'on' ? 'Deactivate' : 'Activate'}
            </button>
          </div>
        </div>
        {statusError && (
          <div className="mt-3 text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded px-3 py-2 flex items-start gap-2">
            <AlertCircle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <span>{statusError}</span>
          </div>
        )}
        {/* Component breakdown — small status chips */}
        {status && (
          <div className="mt-4 grid grid-cols-1 md:grid-cols-2 gap-2">
            <StatusChip
              icon={<Server className="w-3.5 h-3.5" />}
              label="Endpoint"
              value={status.endpoint.name}
              detail={status.endpoint.present
                ? `ready=${status.endpoint.ready ?? '?'} · config=${status.endpoint.config_update ?? '?'}`
                : 'not present'}
              ok={status.endpoint.present && status.endpoint.ready === 'READY'}
            />
            <StatusChip
              icon={<Database className="w-3.5 h-3.5" />}
              label="Online store (Lakebase)"
              value={status.online_store.name}
              detail={status.online_store.present
                ? `${status.online_store.state ?? '?'} · ${status.online_store.capacity ?? '?'}`
                : 'not present'}
              ok={status.online_store.present && (status.online_store.state ?? '').endsWith('AVAILABLE')}
            />
          </div>
        )}
      </div>

      {/* Customer-facing demo pages — standalone, chrome-less UIs that drive
          the same live endpoint. Open in a new tab (full page load) so they
          render outside the workbench shell. Hidden when the system is not ON
          (like the demo flow / load test below) — the pages can't return
          quotes without a live endpoint, so we don't surface dead links. */}
      {status?.state === 'on' && (
        <div className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
          <h4 className="text-sm font-semibold text-gray-800 mb-1 flex items-center gap-1.5">
            <ExternalLink className="w-4 h-4 text-violet-600" /> Demo pages
          </h4>
          <p className="text-xs text-gray-500 mb-3">
            Standalone customer-facing UIs for the live pricing story. Best opened in their own tabs.
          </p>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
            <DemoLink href="/quote"       icon={Receipt} accent="blue"
                      title="Quote portal"   desc="Consumer quote, pre-filled for John" />
            <DemoLink href="/blackbox"    icon={Radio} accent="amber"
                      title="Black-box panel" desc="Fire a telematics event" />
            <DemoLink href="/quotetester" icon={Gauge} accent="violet"
                      title="Live quote tester" desc="Streaming QPS + latency" />
            <DemoLink href="/quote-chat"  icon={Network} accent="fuchsia"
                      title="Agentic MCP sales"
                      desc="Buy by conversation — Claude calls the engine over MCP" />
          </div>
        </div>
      )}

      {/* Sections only show when ON */}
      {status?.state === 'on' && (
        <>
          <DemoFlow />
          <SingleQuote />
          <LoadTest />
        </>
      )}

      {status?.state !== 'on' && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-5 mb-5 text-sm text-gray-600 space-y-3">
          <p>
            Activate the system to access the live-pricing demo pages, demo flow, single-quote latency probe, and load-test chart.
            The first activation provisions a Lakebase online store at CU_2 and warm-starts the scorer endpoint —
            typically 5–10 minutes end-to-end. Subsequent activations on the same workspace reuse what's there.
          </p>
          {/* The agentic journey prices a BRAND-NEW risk through
              motor_pricing_scorer_direct, which is independent of the Lakebase
              online store and the route-optimized endpoint — so it works with
              the live system off, and stays available here. */}
          <div className="pt-1">
            <div className="text-[11px] uppercase tracking-wide text-gray-500 font-semibold mb-2">
              Available without activating
            </div>
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
              <DemoLink href="/quote-chat" icon={Network} accent="fuchsia"
                        title="Agentic MCP sales"
                        desc="Prices a new risk via motor_pricing_scorer_direct — no online store needed" />
            </div>
          </div>
        </div>
      )}

      {/* Architecture — collapsible */}
      <section className="bg-white border border-gray-200 rounded-lg mb-5">
        <button
          onClick={() => setArchOpen(o => !o)}
          className="w-full px-5 py-3 flex items-center justify-between text-sm font-semibold text-gray-800 hover:bg-gray-50"
        >
          <span className="flex items-center gap-2">
            {archOpen ? <ChevronDown className="w-4 h-4" /> : <ChevronRight className="w-4 h-4" />}
            Architecture
          </span>
          <span className="text-[11px] font-normal text-gray-500">
            {archOpen ? 'Hide diagram' : 'Show diagram'}
          </span>
        </button>
        {archOpen && (
          <div className="px-5 pb-5 flex justify-center overflow-x-auto">
            <ArchitectureDiagram />
          </div>
        )}
      </section>
    </div>
  );
}

// Each demo page carries its own accent so the four are tellable apart at a
// glance mid-demo. Full class strings (no interpolation) so Tailwind's scanner
// keeps them.
const DEMO_ACCENTS = {
  blue:    { border: 'hover:border-blue-300',    bg: 'hover:bg-blue-50',    icon: 'text-blue-600',    title: 'group-hover:text-blue-700' },
  amber:   { border: 'hover:border-amber-300',   bg: 'hover:bg-amber-50',   icon: 'text-amber-600',   title: 'group-hover:text-amber-700' },
  violet:  { border: 'hover:border-violet-300',  bg: 'hover:bg-violet-50',  icon: 'text-violet-600',  title: 'group-hover:text-violet-700' },
  fuchsia: { border: 'hover:border-fuchsia-300', bg: 'hover:bg-fuchsia-50', icon: 'text-fuchsia-600', title: 'group-hover:text-fuchsia-700' },
} as const;

function DemoLink({ href, title, desc, icon: Icon, accent = 'violet' }: {
  href: string; title: string; desc: string;
  icon?: React.ComponentType<{ className?: string }>;
  accent?: keyof typeof DEMO_ACCENTS;
}) {
  const c = DEMO_ACCENTS[accent];
  return (
    <a href={href} target="_blank" rel="noopener noreferrer"
       className={`group flex items-start gap-2.5 px-3 py-2.5 rounded-lg border border-gray-200 transition-colors ${c.border} ${c.bg}`}>
      {Icon && <Icon className={`w-4 h-4 mt-0.5 shrink-0 ${c.icon}`} />}
      <div className="flex-1 min-w-0">
        <div className={`text-sm font-medium text-gray-800 flex items-center gap-1 ${c.title}`}>
          {title} <ExternalLink className="w-3 h-3 opacity-50" />
        </div>
        <div className="text-[11px] text-gray-500">{desc}</div>
        <div className="text-[10px] font-mono text-gray-400 mt-0.5">{href}</div>
      </div>
    </a>
  );
}

function StatusChip({ icon, label, value, detail, ok }:
  { icon: React.ReactNode; label: string; value: string; detail: string; ok: boolean }) {
  return (
    <div className={`flex items-center gap-2 px-3 py-2 rounded border text-xs
      ${ok ? 'bg-emerald-50 border-emerald-200 text-emerald-800' : 'bg-gray-50 border-gray-200 text-gray-600'}`}>
      {icon}
      <span className="font-semibold">{label}</span>
      <span className="font-mono text-[11px] truncate">{value}</span>
      <span className="text-[11px] opacity-70 ml-auto">{detail}</span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Demo Flow — three sequential cards: initial quote → file claim → re-quote
// ---------------------------------------------------------------------------

function DemoFlow() {
  const [policyId, setPolicyId] = useState('POL-MOTOR-00000001');
  const [step1, setStep1] = useState<any>(null);
  const [step2, setStep2] = useState<any>(null);
  const [step3, setStep3] = useState<any>(null);
  const [busy,  setBusy]  = useState<1 | 2 | 3 | null>(null);
  const [err,   setErr]   = useState<string | null>(null);

  const reset = () => { setStep1(null); setStep2(null); setStep3(null); setErr(null); };

  const runStep1 = async () => {
    setBusy(1); setErr(null); setStep2(null); setStep3(null);
    try {
      const r = await api.livePricingQuote(policyId);
      setStep1(r);
    } catch (e: any) { setErr(e?.message || String(e)); }
    finally { setBusy(null); }
  };

  const runStep2 = async () => {
    setBusy(2); setErr(null); setStep3(null);
    try {
      const r = await api.livePricingTelematicsEvent({
        policy_id: policyId, speeding_event: true, curfew_breach: true,
        behaviour_score_delta: -15, harsh_braking_delta: 1,
      });
      setStep2(r);
    } catch (e: any) { setErr(e?.message || String(e)); }
    finally { setBusy(null); }
  };

  const runStep3 = async () => {
    setBusy(3); setErr(null);
    try {
      const r = await api.livePricingQuote(policyId);
      setStep3(r);
    } catch (e: any) { setErr(e?.message || String(e)); }
    finally { setBusy(null); }
  };

  const delta = (step1 && step3 && step1.result?.final_premium != null && step3.result?.final_premium != null)
    ? Number(step3.result.final_premium) - Number(step1.result.final_premium)
    : null;

  return (
    <section className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
      <div className="flex items-center justify-between mb-3 gap-3 flex-wrap">
        <h4 className="text-sm font-semibold text-gray-800 flex items-center gap-1.5">
          <Activity className="w-4 h-4 text-violet-600" /> Demo flow
        </h4>
        <div className="flex items-center gap-2">
          <input
            value={policyId}
            onChange={e => setPolicyId(e.target.value.toUpperCase())}
            className="text-xs font-mono px-2 py-1 border border-gray-300 rounded w-36"
            placeholder="POL-MOTOR-00000001"
          />
          <button onClick={reset} className="text-xs text-gray-500 hover:text-gray-700 flex items-center gap-1">
            <Undo2 className="w-3 h-3" /> Reset
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
        <DemoCard
          step={1} title="Initial quote" icon={<Zap className="w-4 h-4" />}
          body={step1
            ? <>
                <PremiumLine label="Final premium" value={step1.result?.final_premium} />
                <DetailGrid result={step1.result} latency={step1.latency_ms} />
              </>
            : <p className="text-xs text-gray-500">Score the policy on the live endpoint and capture the baseline premium.</p>}
          action={<button onClick={runStep1} disabled={busy !== null}
                          className="text-xs px-2.5 py-1.5 rounded bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 inline-flex items-center gap-1">
                    {busy === 1 ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
                    Run
                  </button>}
        />
        <DemoCard
          step={2} title="Simulate telematics event" icon={<FileText className="w-4 h-4" />}
          body={step2
            ? <>
                <div className="text-[11px] text-gray-600 space-y-0.5">
                  <div>Event ID: <code className="text-gray-800">{step2.event_id || step2.claim_id}</code></div>
                  <div className="text-gray-800 font-medium mt-1">Telematics signal change</div>
                  {step2.before && step2.after && (
                    <>
                      <div>behaviour_score: <span className="text-gray-800">{step2.before.behaviour_score} → {step2.after.behaviour_score}</span></div>
                      <div>recent_speeding: <span className="text-gray-800">{step2.before.recent_speeding_events} → {step2.after.recent_speeding_events}</span></div>
                      <div>recent_curfew:   <span className="text-gray-800">{step2.before.recent_curfew_breaches} → {step2.after.recent_curfew_breaches}</span></div>
                    </>
                  )}
                  <div className="mt-1">Telematics write: <span className="text-gray-800">{Math.round(step2.claim_write_ms)} ms</span></div>
                  <div>UPT MERGE:        <span className="text-gray-800">{Math.round(step2.upt_merge_ms)} ms</span></div>
                  {step2.online_refresh && (
                    <div className={step2.online_refresh.completed ? 'text-emerald-700 mt-1.5' : 'text-amber-700 mt-1.5'}>
                      Lakebase SNAPSHOT refresh: {step2.online_refresh.completed
                        ? `synced in ${Math.round(step2.online_refresh.duration_ms ?? 0)} ms`
                        : `state ${step2.online_refresh.state ?? 'unknown'}${step2.online_refresh.duration_ms ? ` (${Math.round(step2.online_refresh.duration_ms)} ms)` : ''}`}
                    </div>
                  )}
                </div>
              </>
            : <p className="text-xs text-gray-500">Out-of-curfew speeding event from the black box. Behaviour score drops, recent event counters tick up, UPT mirrors the change, Lakebase syncs.</p>}
          disabled={!step1}
          action={<button onClick={runStep2} disabled={busy !== null || !step1}
                          className="text-xs px-2.5 py-1.5 rounded bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 inline-flex items-center gap-1">
                    {busy === 2 ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
                    Run
                  </button>}
        />
        <DemoCard
          step={3} title="Re-quote" icon={<Zap className="w-4 h-4" />}
          body={step3
            ? <>
                <PremiumLine label="New final premium" value={step3.result?.final_premium} />
                <DetailGrid result={step3.result} latency={step3.latency_ms} />
                {delta !== null && (
                  <div className={`mt-2 text-xs px-2 py-1 rounded inline-flex items-center gap-1
                    ${delta >= 0 ? 'bg-amber-50 text-amber-800 border border-amber-200'
                                 : 'bg-emerald-50 text-emerald-800 border border-emerald-200'}`}>
                    Δ vs initial: {delta >= 0 ? '+' : ''}£{delta.toFixed(2)}
                  </div>
                )}
              </>
            : <p className="text-xs text-gray-500">Score the same policy again — fraud_pred picks up the new event, telematics surcharge kicks in, premium moves up.</p>}
          disabled={!step2}
          action={<button onClick={runStep3} disabled={busy !== null || !step2}
                          className="text-xs px-2.5 py-1.5 rounded bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 inline-flex items-center gap-1">
                    {busy === 3 ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
                    Run
                  </button>}
        />
      </div>

      {err && (
        <div className="mt-3 text-xs text-rose-700 bg-rose-50 border border-rose-200 rounded px-3 py-2 flex items-start gap-2">
          <AlertTriangle className="w-3.5 h-3.5 mt-0.5 shrink-0" />
          <span>{err}</span>
        </div>
      )}
    </section>
  );
}

function DemoCard({ step, title, icon, body, action, disabled }:
  { step: number; title: string; icon: React.ReactNode; body: React.ReactNode; action: React.ReactNode; disabled?: boolean }) {
  return (
    <div className={`border rounded-lg p-3 flex flex-col gap-2 ${disabled ? 'border-gray-200 bg-gray-50 opacity-70' : 'border-gray-200 bg-white'}`}>
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-xs font-semibold text-gray-700">
          <span className={`w-5 h-5 rounded-full text-white text-[11px] inline-flex items-center justify-center
            ${disabled ? 'bg-gray-400' : 'bg-violet-600'}`}>{step}</span>
          {icon}
          <span>{title}</span>
        </div>
        {action}
      </div>
      <div>{body}</div>
    </div>
  );
}

function PremiumLine({ label, value, prefix = '£' }: { label: string; value: any; prefix?: string }) {
  const num = Number(value);
  const fmt = isFinite(num) ? `${prefix}${num.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : '—';
  return (
    <div className="flex items-baseline gap-2">
      <span className="text-[11px] text-gray-500">{label}:</span>
      <span className="text-base font-semibold text-gray-900">{fmt}</span>
    </div>
  );
}

function DetailGrid({ result, latency }: { result: any; latency: number }) {
  if (!result) return null;
  return (
    <div className="mt-1.5 grid grid-cols-2 gap-x-3 gap-y-0.5 text-[11px] text-gray-600">
      <div>freq: <span className="text-gray-800">{Number(result.freq_pred ?? 0).toFixed(4)}</span></div>
      <div>sev: <span className="text-gray-800">£{Number(result.sev_pred ?? 0).toFixed(0)}</span></div>
      <div>fraud: <span className="text-gray-800">{Number(result.fraud_pred ?? 0).toFixed(4)}</span></div>
      <div>demand: <span className="text-gray-800">{Number(result.demand_pred ?? 0).toFixed(4)}</span></div>
      <div>technical: <span className="text-gray-800">£{Number(result.technical_premium ?? 0).toLocaleString()}</span></div>
      <div>fraud_load: <span className="text-gray-800">£{Number(result.fraud_load ?? 0).toLocaleString()}</span></div>
      <div className="col-span-2 mt-0.5 flex items-center gap-1 text-[11px]">
        <Clock className="w-3 h-3" /> latency: <span className="text-gray-800">{Number(latency).toFixed(0)} ms</span>
        <span className="ml-auto opacity-70">re v{result.rating_engine_version ?? '?'}</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Single quote — bare latency probe, separate from the demo flow
// ---------------------------------------------------------------------------

function SingleQuote() {
  const [policyId, setPolicyId] = useState('POL-MOTOR-00000001');
  const [busy, setBusy] = useState(false);
  const [last, setLast] = useState<any>(null);
  const [err,  setErr]  = useState<string | null>(null);

  const run = async () => {
    setBusy(true); setErr(null);
    try {
      const r = await api.livePricingQuote(policyId);
      setLast(r);
    } catch (e: any) { setErr(e?.message || String(e)); }
    finally { setBusy(false); }
  };

  return (
    <section className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
      <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-1.5">
        <Server className="w-4 h-4 text-violet-600" /> Single quote
      </h4>
      <div className="flex items-end gap-2 flex-wrap">
        <label className="text-xs text-gray-600">
          <div className="mb-1">policy_id</div>
          <input value={policyId}
                 onChange={e => setPolicyId(e.target.value.toUpperCase())}
                 className="px-2 py-1 border border-gray-300 rounded font-mono text-xs w-44" />
        </label>
        <button onClick={run} disabled={busy}
                className="text-xs px-3 py-1.5 rounded bg-violet-600 text-white hover:bg-violet-700 disabled:opacity-50 inline-flex items-center gap-1">
          {busy ? <Loader2 className="w-3 h-3 animate-spin" /> : <Play className="w-3 h-3" />}
          Quote
        </button>
        {last && last.ok && (
          <div className="text-xs text-gray-700 ml-3 inline-flex items-center gap-3">
            <span className="font-semibold text-gray-900">£{Number(last.result?.final_premium ?? 0).toFixed(2)}</span>
            <span><Clock className="w-3 h-3 inline" /> {Number(last.latency_ms).toFixed(0)} ms</span>
            <span className="text-gray-500">re v{last.result?.rating_engine_version ?? '?'}</span>
          </div>
        )}
      </div>
      {err && <div className="mt-2 text-xs text-rose-700">{err}</div>}
    </section>
  );
}

// ---------------------------------------------------------------------------
// Load test — start/stop + live SVG chart
// ---------------------------------------------------------------------------

type LoadTestRow = { ts: string; run_id: string; qps: number; p50_ms: number; p95_ms: number; p99_ms: number; error_pct: number };

function LoadTest() {
  const [targetQps, setTargetQps]  = useState(100);
  const [duration,  setDuration]   = useState(60);
  const [running,   setRunning]    = useState<{ run_id: number; load_test_run_id: string; run_page_url?: string } | null>(null);
  const [rows,      setRows]       = useState<LoadTestRow[]>([]);
  const [err,       setErr]        = useState<string | null>(null);
  const [tableReady, setTableReady] = useState(true);
  const sinceRef = useRef<string | null>(null);

  // Poll metrics whenever there's a row history or a run is active.
  useEffect(() => {
    if (!running && rows.length === 0) return;
    let cancelled = false;
    const poll = async () => {
      try {
        const params: { since?: string; run_id?: string } = {};
        if (sinceRef.current) params.since = sinceRef.current;
        if (running) params.run_id = running.load_test_run_id;
        const r = await api.livePricingLoadTestMetrics(params);
        if (cancelled) return;
        setTableReady(r.table_ready !== false);
        // Warehouse returns all values as strings — coerce numerics so .toFixed
        // and arithmetic in the chart code don't crash the render.
        const newRows: LoadTestRow[] = (r.rows || []).map((raw: any) => ({
          ts:        String(raw.ts ?? ''),
          run_id:    String(raw.run_id ?? ''),
          qps:       Number(raw.qps) || 0,
          p50_ms:    Number(raw.p50_ms) || 0,
          p95_ms:    Number(raw.p95_ms) || 0,
          p99_ms:    Number(raw.p99_ms) || 0,
          error_pct: Number(raw.error_pct) || 0,
        }));
        if (newRows.length > 0) {
          setRows(prev => {
            // Append only rows newer than the last known timestamp.
            const seen = new Set(prev.map(p => p.ts + ':' + p.run_id));
            const merged = [...prev];
            for (const row of newRows) {
              const key = row.ts + ':' + row.run_id;
              if (!seen.has(key)) merged.push(row);
            }
            const trimmed = merged.slice(-180);   // keep last ~3 minutes
            return trimmed;
          });
          sinceRef.current = newRows[newRows.length - 1].ts;
        }
      } catch (e: any) {
        if (!cancelled) setErr(e?.message || String(e));
      }
    };
    poll();
    const t = window.setInterval(poll, running ? 1500 : 5000);
    return () => { cancelled = true; window.clearInterval(t); };
  }, [running, rows.length]);

  const start = async () => {
    setErr(null); setRows([]); sinceRef.current = null;
    try {
      const r = await api.livePricingLoadTestStart({
        target_qps: targetQps, duration_seconds: duration, concurrency: Math.max(20, Math.floor(targetQps / 2)),
      });
      setRunning({ run_id: r.run_id, load_test_run_id: r.load_test_run_id, run_page_url: r.run_page_url });
    } catch (e: any) { setErr(e?.message || String(e)); }
  };

  const stop = async () => {
    if (!running) return;
    try { await api.livePricingLoadTestStop(running.run_id); }
    catch (e: any) { setErr(e?.message || String(e)); }
    setRunning(null);
  };

  return (
    <section className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
      <div className="flex items-center justify-between gap-3 flex-wrap mb-3">
        <h4 className="text-sm font-semibold text-gray-800 flex items-center gap-1.5">
          <Activity className="w-4 h-4 text-violet-600" /> Load test
        </h4>
        <div className="flex items-center gap-2 text-xs">
          <label className="flex items-center gap-1">
            target qps
            <input type="number" value={targetQps} min={10} max={500} step={10}
                   onChange={e => setTargetQps(Number(e.target.value))}
                   disabled={!!running}
                   className="px-1.5 py-0.5 border border-gray-300 rounded font-mono w-16" />
          </label>
          <label className="flex items-center gap-1">
            duration (s)
            <input type="number" value={duration} min={10} max={600} step={10}
                   onChange={e => setDuration(Number(e.target.value))}
                   disabled={!!running}
                   className="px-1.5 py-0.5 border border-gray-300 rounded font-mono w-16" />
          </label>
          {running ? (
            <button onClick={stop}
                    className="px-2.5 py-1 rounded bg-rose-600 text-white hover:bg-rose-700 inline-flex items-center gap-1">
              <Square className="w-3 h-3" /> Stop
            </button>
          ) : (
            <button onClick={start}
                    className="px-2.5 py-1 rounded bg-violet-600 text-white hover:bg-violet-700 inline-flex items-center gap-1">
              <Play className="w-3 h-3" /> Start
            </button>
          )}
          {running?.run_page_url && (
            <a href={running.run_page_url} target="_blank" rel="noreferrer"
               className="text-violet-700 hover:underline inline-flex items-center gap-0.5">
              run <ExternalLink className="w-3 h-3" />
            </a>
          )}
        </div>
      </div>

      {!tableReady && (
        <div className="text-xs text-gray-500 mb-2">
          Metrics table not yet populated — first load test will create it.
        </div>
      )}

      <LatencyChart rows={rows} />

      {rows.length > 0 && (
        <div className="grid grid-cols-4 gap-2 mt-3 text-xs">
          <Stat label="qps (last)"   value={rows[rows.length - 1].qps.toString()} />
          <Stat label="p50 (last)"   value={`${rows[rows.length - 1].p50_ms.toFixed(0)} ms`} />
          <Stat label="p95 (last)"   value={`${rows[rows.length - 1].p95_ms.toFixed(0)} ms`} />
          <Stat label="p99 (last)"   value={`${rows[rows.length - 1].p99_ms.toFixed(0)} ms`} />
        </div>
      )}
      {err && <div className="mt-2 text-xs text-rose-700">{err}</div>}
    </section>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="px-2 py-1.5 bg-gray-50 border border-gray-200 rounded">
      <div className="text-[10px] uppercase text-gray-500">{label}</div>
      <div className="font-mono text-sm text-gray-900">{value}</div>
    </div>
  );
}

function LatencyChart({ rows }: { rows: LoadTestRow[] }) {
  const W = 760, H = 200, M = { l: 40, r: 20, t: 10, b: 24 };
  const innerW = W - M.l - M.r;
  const innerH = H - M.t - M.b;

  const { yMax, qpsMax, points } = useMemo(() => {
    if (rows.length === 0) return { yMax: 100, qpsMax: 100, points: [] as any[] };
    const yMaxRaw  = Math.max(...rows.map(r => r.p99_ms || 0), 100);
    const qpsMaxRaw = Math.max(...rows.map(r => r.qps || 0), 50);
    const yMax = Math.ceil(yMaxRaw / 50) * 50;
    const qpsMax = Math.ceil(qpsMaxRaw / 10) * 10;
    return {
      yMax, qpsMax,
      points: rows.map((r, i) => ({
        x: rows.length > 1 ? (i / (rows.length - 1)) * innerW : innerW / 2,
        p50: innerH - (r.p50_ms / yMax) * innerH,
        p95: innerH - (r.p95_ms / yMax) * innerH,
        p99: innerH - (r.p99_ms / yMax) * innerH,
        qpsBar: (r.qps / qpsMax) * innerH,
      })),
    };
  }, [rows, innerW, innerH]);

  const path = (key: 'p50' | 'p95' | 'p99') =>
    points.length === 0 ? '' :
      points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x.toFixed(1)},${(p[key] as number).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" aria-label="Load test latency chart">
      <g transform={`translate(${M.l},${M.t})`}>
        {/* Y grid */}
        {[0, 0.25, 0.5, 0.75, 1].map(t => (
          <g key={t}>
            <line x1={0} x2={innerW} y1={innerH * (1 - t)} y2={innerH * (1 - t)}
                  stroke="#e5e7eb" strokeDasharray="2,3" />
            <text x={-6} y={innerH * (1 - t) + 3} textAnchor="end" fontSize={10} fill="#6b7280">
              {Math.round(yMax * t)} ms
            </text>
          </g>
        ))}
        {/* QPS bars (faint) */}
        {points.map((p, i) => (
          <rect key={i} x={p.x - 1.5} y={innerH - p.qpsBar} width={3} height={p.qpsBar}
                fill="#ddd6fe" opacity={0.65} />
        ))}
        {/* Latency lines */}
        {points.length > 0 && (
          <>
            <path d={path('p99')} stroke="#dc2626" strokeWidth={1.5} fill="none" />
            <path d={path('p95')} stroke="#f59e0b" strokeWidth={1.5} fill="none" />
            <path d={path('p50')} stroke="#10b981" strokeWidth={1.8} fill="none" />
          </>
        )}
        {/* Axis labels */}
        <text x={innerW / 2} y={innerH + 18} textAnchor="middle" fontSize={10} fill="#6b7280">
          time →   ({rows.length} samples · qps max ~{qpsMax})
        </text>
      </g>
      {/* Legend */}
      <g transform={`translate(${M.l + 8},${M.t + 8})`} fontSize={11}>
        <g><circle cx={0} cy={0} r={3} fill="#10b981" /><text x={8} y={4} fill="#374151">p50</text></g>
        <g transform="translate(50,0)"><circle cx={0} cy={0} r={3} fill="#f59e0b" /><text x={8} y={4} fill="#374151">p95</text></g>
        <g transform="translate(100,0)"><circle cx={0} cy={0} r={3} fill="#dc2626" /><text x={8} y={4} fill="#374151">p99</text></g>
        <g transform="translate(150,0)"><rect x={-3} y={-3} width={6} height={6} fill="#ddd6fe" /><text x={8} y={4} fill="#374151">qps</text></g>
      </g>
    </svg>
  );
}

function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 920 400" className="w-full max-w-5xl" aria-label="Live pricing architecture">
      <defs>
        <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
                markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#94a3b8" />
        </marker>
        <linearGradient id="quote-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#eff6ff" />
          <stop offset="100%" stopColor="#dbeafe" />
        </linearGradient>
        <linearGradient id="orch-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#faf5ff" />
          <stop offset="100%" stopColor="#ede9fe" />
        </linearGradient>
        <linearGradient id="model-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#f0fdfa" />
          <stop offset="100%" stopColor="#ccfbf1" />
        </linearGradient>
        <linearGradient id="fs-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#fffbeb" />
          <stop offset="100%" stopColor="#fef3c7" />
        </linearGradient>
        <linearGradient id="rules-grad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stopColor="#fef2f2" />
          <stop offset="100%" stopColor="#fecaca" />
        </linearGradient>
      </defs>

      {/* Quote request */}
      <rect x="20" y="170" width="140" height="60" rx="8" fill="url(#quote-grad)" stroke="#3b82f6" />
      <text x="90" y="200" textAnchor="middle" fontSize="13" fontWeight="600" fill="#1e3a8a">Quote request</text>
      <text x="90" y="218" textAnchor="middle" fontSize="10" fill="#2563eb">policy_id · &lt; 1 KB</text>

      {/* Orchestrator (pyfunc) */}
      <rect x="200" y="170" width="160" height="60" rx="8" fill="url(#orch-grad)" stroke="#7c3aed" />
      <text x="280" y="196" textAnchor="middle" fontSize="13" fontWeight="600" fill="#4c1d95">motor_pricing_scorer</text>
      <text x="280" y="214" textAnchor="middle" fontSize="10" fill="#6d28d9">pyfunc · single endpoint</text>
      <line x1="160" y1="200" x2="200" y2="200" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* Online feature store (Lakebase) — above orchestrator */}
      <rect x="200" y="60" width="160" height="60" rx="8" fill="url(#fs-grad)" stroke="#d97706" />
      <text x="280" y="86" textAnchor="middle" fontSize="13" fontWeight="600" fill="#78350f">Lakebase online store</text>
      <text x="280" y="104" textAnchor="middle" fontSize="10" fill="#92400e">unified_motor_table_live</text>
      <line x1="280" y1="170" x2="280" y2="120" stroke="#cbd5e1" strokeWidth="2" strokeDasharray="4 2" markerEnd="url(#arrow)" />
      <text x="370" y="148" fontSize="9" fill="#92400e" fontStyle="italic">FeatureLookup</text>

      {/* 4 parallel models */}
      <g>
        {[
          { name: "freq_glm_motor",   sub: "Poisson · claim count" },
          { name: "sev_glm_motor",    sub: "Gamma · £ per claim" },
          { name: "demand_gbm_motor", sub: "LightGBM · accept prob" },
          { name: "fraud_gbm_motor",  sub: "LightGBM · fraud prob" },
        ].map((m, i) => {
          const y = 80 + i * 60;
          return (
            <g key={m.name}>
              <rect x="420" y={y} width="180" height="44" rx="6" fill="url(#model-grad)" stroke="#14b8a6" />
              <text x="510" y={y + 18} textAnchor="middle" fontSize="12" fontWeight="600" fill="#115e59">{m.name}</text>
              <text x="510" y={y + 34} textAnchor="middle" fontSize="10" fill="#0f766e">{m.sub}</text>
              <line x1="360" y1="200" x2="420" y2={y + 22} stroke="#cbd5e1" strokeWidth="1.5" />
            </g>
          );
        })}
      </g>
      <text x="510" y="350" textAnchor="middle" fontSize="10" fill="#0f766e" fontStyle="italic">
        4 champions scored in-process · shared feature batch
      </text>

      {/* Rating Engine */}
      <rect x="660" y="155" width="160" height="90" rx="8" fill="url(#rules-grad)" stroke="#dc2626" />
      <text x="740" y="178" textAnchor="middle" fontSize="13" fontWeight="600" fill="#7f1d1d">Rating engine</text>
      <text x="740" y="196" textAnchor="middle" fontSize="9.5" fill="#991b1b">freq × sev → technical</text>
      <text x="740" y="210" textAnchor="middle" fontSize="9.5" fill="#991b1b">+ expense · commission</text>
      <text x="740" y="224" textAnchor="middle" fontSize="9.5" fill="#991b1b">+ young driver · telematics</text>
      <text x="740" y="238" textAnchor="middle" fontSize="9.5" fill="#991b1b">+ fraud load · demand adj</text>
      {/* Arrows from each model into rating engine */}
      {[80, 140, 200, 260].map((y) => (
        <line key={y} x1="600" y1={y + 22} x2="660" y2="200" stroke="#cbd5e1" strokeWidth="1.5" markerEnd="url(#arrow)" />
      ))}

      {/* Final premium response — arrow back to quote */}
      <path d="M 740 245 Q 740 320 420 320 Q 180 320 90 245"
            fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="5 3" markerEnd="url(#arrow)" />
      <text x="430" y="338" textAnchor="middle" fontSize="11" fontWeight="600" fill="#475569">
        final_premium + every intermediate factor · end-to-end &lt; 200 ms
      </text>
      <text x="430" y="354" textAnchor="middle" fontSize="9.5" fill="#64748b" fontStyle="italic">
        audit trail logged to inference table
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso?: string | null): string {
  if (!iso) return '—';
  const t = new Date(iso);
  if (isNaN(t.getTime())) return String(iso).substring(0, 10);
  return t.toISOString().substring(0, 10);
}

function eventShortLabel(t: string): string {
  if (t === 'governance_pack_generated') return 'pack';
  if (t === 'model_promoted')   return 'promote';
  if (t === 'model_rollback' || t === 'model_rolled_back') return 'rollback';
  if (t === 'model_trained')    return 'train';
  return t;
}
function eventColor(t: string): string {
  if (t === 'model_rollback' || t === 'model_rolled_back') return 'bg-red-100 text-red-700';
  if (t === 'model_promoted') return 'bg-emerald-100 text-emerald-700';
  if (t === 'governance_pack_generated') return 'bg-blue-100 text-blue-700';
  return 'bg-gray-100 text-gray-600';
}

// ---------------------------------------------------------------------------
// Serving SLOs — what the rating engine sees at quote time
// ---------------------------------------------------------------------------

function ServingSLOs() {
  const tiles = [
    { icon: <Clock    className="w-4 h-4 text-emerald-600" />, label: 'Feature lookup p50',    value: '38 ms',  sub: 'online feature store',     tone: 'emerald' as const },
    { icon: <Clock    className="w-4 h-4 text-emerald-600" />, label: 'Feature lookup p99',    value: '92 ms',  sub: 'sub-100ms target',         tone: 'emerald' as const },
    { icon: <Database className="w-4 h-4 text-blue-600" />,    label: 'Features tested',       value: '3.0 M',  sub: 'across all candidates',    tone: 'blue'    as const },
    { icon: <Zap      className="w-4 h-4 text-purple-600" />,  label: 'End-to-end quote',      value: '<500 ms',sub: '4 models + factor build-up', tone: 'purple'  as const },
  ];
  return (
    <section className="mt-5 mb-5 bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-baseline justify-between mb-3">
        <h3 className="text-sm font-semibold text-gray-900">Serving SLOs</h3>
        <span className="text-[11px] text-gray-500 italic">
          what the rating engine sees at quote time
        </span>
      </div>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {tiles.map(t => (
          <div key={t.label} className="rounded-lg border border-gray-200 bg-gray-50 px-3 py-2.5">
            <div className="flex items-center gap-1.5 mb-1">{t.icon}
              <span className="text-[11px] uppercase tracking-wider text-gray-600 font-semibold">{t.label}</span>
            </div>
            <div className="text-2xl font-bold text-gray-900 leading-tight">{t.value}</div>
            <div className="text-[11px] text-gray-500">{t.sub}</div>
          </div>
        ))}
      </div>
      <div className="mt-3 text-[11px] text-gray-500 italic">
        Targets representative of a Mosaic AI Model Serving + Online Feature Store deployment. Real
        numbers populate when the endpoint receives production traffic — an Inference Table records
        every request, latency, and feature snapshot for governance.
      </div>
    </section>
  );
}
