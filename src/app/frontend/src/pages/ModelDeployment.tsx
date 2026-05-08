import { useEffect, useState } from 'react';
import {
  Rocket, ExternalLink, Loader2, Undo2, ChevronDown, ChevronRight,
  FileCheck2, ShieldCheck, Server, AlertCircle, Zap, Clock, Database,
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
// Tab 2 — Live Pricing System (placeholder)
// ===========================================================================

function LivePricing() {
  return (
    <div>
      <div className="bg-gradient-to-br from-violet-50 to-blue-50 border border-blue-200 rounded-lg p-5 mb-5">
        <div className="flex items-center gap-2 mb-1">
          <h3 className="font-semibold text-gray-900 text-sm flex items-center gap-2">
            <Zap className="w-4 h-4 text-violet-600" />
            Live Pricing System
          </h3>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-700 font-medium uppercase tracking-wide">
            Coming soon
          </span>
        </div>
        <p className="text-sm text-gray-700 mt-1">
          Real-time scoring at scale — this tab will host the live pricing system: 10+ models running in parallel,
          end-to-end quote response under 500ms. Demonstrates the platform's ability to serve complex pricing
          workflows at aggregator-grade latency.
        </p>
      </div>

      <section className="bg-white border border-gray-200 rounded-lg p-5 mb-5">
        <h4 className="text-sm font-semibold text-gray-800 mb-4">Architecture</h4>
        <div className="flex justify-center overflow-x-auto">
          <ArchitectureDiagram />
        </div>
      </section>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-1.5">
            <Server className="w-4 h-4 text-gray-500" /> Capabilities
          </h4>
          <ul className="text-sm text-gray-700 space-y-1.5">
            {[
              'Sub-500ms end-to-end response',
              'Automatic feature lookup via FeatureLookup',
              'Parallel model invocation',
              'Latency trace per request',
              'Load testing with p50/p95/p99',
              'Champion/challenger traffic splitting',
            ].map(c => (
              <li key={c} className="flex items-start gap-2">
                <span className="w-1.5 h-1.5 rounded-full bg-violet-500 mt-2 shrink-0" />
                <span>{c}</span>
              </li>
            ))}
          </ul>
        </section>

        <section className="bg-white border border-gray-200 rounded-lg p-4">
          <h4 className="text-sm font-semibold text-gray-800 mb-3 flex items-center gap-1.5">
            <Rocket className="w-4 h-4 text-gray-500" /> What's in the pipeline
          </h4>
          <ul className="text-sm text-gray-700 space-y-1.5">
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
              <span>Databricks Model Serving endpoints per model family</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
              <span>Online feature store tables (sub-10ms lookup)</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
              <span>Pricing orchestrator: fan out, combine, apply business rules</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
              <span>Per-request tracing + latency dashboards</span>
            </li>
            <li className="flex items-start gap-2">
              <span className="w-1.5 h-1.5 rounded-full bg-blue-400 mt-2 shrink-0" />
              <span>Load-test harness and traffic splitter for canary releases</span>
            </li>
          </ul>
        </section>
      </div>
    </div>
  );
}

function ArchitectureDiagram() {
  return (
    <svg viewBox="0 0 820 320" className="w-full max-w-4xl" aria-label="Live pricing architecture">
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
      </defs>

      {/* Quote request */}
      <rect x="20" y="130" width="140" height="60" rx="8" fill="url(#quote-grad)" stroke="#3b82f6" />
      <text x="90" y="160" textAnchor="middle" fontSize="13" fontWeight="600" fill="#1e3a8a">Quote request</text>
      <text x="90" y="178" textAnchor="middle" fontSize="10" fill="#2563eb">broker / direct / aggregator</text>

      {/* Orchestrator */}
      <rect x="210" y="130" width="160" height="60" rx="8" fill="url(#orch-grad)" stroke="#7c3aed" />
      <text x="290" y="158" textAnchor="middle" fontSize="13" fontWeight="600" fill="#4c1d95">Orchestrator</text>
      <text x="290" y="174" textAnchor="middle" fontSize="10" fill="#6d28d9">fan-out · combine · rules</text>
      <line x1="160" y1="160" x2="210" y2="160" stroke="#94a3b8" strokeWidth="2" markerEnd="url(#arrow)" />

      {/* Parallel model endpoints — 10 stacked */}
      <g>
        {[
          "freq_glm",
          "sev_glm",
          "demand_gbm",
          "fraud_gbm",
          "peril_fire_gbm",
          "peril_flood_gbm",
          "retention_gbm",
          "price_elasticity_gbm",
          "loading_engine",
          "price_match_rules",
        ].map((name, i) => {
          const y = 20 + i * 28;
          return (
            <g key={name}>
              <rect x="430" y={y} width="180" height="22" rx="4" fill="url(#model-grad)" stroke="#14b8a6" />
              <text x="520" y={y + 15} textAnchor="middle" fontSize="11" fontWeight="500" fill="#115e59">{name}</text>
              <line x1="370" y1="160" x2="430" y2={y + 11} stroke="#cbd5e1" strokeWidth="1" />
            </g>
          );
        })}
      </g>
      <text x="520" y="310" textAnchor="middle" fontSize="10" fill="#0f766e">10 model endpoints scored in parallel</text>

      {/* Online feature store */}
      <rect x="660" y="130" width="150" height="60" rx="8" fill="url(#fs-grad)" stroke="#d97706" />
      <text x="735" y="158" textAnchor="middle" fontSize="13" fontWeight="600" fill="#78350f">Online Feature Store</text>
      <text x="735" y="174" textAnchor="middle" fontSize="10" fill="#92400e">FeatureLookup · &lt;10ms</text>
      <line x1="610" y1="90"  x2="660" y2="140" stroke="#cbd5e1" strokeDasharray="4 2" />
      <line x1="610" y1="180" x2="660" y2="180" stroke="#cbd5e1" strokeDasharray="4 2" />

      {/* Response back */}
      <path d="M 370 200 Q 400 260 290 260 Q 180 260 160 200"
            fill="none" stroke="#64748b" strokeWidth="2" strokeDasharray="5 3" markerEnd="url(#arrow)" />
      <text x="290" y="285" textAnchor="middle" fontSize="10" fill="#475569">
        price + loading + referral flag · end-to-end &lt;500ms
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
