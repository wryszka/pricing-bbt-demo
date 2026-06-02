import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Activity, Play, Square, ArrowRight, Zap, Server, Cpu } from 'lucide-react';
import { api } from '../lib/api';

// Standalone live quote-tester. Drives the app-side continuous quote stream
// (no job startup lag) and polls in-memory rolling metrics sub-second so the
// numbers + chart move continuously while it runs.
export default function QuoteTester() {
  const [targetQps, setTargetQps] = useState(25);
  const [running, setRunning]     = useState(false);
  const [m, setM]                 = useState<any>(null);
  const [scale, setScale]         = useState<any>(null);
  const [hist, setHist]           = useState<number[]>([]);   // p50 over time for the chart
  const pollRef = useRef<number | null>(null);
  const scaleRef = useRef<number | null>(null);

  const poll = async () => {
    try {
      const s = await api.livePricingStreamMetrics();
      setM(s);
      setRunning(s.running);
      if (s.running) setHist(h => [...h, s.p50_ms].slice(-120));
    } catch { /* ignore */ }
  };
  // Endpoint scale (provisioned concurrency) refreshes on a ~1-min cadence
  // server-side, so poll it less often than the stream metrics.
  const pollScale = async () => {
    try { setScale(await api.livePricingEndpointScale()); } catch { /* ignore */ }
  };

  useEffect(() => {
    poll(); pollScale();
    pollRef.current  = window.setInterval(poll, 700);
    scaleRef.current = window.setInterval(pollScale, 4000);
    return () => {
      if (pollRef.current)  window.clearInterval(pollRef.current);
      if (scaleRef.current) window.clearInterval(scaleRef.current);
    };
  }, []);

  const start = async () => {
    setHist([]);
    await api.livePricingStreamStart(targetQps);
    setRunning(true);
    poll();
  };
  const stop = async () => {
    await api.livePricingStreamStop();
    setRunning(false);
    poll();
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <header className="flex items-center justify-between px-8 py-5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-violet-500 flex items-center justify-center">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight">Live Quote Tester</div>
            <div className="text-[11px] text-slate-400">continuous load against <code>motor_pricing_scorer</code></div>
          </div>
        </div>
        <Link to="/quote" className="text-[12px] text-slate-400 hover:text-white inline-flex items-center gap-1">
          quote portal <ArrowRight className="w-3 h-3" />
        </Link>
      </header>

      <div className="max-w-5xl mx-auto px-8 py-10">
        {/* Controls */}
        <div className="flex items-center gap-4 mb-8">
          <label className="text-sm text-slate-300 flex items-center gap-3">
            Target QPS
            <input type="range" min={5} max={100} step={5} value={targetQps}
                   disabled={running}
                   onChange={e => setTargetQps(Number(e.target.value))}
                   className="w-48 accent-violet-500" />
            <span className="font-mono text-lg w-10">{targetQps}</span>
          </label>
          {running ? (
            <button onClick={stop} className="ml-auto px-5 py-2.5 rounded-xl bg-rose-500 hover:bg-rose-600 font-semibold inline-flex items-center gap-2">
              <Square className="w-4 h-4" /> Stop
            </button>
          ) : (
            <button onClick={start} className="ml-auto px-5 py-2.5 rounded-xl bg-violet-500 hover:bg-violet-600 font-semibold inline-flex items-center gap-2">
              <Play className="w-4 h-4" /> Start streaming
            </button>
          )}
        </div>

        {/* Live tiles */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
          <Tile label="QPS (live)" value={m ? m.qps.toFixed(0) : '0'}
                accent={running ? 'text-violet-300' : 'text-slate-500'}
                pulse={running} />
          <Tile label="p50 latency" value={m ? `${m.p50_ms.toFixed(0)} ms` : '—'} accent="text-emerald-300" />
          <Tile label="p95 latency" value={m ? `${m.p95_ms.toFixed(0)} ms` : '—'} accent="text-amber-300" />
          <Tile label="p99 latency" value={m ? `${m.p99_ms.toFixed(0)} ms` : '—'} accent="text-amber-300" />
        </div>
        <div className="grid grid-cols-3 gap-4 mb-6">
          <Tile label="Total served" value={m ? m.total.toLocaleString() : '0'} small />
          <Tile label="Errors" value={m ? `${m.errors} (${m.error_pct}%)` : '0'} small
                accent={m && m.error_pct > 1 ? 'text-red-300' : 'text-slate-300'} />
          <Tile label="Uptime" value={m ? `${Math.round(m.uptime_s)}s` : '0s'} small />
        </div>

        {/* Compute size — live autoscale within min..max */}
        <ComputeCard scale={scale} />

        {/* Live chart */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-6">
          <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-3 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5" /> p50 latency — live ({hist.length} pts)
          </div>
          <Sparkline data={hist} />
          <div className="text-[11px] text-slate-500 mt-3">
            Measures <span className="text-slate-400">client round-trip from the app</span> (network + serving),
            a little above the server-side latency in the Databricks Serving UI. Clean to ~60 QPS from this app
            container; pushing toward 100 saturates the container (latency/errors climb) — that's the driver, not
            the endpoint. For sustained 100+ QPS and true endpoint-ceiling numbers, use the in-region job-based
            load test in the workbench.
          </div>
        </div>
      </div>
    </div>
  );
}

function ComputeCard({ scale }: { scale: any }) {
  const min  = scale?.min ?? 4;
  const max  = scale?.max ?? 64;
  const cur  = scale?.provisioned_concurrency;
  const cpu  = scale?.cpu_pct;
  const pct  = cur != null ? Math.max(0, Math.min(100, ((cur - min) / Math.max(1, max - min)) * 100)) : 0;
  return (
    <div className="bg-white/5 border border-white/10 rounded-2xl p-5 mb-6">
      <div className="flex items-center justify-between mb-2">
        <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 flex items-center gap-1.5">
          <Server className="w-3.5 h-3.5" /> Endpoint compute — live autoscale
        </div>
        {cpu != null && (
          <div className="text-[11px] text-slate-400 flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5" /> CPU {cpu}%
          </div>
        )}
      </div>
      <div className="flex items-end gap-3 mb-2">
        <div className="text-3xl font-bold text-violet-300">{cur != null ? cur : '—'}</div>
        <div className="text-sm text-slate-400 mb-1">/ {max} provisioned concurrency (slots)</div>
      </div>
      {/* scale bar from min..max */}
      <div className="relative h-2 rounded-full bg-white/10 overflow-hidden">
        <div className="absolute inset-y-0 left-0 bg-violet-500 rounded-full transition-all duration-500"
             style={{ width: `${pct}%` }} />
      </div>
      <div className="flex justify-between text-[10px] text-slate-500 mt-1">
        <span>min {min}</span>
        <span>scales with load</span>
        <span>max {max}</span>
      </div>
    </div>
  );
}

function Tile({ label, value, accent = 'text-white', small, pulse }:
  { label: string; value: string; accent?: string; small?: boolean; pulse?: boolean }) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4">
      <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1 flex items-center gap-1.5">
        {label}{pulse && <span className="w-1.5 h-1.5 rounded-full bg-violet-400 animate-pulse" />}
      </div>
      <div className={`font-bold ${small ? 'text-xl' : 'text-3xl'} ${accent}`}>{value}</div>
    </div>
  );
}

function Sparkline({ data }: { data: number[] }) {
  const W = 900, H = 160, pad = 8;
  if (data.length < 2) {
    return <div className="h-[160px] flex items-center justify-center text-slate-600 text-sm">Start the stream to see live latency…</div>;
  }
  const max = Math.max(...data, 50);
  const min = 0;
  const pts = data.map((v, i) => {
    const x = pad + (i / (data.length - 1)) * (W - 2 * pad);
    const y = H - pad - ((v - min) / (max - min)) * (H - 2 * pad);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full" preserveAspectRatio="none" style={{ height: 160 }}>
      <polyline points={pts} fill="none" stroke="#a78bfa" strokeWidth="2" />
      <text x={pad} y={14} fill="#64748b" fontSize="11">{max.toFixed(0)} ms</text>
    </svg>
  );
}
