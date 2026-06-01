import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Radio, AlertTriangle, Loader2, Gauge, ArrowRight, Activity } from 'lucide-react';
import { api } from '../lib/api';

const JOHN = 'POL-MOTOR-00000001';

// Standalone telematics black-box ops panel. Fires a simulated driving event
// (speeding + curfew breach), MERGEs it into the live feature table, refreshes
// the online store, and shows the before/after so the operator can flip back
// to /quote and re-price.
export default function BlackBox() {
  const [tel, setTel]       = useState<any>(null);
  const [firing, setFiring] = useState(false);
  const [last, setLast]     = useState<any>(null);
  const [error, setError]   = useState<string | null>(null);

  const load = () => { api.livePricingPolicy(JOHN).then(d => setTel(d.telematics)).catch(() => {}); };
  useEffect(load, []);

  const fire = async () => {
    setFiring(true); setError(null);
    try {
      const r = await api.livePricingTelematicsEvent({
        policy_id: JOHN, speeding_event: true, curfew_breach: true, behaviour_score_delta: -8,
      });
      if (!r.ok) { setError(r.error || 'event failed'); }
      else { setLast(r); load(); }
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setFiring(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      <header className="flex items-center justify-between px-8 py-5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-amber-500 flex items-center justify-center">
            <Radio className="w-5 h-5 text-slate-900" />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight">Telematics Black-Box <span className="text-amber-400">Ops</span></div>
            <div className="text-[11px] text-slate-400">Live driving-event ingestion · {JOHN}</div>
          </div>
        </div>
        <Link to="/quote" className="text-[12px] text-slate-400 hover:text-white inline-flex items-center gap-1">
          quote portal <ArrowRight className="w-3 h-3" />
        </Link>
      </header>

      <div className="max-w-3xl mx-auto px-8 py-10">
        {/* Current live signal */}
        <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Current driving signal</h2>
        {!tel ? <div className="text-slate-500 text-sm">Loading…</div> : (
          <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 mb-8">
            <Metric label="Behaviour score" value={`${tel.behaviour_score}`} suffix="/100"
                    tone={tel.behaviour_score >= 70 ? 'good' : tel.behaviour_score >= 45 ? 'warn' : 'bad'} />
            <Metric label="Speeding events" value={`${tel.recent_speeding_events}`} tone={tel.recent_speeding_events ? 'warn' : 'good'} />
            <Metric label="Curfew breaches" value={`${tel.recent_curfew_breaches}`} tone={tel.recent_curfew_breaches ? 'warn' : 'good'} />
            <Metric label="Harsh braking" value={`${tel.recent_harsh_braking_30d}`} tone={tel.recent_harsh_braking_30d ? 'warn' : 'good'} />
          </div>
        )}

        {/* Fire event */}
        <div className="bg-white/5 border border-white/10 rounded-2xl p-8 text-center">
          <AlertTriangle className="w-10 h-10 text-amber-400 mx-auto mb-3" />
          <h3 className="text-lg font-semibold mb-1">Simulate a black-box event</h3>
          <p className="text-slate-400 text-sm mb-6 max-w-md mx-auto">
            Records a speeding incident + curfew breach, drops the behaviour score, and pushes the new
            signal into the live feature store. Re-quoting will reflect the higher risk.
          </p>
          <button onClick={fire} disabled={firing}
                  className="px-8 py-3 rounded-xl bg-amber-500 hover:bg-amber-600 text-slate-900 disabled:opacity-50 font-semibold inline-flex items-center gap-2">
            {firing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Activity className="w-4 h-4" />}
            {firing ? 'Recording event + refreshing feature store…' : 'Trigger driving event'}
          </button>
          {error && <div className="mt-4 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{error}</div>}
        </div>

        {/* Before/after */}
        {last && (
          <div className="mt-6 bg-white/5 border border-white/10 rounded-2xl p-6">
            <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-3">
              Event {last.event_id} · feature store {last.online_refresh?.completed ? 'refreshed' : 'refreshing'}
              {last.online_refresh?.duration_ms ? ` (${Math.round(last.online_refresh.duration_ms/1000)}s)` : ''}
            </div>
            <div className="grid grid-cols-2 gap-6 text-sm">
              <BA title="Before" d={last.before} />
              <BA title="After"  d={last.after} highlight />
            </div>
          </div>
        )}

        <div className="mt-8 text-center">
          <Link to="/quote"
                className="inline-flex items-center gap-2 px-6 py-3 rounded-xl bg-blue-500 hover:bg-blue-600 font-semibold">
            <Gauge className="w-4 h-4" /> Go re-quote John
          </Link>
          <p className="text-[11px] text-slate-500 mt-3">
            Demo · events MERGE into <code>unified_motor_table_live</code> and republish to the Lakebase online store.
          </p>
        </div>
      </div>
    </div>
  );
}

function Metric({ label, value, suffix, tone }: { label: string; value: string; suffix?: string; tone: 'good'|'warn'|'bad' }) {
  const c = tone === 'good' ? 'text-emerald-300' : tone === 'warn' ? 'text-amber-300' : 'text-red-300';
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4">
      <div className="text-[10px] uppercase tracking-wider text-slate-400 mb-1">{label}</div>
      <div className={`text-2xl font-bold ${c}`}>{value}<span className="text-sm text-slate-500">{suffix}</span></div>
    </div>
  );
}
function BA({ title, d, highlight }: { title: string; d: any; highlight?: boolean }) {
  if (!d) return null;
  return (
    <div className={highlight ? 'rounded-lg bg-amber-500/10 border border-amber-500/20 p-3' : 'rounded-lg bg-white/5 p-3'}>
      <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">{title}</div>
      <Row k="Behaviour" v={d.behaviour_score} />
      <Row k="Speeding" v={d.recent_speeding_events} />
      <Row k="Curfew" v={d.recent_curfew_breaches} />
      <Row k="Harsh braking" v={d.recent_harsh_braking_30d} />
    </div>
  );
}
function Row({ k, v }: { k: string; v: any }) {
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400">{k}</span>
      <span className="font-medium">{String(v)}</span>
    </div>
  );
}
