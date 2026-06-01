import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Car, Zap, ShieldCheck, Loader2, RefreshCw, Gauge, ArrowRight } from 'lucide-react';
import { api } from '../lib/api';

const JOHN = 'POL-MOTOR-00000001';

// Standalone, consumer-facing motor quote portal. No workbench chrome — this
// is meant to read like a real insurer's online quote page. Pre-filled for
// our demo driver (John) and wired to the live motor_pricing_scorer endpoint.
export default function QuoteSystem() {
  const [profile, setProfile] = useState<any>(null);
  const [quote, setQuote]     = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError]     = useState<string | null>(null);

  const loadProfile = () => {
    api.livePricingPolicy(JOHN).then(setProfile).catch(() => setProfile(null));
  };
  useEffect(loadProfile, []);

  const getQuote = async () => {
    setLoading(true); setError(null);
    try {
      const r = await api.livePricingQuote(JOHN);
      if (!r.ok || r.status_code !== 200) {
        setError(r.detail || r.error || `Pricing service returned ${r.status_code}`);
      } else {
        setQuote(r);
        loadProfile(); // refresh telematics shown alongside
      }
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  const r = quote?.result || {};
  const premium = r.final_premium;
  const latency = quote?.latency_ms;

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-slate-800 to-slate-900 text-white">
      {/* Brand bar */}
      <header className="flex items-center justify-between px-8 py-5 border-b border-white/10">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-500 flex items-center justify-center">
            <Car className="w-5 h-5" />
          </div>
          <div>
            <div className="text-lg font-bold tracking-tight">Bricksurance <span className="text-blue-400">Motor</span></div>
            <div className="text-[11px] text-slate-400">Telematics-based car insurance · instant quote</div>
          </div>
        </div>
        <Link to="/blackbox" className="text-[12px] text-slate-400 hover:text-white inline-flex items-center gap-1">
          black-box panel <ArrowRight className="w-3 h-3" />
        </Link>
      </header>

      <div className="max-w-5xl mx-auto px-8 py-10 grid md:grid-cols-2 gap-8">
        {/* Left — driver + vehicle */}
        <section>
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Your details</h2>
          {!profile ? (
            <div className="text-slate-500 text-sm">Loading…</div>
          ) : (
            <div className="space-y-4">
              <Card title="Driver">
                <Row k="Age" v={`${profile.driver.age}`} />
                <Row k="Licence held" v={`${profile.driver.license_years} yr`} />
                <Row k="No-claims" v={`${profile.driver.no_claims_years} yr`} />
                <Row k="Occupation" v={profile.driver.occupation} />
                <Row k="Location" v={`${profile.driver.postcode_area} · ${profile.driver.region}`} />
              </Card>
              <Card title="Vehicle">
                <Row k="Car" v={`${profile.vehicle.make} ${profile.vehicle.model} (${profile.vehicle.year})`} />
                <Row k="Value" v={`£${profile.vehicle.value.toLocaleString()}`} />
                <Row k="Insurance group" v={`${profile.vehicle.group}`} />
                <Row k="Annual mileage" v={`${profile.vehicle.mileage.toLocaleString()} mi`} />
                <Row k="Overnight parking" v={profile.vehicle.parking} />
              </Card>
              <Card title="Telematics (live black-box)">
                <Row k="Behaviour score" v={<ScoreBadge score={profile.telematics.behaviour_score} />} />
                <Row k="Avg speed" v={`${profile.telematics.avg_speed_mph} mph`} />
                <Row k="Speeding events" v={`${profile.telematics.recent_speeding_events}`} />
                <Row k="Curfew breaches" v={`${profile.telematics.recent_curfew_breaches}`} />
              </Card>
            </div>
          )}
        </section>

        {/* Right — quote action + result */}
        <section className="flex flex-col">
          <h2 className="text-sm font-semibold uppercase tracking-wider text-slate-400 mb-4">Your quote</h2>
          <div className="bg-white/5 border border-white/10 rounded-2xl p-8 flex-1 flex flex-col">
            {premium != null ? (
              <div className="flex-1 flex flex-col">
                <div className="text-slate-400 text-sm mb-1">Annual premium</div>
                <div className="text-6xl font-bold tracking-tight mb-1">£{Math.round(premium).toLocaleString()}</div>
                <div className="text-[12px] text-emerald-400 inline-flex items-center gap-1 mb-6">
                  <Zap className="w-3.5 h-3.5" /> priced live in {latency != null ? `${Math.round(latency)} ms` : '—'}
                </div>
                <div className="space-y-1.5 text-sm border-t border-white/10 pt-4">
                  <Build k="Technical premium" v={r.technical_premium} />
                  {r.young_driver_load > 0 && <Build k="Young-driver loading" v={r.young_driver_load} warn />}
                  {r.telematics_event_load > 0 && <Build k="Telematics surcharge" v={r.telematics_event_load} warn />}
                  {r.fraud_load > 0 && <Build k="Risk loading" v={r.fraud_load} warn />}
                  {r.demand_adj != null && r.demand_adj !== 0 && <Build k="Market adjustment" v={r.demand_adj} />}
                </div>
                <div className="mt-auto pt-6">
                  <button onClick={getQuote} disabled={loading}
                          className="w-full py-3 rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-50 font-semibold inline-flex items-center justify-center gap-2">
                    {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <RefreshCw className="w-4 h-4" />}
                    Re-quote
                  </button>
                </div>
              </div>
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-center">
                <Gauge className="w-12 h-12 text-blue-400 mb-4" />
                <p className="text-slate-300 mb-6 max-w-xs">Get an instant, telematics-based quote priced by our live engine.</p>
                <button onClick={getQuote} disabled={loading}
                        className="px-8 py-3 rounded-xl bg-blue-500 hover:bg-blue-600 disabled:opacity-50 font-semibold inline-flex items-center gap-2">
                  {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <ShieldCheck className="w-4 h-4" />}
                  Get my quote
                </button>
              </div>
            )}
            {error && <div className="mt-4 text-sm text-red-400 bg-red-500/10 border border-red-500/20 rounded-lg p-3">{error}</div>}
          </div>
          <p className="text-[11px] text-slate-500 mt-4 text-center">
            Demo · priced by the live <code>motor_pricing_scorer</code> endpoint on Databricks Model Serving.
          </p>
        </section>
      </div>
    </div>
  );
}

function Card({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white/5 border border-white/10 rounded-xl p-4">
      <div className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 mb-2">{title}</div>
      <div className="space-y-1">{children}</div>
    </div>
  );
}
function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div className="flex items-center justify-between text-sm">
      <span className="text-slate-400">{k}</span>
      <span className="font-medium">{v}</span>
    </div>
  );
}
function Build({ k, v, warn }: { k: string; v: number; warn?: boolean }) {
  const neg = v < 0;
  return (
    <div className="flex items-center justify-between">
      <span className="text-slate-400">{k}</span>
      <span className={warn ? 'text-amber-400' : neg ? 'text-emerald-400' : 'text-slate-200'}>
        {neg ? '−' : '+'}£{Math.abs(Math.round(v)).toLocaleString()}
      </span>
    </div>
  );
}
function ScoreBadge({ score }: { score: number }) {
  const tone = score >= 70 ? 'bg-emerald-500/20 text-emerald-300'
             : score >= 45 ? 'bg-amber-500/20 text-amber-300'
             : 'bg-red-500/20 text-red-300';
  return <span className={`px-2 py-0.5 rounded-full text-xs font-semibold ${tone}`}>{score}/100</span>;
}
