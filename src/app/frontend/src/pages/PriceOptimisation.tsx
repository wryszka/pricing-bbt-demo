import { useEffect, useMemo, useState } from 'react';
import { Target, ChevronDown, ChevronUp, ShieldCheck, Info, SlidersHorizontal } from 'lucide-react';
import { api } from '../lib/api';

// Price Optimisation — an INTERACTIVE optimizer, not a dashboard. The demand +
// profit curves per segment are precomputed and governed by the optimiser job;
// the optimisation DECISION (objective + constraints) runs live in the browser
// against those governed curves, with a portfolio roll-up and efficient frontier.
// This is what a pricing practitioner needs to believe the platform replaces a
// closed optimiser: turn the levers, watch the whole book move.

type Seg = {
  segment: string; n_quotes: number; elasticity: number; market_ref: number;
  cost_line: number; current_multiplier: number; current_conversion: number;
  current_profit_per_quote: number; optimal_multiplier: number;
  optimal_conversion: number; optimal_profit_per_quote: number;
  profit_uplift_per_quote: number; profit_uplift_pct: number; binding_constraint: string;
};
type CurvePt = {
  segment: string; price_multiplier: number; expected_conversion: number;
  price: number; expected_profit_per_quote: number; within_rate_cap: boolean;
};
type Levers = { alpha: number; rateCap: number; marginFloor: number };

const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`;
const fmtMult = (v: number) => `${v.toFixed(2)}×`;
const gbpM = (v: number) => `£${(v / 1e6).toFixed(2)}m`;
const num = (v: any) => (v === null || v === undefined || v === '' ? v : Number(v));

// Pick the objective-optimal curve point for a segment under the current levers.
function choose(seg: Seg, pts: CurvePt[], L: Levers): CurvePt | null {
  const within = pts.filter((p) => Math.abs(p.price_multiplier - seg.current_multiplier) <= L.rateCap + 1e-9);
  const ok = within.filter((p) => p.price > 0 && (p.price - seg.cost_line) / p.price >= L.marginFloor - 1e-9);
  const pool = ok.length ? ok : within;
  if (!pool.length) return null;
  const maxProfit = Math.max(...pts.map((p) => p.expected_profit_per_quote), 1e-9);
  let best = pool[0], bestVal = -Infinity;
  for (const p of pool) {
    const val = L.alpha * (p.expected_profit_per_quote / maxProfit) + (1 - L.alpha) * p.expected_conversion;
    if (val > bestVal) { bestVal = val; best = p; }
  }
  return best;
}
function nearestCurrent(seg: Seg, pts: CurvePt[]): CurvePt | null {
  if (!pts.length) return null;
  return pts.reduce((a, b) =>
    Math.abs(b.price_multiplier - seg.current_multiplier) < Math.abs(a.price_multiplier - seg.current_multiplier) ? b : a);
}

export default function PriceOptimisation() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [seg, setSeg] = useState<string>('');
  const [showHelp, setShowHelp] = useState(false);
  const [objective, setObjective] = useState<'profit' | 'volume' | 'blend'>('profit');
  const [alpha, setAlpha] = useState(0.6);
  const [rateCap, setRateCap] = useState(0.15);
  const [marginFloor, setMarginFloor] = useState(0.05);

  useEffect(() => {
    const numify = (r: any) => {
      const o = { ...r };
      for (const k of ['n_quotes', 'elasticity', 'market_ref', 'cost_line', 'current_multiplier',
        'current_conversion', 'current_profit_per_quote', 'optimal_multiplier', 'optimal_conversion',
        'optimal_profit_per_quote', 'profit_uplift_per_quote', 'profit_uplift_pct',
        'price_multiplier', 'expected_conversion', 'price', 'expected_profit_per_quote']) {
        if (k in o) o[k] = num(o[k]);
      }
      return o;
    };
    api.optimisationSummary()
      .then((d) => {
        if (d?.segments) d.segments = d.segments.map(numify);
        if (d?.curve) d.curve = d.curve.map(numify);
        setData(d);
        if (d?.segments?.length) setSeg(d.segments[0].segment);
      })
      .catch((e) => setErr(String(e)));
  }, []);

  const segments: Seg[] = data?.segments || [];
  const curve: CurvePt[] = data?.curve || [];
  const cfg = data?.config;
  const effAlpha = objective === 'profit' ? 1 : objective === 'volume' ? 0 : alpha;
  const levers: Levers = { alpha: effAlpha, rateCap, marginFloor };

  const ptsBySeg = useMemo(() => {
    const m: Record<string, CurvePt[]> = {};
    for (const c of curve) (m[c.segment] ||= []).push(c);
    for (const k in m) m[k].sort((a, b) => a.price_multiplier - b.price_multiplier);
    return m;
  }, [curve]);

  // Live per-segment choice + portfolio roll-up under the current levers.
  const roll = useMemo(() => {
    let curProfit = 0, optProfit = 0, curPol = 0, optPol = 0, gwp = 0, wRate = 0, pool = 0;
    const perSeg: Record<string, { chosen: CurvePt | null; cur: CurvePt | null }> = {};
    for (const s of segments) {
      const pts = ptsBySeg[s.segment] || [];
      const chosen = choose(s, pts, levers);
      const cur = nearestCurrent(s, pts);
      perSeg[s.segment] = { chosen, cur };
      const n = s.n_quotes || 0; pool += n;
      if (cur) { curProfit += cur.expected_profit_per_quote * n; curPol += cur.expected_conversion * n; }
      if (chosen) {
        optProfit += chosen.expected_profit_per_quote * n;
        optPol += chosen.expected_conversion * n;
        gwp += chosen.price * chosen.expected_conversion * n;
        wRate += chosen.price_multiplier * n;
      }
    }
    return { curProfit, optProfit, curPol, optPol, gwp, avgRate: pool ? wRate / pool : 1, perSeg };
  }, [segments, ptsBySeg, levers]);

  // Efficient frontier: sweep the profit/volume blend, plot (policies, profit).
  const frontier = useMemo(() => {
    const out: { alpha: number; pol: number; profit: number }[] = [];
    for (let a = 0; a <= 1.0001; a += 0.1) {
      let profit = 0, pol = 0;
      for (const s of segments) {
        const c = choose(s, ptsBySeg[s.segment] || [], { alpha: a, rateCap, marginFloor });
        if (c) { profit += c.expected_profit_per_quote * s.n_quotes; pol += c.expected_conversion * s.n_quotes; }
      }
      out.push({ alpha: a, pol, profit });
    }
    return out;
  }, [segments, ptsBySeg, rateCap, marginFloor]);

  if (err) return <div className="p-8 text-red-600">Failed to load: {err}</div>;
  if (!data) return <div className="p-8 text-gray-500">Loading…</div>;
  if (!data.available)
    return (
      <div className="max-w-3xl mx-auto p-8">
        <h1 className="text-2xl font-bold mb-3 flex items-center gap-2">
          <Target className="w-6 h-6 text-teal-600" /> Price Optimisation
        </h1>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-900">{data.message}</div>
      </div>
    );

  const active = segments.find((s) => s.segment === seg);
  const segChoice = roll.perSeg[seg];
  const profitDelta = roll.optProfit - roll.curProfit;
  const polDelta = roll.optPol - roll.curPol;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center gap-2 mb-1">
        <Target className="w-6 h-6 text-teal-600" />
        <h1 className="text-2xl font-bold">Price Optimisation</h1>
        <span className="ml-2 text-[11px] uppercase tracking-wide bg-teal-100 text-teal-800 px-2 py-0.5 rounded">interactive</span>
      </div>
      <p className="text-gray-600 text-sm mb-4 max-w-3xl">
        Set the objective and the guardrails; the optimiser re-solves the profit-optimal
        price per segment against the governed demand curves and rolls the impact up
        across the book — live. Your models, transparent, constraint-aware.
      </p>

      <button onClick={() => setShowHelp(!showHelp)} className="text-xs text-teal-700 flex items-center gap-1 mb-3">
        <Info className="w-3.5 h-3.5" /> What am I looking at?
        {showHelp ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {showHelp && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-700 mb-4 space-y-2 max-w-3xl">
          <p><b>Objective</b> — maximise profit, volume, or a blend. <b>Guardrails</b> — a
            rate-change cap around today's book price and a margin floor, enforced as
            first-class constraints. Move any lever and every number below re-solves.</p>
          <p><b>Portfolio impact</b> — the whole-book roll-up of the chosen strategy vs the
            current book. <b>Efficient frontier</b> — the profit↔volume trade-off; each dot is
            a blend weight, the ring is where you are now.</p>
          <p className="text-gray-500">Bricksurance SE is fictional; data synthetic; the cost
            line + method are illustrative, not a certified rate.</p>
        </div>
      )}

      {/* Levers */}
      <div className="bg-white rounded-lg border border-gray-200 p-4 mb-4">
        <div className="flex items-center gap-2 mb-3 text-sm font-semibold text-gray-700">
          <SlidersHorizontal className="w-4 h-4 text-teal-600" /> Objective &amp; guardrails
        </div>
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 items-start">
          <div>
            <div className="text-[11px] uppercase tracking-wide text-gray-400 mb-1">Objective</div>
            <div className="flex gap-1">
              {(['profit', 'volume', 'blend'] as const).map((o) => (
                <button key={o} onClick={() => setObjective(o)}
                  className={`px-2.5 py-1 rounded text-xs border capitalize ${objective === o
                    ? 'bg-teal-600 text-white border-teal-600' : 'bg-white text-gray-600 border-gray-300 hover:border-teal-400'}`}>{o}</button>
              ))}
            </div>
            {objective === 'blend' && (
              <div className="mt-2">
                <input type="range" min={0} max={1} step={0.05} value={alpha}
                  onChange={(e) => setAlpha(Number(e.target.value))} className="w-full accent-teal-600" />
                <div className="text-[11px] text-gray-500">{fmtPct(alpha)} profit / {fmtPct(1 - alpha)} volume</div>
              </div>
            )}
          </div>
          <Slider label="Rate-change cap" value={rateCap} min={0} max={0.3} step={0.01}
            onChange={setRateCap} display={`±${fmtPct(rateCap)}`} />
          <Slider label="Margin floor" value={marginFloor} min={0} max={0.25} step={0.01}
            onChange={setMarginFloor} display={fmtPct(marginFloor)} />
          <div className="text-[11px] text-gray-500 self-end">
            Every lever re-solves the optimiser client-side against the governed curves —
            no black box.
          </div>
        </div>
      </div>

      {/* Portfolio roll-up */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <Kpi label="Expected profit (book)" value={gbpM(roll.optProfit)}
          delta={`${profitDelta >= 0 ? '+' : ''}${gbpM(profitDelta)} vs current`} up={profitDelta >= 0} big />
        <Kpi label="Policies bound (book)" value={Math.round(roll.optPol).toLocaleString()}
          delta={`${polDelta >= 0 ? '+' : ''}${Math.round(polDelta).toLocaleString()} vs current`} up={polDelta >= 0} big />
        <Kpi label="Avg rate change" value={`${roll.avgRate >= 1 ? '+' : ''}${fmtPct(roll.avgRate - 1)}`}
          delta="vs today's book" />
        <Kpi label="Gross written premium" value={gbpM(roll.gwp)} delta="at the optimised prices" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {/* Efficient frontier */}
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-semibold text-gray-700 mb-2">Efficient frontier — volume vs profit</h2>
          <FrontierChart frontier={frontier} alpha={effAlpha} />
        </div>
        {/* Per-segment curve */}
        <div className="lg:col-span-2 bg-white rounded-lg border border-gray-200 p-4">
          <div className="flex items-center justify-between mb-2">
            <h2 className="text-sm font-semibold text-gray-700">Demand &amp; profit — {seg}</h2>
            <div className="flex gap-1 flex-wrap">
              {segments.map((s) => (
                <button key={s.segment} onClick={() => setSeg(s.segment)}
                  className={`px-2 py-0.5 rounded text-[11px] border ${seg === s.segment
                    ? 'bg-teal-600 text-white border-teal-600' : 'bg-white text-gray-600 border-gray-300'}`}>
                  {s.segment.replace(/^\d+ /, '')}</button>
              ))}
            </div>
          </div>
          {active && segChoice && (
            <CurveChart pts={ptsBySeg[seg] || []} current={segChoice.cur} chosen={segChoice.chosen}
              currentMult={active.current_multiplier} rateCap={rateCap} />
          )}
        </div>
      </div>

      {/* Per-segment table (live under levers) */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden mb-6">
        <div className="px-4 py-2 border-b border-gray-200 text-sm font-semibold text-gray-700">
          Per-segment recommendation — under the current levers
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
              <tr>{['Segment', 'Quotes', 'Elasticity', 'Current', 'Optimal', 'Conversion →', 'Profit/quote →'].map((h) => (
                <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap">{h}</th>))}</tr>
            </thead>
            <tbody>
              {segments.map((s) => {
                const ch = roll.perSeg[s.segment]?.chosen; const cu = roll.perSeg[s.segment]?.cur;
                return (
                  <tr key={s.segment} className={`border-t border-gray-100 ${seg === s.segment ? 'bg-teal-50/40' : ''}`}>
                    <td className="px-3 py-2 font-medium">{s.segment}</td>
                    <td className="px-3 py-2 text-gray-500">{s.n_quotes.toLocaleString()}</td>
                    <td className="px-3 py-2">{s.elasticity.toFixed(1)}</td>
                    <td className="px-3 py-2">{fmtMult(s.current_multiplier)}</td>
                    <td className="px-3 py-2 font-semibold text-teal-700">{ch ? fmtMult(ch.price_multiplier) : '—'}</td>
                    <td className="px-3 py-2 text-gray-600">{cu ? fmtPct(cu.expected_conversion) : '—'} → {ch ? fmtPct(ch.expected_conversion) : '—'}</td>
                    <td className="px-3 py-2 text-gray-600">£{cu ? Math.round(cu.expected_profit_per_quote).toLocaleString() : '—'} → £{ch ? Math.round(ch.expected_profit_per_quote).toLocaleString() : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Governed scenario */}
      {cfg && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-teal-600" /> This scenario, governed
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <Cfg label="Objective" value={objective === 'blend' ? `blend ${fmtPct(alpha)} profit` : objective} />
            <Cfg label="Rate-change cap" value={`±${fmtPct(rateCap)}`} />
            <Cfg label="Margin floor" value={fmtPct(marginFloor)} />
            <Cfg label="Demand source" value={cfg.demand_source} />
          </div>
          <p className="text-xs text-gray-500 mt-3">
            In production this lever set is saved as a versioned, audited config (like the
            baseline in <code>optimisation_config</code>) — the "why" of every price is a
            diffable governed artefact a closed optimiser can't evidence. A fair-value /
            no-price-walking rule plugs in as another constraint, and the accepted strategy
            becomes a rate release.
          </p>
        </div>
      )}
    </div>
  );
}

function Slider({ label, value, min, max, step, onChange, display }: {
  label: string; value: number; min: number; max: number; step: number;
  onChange: (v: number) => void; display: string;
}) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-gray-400 mb-1 flex justify-between">
        <span>{label}</span><span className="text-teal-700 font-semibold">{display}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(Number(e.target.value))} className="w-full accent-teal-600" />
    </div>
  );
}

function Kpi({ label, value, delta, up, big }: { label: string; value: string; delta?: string; up?: boolean; big?: boolean }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 p-3">
      <div className="text-[11px] uppercase tracking-wide text-gray-400">{label}</div>
      <div className={`font-bold ${big ? 'text-xl' : 'text-lg'} text-gray-800`}>{value}</div>
      {delta && <div className={`text-xs ${up === undefined ? 'text-gray-500' : up ? 'text-emerald-600' : 'text-rose-600'}`}>{delta}</div>}
    </div>
  );
}
function Cfg({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-gray-400">{label}</div>
      <div className="text-gray-700 capitalize">{value}</div>
    </div>
  );
}

function FrontierChart({ frontier, alpha }: { frontier: { alpha: number; pol: number; profit: number }[]; alpha: number }) {
  if (frontier.length < 2) return <div className="text-sm text-gray-400">No frontier.</div>;
  const W = 300, H = 240, pad = 40;
  const pols = frontier.map((f) => f.pol), profits = frontier.map((f) => f.profit);
  const minX = Math.min(...pols), maxX = Math.max(...pols), minY = Math.min(...profits), maxY = Math.max(...profits);
  const x = (v: number) => pad + (maxX === minX ? 0.5 : (v - minX) / (maxX - minX)) * (W - pad - 12);
  const y = (v: number) => H - pad - (maxY === minY ? 0.5 : (v - minY) / (maxY - minY)) * (H - pad - 12);
  const path = frontier.map((f, i) => `${i ? 'L' : 'M'}${x(f.pol).toFixed(1)},${y(f.profit).toFixed(1)}`).join(' ');
  const here = frontier.reduce((a, b) => Math.abs(b.alpha - alpha) < Math.abs(a.alpha - alpha) ? b : a);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      <line x1={pad} y1={H - pad} x2={W - 12} y2={H - pad} stroke="#d1d5db" />
      <line x1={pad} y1={12} x2={pad} y2={H - pad} stroke="#d1d5db" />
      <text x={(pad + W) / 2} y={H - 8} textAnchor="middle" fontSize="10" fill="#6b7280">policies bound →</text>
      <text x={12} y={H / 2} textAnchor="middle" fontSize="10" fill="#6b7280" transform={`rotate(-90 12 ${H / 2})`}>expected profit →</text>
      <path d={path} fill="none" stroke="#0d9488" strokeWidth={2} />
      {frontier.map((f, i) => <circle key={i} cx={x(f.pol)} cy={y(f.profit)} r={2.5} fill="#0d9488" opacity={0.6} />)}
      <circle cx={x(here.pol)} cy={y(here.profit)} r={6} fill="none" stroke="#0d9488" strokeWidth={2.5} />
    </svg>
  );
}

function CurveChart({ pts, current, chosen, currentMult, rateCap }: {
  pts: CurvePt[]; current: CurvePt | null; chosen: CurvePt | null; currentMult: number; rateCap: number;
}) {
  if (!pts.length) return <div className="text-sm text-gray-400">No curve.</div>;
  const W = 620, H = 300, padL = 44, padR = 44, padT = 16, padB = 34;
  const xs = pts.map((c) => c.price_multiplier);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const profits = pts.map((c) => c.expected_profit_per_quote);
  const maxProfit = Math.max(...profits), minProfit = Math.min(0, ...profits);
  const x = (m: number) => padL + ((m - minX) / (maxX - minX)) * (W - padL - padR);
  const yC = (c: number) => padT + (1 - c) * (H - padT - padB);
  const yP = (p: number) => padT + (1 - (p - minProfit) / (maxProfit - minProfit)) * (H - padT - padB);
  const capX0 = x(Math.max(minX, currentMult - rateCap)), capX1 = x(Math.min(maxX, currentMult + rateCap));
  const dPath = pts.map((c, i) => `${i ? 'L' : 'M'}${x(c.price_multiplier).toFixed(1)},${yC(c.expected_conversion).toFixed(1)}`).join(' ');
  const pPath = pts.map((c, i) => `${i ? 'L' : 'M'}${x(c.price_multiplier).toFixed(1)},${yP(c.expected_profit_per_quote).toFixed(1)}`).join(' ');
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      <rect x={capX0} y={padT} width={Math.max(0, capX1 - capX0)} height={H - padT - padB} fill="#14b8a6" opacity={0.08} />
      <text x={(capX0 + capX1) / 2} y={padT + 12} textAnchor="middle" fontSize="10" fill="#0d9488">within rate cap</text>
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#d1d5db" />
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="#6b7280">price vs market</text>
      <path d={pPath} fill="none" stroke="#0d9488" strokeWidth={2.5} />
      <path d={dPath} fill="none" stroke="#6366f1" strokeWidth={2} strokeDasharray="4 3" />
      {current && (<>
        <line x1={x(current.price_multiplier)} y1={padT} x2={x(current.price_multiplier)} y2={H - padB} stroke="#9ca3af" strokeDasharray="3 3" />
        <text x={x(current.price_multiplier)} y={H - padB + 14} textAnchor="middle" fontSize="10" fill="#6b7280">current {current.price_multiplier.toFixed(2)}×</text>
      </>)}
      {chosen && (<>
        <line x1={x(chosen.price_multiplier)} y1={padT} x2={x(chosen.price_multiplier)} y2={H - padB} stroke="#0d9488" strokeWidth={1.5} />
        <text x={x(chosen.price_multiplier)} y={padT - 4} textAnchor="middle" fontSize="10" fill="#0d9488" fontWeight="bold">optimal {chosen.price_multiplier.toFixed(2)}×</text>
        <circle cx={x(chosen.price_multiplier)} cy={yP(chosen.expected_profit_per_quote)} r={4} fill="#0d9488" />
      </>)}
      <g fontSize="10">
        <rect x={W - padR - 150} y={padT} width={10} height={3} fill="#0d9488" /><text x={W - padR - 135} y={padT + 4} fill="#374151">expected profit</text>
        <rect x={W - padR - 150} y={padT + 14} width={10} height={3} fill="#6366f1" /><text x={W - padR - 135} y={padT + 18} fill="#374151">conversion</text>
      </g>
    </svg>
  );
}
