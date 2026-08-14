import { useEffect, useMemo, useState } from 'react';
import { Target, ChevronDown, ChevronUp, ShieldCheck, Info } from 'lucide-react';
import { api } from '../lib/api';

// Price Optimisation — a demo OF optimisation. Per segment: the demand curve
// (conversion vs price-to-market), the cost line, the profit curve, the
// profit-optimal price under governed constraints, and the volume/profit
// trade-off. Every number comes from the governed optimiser tables.

type Seg = {
  segment: string; n_quotes: number; elasticity: number; market_ref: number;
  cost_line: number; current_multiplier: number; current_conversion: number;
  current_profit_per_quote: number; optimal_multiplier: number;
  optimal_conversion: number; optimal_profit_per_quote: number;
  profit_uplift_per_quote: number; profit_uplift_pct: number;
  binding_constraint: string;
};
type CurvePt = {
  segment: string; price_multiplier: number; expected_conversion: number;
  price: number; expected_profit_per_quote: number; within_rate_cap: boolean;
};

const fmtPct = (v: number) => `${(v * 100).toFixed(0)}%`;
const fmtMult = (v: number) => `${v.toFixed(2)}×`;
const fmtGbp = (v: number) =>
  v >= 1000 ? `£${(v / 1000).toFixed(1)}k` : `£${v.toFixed(0)}`;

export default function PriceOptimisation() {
  const [data, setData] = useState<any>(null);
  const [err, setErr] = useState<string | null>(null);
  const [seg, setSeg] = useState<string>('');
  const [showHelp, setShowHelp] = useState(false);

  useEffect(() => {
    const num = (v: any) => (v === null || v === undefined || v === '' ? v : Number(v));
    const numify = (r: any) => {
      const o = { ...r };
      for (const k of ['n_quotes', 'elasticity', 'market_ref', 'cost_line',
        'current_multiplier', 'current_conversion', 'current_profit_per_quote',
        'optimal_multiplier', 'optimal_conversion', 'optimal_profit_per_quote',
        'profit_uplift_per_quote', 'profit_uplift_pct', 'price_multiplier',
        'expected_conversion', 'price', 'expected_profit_per_quote']) {
        if (k in o) o[k] = num(o[k]);
      }
      return o;
    };
    api.optimisationSummary()
      .then((d) => {
        // Belt-and-suspenders: coerce numbers client-side too so a stray string
        // can never white-screen the page.
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
  const active = segments.find((s) => s.segment === seg);
  const segCurve = useMemo(
    () => curve.filter((c) => c.segment === seg).sort((a, b) => a.price_multiplier - b.price_multiplier),
    [curve, seg],
  );

  if (err) return <div className="p-8 text-red-600">Failed to load: {err}</div>;
  if (!data) return <div className="p-8 text-gray-500">Loading…</div>;
  if (!data.available)
    return (
      <div className="max-w-3xl mx-auto p-8">
        <h1 className="text-2xl font-bold mb-3 flex items-center gap-2">
          <Target className="w-6 h-6 text-teal-600" /> Price Optimisation
        </h1>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 text-sm text-amber-900">
          {data.message}
        </div>
      </div>
    );

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center gap-2 mb-1">
        <Target className="w-6 h-6 text-teal-600" />
        <h1 className="text-2xl font-bold">Price Optimisation</h1>
        <span className="ml-2 text-[11px] uppercase tracking-wide bg-teal-100 text-teal-800 px-2 py-0.5 rounded">
          worked example
        </span>
      </div>
      <p className="text-gray-600 text-sm mb-4 max-w-3xl">
        The profit-maximising price per segment, from your own demand and cost
        models — transparent, governed, constraint-aware. The wedge against a
        black-box optimiser: every price is readable code over governed tables.
      </p>

      {/* Explainer */}
      <button onClick={() => setShowHelp(!showHelp)}
        className="text-xs text-teal-700 flex items-center gap-1 mb-3">
        <Info className="w-3.5 h-3.5" /> What am I looking at?
        {showHelp ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
      </button>
      {showHelp && (
        <div className="bg-gray-50 border border-gray-200 rounded-lg p-4 text-sm text-gray-700 mb-4 space-y-2 max-w-3xl">
          <p><b>Demand curve</b> — how conversion falls as we price above market
            (fitted per segment from the quote stream; the slope is the price
            elasticity).</p>
          <p><b>Cost line</b> — expected claims cost for the segment. Profit is
            demand × (price − cost); the <b>optimal price</b> is where that peaks
            inside the guardrails.</p>
          <p><b>Guardrails</b> — a rate-change cap around the current book price
            and a margin floor, applied as first-class constraints and audited in
            the optimisation config. The binding one is shown per segment.</p>
          <p className="text-gray-500">Bricksurance SE is fictional; data synthetic;
            the cost line and method are illustrative, not a certified rate.</p>
        </div>
      )}

      {/* Segment selector */}
      <div className="flex gap-2 flex-wrap mb-4">
        {segments.map((s) => (
          <button key={s.segment} onClick={() => setSeg(s.segment)}
            className={`px-3 py-1.5 rounded-lg text-sm border transition-colors ${
              seg === s.segment
                ? 'bg-teal-600 text-white border-teal-600'
                : 'bg-white text-gray-700 border-gray-300 hover:border-teal-400'}`}>
            {s.segment}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
        {/* Curve chart */}
        <div className="lg:col-span-2 bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-semibold text-gray-700 mb-2">
            Demand & profit vs price — {seg}
          </h2>
          {active && <CurveChart curve={segCurve} active={active} />}
        </div>

        {/* Active segment KPIs */}
        {active && (
          <div className="bg-white rounded-lg border border-gray-200 p-4 space-y-3">
            <Kpi label="Price elasticity" value={active.elasticity.toFixed(1)}
              sub="conversion sensitivity to price (more negative = more elastic)" />
            <div className="grid grid-cols-2 gap-3">
              <Kpi label="Current price" value={fmtMult(active.current_multiplier)}
                sub={`${fmtPct(active.current_conversion)} convert`} />
              <Kpi label="Optimal price" value={fmtMult(active.optimal_multiplier)}
                sub={`${fmtPct(active.optimal_conversion)} convert`} accent />
            </div>
            <Kpi label="Expected profit / quote"
              value={`${fmtGbp(active.current_profit_per_quote)} → ${fmtGbp(active.optimal_profit_per_quote)}`}
              sub={`${active.profit_uplift_pct >= 0 ? '+' : ''}${active.profit_uplift_pct}% at the optimum`} accent />
            <div className="text-xs text-gray-500 border-t pt-2 flex items-center gap-1.5">
              <ShieldCheck className="w-3.5 h-3.5 text-teal-600" />
              Binding constraint: <b className="text-gray-700">{active.binding_constraint}</b>
            </div>
          </div>
        )}
      </div>

      {/* Portfolio table */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden mb-6">
        <div className="px-4 py-2 border-b border-gray-200 text-sm font-semibold text-gray-700">
          Per-segment recommendation
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-gray-50 text-gray-500 text-xs uppercase">
              <tr>
                {['Segment', 'Quotes', 'Elasticity', 'Current', 'Optimal', 'Conv. →', 'Profit/quote →', 'Uplift', 'Binding'].map((h) => (
                  <th key={h} className="text-left px-3 py-2 font-medium whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {segments.map((s) => (
                <tr key={s.segment} className={`border-t border-gray-100 ${seg === s.segment ? 'bg-teal-50/40' : ''}`}>
                  <td className="px-3 py-2 font-medium">{s.segment}</td>
                  <td className="px-3 py-2 text-gray-500">{s.n_quotes.toLocaleString()}</td>
                  <td className="px-3 py-2">{s.elasticity.toFixed(1)}</td>
                  <td className="px-3 py-2">{fmtMult(s.current_multiplier)}</td>
                  <td className="px-3 py-2 font-semibold text-teal-700">{fmtMult(s.optimal_multiplier)}</td>
                  <td className="px-3 py-2 text-gray-600">{fmtPct(s.current_conversion)} → {fmtPct(s.optimal_conversion)}</td>
                  <td className="px-3 py-2 text-gray-600">{fmtGbp(s.current_profit_per_quote)} → {fmtGbp(s.optimal_profit_per_quote)}</td>
                  <td className={`px-3 py-2 font-medium ${s.profit_uplift_pct > 0 ? 'text-emerald-600' : 'text-gray-400'}`}>
                    {s.profit_uplift_pct >= 0 ? '+' : ''}{s.profit_uplift_pct}%
                  </td>
                  <td className="px-3 py-2 text-xs text-gray-500">{s.binding_constraint}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>

      {/* Governed objective/constraint config */}
      {cfg && (
        <div className="bg-white rounded-lg border border-gray-200 p-4">
          <h2 className="text-sm font-semibold text-gray-700 mb-2 flex items-center gap-1.5">
            <ShieldCheck className="w-4 h-4 text-teal-600" /> Governed objective & constraints
            <span className="ml-1 text-[11px] font-normal text-gray-400">version {cfg.version}</span>
          </h2>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-3 text-sm">
            <Cfg label="Objective" value={cfg.objective} />
            <Cfg label="Rate-change cap" value={`±${(cfg.rate_change_cap * 100).toFixed(0)}%`} />
            <Cfg label="Margin floor" value={`${(cfg.margin_floor * 100).toFixed(0)}%`} />
            <Cfg label="Cost basis (LR)" value={String(cfg.target_loss_ratio)} />
            <Cfg label="Demand source" value={cfg.demand_source} wide />
            <Cfg label="Cost source" value={cfg.cost_source} wide />
          </div>
          <p className="text-xs text-gray-500 mt-3">
            This config is a versioned, audited table — the "why" of every
            recommended price is a diffable governed artefact, which a closed
            vendor optimiser cannot evidence. A fair-value / no-price-walking
            guardrail plugs in here as another constraint.
          </p>
        </div>
      )}
    </div>
  );
}

function Kpi({ label, value, sub, accent }: { label: string; value: string; sub?: string; accent?: boolean }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-gray-400">{label}</div>
      <div className={`text-lg font-bold ${accent ? 'text-teal-700' : 'text-gray-800'}`}>{value}</div>
      {sub && <div className="text-xs text-gray-500">{sub}</div>}
    </div>
  );
}

function Cfg({ label, value, wide }: { label: string; value: string; wide?: boolean }) {
  return (
    <div className={wide ? 'md:col-span-2' : ''}>
      <div className="text-[11px] uppercase tracking-wide text-gray-400">{label}</div>
      <div className="text-gray-700">{value}</div>
    </div>
  );
}

// Inline SVG: demand (conversion) and profit vs price multiplier, with the
// current price, optimal price, and rate-cap band marked. No chart lib (CSP).
function CurveChart({ curve, active }: { curve: CurvePt[]; active: Seg }) {
  if (!curve.length) return <div className="text-sm text-gray-400">No curve data.</div>;
  const W = 620, H = 300, padL = 44, padR = 44, padT = 16, padB = 34;
  const xs = curve.map((c) => c.price_multiplier);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const profits = curve.map((c) => c.expected_profit_per_quote);
  const maxProfit = Math.max(...profits), minProfit = Math.min(0, ...profits);
  const x = (m: number) => padL + ((m - minX) / (maxX - minX)) * (W - padL - padR);
  const yConv = (c: number) => padT + (1 - c) * (H - padT - padB);           // 0..1
  const yProf = (p: number) => padT + (1 - (p - minProfit) / (maxProfit - minProfit)) * (H - padT - padB);

  const capPts = curve.filter((c) => c.within_rate_cap);
  const capX0 = capPts.length ? x(Math.min(...capPts.map((c) => c.price_multiplier))) : padL;
  const capX1 = capPts.length ? x(Math.max(...capPts.map((c) => c.price_multiplier))) : W - padR;

  const demandPath = curve.map((c, i) => `${i ? 'L' : 'M'}${x(c.price_multiplier).toFixed(1)},${yConv(c.expected_conversion).toFixed(1)}`).join(' ');
  const profitPath = curve.map((c, i) => `${i ? 'L' : 'M'}${x(c.price_multiplier).toFixed(1)},${yProf(c.expected_profit_per_quote).toFixed(1)}`).join(' ');

  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="w-full h-auto">
      {/* rate-cap band */}
      <rect x={capX0} y={padT} width={Math.max(0, capX1 - capX0)} height={H - padT - padB}
        fill="#14b8a6" opacity={0.07} />
      <text x={(capX0 + capX1) / 2} y={padT + 12} textAnchor="middle" fontSize="10" fill="#0d9488">
        within rate cap
      </text>
      {/* axes */}
      <line x1={padL} y1={H - padB} x2={W - padR} y2={H - padB} stroke="#d1d5db" />
      <text x={(padL + W - padR) / 2} y={H - 6} textAnchor="middle" fontSize="11" fill="#6b7280">
        price vs market
      </text>
      {/* profit curve */}
      <path d={profitPath} fill="none" stroke="#0d9488" strokeWidth={2.5} />
      {/* demand curve */}
      <path d={demandPath} fill="none" stroke="#6366f1" strokeWidth={2} strokeDasharray="4 3" />
      {/* current + optimal markers */}
      <line x1={x(active.current_multiplier)} y1={padT} x2={x(active.current_multiplier)} y2={H - padB}
        stroke="#9ca3af" strokeWidth={1} strokeDasharray="3 3" />
      <text x={x(active.current_multiplier)} y={H - padB + 14} textAnchor="middle" fontSize="10" fill="#6b7280">
        current {active.current_multiplier.toFixed(2)}×
      </text>
      <line x1={x(active.optimal_multiplier)} y1={padT} x2={x(active.optimal_multiplier)} y2={H - padB}
        stroke="#0d9488" strokeWidth={1.5} />
      <text x={x(active.optimal_multiplier)} y={padT - 4} textAnchor="middle" fontSize="10" fill="#0d9488" fontWeight="bold">
        optimal {active.optimal_multiplier.toFixed(2)}×
      </text>
      <circle cx={x(active.optimal_multiplier)} cy={yProf(active.optimal_profit_per_quote)} r={4} fill="#0d9488" />
      {/* legend */}
      <g fontSize="10">
        <rect x={W - padR - 150} y={padT} width={10} height={3} fill="#0d9488" />
        <text x={W - padR - 135} y={padT + 4} fill="#374151">expected profit</text>
        <rect x={W - padR - 150} y={padT + 14} width={10} height={3} fill="#6366f1" />
        <text x={W - padR - 135} y={padT + 18} fill="#374151">conversion (demand)</text>
      </g>
    </svg>
  );
}
