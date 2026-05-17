import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Database, FlaskConical, Table2, Shield, ArrowRight, Code, Rocket, Package, Sparkles,
  RotateCcw, Loader2, ExternalLink, CheckCircle2, AlertCircle,
} from 'lucide-react';
import { api } from '../lib/api';

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Hero */}
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Pricing Workbench</h1>
        <p className="text-lg text-blue-600 font-medium">Databricks Accelerator</p>
        <p className="text-gray-500 mt-3 max-w-3xl mx-auto">
          End-to-end commercial pricing on a single platform. Every step of the real data flow is
          traceable, auditable, and governed — from ingestion through promotion, deployment, and
          regulator-facing defence.
        </p>
        <div className="mt-4 flex justify-center">
          <ResetDemoButton />
        </div>
      </div>

      {/* About this demo — single landing-page disclaimer */}
      <div className="max-w-3xl mx-auto mb-10 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
        <div className="text-xs font-semibold text-amber-900 uppercase tracking-wide mb-1">
          About this demo
        </div>
        <p className="text-sm text-amber-900/90 leading-relaxed">
          Bricksurance SE is a synthetic insurance carrier. All policies, quotes, claims, and
          director demographics are generated; the UK postcode enrichment is real public data.
          Production models, agents (Claude Sonnet 4.6 via the Databricks Foundation Model API),
          governance packs, audit logs, and the scoring flow are real — everything else is
          illustrative.
        </p>
      </div>

      {/* Main pricing flow — linear spine */}
      <FlowSpine />

      {/* Section cards — mirror the sidebar order */}
      <div className="mb-3 mt-10 flex items-end justify-between">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">The pricing spine</h2>
        <span className="text-[11px] text-gray-500">left to right, every stage is a tab</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4 mb-8">
        <SectionCard
          to="/datasets"
          icon={Database}
          color="blue"
          title="Data Ingestion"
          description="Internal book + vendor feeds + public reference data. Vendor data passes through an actuary approval gate with DQ checks."
          features={['Internal + vendor + public', 'DQ expectations', 'Actuary approval gate']}
        />
        <SectionCard
          to="/pricing-table"
          icon={Table2}
          color="green"
          title="Modelling Mart"
          description="Engineered feature table — every approved source joined on the active book. Factor catalog with per-factor provenance and an embedded AI/BI Genie."
          features={['Contributing sources', 'Factor catalog + lineage', 'AI/BI Genie']}
        />
        <SectionCard
          to="/development"
          icon={Code}
          color="purple"
          title="Model Development"
          description="Train, compare, promote. Three tabs: reference notebooks + model library for actuaries; candidate vs champion comparison; pack generation on promotion."
          features={['Train', 'Compare & test', 'Promote → governance pack']}
        />
        <SectionCard
          to="/deployment"
          icon={Rocket}
          color="red"
          title="Model Deployment"
          description="Production champions across all 4 model families, with rollback. Second tab: roadmap for the live pricing system (sub-500ms, 10+ models, online feature store)."
          features={['UC alias-based versioning', 'One-click rollback', 'Live endpoint metrics']}
        />
        <SectionCard
          to="/governance"
          icon={Shield}
          color="amber"
          title="Model Governance"
          description="Post-promotion defence for regulators. Browse by model, by date, or by policy — with an LLM assistant grounded in the governance pack (Claude Sonnet 4.6)."
          features={['By model / date / policy', 'Immutable audit trail', 'Agent-assisted review']}
        />
        <SectionCard
          to="/pricing-ai"
          icon={Sparkles}
          color="purple"
          title="Pricing AI"
          description="One chat surface fronting every agent in the workbench — governance, bias, ingestion-impact, factory-review, plus AI/BI Genie. Auto-routes via a fast classifier or pin a sub-agent yourself. Every dispatch is audit-logged."
          features={['Mosaic AI Agent Framework', 'AI/BI Genie', 'Auto-routing classifier']}
        />
        <SectionCard
          to="/add-ons"
          icon={Package}
          color="gray"
          title="Add-ons"
          description="Tools that sit alongside the pricing spine: transaction-level Quote Review, and the New Data Impact module for data scientists measuring lift from external feeds."
          features={['Quote Review', 'New Data Impact']}
        />
      </div>

      {/* Model factory — own row so the 4-step flow gets focus */}
      <div className="mb-8">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">Bulk model exploration</h2>
        <SectionCard
          to="/models"
          icon={FlaskConical}
          color="indigo"
          title="Model Factory"
          description="Systematic generation of 50+ candidate models in four steps — agent-analysed plan, virtual training, three-tier review (leaderboard → shortlist → portfolio what-if), and selective packaging that hands off to the Promote tab."
          features={['4-step actuary wizard', 'Claude-narrated plan', 'Grounded review agent', 'Hands off to Promote']}
          full
        />
      </div>

      {/* Reference architecture — closes the platform-narrative beat */}
      <div className="mb-8">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide mb-3">
          Reference architecture · one platform, integrated
        </h2>
        <ArchBlock />
      </div>

      {/* About */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
        <h3 className="font-semibold text-gray-800 mb-2">About this demo</h3>
        <p className="text-sm text-gray-600 mb-3">
          <strong>This is not a Databricks product.</strong> It's an example of what can be built on the
          Databricks platform using standard capabilities (Unity Catalog, Delta Lake, MLflow, Mosaic AI,
          Databricks Apps, Feature Engineering, Foundation Model API). The full source code is public —
          fork it, adapt it, use it as a starting point.
        </p>
        <p className="text-sm text-gray-600">
          All company names (Bricksurance SE), policy data and financial figures are fictional. No real
          customer data. The optional postcode enrichment in Add-ons uses genuine UK public data (ONSPD +
          IMD 2019 + ONS RUC) under the Open Government Licence.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Flow spine — simple, left-to-right, reflects the new sidebar structure
// ---------------------------------------------------------------------------

function FlowSpine() {
  const steps = [
    { to: '/datasets',      icon: Database,     label: 'Data Ingestion',         sub: 'approved sources' },
    { to: '/pricing-table', icon: Table2,       label: 'Modelling Mart',    sub: 'feature table' },
    { to: '/development',   icon: Code,         label: 'Model Development', sub: 'train · compare · promote' },
    { to: '/deployment',    icon: Rocket,       label: 'Deployment',        sub: 'UC champions · rollback' },
    { to: '/governance',    icon: Shield,       label: 'Governance',        sub: 'defend to regulators' },
  ];
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 overflow-x-auto">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4">
        The pricing flow, end to end
      </h3>
      <div className="flex items-stretch gap-2 min-w-[720px]">
        {steps.map((s, i) => (
          <>
            <Link key={s.to} to={s.to}
                  className="flex-1 rounded-lg border border-gray-200 bg-gray-50 hover:bg-blue-50 hover:border-blue-300 p-3 transition">
              <s.icon className="w-4 h-4 text-gray-600 mb-1.5" />
              <div className="text-sm font-semibold text-gray-900 leading-tight">{s.label}</div>
              <div className="text-[11px] text-gray-500 mt-0.5 leading-snug">{s.sub}</div>
            </Link>
            {i < steps.length - 1 && (
              <div key={`arrow-${i}`} className="flex items-center shrink-0 px-1">
                <ArrowRight className="w-4 h-4 text-gray-400" />
              </div>
            )}
          </>
        ))}
      </div>
      <p className="text-xs text-gray-500 mt-4 leading-relaxed">
        Every stage has its own tab — click a card to jump in. Add-ons and the legacy Model Factory live
        off to the side; they aren't part of the main spine.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section card
// ---------------------------------------------------------------------------

function SectionCard({ to, icon: Icon, color, title, description, features, full }: {
  to: string; icon: any; color: string; title: string; description: string;
  features: string[]; full?: boolean;
}) {
  const colorMap: Record<string, { bg: string; border: string; icon: string; badge: string }> = {
    blue:   { bg: 'bg-blue-50',   border: 'border-blue-200',   icon: 'text-blue-600',   badge: 'bg-blue-100 text-blue-700' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-200', icon: 'text-purple-600', badge: 'bg-purple-100 text-purple-700' },
    green:  { bg: 'bg-green-50',  border: 'border-green-200',  icon: 'text-green-600',  badge: 'bg-green-100 text-green-700' },
    amber:  { bg: 'bg-amber-50',  border: 'border-amber-200',  icon: 'text-amber-600',  badge: 'bg-amber-100 text-amber-700' },
    red:    { bg: 'bg-red-50',    border: 'border-red-200',    icon: 'text-red-600',    badge: 'bg-red-100 text-red-700' },
    indigo: { bg: 'bg-indigo-50', border: 'border-indigo-200', icon: 'text-indigo-600', badge: 'bg-indigo-100 text-indigo-700' },
    gray:   { bg: 'bg-gray-50',   border: 'border-gray-200',   icon: 'text-gray-600',   badge: 'bg-gray-100 text-gray-700' },
  };
  const c = colorMap[color] || colorMap.blue;
  return (
    <Link to={to}
          className={`group block ${c.bg} border ${c.border} rounded-lg p-5 hover:shadow-md transition-all ${full ? 'col-span-full' : ''}`}>
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${c.icon}`} />
        <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">{title}</h3>
        <ArrowRight className="w-4 h-4 text-gray-400 ml-auto group-hover:translate-x-1 transition-transform" />
      </div>
      <p className="text-sm text-gray-600 mb-3">{description}</p>
      <div className="flex flex-wrap gap-1.5">
        {features.map((f, i) => (
          <span key={i} className={`px-2 py-0.5 rounded text-[10px] font-medium ${c.badge}`}>{f}</span>
        ))}
      </div>
    </Link>
  );
}

// ---------------------------------------------------------------------------
// Reset Demo button — fires the demo_reset job
// ---------------------------------------------------------------------------

function ResetDemoButton() {
  const [confirming, setConfirming] = useState(false);
  const [running, setRunning]       = useState(false);
  const [result, setResult]         = useState<any>(null);
  const [status, setStatus]         = useState<any>(null);
  const [error, setError]           = useState<string | null>(null);

  const fire = async () => {
    setRunning(true); setError(null); setResult(null); setStatus(null);
    try {
      const r = await api.resetDemo();
      setResult(r);
    } catch (e: any) {
      setError(e.message || String(e));
    } finally {
      setRunning(false);
      setConfirming(false);
    }
  };

  // Poll the reset run so the user can see when it lands without leaving the page.
  useEffect(() => {
    if (!result?.run_id) return;
    if (status?.life_cycle === 'TERMINATED') return;
    const t = setInterval(async () => {
      try {
        const r = await fetch(`/api/admin/reset-demo/status?run_id=${result.run_id}`);
        if (!r.ok) return;
        const s = await r.json();
        setStatus(s);
        if (s.life_cycle === 'TERMINATED') clearInterval(t);
      } catch {/* swallow */}
    }, 6000);
    return () => clearInterval(t);
  }, [result, status?.life_cycle]);

  if (result) {
    const finished = status?.life_cycle === 'TERMINATED';
    const success  = status?.result === 'SUCCESS';
    const phase    = status?.life_cycle || 'PENDING';
    const warm     = status?.summary?.ai_cache_warm;
    const bgClass  = !finished ? 'bg-blue-50 border-blue-200 text-blue-800'
                  : success    ? 'bg-emerald-50 border-emerald-200 text-emerald-800'
                               : 'bg-red-50 border-red-200 text-red-800';
    return (
      <div className={`inline-flex items-center gap-2 px-4 py-2 rounded border text-sm ${bgClass}`}>
        {finished
          ? (success ? <CheckCircle2 className="w-4 h-4" /> : <AlertCircle className="w-4 h-4" />)
          : <Loader2 className="w-4 h-4 animate-spin" />}
        <span>
          Reset {finished ? (success ? 'complete' : 'failed') : phase.toLowerCase()} —
          {' '}
          <a href={result.run_page_url} target="_blank" rel="noreferrer"
             className="font-medium underline inline-flex items-center gap-1">
            run #{result.run_id} <ExternalLink className="w-3 h-3" />
          </a>
        </span>
        {finished && warm?.called && (
          <span className="text-[11px] opacity-80">
            · AI cache warmed
          </span>
        )}
        <button onClick={() => { setResult(null); setStatus(null); }}
                className="ml-2 text-[11px] hover:underline">
          dismiss
        </button>
      </div>
    );
  }
  if (error) {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded bg-red-50 border border-red-200 text-sm text-red-800">
        <AlertCircle className="w-4 h-4" />
        Reset failed: {error.slice(0, 200)}
        <button onClick={() => setError(null)} className="ml-2 text-[11px] text-red-700 hover:underline">
          dismiss
        </button>
      </div>
    );
  }
  if (confirming) {
    return (
      <div className="inline-flex items-center gap-2 px-4 py-2 rounded bg-amber-50 border border-amber-300 text-sm text-amber-900">
        Reset will: revert champion aliases, clear compare results, drop simulated MTAs, restore the geospatial vendor refresh story.
        <button onClick={fire}
                disabled={running}
                className="ml-2 inline-flex items-center gap-1 px-3 py-1 rounded bg-amber-600 text-white hover:bg-amber-700 disabled:opacity-50">
          {running ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <RotateCcw className="w-3.5 h-3.5" />}
          Confirm reset
        </button>
        <button onClick={() => setConfirming(false)}
                className="text-[11px] text-amber-800 hover:underline">cancel</button>
      </div>
    );
  }
  return (
    <button onClick={() => setConfirming(true)}
            className="inline-flex items-center gap-2 px-3 py-1.5 rounded border border-gray-300 bg-white text-sm text-gray-700 hover:bg-gray-50">
      <RotateCcw className="w-4 h-4" />
      Reset demo state
    </button>
  );
}

// ---------------------------------------------------------------------------
// Reference architecture block — narrative closer for the platform pitch
// ---------------------------------------------------------------------------

function ArchBlock() {
  const layers = [
    {
      title: 'Sources',
      colour: 'blue',
      items: ['Internal book (policies, claims)', 'Vendor feeds (geospatial, credit, market)', 'Public data (ONSPD, IMD)'],
    },
    {
      title: 'Bronze → Silver',
      colour: 'cyan',
      items: ['DLT pipeline · expectations', 'HITL approval gate', 'Versioned, audited'],
    },
    {
      title: 'Modelling Mart',
      colour: 'green',
      items: ['Per-LOB feature tables (Delta)', 'Online Feature Store', 'AI/BI Genie over the mart'],
    },
    {
      title: 'Models',
      colour: 'purple',
      items: ['GLMs · GBMs (4 families)', 'MLflow + Unity Catalog registry', 'Model Factory · agent-narrated'],
    },
    {
      title: 'Champions',
      colour: 'red',
      items: ['UC alias-based versioning', 'One-click rollback', 'Mosaic AI Model Serving'],
    },
    {
      title: 'Rating engine',
      colour: 'emerald',
      items: ['Real-time API · <500ms', 'Batch enrichment', 'Factor-table export (Radar)'],
    },
  ];
  const colourMap: Record<string, { bg: string; border: string; pill: string; head: string }> = {
    blue:    { bg: 'bg-blue-50',    border: 'border-blue-200',    pill: 'bg-blue-200    text-blue-900',    head: 'text-blue-700' },
    cyan:    { bg: 'bg-cyan-50',    border: 'border-cyan-200',    pill: 'bg-cyan-200    text-cyan-900',    head: 'text-cyan-700' },
    green:   { bg: 'bg-green-50',   border: 'border-green-200',   pill: 'bg-green-200   text-green-900',   head: 'text-green-700' },
    purple:  { bg: 'bg-purple-50',  border: 'border-purple-200',  pill: 'bg-purple-200  text-purple-900',  head: 'text-purple-700' },
    red:     { bg: 'bg-red-50',     border: 'border-red-200',     pill: 'bg-red-200     text-red-900',     head: 'text-red-700' },
    emerald: { bg: 'bg-emerald-50', border: 'border-emerald-200', pill: 'bg-emerald-200 text-emerald-900', head: 'text-emerald-700' },
  };
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 overflow-x-auto">
      <div className="flex items-stretch gap-2 min-w-[860px]">
        {layers.map((l, i) => {
          const c = colourMap[l.colour];
          return (
            <>
              <div key={l.title} className={`flex-1 rounded-lg border ${c.border} ${c.bg} p-3`}>
                <div className={`text-[10px] uppercase tracking-wider font-bold ${c.head} mb-1.5`}>
                  Layer {i + 1}
                </div>
                <div className="text-sm font-semibold text-gray-900 mb-2">{l.title}</div>
                <div className="space-y-1">
                  {l.items.map(it => (
                    <div key={it} className={`text-[11px] px-1.5 py-0.5 rounded ${c.pill}`}>{it}</div>
                  ))}
                </div>
              </div>
              {i < layers.length - 1 && (
                <div key={`arrow-${i}`} className="flex items-center shrink-0 px-1">
                  <ArrowRight className="w-4 h-4 text-gray-400" />
                </div>
              )}
            </>
          );
        })}
      </div>
      <div className="mt-4 grid grid-cols-1 md:grid-cols-3 gap-2 text-[11px] text-gray-600">
        <div className="px-3 py-2 rounded bg-amber-50 border border-amber-200">
          <span className="font-semibold text-amber-900">Cross-cutting:</span>{' '}
          Unity Catalog lineage · audit log · governance packs · bias monitor
        </div>
        <div className="px-3 py-2 rounded bg-violet-50 border border-violet-200">
          <span className="font-semibold text-violet-900">Agents:</span>{' '}
          Claude Sonnet 4.6 · grounded over packs, audit log, mart
        </div>
        <div className="px-3 py-2 rounded bg-gray-50 border border-gray-200">
          <span className="font-semibold text-gray-800">Integration footprint:</span>{' '}
          one scoring adapter to your rating engine — nothing else changes
        </div>
      </div>
    </div>
  );
}
