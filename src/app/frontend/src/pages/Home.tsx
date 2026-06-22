import { Link } from 'react-router-dom';
import { Database, Table2, ArrowRight, Code } from 'lucide-react';

// AXA edition: data prep -> modelling mart -> model development.
// Live serving, deployment, governance, pricing AI, factory and add-ons are
// excluded from this build, so this landing page reflects only the kept spine.
export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Hero */}
      <div className="text-center mb-6">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Pricing Workbench</h1>
        <p className="text-lg text-blue-600 font-medium">Databricks Accelerator</p>
        <p className="text-gray-500 mt-3 max-w-3xl mx-auto">
          Commercial pricing data preparation and modelling on a single platform. Every step of the
          real data flow is traceable, auditable, and governed - from ingestion through the modelling
          mart to model development.
        </p>
      </div>

      {/* About this demo - single landing-page disclaimer */}
      <div className="max-w-3xl mx-auto mb-10 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3">
        <div className="text-xs font-semibold text-amber-900 uppercase tracking-wide mb-1">
          About this demo
        </div>
        <p className="text-sm text-amber-900/90 leading-relaxed">
          Bricksurance SE is a synthetic insurance carrier. All policies, quotes, claims, and
          director demographics are generated; the UK postcode enrichment is real public data.
          Everything here is illustrative.
        </p>
      </div>

      {/* Main flow - linear spine */}
      <FlowSpine />

      {/* Section cards */}
      <div className="mb-3 mt-10 flex items-end justify-between">
        <h2 className="text-sm font-semibold text-gray-500 uppercase tracking-wide">The pricing spine</h2>
        <span className="text-[11px] text-gray-500">left to right, every stage is a tab</span>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-8">
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
          description="Engineered feature table - every approved source joined on the active book. Factor catalog with per-factor provenance and an embedded AI/BI Genie."
          features={['Contributing sources', 'Factor catalog + lineage', 'AI/BI Genie']}
        />
        <SectionCard
          to="/development"
          icon={Code}
          color="purple"
          title="Model Development"
          description="Reference notebooks and a model library for actuaries and data scientists to build pricing models on the Modelling Mart."
          features={['Reference notebooks', 'GLM + GBM examples', 'MLflow + UC registry']}
        />
      </div>

      {/* Reference architecture */}
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
          <strong>This is not a Databricks product.</strong> It is an example of what can be built on the
          Databricks platform using standard capabilities (Unity Catalog, Delta Lake, MLflow,
          Databricks Apps, Feature Engineering). The full source code is public - fork it, adapt it,
          use it as a starting point.
        </p>
        <p className="text-sm text-gray-600">
          All company names (Bricksurance SE), policy data and financial figures are fictional. No real
          customer data. The postcode enrichment uses genuine UK public data (ONSPD + IMD 2019 + ONS RUC)
          under the Open Government Licence.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Flow spine
// ---------------------------------------------------------------------------

function FlowSpine() {
  const steps = [
    { to: '/datasets',      icon: Database, label: 'Data Ingestion',    sub: 'approved sources' },
    { to: '/pricing-table', icon: Table2,   label: 'Modelling Mart',    sub: 'feature table' },
    { to: '/development',   icon: Code,     label: 'Model Development', sub: 'build models on the mart' },
  ];
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 overflow-x-auto">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4">
        The data flow, end to end
      </h3>
      <div className="flex items-stretch gap-2 min-w-[520px]">
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
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section card
// ---------------------------------------------------------------------------

function SectionCard({ to, icon: Icon, color, title, description, features }: {
  to: string; icon: any; color: string; title: string; description: string;
  features: string[];
}) {
  const colorMap: Record<string, { bg: string; border: string; icon: string; badge: string }> = {
    blue:   { bg: 'bg-blue-50',   border: 'border-blue-200',   icon: 'text-blue-600',   badge: 'bg-blue-100 text-blue-700' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-200', icon: 'text-purple-600', badge: 'bg-purple-100 text-purple-700' },
    green:  { bg: 'bg-green-50',  border: 'border-green-200',  icon: 'text-green-600',  badge: 'bg-green-100 text-green-700' },
  };
  const c = colorMap[color] || colorMap.blue;
  return (
    <Link to={to}
          className={`group block ${c.bg} border ${c.border} rounded-lg p-5 hover:shadow-md transition-all`}>
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
// Reference architecture block (data prep -> mart -> models)
// ---------------------------------------------------------------------------

function ArchBlock() {
  const layers = [
    { title: 'Sources',        colour: 'blue',   items: ['Internal book (policies, claims)', 'Vendor feeds (geospatial, credit, market)', 'Public data (ONSPD, IMD)'] },
    { title: 'Bronze → Silver', colour: 'cyan',   items: ['DLT pipeline · expectations', 'HITL approval gate', 'Versioned, audited'] },
    { title: 'Modelling Mart', colour: 'green',  items: ['Per-LOB feature tables (Delta)', 'Factor catalog + lineage', 'AI/BI Genie over the mart'] },
    { title: 'Models',         colour: 'purple', items: ['GLMs · GBMs', 'MLflow + Unity Catalog registry', 'Reference notebooks'] },
  ];
  const colourMap: Record<string, { bg: string; border: string; pill: string; head: string }> = {
    blue:   { bg: 'bg-blue-50',   border: 'border-blue-200',   pill: 'bg-blue-200   text-blue-900',   head: 'text-blue-700' },
    cyan:   { bg: 'bg-cyan-50',   border: 'border-cyan-200',   pill: 'bg-cyan-200   text-cyan-900',   head: 'text-cyan-700' },
    green:  { bg: 'bg-green-50',  border: 'border-green-200',  pill: 'bg-green-200  text-green-900',  head: 'text-green-700' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-200', pill: 'bg-purple-200 text-purple-900', head: 'text-purple-700' },
  };
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 overflow-x-auto">
      <div className="flex items-stretch gap-2 min-w-[720px]">
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
    </div>
  );
}
