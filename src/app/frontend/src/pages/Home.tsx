import { Link } from 'react-router-dom';
import {
  Database, FlaskConical, Table2, Shield, ArrowRight, Receipt, BookOpen, ExternalLink,
  Phone, AlertTriangle, Clock, CheckSquare,
} from 'lucide-react';

const GITHUB_REPO_URL = 'https://github.com/wryszka/pricing-workbench';

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Hero */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">Pricing Workbench</h1>
        <p className="text-lg text-blue-600 font-medium">Databricks Accelerator</p>
        <p className="text-gray-500 mt-3 max-w-3xl mx-auto">
          End-to-end commercial pricing on a single platform. Every step of the real data flow is
          traceable, auditable, and governed — from the first quote request to the policy renewal.
        </p>
      </div>

      {/* Data flow diagram */}
      <FlowDiagram />

      {/* Section cards */}
      <div className="grid grid-cols-2 gap-5 mb-8">
        <SectionCard
          to="/datasets"
          icon={Database}
          color="blue"
          title="External Data"
          description="Reference and vendor data joined into the feature vector at both quote AND policy time: ONS Postcode Directory + IMD 2019, market benchmarks, geospatial hazard, credit bureau. HITL approval flow with DQ checks."
          features={["1.5M postcode enrichment (real UK data)", "DLT expectations", "Actuary approval gate"]}
        />
        <SectionCard
          to="/quote-review"
          icon={Receipt}
          color="red"
          title="Quote Stream"
          description="The serving-time feature shape. Every quote — Jane's form, the rating engine call, the pricing response — captured as three JSON payloads. Investigation flow for outliers; training data for the demand model."
          features={["3 JSON payloads per transaction", "Simulated model replay", "AI-analyst RCA"]}
        />
        <SectionCard
          to="/pricing-table"
          icon={Table2}
          color="green"
          title="Pricing Feature Table"
          description="Engineered from every approved source — policies, claims, vendor feeds (market + geo + credit), plus the real UK postcode enrichment. Rebuild pipeline, feature catalog with per-feature provenance, and promotable online store."
          features={["8 contributing sources", "Feature catalog + lineage", "Offline ↔ Online toggle"]}
        />
        <SectionCard
          to="/models"
          icon={FlaskConical}
          color="purple"
          title="Model Factory"
          description="Train, evaluate, and approve pricing models. Frequency GLMs, severity GBMs, demand GBMs. 50-spec factory for Radar-class ranking. MLflow + UC registry with full lineage."
          features={["MLflow experiment tracking", "Unity Catalog model registry", "AI-assisted model selection"]}
        />
        <SectionCard
          external
          to={`${GITHUB_REPO_URL}/tree/main/src/new_data_impact`}
          icon={BookOpen}
          color="indigo"
          title="New Data Impact"
          description="For data scientists and actuaries: six notebooks that answer 'does adding real external data actually make pricing models better?' Builds the ~1.5M postcode enrichment, trains standard vs enriched models, quantifies the lift."
          features={["Gini 0.11 → 0.25", "50-spec Model Factory", "Claude review agent", "Governance PDF"]}
        />
        <SectionCard
          to="/governance"
          icon={Shield}
          color="amber"
          title="Governance & Audit"
          description="Complete audit trail across the whole flow — data approvals, feature provenance, model decisions, LLM calls. Regulatory-grade PDF export."
          features={["Feature catalog + lineage", "Immutable audit log", "Regulatory export (PDF + JSON)"]}
        />
      </div>

      {/* About */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
        <h3 className="font-semibold text-gray-800 mb-2">About this demo</h3>
        <p className="text-sm text-gray-600 mb-3">
          <strong>This is not a Databricks product.</strong> It is an example of what can be built
          on the Databricks platform using standard capabilities (Unity Catalog, Delta Lake, MLflow,
          Mosaic AI, Databricks Apps, Feature Engineering, Lakebase). The full source code is
          available on GitHub — feel free to fork it, adapt it to your own data, and use it as a
          starting point.
        </p>
        <p className="text-sm text-gray-600">
          All company names (Bricksurance SE), policy data, and financial figures are entirely
          fictional and generated for illustrative purposes. No real customer data is used. The
          postcode enrichment is built from genuine UK public data (ONSPD + IMD 2019 + ONS RUC) —
          freely available under the Open Government Licence.
        </p>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Flow diagram — the real data flow, step by step
// ---------------------------------------------------------------------------

function FlowDiagram() {
  return (
    <div className="mb-10 bg-white border border-gray-200 rounded-lg p-6">
      <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wide mb-4">
        How the data flows
      </h3>
      <div className="flex items-stretch gap-2 overflow-x-auto pb-2">
        <FlowStep
          icon={Database}
          title="Sources"
          subtitle="approved vendor feeds + internal systems + enrichment"
          color="blue"
          to="/datasets"
        />
        <FlowArrow label="build" />
        <FlowStep
          icon={Table2}
          title="Feature table"
          subtitle="engineered from every approved source"
          color="green"
          to="/pricing-table"
        />
        <FlowArrow />
        <FlowStep
          icon={Phone}
          title="Quote request"
          subtitle="Jane submits — rating factors arrive"
          color="red"
          to="/quote-review"
        />
        <FlowArrow label="+ enrichment" />
        <FlowStep
          icon={Receipt}
          title="Pricing model"
          subtitle="feature vector → Freq × Severity → quote"
          color="purple"
          to="/deployment"
        />
        <FlowArrow label="if bound" />
        <FlowStep
          icon={CheckSquare}
          title="Policy"
          subtitle="feature snapshot stored"
          color="green"
          to="/pricing-table"
        />
        <FlowArrow label="+ claims" />
        <FlowStep
          icon={AlertTriangle}
          title="Outcomes"
          subtitle="claim count + loss observed"
          color="amber"
        />
        <FlowArrow label="feedback" />
        <FlowStep
          icon={Clock}
          title="Retrain"
          subtitle="training feature store updated"
          color="indigo"
          to="/models"
        />
      </div>
      <p className="text-xs text-gray-500 mt-4 leading-relaxed">
        The <strong className="text-gray-700">Feature Table</strong> is an engineered artifact —
        built by the <code className="bg-gray-100 px-1 rounded">build_upt</code> pipeline from every
        approved source. Policies is just one contributor; market, geo, credit, the real UK postcode
        enrichment, and derived factors all join together.{' '}
        <strong className="text-gray-700">Training</strong> reads the feature table with observed
        outcomes attached. <strong className="text-gray-700">Serving</strong> reads the same table
        (online, via Lakebase) for renewals, or takes the feature vector directly for fresh business.
      </p>
    </div>
  );
}

function FlowStep({
  icon: Icon, title, subtitle, color, to,
}: {
  icon: any; title: string; subtitle: string; color: string; to?: string;
}) {
  const colorMap: Record<string, string> = {
    blue:   'bg-blue-50 border-blue-200 text-blue-700',
    red:    'bg-red-50 border-red-200 text-red-700',
    purple: 'bg-purple-50 border-purple-200 text-purple-700',
    green:  'bg-green-50 border-green-200 text-green-700',
    amber:  'bg-amber-50 border-amber-200 text-amber-700',
    indigo: 'bg-indigo-50 border-indigo-200 text-indigo-700',
  };
  const card = (
    <div className={`w-40 shrink-0 rounded-lg border p-3 ${colorMap[color]} hover:shadow-sm transition-all cursor-${to ? 'pointer' : 'default'}`}>
      <Icon className="w-4 h-4 mb-1.5" />
      <div className="text-xs font-semibold text-gray-900 leading-tight">{title}</div>
      <div className="text-[10px] text-gray-600 mt-0.5 leading-snug">{subtitle}</div>
    </div>
  );
  return to ? <Link to={to} className="block">{card}</Link> : card;
}

function FlowArrow({ label }: { label?: string }) {
  return (
    <div className="flex flex-col items-center justify-center shrink-0 pt-4 min-w-[36px]">
      <ArrowRight className="w-4 h-4 text-gray-400" />
      {label && <div className="text-[9px] text-gray-400 font-medium mt-0.5 whitespace-nowrap">{label}</div>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section card — supports internal Link or external <a>
// ---------------------------------------------------------------------------

function SectionCard({ to, icon: Icon, color, title, description, features, external }: {
  to: string; icon: any; color: string; title: string; description: string; features: string[];
  external?: boolean;
}) {
  const colorMap: Record<string, { bg: string; border: string; icon: string; badge: string }> = {
    blue:   { bg: 'bg-blue-50',   border: 'border-blue-200',   icon: 'text-blue-600',   badge: 'bg-blue-100 text-blue-700' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-200', icon: 'text-purple-600', badge: 'bg-purple-100 text-purple-700' },
    green:  { bg: 'bg-green-50',  border: 'border-green-200',  icon: 'text-green-600',  badge: 'bg-green-100 text-green-700' },
    amber:  { bg: 'bg-amber-50',  border: 'border-amber-200',  icon: 'text-amber-600',  badge: 'bg-amber-100 text-amber-700' },
    red:    { bg: 'bg-red-50',    border: 'border-red-200',    icon: 'text-red-600',    badge: 'bg-red-100 text-red-700' },
    indigo: { bg: 'bg-indigo-50', border: 'border-indigo-200', icon: 'text-indigo-600', badge: 'bg-indigo-100 text-indigo-700' },
  };
  const c = colorMap[color] || colorMap.blue;

  const body = (
    <>
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${c.icon}`} />
        <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">{title}</h3>
        {external
          ? <ExternalLink className="w-4 h-4 text-gray-400 ml-auto" />
          : <ArrowRight className="w-4 h-4 text-gray-400 ml-auto group-hover:translate-x-1 transition-transform" />}
      </div>
      <p className="text-sm text-gray-600 mb-3">{description}</p>
      <div className="flex flex-wrap gap-1.5">
        {features.map((f, i) => (
          <span key={i} className={`px-2 py-0.5 rounded text-[10px] font-medium ${c.badge}`}>{f}</span>
        ))}
      </div>
    </>
  );

  const className = `block ${c.bg} border ${c.border} rounded-lg p-5 hover:shadow-md transition-all group`;
  return external
    ? <a href={to} target="_blank" rel="noopener noreferrer" className={className}>{body}</a>
    : <Link to={to} className={className}>{body}</Link>;
}
