import { Link } from 'react-router-dom';
import { Receipt, BookOpen, ArrowRight, Package, Radar, Network } from 'lucide-react';

/**
 * Add-ons landing page — the home for useful tools that sit alongside the
 * core pricing flow but aren't part of the train → promote → deploy →
 * govern spine. Each tile opens a full feature underneath.
 */
export default function Addons() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Package className="w-6 h-6 text-gray-700" /> Add-ons
        </h2>
        <p className="text-gray-500 mt-1">
          Supporting tools and companion workflows. Not part of the main pricing flow, but things
          different audiences reach for at specific moments.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-5">
        <AddonCard
          to="/add-ons/quote-review"
          icon={Receipt}
          accent="red"
          title="Quote Review"
          description="Inspect individual quotes end to end — the three JSON payloads (request, rating engine call, pricing response), model replay, and AI-assisted root-cause analysis for outliers."
          audience="underwriters · pricing ops · complaint handlers"
          tags={['Transaction-level drill-down', 'Simulated model replay', 'RCA agent']}
        />
        <AddonCard
          to="/add-ons/new-data-impact"
          icon={BookOpen}
          accent="indigo"
          title="New Data Impact"
          description="For data scientists and actuaries: six notebooks that answer 'does adding real external data make pricing models better?'. Builds ~1.5M-row postcode enrichment, trains standard vs enriched models, quantifies the lift."
          audience="data scientists · senior actuaries"
          tags={['Gini 0.11 → 0.25', 'ONSPD + IMD enrichment', 'Claude review agent', 'Governance PDF']}
        />
        <AddonCard
          to="/add-ons/rating-engine"
          icon={Radar}
          accent="emerald"
          title="Rating Engine Integration"
          description="The workbench acts as an enrichment layer for commercial rating engines like Willis Towers Watson Radar and Earnix — delivering scored factors, features, and loading signals into the rating engine without disrupting the actuary's existing workflow."
          audience="pricing actuaries · rating-engine administrators"
          tags={['Radar / Earnix', 'Feature enrichment', 'Reference architecture']}
        />
        <AddonCard
          to="/add-ons/agentic-distribution"
          icon={Network}
          accent="violet"
          title="Agentic Distribution"
          description="Being present and priceable when the customer arrives through an AI agent instead of a website. The pricing engine published as services — discover requirements, get a real price, ask why — over a conversational journey and over MCP for outside agents."
          audience="distribution · digital · pricing leadership"
          tags={['MCP server', 'Claude-driven journey', 'Real engine prices', 'Channel telemetry']}
        />
      </div>
    </div>
  );
}

function AddonCard({ to, icon: Icon, accent, title, description, audience, tags }: {
  to: string; icon: any; accent: 'red' | 'indigo' | 'blue' | 'emerald' | 'violet';
  title: string; description: string; audience: string; tags: string[];
}) {
  const colors = {
    red:     { bg: 'bg-red-50',     border: 'border-red-200',     icon: 'text-red-600',     badge: 'bg-red-100 text-red-700' },
    indigo:  { bg: 'bg-indigo-50',  border: 'border-indigo-200',  icon: 'text-indigo-600',  badge: 'bg-indigo-100 text-indigo-700' },
    blue:    { bg: 'bg-blue-50',    border: 'border-blue-200',    icon: 'text-blue-600',    badge: 'bg-blue-100 text-blue-700' },
    emerald: { bg: 'bg-emerald-50', border: 'border-emerald-200', icon: 'text-emerald-600', badge: 'bg-emerald-100 text-emerald-700' },
    violet:  { bg: 'bg-violet-50',  border: 'border-violet-200',  icon: 'text-violet-600',  badge: 'bg-violet-100 text-violet-700' },
  }[accent];

  return (
    <Link to={to}
          className={`group block rounded-lg border p-5 hover:shadow-md transition-all ${colors.bg} ${colors.border}`}>
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${colors.icon}`} />
        <h3 className="font-semibold text-gray-900 group-hover:text-blue-700">{title}</h3>
        <ArrowRight className="w-4 h-4 text-gray-400 ml-auto group-hover:translate-x-1 transition-transform" />
      </div>
      <p className="text-sm text-gray-700 mb-2">{description}</p>
      <div className="text-[11px] text-gray-500 italic mb-3">For: {audience}</div>
      <div className="flex flex-wrap gap-1.5">
        {tags.map(t => (
          <span key={t} className={`px-2 py-0.5 rounded text-[10px] font-medium ${colors.badge}`}>{t}</span>
        ))}
      </div>
    </Link>
  );
}
