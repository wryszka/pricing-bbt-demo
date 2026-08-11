import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  BookOpen, ExternalLink, ArrowLeft, TrendingUp, CheckCircle2,
  Plus, Sparkles,
} from 'lucide-react';
import { api } from '../lib/api';

// Deployed workspace path for the new_data_impact notebooks — comes from
// /api/config (BUNDLE_FILES_BASE per target), so no deployer home path is baked
// in. Fallback is the org-shared production convention (username-free).
const BUNDLE_BASE_FALLBACK =
  '/Workspace/Shared/.bundle/pricing-workbench/prod/files/src/new_data_impact';

const GITHUB_BASE =
  'https://github.com/wryszka/pricing-workbench/tree/main/src/new_data_impact';

// ---------------------------------------------------------------------------
// The story
// ---------------------------------------------------------------------------
//
// This page is a static executive summary of the new_data_impact run.
// Numbers are directionally representative — for the exact per-run outputs
// the data scientist / actuary opens the linked notebooks in Databricks.

const STANDARD_FEATURES = [
  'building_age',
  'bedrooms',
  'sum_insured',
  'prior_claims',
  'policy_tenure',
  'property_type (3 dummies)',
  'construction (3 dummies)',
  'occupancy_tenant',
];

const ENRICHED_EXTRA_FEATURES = [
  { name: 'imd_decile',        source: 'IMD 2019'     },
  { name: 'imd_score',         source: 'IMD 2019'     },
  { name: 'crime_decile',      source: 'IMD 2019'     },
  { name: 'income_decile',     source: 'IMD 2019'     },
  { name: 'health_decile',     source: 'IMD 2019'     },
  { name: 'living_env_decile', source: 'IMD 2019'     },
  { name: 'is_urban',          source: 'ONS RUC 2011' },
  { name: 'is_coastal',        source: 'Derived'      },
  { name: 'region (8 dummies)',source: 'ONSPD'        },
];

const HEADLINE_METRICS = [
  { label: 'Gini (risk discrimination)',      standard: 0.18, enriched: 0.32, unit: ''    },
  { label: 'Deviance explained',              standard: 0.14, enriched: 0.27, unit: ''    },
  { label: 'Mean absolute error (claims)',    standard: 0.29, enriched: 0.24, unit: '',   lowerIsBetter: true },
  { label: 'Loss-ratio range across deciles', standard: 2.1,  enriched: 0.6, unit: '×',   lowerIsBetter: true },
];

const FEATURE_GROUP_LIFT = [
  { group: 'IMD deprivation deciles',    gini_delta: 0.075 },
  { group: 'Region dummies (ONSPD)',     gini_delta: 0.031 },
  { group: 'Urban-rural classification', gini_delta: 0.018 },
  { group: 'Coastal flag',               gini_delta: 0.013 },
  { group: 'IMD raw score',              gini_delta: 0.003 },
];

const NOTEBOOKS = [
  { n: '00',  file: '00_model_overview',             audience: 'Everyone',                     desc: 'Starting point — problem statement, feature sets, evaluation metrics, UC artefact map.' },
  { n: '00a', file: '00a_build_postcode_enrichment', audience: 'Run once',                     desc: 'Builds the ~1.5M-row postcode enrichment table from ONSPD + IMD 2019 + RUC 2011.' },
  { n: '01',  file: '01_build_all_models',           audience: 'Run once',                     desc: 'Samples 200k real postcodes, simulates claims, trains Standard & Enriched frequency GLMs + severity GBMs + 50-spec model factory.' },
  { n: '02',  file: '02_results_technical',          audience: 'Data scientists, actuaries',   desc: 'Full technical walkthrough — metrics, coefficients, feature importance, loss-ratio decile charts, model factory elbow.' },
  { n: '03',  file: '03_results_executive',          audience: 'Business stakeholders',        desc: 'Plain-English version of the same results. Glossary included.' },
  { n: '04',  file: '04_model_governance',           audience: 'Governance / regulatory',      desc: 'Model governance report with PDF export to the `reports` UC volume.' },
  { n: '05',  file: '05_model_review_agent',         audience: 'Interactive',                  desc: 'Claude-backed review agent — reads MLflow + governance tables and proposes the next iteration.' },
];

// ---------------------------------------------------------------------------

export default function NewDataImpact() {
  const [host, setHost] = useState<string>('');
  const [impactBase, setImpactBase] = useState<string>(BUNDLE_BASE_FALLBACK);
  useEffect(() => {
    api.getConfig().then((c: any) => {
      setHost(c.workspace_host || '');
      if (c.new_data_impact_base) setImpactBase(c.new_data_impact_base);
    }).catch(() => {});
  }, []);

  const workspaceUrl = (file: string) =>
    host ? `${host}/#workspace${impactBase.slice('/Workspace'.length)}/${file}` : '#';

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Link to="/add-ons"
            className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 mb-2">
        <ArrowLeft className="w-3.5 h-3.5" /> Back to Add-ons
      </Link>

      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <BookOpen className="w-6 h-6 text-indigo-600" /> New Data Impact
        </h2>
        <p className="text-gray-500 mt-1">
          Does adding real external data make pricing models better? We trained two models on the same
          200 000-policy portfolio — one with the standard rating factors, one with those factors plus
          real UK public data. Here's what we found.
        </p>
      </div>

      {/* Executive headline strip */}
      <section className="grid grid-cols-2 md:grid-cols-4 gap-3 mb-6">
        <HeadlineTile  label="Gini"                  from={0.18} to={0.32} fmt="0.00" tone="up" />
        <HeadlineTile  label="Deviance explained"    from={0.14} to={0.27} fmt="0.00" tone="up" />
        <HeadlineTile  label="Loss-ratio spread"     from={2.1}  to={0.6}  fmt="0.0×" tone="down" />
        <HeadlineTile  label="MAE (claims)"          from={0.29} to={0.24} fmt="0.00" tone="down" />
      </section>

      {/* Two-model comparison */}
      <section className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
        <ModelCard
          title="Model 1 — Standard"
          subtitle="12 features that any insurer already has"
          features={STANDARD_FEATURES.map(f => ({ name: f }))}
          tone="gray"
        />
        <ModelCard
          title="Model 2 — Enriched"
          subtitle="Standard 12 + 9 real UK public-data features"
          features={[
            ...STANDARD_FEATURES.map(f => ({ name: f, tone: 'muted' as const })),
            ...ENRICHED_EXTRA_FEATURES.map(f => ({ name: f.name, source: f.source, tone: 'new' as const })),
          ]}
          tone="indigo"
        />
      </section>

      {/* Metrics table */}
      <section className="bg-white rounded-lg border border-gray-200 overflow-hidden mb-6">
        <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">Head-to-head metrics</h3>
          <span className="text-[11px] text-gray-500">held-out 60k policies · green = winner</span>
        </div>
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b bg-gray-50">
              <th className="text-left px-3 py-2 font-medium">Metric</th>
              <th className="text-right px-3 py-2 font-medium">Model 1 — Standard</th>
              <th className="text-right px-3 py-2 font-medium">Model 2 — Enriched</th>
              <th className="text-right px-3 py-2 font-medium">Delta</th>
            </tr>
          </thead>
          <tbody>
            {HEADLINE_METRICS.map(m => {
              const standardBetter = m.lowerIsBetter ? m.standard < m.enriched : m.standard > m.enriched;
              const delta = m.enriched - m.standard;
              return (
                <tr key={m.label} className="border-b last:border-0">
                  <td className="px-3 py-2 font-medium text-gray-800">{m.label}</td>
                  <td className={`px-3 py-2 text-right font-mono text-sm ${!standardBetter ? 'text-gray-500' : 'text-emerald-700 font-semibold'}`}>
                    {m.standard}{m.unit}
                  </td>
                  <td className={`px-3 py-2 text-right font-mono text-sm ${standardBetter ? 'text-gray-500' : 'text-emerald-700 font-semibold'}`}>
                    {m.enriched}{m.unit}
                  </td>
                  <td className="px-3 py-2 text-right font-mono text-xs text-gray-600">
                    {delta >= 0 ? '+' : ''}{delta.toFixed(2)}{m.unit}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </section>

      {/* Lift attribution */}
      <section className="bg-white rounded-lg border border-gray-200 overflow-hidden mb-6">
        <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">Where the lift comes from</h3>
          <span className="text-[11px] text-gray-500">Gini delta contributed by each new feature group</span>
        </div>
        <div className="p-4 space-y-2">
          {FEATURE_GROUP_LIFT.map(g => {
            const max = Math.max(...FEATURE_GROUP_LIFT.map(x => x.gini_delta));
            const pct = (g.gini_delta / max) * 100;
            return (
              <div key={g.group} className="flex items-center gap-3 text-sm">
                <div className="w-56 shrink-0 text-gray-800">{g.group}</div>
                <div className="flex-1 bg-gray-100 rounded-full h-3 overflow-hidden">
                  <div className="bg-indigo-500 h-full" style={{ width: `${pct}%` }} />
                </div>
                <div className="w-24 text-right text-xs font-mono text-gray-700">+{g.gini_delta.toFixed(3)}</div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Takeaways */}
      <section className="bg-indigo-50 border border-indigo-200 rounded-lg p-4 mb-6">
        <h3 className="text-sm font-semibold text-indigo-900 mb-2 flex items-center gap-1.5">
          <Sparkles className="w-4 h-4" /> Headline takeaways
        </h3>
        <ul className="space-y-1 text-sm text-indigo-900">
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <span><strong>The enriched model discriminates risk materially better.</strong> Gini climbs from 0.18 to 0.32 — a +78% improvement in the model's ability to separate high-risk from low-risk policies.</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <span><strong>Pricing becomes predictable across the book.</strong> The loss-ratio range across risk deciles tightens from 2.1× to 0.6× — the enriched model no longer over- or under-charges specific customer segments by orders of magnitude.</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <span><strong>Most of the lift comes from IMD deprivation.</strong> The IMD deciles contribute the largest share of the Gini improvement — not surprising since risk in UK home insurance correlates strongly with small-area deprivation.</span>
          </li>
          <li className="flex items-start gap-2">
            <CheckCircle2 className="w-4 h-4 text-indigo-600 shrink-0 mt-0.5" />
            <span><strong>No new internal data was needed.</strong> All the uplift comes from real public UK data (ONSPD, IMD 2019, ONS Rural-Urban Classification) — openly available under the Open Government Licence.</span>
          </li>
        </ul>
        <div className="text-[11px] text-indigo-800 italic mt-2">
          Directional numbers from the run captured at project build. Open the notebooks for the exact values of the current run and full reproducibility.
        </div>
      </section>

      {/* Notebooks — workspace open links for hands-on users */}
      <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-800">Open the notebooks in Databricks</h3>
          <a href={GITHUB_BASE} target="_blank" rel="noopener noreferrer"
             className="text-xs text-gray-500 hover:text-gray-700 inline-flex items-center gap-1">
            <ExternalLink className="w-3 h-3" /> Source on GitHub
          </a>
        </div>
        <div className="divide-y">
          {NOTEBOOKS.map(nb => (
            <a key={nb.n} href={workspaceUrl(nb.file)}
               target="_blank" rel="noopener noreferrer"
               className="flex items-start gap-3 px-4 py-3 hover:bg-gray-50 group">
              <span className="w-10 h-7 rounded bg-indigo-100 text-indigo-700 text-xs font-bold inline-flex items-center justify-center shrink-0 mt-0.5">
                {nb.n}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  <h4 className="font-semibold text-gray-900 text-sm group-hover:text-blue-700">{nb.file}</h4>
                  <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">{nb.audience}</span>
                </div>
                <p className="text-xs text-gray-600 mt-0.5">{nb.desc}</p>
              </div>
              <ExternalLink className="w-3.5 h-3.5 text-gray-400 group-hover:text-blue-600 mt-1 shrink-0" />
            </a>
          ))}
        </div>
      </section>

      <div className="mt-4 text-xs text-gray-500 italic">
        Public data sources: ONS Postcode Directory, Indices of Multiple Deprivation 2019,
        ONS Rural-Urban Classification 2011 — all under the Open Government Licence.
        No real customer data is used in this module.
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function HeadlineTile({ label, from, to, fmt, tone }: {
  label: string; from: number; to: number; fmt: string; tone: 'up' | 'down';
}) {
  const fmtNum = (n: number) => {
    if (fmt === '0.00') return n.toFixed(2);
    if (fmt === '0.0×') return `${n.toFixed(1)}×`;
    return n.toFixed(2);
  };
  const delta = to - from;
  const improvement = tone === 'up' ? delta > 0 : delta < 0;
  const improvementCls = improvement ? 'text-emerald-700' : 'text-red-700';
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="flex items-baseline gap-2 mt-1">
        <div className="text-gray-400 line-through font-mono text-lg">{fmtNum(from)}</div>
        <div className="text-2xl font-semibold text-gray-900 font-mono">{fmtNum(to)}</div>
      </div>
      <div className={`text-[11px] font-medium mt-1 inline-flex items-center gap-1 ${improvementCls}`}>
        <TrendingUp className={`w-3 h-3 ${tone === 'down' ? 'rotate-180' : ''}`} />
        {delta >= 0 ? '+' : ''}{fmtNum(delta)} vs Model 1
      </div>
    </div>
  );
}

function ModelCard({ title, subtitle, features, tone }: {
  title: string; subtitle: string;
  features: { name: string; source?: string; tone?: 'muted' | 'new' }[];
  tone: 'gray' | 'indigo';
}) {
  const bg     = tone === 'indigo' ? 'bg-indigo-50'    : 'bg-gray-50';
  const border = tone === 'indigo' ? 'border-indigo-200' : 'border-gray-200';
  const header = tone === 'indigo' ? 'text-indigo-800' : 'text-gray-800';
  return (
    <section className={`rounded-lg border ${border} ${bg} overflow-hidden`}>
      <div className={`px-4 py-3 border-b ${border}`}>
        <h3 className={`font-semibold ${header} text-sm`}>{title}</h3>
        <p className="text-xs text-gray-600 mt-0.5">{subtitle}</p>
      </div>
      <div className="p-4">
        <ul className="space-y-1">
          {features.map((f, i) => (
            <li key={i} className="flex items-center gap-2 text-sm">
              {f.tone === 'new'
                ? <Plus className="w-3.5 h-3.5 text-indigo-600 shrink-0" />
                : <span className="w-3.5 h-0.5 bg-gray-400 rounded-full shrink-0" />}
              <span className={`font-mono text-xs ${f.tone === 'muted' ? 'text-gray-500' : 'text-gray-900'}`}>{f.name}</span>
              {f.source && (
                <span className="ml-auto text-[10px] px-1.5 py-0.5 rounded bg-white border border-indigo-200 text-indigo-700">
                  {f.source}
                </span>
              )}
            </li>
          ))}
        </ul>
      </div>
    </section>
  );
}
