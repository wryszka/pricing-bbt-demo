import { useEffect, useState } from 'react';
import {
  Code, ExternalLink, FlaskConical, GitCompare, ChevronDown, Library, Clock,
  ArrowUpRight,
} from 'lucide-react';
import { api } from '../lib/api';

const GITHUB_REPO_URL = 'https://github.com/wryszka/pricing-workbench';

export default function ModelDevelopment() {
  const [notebooks,   setNotebooks]  = useState<any[]>([]);
  const [libraries,   setLibraries]  = useState<any[]>([]);
  const [recentRuns,  setRecentRuns] = useState<any[]>([]);
  const [challenger,  setChallenger] = useState<any>(null);
  const [config,      setConfig]     = useState<any>(null);
  const [opening,     setOpening]    = useState<string | null>(null);
  const [showLibs,    setShowLibs]   = useState(false);
  const [showLibrary, setShowLibrary]= useState(false);

  useEffect(() => {
    api.getDevelopmentNotebooks().then((d: any) => {
      setNotebooks(d.notebooks || []);
      setLibraries(d.libraries || []);
    }).catch(() => {});
    api.getRecentMlflowRuns(10).then((d: any) => setRecentRuns(d.runs || [])).catch(() => {});
    api.getChallengerComparison().then(setChallenger).catch(() => {});
    api.getConfig().then(setConfig).catch(() => {});
  }, []);

  const featured   = notebooks.filter(n => n.is_featured);
  const moreBuilt  = notebooks.filter(n => !n.is_featured && n.status === 'built');
  const onRequest  = notebooks.filter(n => n.status === 'on_request');

  const openNotebook = async (id: string) => {
    setOpening(id);
    try {
      const r: any = await api.openNotebook(id);
      if (r?.workspace_url) window.open(r.workspace_url, '_blank', 'noopener,noreferrer');
    } finally {
      setOpening(null);
    }
  };

  const workspaceHost = config?.workspace_host || '';

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Header */}
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Model Development</h2>
        <p className="text-gray-500 mt-1">
          Where actuaries and data scientists build pricing models. Every notebook reads the Modelling Mart, runs on serverless ML compute, and logs to MLflow for governance.
        </p>
      </div>

      {/* Databricks features callout */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 mb-6">
        <h3 className="font-semibold text-blue-800 mb-2 text-sm">Databricks features demonstrated</h3>
        <div className="flex flex-wrap gap-1.5">
          {['MLflow experiment tracking', 'Unity Catalog Model Registry',
            'FeatureLookup (auto-binding at serving)', 'Serverless ML compute',
            'Delta-backed training sets', 'Unity Catalog governance'].map(f => (
            <span key={f} className="px-2 py-0.5 rounded text-[11px] font-medium bg-blue-100 text-blue-700">{f}</span>
          ))}
        </div>
      </div>

      {/* Featured notebooks — 4 headline cards */}
      <section className="mb-8">
        <h3 className="text-base font-semibold text-gray-800 mb-3">Start here — reference notebooks</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {featured.map(nb => (
            <FeaturedCard key={nb.id} nb={nb} opening={opening === nb.id} onOpen={() => openNotebook(nb.id)} />
          ))}
        </div>
      </section>

      {/* Model library — "Can you also do this?" tiles */}
      <section className="mb-8 bg-white border border-gray-200 rounded-lg overflow-hidden">
        <button onClick={() => setShowLibrary(v => !v)}
                className="w-full px-5 py-3 bg-gray-50 border-b flex items-center justify-between hover:bg-gray-100">
          <div className="text-left">
            <h3 className="text-base font-semibold text-gray-800">Can you also do…?</h3>
            <p className="text-xs text-gray-500 mt-0.5">
              Every pricing-model type an actuary asks about. <span className="text-green-600">✓</span> = runnable notebook in this demo · <span className="text-amber-600">🚧</span> = supported, drop it in when needed.
            </p>
          </div>
          <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform shrink-0 ml-3 ${showLibrary ? 'rotate-180' : ''}`} />
        </button>
        {showLibrary && (
          <div className="p-5 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
            {[...moreBuilt, ...onRequest].map(nb => (
              <ModelTile key={nb.id} nb={nb} opening={opening === nb.id} onOpen={() => openNotebook(nb.id)} />
            ))}
          </div>
        )}
      </section>

      {/* Proof of lift — preserved from the old page, this is high-value actuarial content */}
      <ChallengerPanel data={challenger} />

      {/* Recent MLflow runs — live */}
      <section className="mb-6 bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="px-5 py-3 bg-gray-50 border-b flex items-center justify-between">
          <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2">
            <Clock className="w-4 h-4 text-gray-600" /> Recent training runs
          </h3>
          <span className="text-xs text-gray-500">Live from MLflow · experiments matching <code>pricing_workbench_*</code></span>
        </div>
        <div className="p-5">
          {recentRuns.length === 0 ? (
            <div className="text-xs text-gray-500 italic py-2">
              No runs yet. Open one of the notebooks above and train a model — it'll appear here.
            </div>
          ) : (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-500 border-b">
                  <th className="text-left py-1.5 pr-3 font-medium">Run</th>
                  <th className="text-left py-1.5 pr-3 font-medium">Experiment</th>
                  <th className="text-left py-1.5 pr-3 font-medium">Started</th>
                  <th className="text-left py-1.5 pr-3 font-medium">User</th>
                  <th className="text-right py-1.5 pr-3 font-medium">Key metric</th>
                  <th className="text-right py-1.5 font-medium">&nbsp;</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.map((r: any) => (
                  <tr key={r.run_id} className="border-b last:border-b-0 hover:bg-gray-50">
                    <td className="py-1.5 pr-3 font-medium text-gray-800">{r.run_name}</td>
                    <td className="py-1.5 pr-3 text-xs text-gray-500 font-mono truncate max-w-xs">
                      {(r.experiment_name || '').split('/').pop()}
                    </td>
                    <td className="py-1.5 pr-3 text-xs text-gray-600">{formatRelative(r.start_time)}</td>
                    <td className="py-1.5 pr-3 text-xs text-gray-600">{(r.user || '—').split('@')[0]}</td>
                    <td className="py-1.5 pr-3 text-xs text-right">
                      {r.key_metric ? (
                        <span className="font-mono">
                          <span className="text-gray-500">{r.key_metric.name}:</span>{' '}
                          <span className="text-gray-800 font-medium">{r.key_metric.value}</span>
                        </span>
                      ) : <span className="text-gray-400">—</span>}
                    </td>
                    <td className="py-1.5 text-xs text-right">
                      <a href={r.url} target="_blank" rel="noopener noreferrer"
                         className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800">
                        Open <ExternalLink className="w-3 h-3" />
                      </a>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {/* Libraries — collapsible */}
      <section className="mb-6 bg-white rounded-lg border border-gray-200 overflow-hidden">
        <button onClick={() => setShowLibs(v => !v)}
                className="w-full px-5 py-3 bg-gray-50 border-b flex items-center justify-between hover:bg-gray-100">
          <h3 className="font-semibold text-gray-800 text-sm flex items-center gap-2">
            <Library className="w-4 h-4 text-gray-600" /> Libraries &amp; runtime
          </h3>
          <div className="flex items-center gap-3 text-xs text-gray-500">
            <span>{libraries.length} libraries pinned</span>
            <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${showLibs ? 'rotate-180' : ''}`} />
          </div>
        </button>
        {showLibs && (
          <div className="p-5">
            <table className="w-full text-sm">
              <thead>
                <tr className="text-xs text-gray-500 border-b">
                  <th className="text-left py-1.5 pr-3 font-medium">Library</th>
                  <th className="text-left py-1.5 pr-3 font-medium">Version</th>
                  <th className="text-left py-1.5 font-medium">Purpose</th>
                </tr>
              </thead>
              <tbody>
                {libraries.map((l: any) => (
                  <tr key={l.name} className="border-b last:border-b-0">
                    <td className="py-1.5 pr-3 font-mono text-xs text-gray-800">{l.name}</td>
                    <td className="py-1.5 pr-3 font-mono text-xs text-gray-600">{l.version}</td>
                    <td className="py-1.5 text-xs text-gray-600">{l.purpose}</td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="text-xs text-gray-500 italic mt-3">
              Target versions for the Databricks Serverless ML runtime. Any other library can be installed in a notebook via <code className="bg-gray-100 px-1 rounded">%pip install</code>; serverless compute isolates each run's environment.
            </p>
          </div>
        )}
      </section>

      {/* Browse all experiments */}
      <div className="text-center mt-6">
        <a href={workspaceHost ? `${workspaceHost}/ml/experiments` : '#'}
           target="_blank" rel="noopener noreferrer"
           className="inline-flex items-center gap-2 px-5 py-2.5 bg-white border border-gray-300 rounded-lg text-sm font-medium text-gray-700 hover:bg-gray-50">
          <ExternalLink className="w-4 h-4" /> Browse all MLflow experiments
        </a>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Featured card — the 4 headline notebooks
// -------------------------------------------------------------

function FeaturedCard({ nb, opening, onOpen }: { nb: any; opening: boolean; onOpen: () => void }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-5 flex flex-col hover:border-blue-300 hover:shadow-sm transition">
      <div className="flex items-start justify-between mb-2 gap-3">
        <h4 className="font-semibold text-gray-900 leading-tight">{nb.title}</h4>
        <FlaskConical className="w-4 h-4 text-blue-600 shrink-0 mt-0.5" />
      </div>
      <p className="text-sm text-gray-600 leading-relaxed flex-1">{nb.description}</p>
      <div className="flex flex-wrap gap-1 mt-3 mb-3">
        {(nb.tags || []).map((t: string) => (
          <span key={t} className="px-2 py-0.5 rounded text-[10px] font-medium bg-gray-100 text-gray-700">{t}</span>
        ))}
      </div>
      <div className="flex items-center gap-3 mt-auto pt-3 border-t border-gray-100">
        <button onClick={onOpen} disabled={opening}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded text-xs font-medium hover:bg-blue-700 disabled:opacity-50">
          <ExternalLink className="w-3 h-3" /> {opening ? 'Opening…' : 'Open notebook'}
        </button>
        <a href={`${GITHUB_REPO_URL}/blob/main/src/04_models/${nb.id}.py`}
           target="_blank" rel="noopener noreferrer"
           className="text-xs text-gray-500 hover:text-blue-600 inline-flex items-center gap-1">
          <Code className="w-3 h-3" /> View on GitHub
        </a>
      </div>
    </div>
  );
}

// -------------------------------------------------------------
// Model library tile — compact "can you also do this?"
// -------------------------------------------------------------

function ModelTile({ nb, opening, onOpen }: { nb: any; opening: boolean; onOpen: () => void }) {
  const isBuilt = nb.status === 'built';
  return (
    <div className={`border rounded-lg p-3 ${isBuilt ? 'bg-white border-gray-200' : 'bg-gray-50 border-gray-200'}`}>
      <div className="flex items-start justify-between gap-2 mb-1">
        <h5 className="text-sm font-semibold text-gray-900 leading-tight">{nb.title}</h5>
        <span className={`text-xs shrink-0 ${isBuilt ? 'text-green-600' : 'text-amber-600'}`}>
          {isBuilt ? '✓' : '🚧'}
        </span>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed mb-2">{nb.description}</p>
      <div className="flex flex-wrap gap-1 mb-2">
        {(nb.tags || []).map((t: string) => (
          <span key={t} className="px-1.5 py-0.5 rounded text-[10px] bg-gray-100 text-gray-600">{t}</span>
        ))}
      </div>
      {isBuilt && (
        <button onClick={onOpen} disabled={opening}
                className="text-xs text-blue-600 hover:text-blue-800 font-medium disabled:opacity-50 inline-flex items-center gap-1">
          {opening ? 'Opening…' : 'Open notebook →'}
        </button>
      )}
    </div>
  );
}

// -------------------------------------------------------------
// Helpers
// -------------------------------------------------------------

function formatRelative(iso?: string): string {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (isNaN(t)) return '—';
  const diff = Date.now() - t;
  if (diff < 60_000) return 'just now';
  const mins = Math.floor(diff / 60_000);
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24) return `${hrs}h ago`;
  const days = Math.floor(hrs / 24);
  return `${days}d ago`;
}

// -------------------------------------------------------------
// Challenger comparison: baseline vs +urban_score vs +both factors.
// Preserved from the previous version of this page — this is high-
// value actuarial content showing lift attribution per factor.
// -------------------------------------------------------------

function ChallengerPanel({ data }: { data: any }) {
  if (!data || !data.cohorts || data.cohorts.length === 0) {
    return (
      <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-1">
          <GitCompare className="w-5 h-5 text-amber-600" />
          <h3 className="font-semibold text-amber-800">Adding factors → model lift</h3>
        </div>
        <p className="text-sm text-amber-700">
          Run <code className="bg-white px-1.5 py-0.5 rounded text-xs">src/04_models/challenger_comparison.py</code>{' '}
          to populate this panel. It trains baseline vs +urban_score vs +both, and attributes
          the Gini lift to each derived factor.
        </p>
      </div>
    );
  }

  const baseline = data.baseline_gini || 0;
  const fullGini = data.full_gini || data.plus_both_gini || 0;
  const totalLift = data.total_lift || 0;
  const totalLiftPct = data.total_lift_pct || 0;
  const attribution = data.attribution || [];
  const cohorts = data.cohorts || [];
  const positive = totalLift >= 0;

  return (
    <div className="mb-6 bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="px-5 py-3 bg-emerald-50 border-b border-emerald-200 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <GitCompare className="w-5 h-5 text-emerald-700" />
          <div>
            <h3 className="font-semibold text-emerald-800">Adding real UK data → model lift</h3>
            <p className="text-xs text-emerald-700">
              Baseline vs challenger with real ONSPD / IMD / coastal factors — Gini on held-out sample
            </p>
          </div>
        </div>
        <span className="text-xs text-gray-500">
          Modelling Mart v{data.upt_delta_version ?? '—'}
        </span>
      </div>

      <div className="p-5 grid grid-cols-3 gap-4">
        <GiniCard label="Baseline (synthetic features)" value={baseline} />
        <GiniCard label={`Full challenger (${cohorts.length} cohorts)`} value={fullGini} delta={fullGini - baseline} highlight />
        <div className={`rounded-lg p-3 border ${positive ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <div className="text-xs text-gray-500">Total lift</div>
          <div className={`text-2xl font-bold mt-1 ${positive ? 'text-green-700' : 'text-red-700'}`}>
            {positive ? '+' : ''}{totalLift.toFixed(4)}
          </div>
          <div className={`text-xs mt-0.5 ${positive ? 'text-green-600' : 'text-red-600'}`}>
            {positive ? '+' : ''}{totalLiftPct.toFixed(2)}% vs baseline
          </div>
        </div>
      </div>

      <div className="px-5 pb-5">
        <div className="text-xs font-medium text-gray-600 mb-2">Lift attribution</div>
        <div className="space-y-1.5">
          {attribution.map((a: any) => {
            const share = Math.max(0, Math.min(100, a.share_pct || 0));
            return (
              <div key={a.factor} className="flex items-center gap-3">
                <div className="w-60 text-xs font-mono text-gray-700">{a.factor}</div>
                <div className="flex-1 bg-gray-100 rounded-full h-2 overflow-hidden">
                  <div className="bg-emerald-500 h-full" style={{ width: `${share}%` }} />
                </div>
                <div className="w-32 text-right text-xs text-gray-600">
                  <span className="font-medium">{a.lift >= 0 ? '+' : ''}{Number(a.lift).toFixed(4)}</span>
                  <span className="text-gray-400 ml-1">({share.toFixed(0)}% share)</span>
                </div>
              </div>
            );
          })}
        </div>
        <div className="flex items-center justify-between mt-3 text-xs text-gray-500">
          <div>Ablation: each factor's lift is the Gini delta when it's added to the previous model.</div>
          <a href={`${GITHUB_REPO_URL}/blob/main/src/04_models/challenger_comparison.py`}
            target="_blank" rel="noopener noreferrer"
            className="inline-flex items-center gap-1 text-emerald-600 hover:text-emerald-800 font-medium">
            View challenger_comparison.py <ArrowUpRight className="w-3 h-3" />
          </a>
        </div>
      </div>
    </div>
  );
}

function GiniCard({ label, value, delta, highlight }: { label: string; value: number; delta?: number; highlight?: boolean }) {
  return (
    <div className={`rounded-lg p-3 border ${highlight ? 'bg-emerald-50 border-emerald-200' : 'bg-gray-50 border-gray-200'}`}>
      <div className="text-xs text-gray-500">{label}</div>
      <div className={`text-2xl font-bold mt-1 ${highlight ? 'text-emerald-700' : 'text-gray-800'}`}>
        {value.toFixed(4)}
      </div>
      {delta !== undefined && (
        <div className={`text-xs mt-0.5 ${delta >= 0 ? 'text-green-600' : 'text-red-600'}`}>
          {delta >= 0 ? '+' : ''}{delta.toFixed(4)}
        </div>
      )}
    </div>
  );
}
