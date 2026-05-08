import { useEffect, useMemo, useState, FormEvent } from 'react';
import {
  FlaskConical, Loader2, Check, ArrowRight, Play, Sparkles, ClipboardList,
  Trophy, Target, Layers, PackageCheck, AlertTriangle, ChevronRight, Send, Bot,
  UserCircle2, MessageSquare, TrendingDown, TrendingUp, Info,
} from 'lucide-react';
import { api } from '../lib/api';

type Variant = {
  variant_id: string;
  name: string;
  category: 'feature_subset' | 'interactions' | 'banding';
  features?: string[];
  interactions?: [string, string][];
  banding?: string;
  glm?: { family: string; link: string };
  notes?: string;
  n_features?: number;
  metrics?: Record<string, number>;
  config?: any;
  cv?: { cv_gini_mean: number; cv_gini_std: number; stability: string; cv_folds: number };
  sign_checks?: Record<string, string>;
};

type Step = 'plan' | 'train' | 'review' | 'pack';

const FAMILIES = [
  { key: 'freq_glm',   label: 'Frequency (GLM)',  supported: true  },
  { key: 'sev_glm',    label: 'Severity (GLM)',   supported: false },
  { key: 'demand_gbm', label: 'Demand (GBM)',     supported: false },
  { key: 'fraud_gbm',  label: 'Fraud (GBM)',      supported: false },
];

type Mode = 'demo' | 'real';

const API_BY_MODE = {
  demo: {
    propose:      (family: string) => api.factoryPropose(family),
    approve:      (family: string, plan: any[], narrative: string) => api.factoryApprove(family, plan, narrative),
    getRun:       (runId: string) => api.factoryGetRun(runId),
    leaderboard:  (runId: string) => api.factoryLeaderboard(runId),
    shortlist:    (runId: string) => api.factoryShortlist(runId),
    portfolio:    (runId: string) => api.factoryPortfolio(runId),
    chat:         (runId: string, q: string) => api.factoryChat(runId, q),
    promoteFor:   (runId: string, variantId: string) => api.factoryPromoteVariant(runId, variantId),
    pollIntervalMs: 1200,
    pollTimesOutAfter: undefined as number | undefined,
  },
  real: {
    propose:      (family: string) => api.factoryRealPropose(family),
    approve:      (family: string, plan: any[], narrative: string) => api.factoryRealApprove(family, plan, narrative),
    getRun:       (runId: string) => api.factoryRealGetRun(runId),
    leaderboard:  (runId: string) => api.factoryRealLeaderboard(runId),
    shortlist:    (runId: string) => api.factoryRealShortlist(runId),
    portfolio:    async (_runId: string) => ({ results: [], notes: "Portfolio what-if not wired for Real tab in MVP." }),
    chat:         (runId: string, q: string) => api.factoryRealChat(runId, q),
    promoteFor:   (runId: string, variantId: string) => api.factoryRealPromoteVariant(runId, variantId),
    pollIntervalMs: 3000,
    pollTimesOutAfter: undefined as number | undefined,
  },
};

export default function ModelFactory() {
  const [mode, setMode] = useState<Mode>('demo');
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <FlaskConical className="w-6 h-6 text-indigo-600" /> Model Factory
        </h2>
        <p className="text-gray-500 mt-1">
          Systematic generation and review of many candidate models. Four steps: analyse &amp; plan,
          train, review, selectively package for promotion.
        </p>
      </div>

      <div className="flex gap-1 border-b border-gray-200 mb-5">
        <button onClick={() => setMode('demo')}
                className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 transition ${
                  mode === 'demo' ? 'border-indigo-600 text-indigo-700 bg-white'
                                  : 'border-transparent text-gray-500 hover:text-gray-800'}`}>
          Demo <span className="text-[10px] ml-1 px-1.5 py-0.5 rounded bg-amber-100 text-amber-700 font-medium">virtual</span>
        </button>
        <button onClick={() => setMode('real')}
                className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 transition ${
                  mode === 'real' ? 'border-emerald-600 text-emerald-700 bg-white'
                                  : 'border-transparent text-gray-500 hover:text-gray-800'}`}>
          Real <span className="text-[10px] ml-1 px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">fits models</span>
        </button>
      </div>

      <FactoryFlow key={mode} mode={mode} />
    </div>
  );
}

function FactoryFlow({ mode }: { mode: Mode }) {
  const apiSet = API_BY_MODE[mode];
  const [step, setStep]       = useState<Step>('plan');
  const [family, setFamily]   = useState<string>('freq_glm');
  const [plan, setPlan]       = useState<Variant[] | null>(null);
  const [narrative, setNarrative] = useState<string>('');
  const [review, setReview] = useState<{ recommended_count: number; reasoning: string; breakdown: Record<string, number> } | null>(null);
  const [proposing, setProposing] = useState(false);
  const [approving, setApproving] = useState(false);
  const [unsupportedMsg, setUnsupportedMsg] = useState<string | null>(null);
  const [runId, setRunId]     = useState<string | null>(null);
  const [runStatus, setRunStatus] = useState<any>(null);
  const [selected, setSelected]  = useState<Set<string>>(new Set());
  const [toast, setToast]     = useState<string | null>(null);

  const propose = async () => {
    setProposing(true);
    setUnsupportedMsg(null);
    setPlan(null);
    try {
      const r = await apiSet.propose(family);
      if (r.status === 'unsupported') {
        setUnsupportedMsg(r.message);
      } else {
        setPlan(r.plan);
        setNarrative(r.narrative);
        setReview(r.review || null);
      }
    } catch (e: any) {
      setToast(`Plan failed: ${e.message}`);
    } finally {
      setProposing(false);
    }
  };

  const approve = async () => {
    if (!plan || approving) return;
    setApproving(true);
    try {
      const r = await apiSet.approve(family, plan, narrative);
      setRunId(r.run_id);
      setStep('train');
    } catch (e: any) {
      setToast(`Approve failed: ${e.message}`);
    } finally {
      setApproving(false);
    }
  };

  useEffect(() => {
    if (!runId || step !== 'train') return;
    let done = false;
    const poll = async () => {
      try {
        const r = await apiSet.getRun(runId);
        setRunStatus(r);
        if (r.status === 'COMPLETED' && !done) {
          done = true;
          setTimeout(() => setStep('review'), 600);
        }
      } catch {}
    };
    poll();
    const t = setInterval(poll, apiSet.pollIntervalMs);
    return () => clearInterval(t);
  }, [runId, step]);

  const familyMeta = FAMILIES.find(f => f.key === family);

  return (
    <>
      <Stepper step={step} />

      {mode === 'demo' ? (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-xs text-amber-900 mb-4 flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5 text-amber-700" />
          <div>
            <strong>Demo — virtual training.</strong> Variants are generated from the plan and
            metrics are synthesised deterministically — no actual model fit runs. Portfolio
            what-if and pack generation are synthesised too. Switch to the <strong>Real</strong>{' '}
            tab above to fit variants against the live Modelling Mart.
          </div>
        </div>
      ) : (
        <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3 text-xs text-emerald-900 mb-4 flex items-start gap-2">
          <Sparkles className="w-4 h-4 shrink-0 mt-0.5 text-emerald-700" />
          <div>
            <strong>Real — fits models.</strong> Each variant is fitted on the real Modelling Mart
            via the <code className="bg-white px-1 rounded">v1 — Factory training (real)</code> job
            on serverless compute. Metrics, CV Gini, relativities are all real. Registered UC models
            land as <code className="bg-white px-1 rounded">factory_freq_glm_*</code> — separate
            namespace from the 4 production champions. <strong>Factory candidates never claim the{' '}
            <code>champion</code> alias</strong>; they stay out of deployment. Training a 15-variant
            plan takes ~3-5 min.
          </div>
        </div>
      )}

      {step === 'plan' && (
        <StepPlan
          family={family}
          setFamily={setFamily}
          proposing={proposing}
          propose={propose}
          plan={plan}
          narrative={narrative}
          review={review}
          unsupported={unsupportedMsg}
          approve={approve}
          approving={approving}
          supported={mode === 'demo' ? !!familyMeta?.supported : family === 'freq_glm'}
          mode={mode}
        />
      )}

      {step === 'train' && runId && (
        <StepTrain runId={runId} runStatus={runStatus} mode={mode} />
      )}

      {(step === 'review' || step === 'pack') && runId && (
        <StepReview
          runId={runId}
          mode={mode}
          apiSet={apiSet}
          selected={selected}
          setSelected={setSelected}
          onPackFor={async (variantId) => {
            try {
              const r = await apiSet.promoteFor(runId, variantId);
              setToast(r.message || 'Queued');
              setStep('pack');
            } catch (e: any) {
              setToast(`Queue failed: ${e.message}`);
            }
          }}
          atPackStep={step === 'pack'}
        />
      )}

      {toast && (
        <div onClick={() => setToast(null)}
             className="fixed bottom-4 right-4 bg-gray-900 text-white text-sm px-4 py-2 rounded-lg shadow-lg z-50 cursor-pointer max-w-md">
          {toast}
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Stepper header
// ---------------------------------------------------------------------------

function Stepper({ step }: { step: Step }) {
  const stages: { id: Step; label: string; icon: any }[] = [
    { id: 'plan',   label: 'Analyse & plan',    icon: ClipboardList },
    { id: 'train',  label: 'Train',             icon: Play },
    { id: 'review', label: 'Review',            icon: Target },
    { id: 'pack',   label: 'Selective packaging', icon: PackageCheck },
  ];
  const current = stages.findIndex(s => s.id === step);
  return (
    <div className="flex items-center gap-2 mb-5">
      {stages.map((s, i) => (
        <>
          <div key={s.id}
               className={`flex-1 flex items-center gap-2 rounded-lg px-3 py-2 border ${
                 i === current ? 'bg-indigo-50 border-indigo-200 text-indigo-800' :
                 i <  current ? 'bg-emerald-50 border-emerald-200 text-emerald-800' :
                                'bg-white border-gray-200 text-gray-500'
               }`}>
            {i < current
              ? <Check className="w-4 h-4" />
              : <s.icon className="w-4 h-4" />}
            <span className="text-sm font-medium">{i + 1}. {s.label}</span>
          </div>
          {i < stages.length - 1 &&
            <ChevronRight key={`sep-${i}`} className="w-4 h-4 text-gray-400 shrink-0" />}
        </>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 1 — Analyse & plan
// ---------------------------------------------------------------------------

function StepPlan({ family, setFamily, proposing, propose, plan, narrative, review, unsupported, approve, approving = false, supported, mode = 'demo' }:
  {
    family: string; setFamily: (f: string) => void;
    proposing: boolean; propose: () => void;
    plan: Variant[] | null; narrative: string;
    review: { recommended_count: number; reasoning: string; breakdown: Record<string, number> } | null;
    unsupported: string | null;
    approve: () => void;
    approving?: boolean;
    supported: boolean;
    mode?: 'demo' | 'real';
  }) {
  return (
    <div className="space-y-4">
      <section className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-end gap-3">
          <div className="flex-1">
            <label className="text-xs text-gray-500 font-medium block mb-1">Model family</label>
            <div className="flex flex-wrap gap-2">
              {FAMILIES.map(f => (
                <button key={f.key}
                        onClick={() => setFamily(f.key)}
                        className={`px-3 py-1.5 rounded text-sm font-medium border transition ${
                          family === f.key
                            ? 'bg-indigo-600 text-white border-indigo-600'
                            : 'bg-gray-50 text-gray-700 border-gray-200 hover:bg-gray-100'
                        } ${!f.supported ? 'opacity-70' : ''}`}>
                  {f.label}
                  {!f.supported && <span className={`ml-1.5 text-[9px] px-1 rounded ${family === f.key ? 'bg-white/20 text-white' : 'bg-amber-100 text-amber-800'}`}>
                    soon
                  </span>}
                </button>
              ))}
            </div>
          </div>
          <button onClick={propose} disabled={proposing || !supported}
                  className="inline-flex items-center gap-1.5 px-4 py-2 bg-indigo-600 text-white rounded text-sm font-medium hover:bg-indigo-700 disabled:opacity-50 shrink-0">
            {proposing ? <Loader2 className="w-4 h-4 animate-spin" /> : <Sparkles className="w-4 h-4" />}
            Propose plan
          </button>
        </div>
      </section>

      {unsupported && (
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-3 text-sm text-amber-900">
          <AlertTriangle className="w-4 h-4 inline mr-1" /> {unsupported}
        </div>
      )}

      {plan && (
        <>
          {review && review.recommended_count > 0 && (
            <section className="bg-violet-50 border border-violet-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-violet-900 mb-2 flex items-center gap-1.5">
                <Sparkles className="w-4 h-4 text-violet-600" /> Pricing AI · data review
                <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-200 text-violet-900 font-medium ml-1">
                  {review.recommended_count} variants recommended
                </span>
              </h3>
              <p className="text-sm text-violet-900 leading-relaxed">{review.reasoning}</p>
            </section>
          )}

          {/* Narrative */}
          <section className="bg-white border border-gray-200 rounded-lg p-4">
            <h3 className="text-sm font-semibold text-gray-800 mb-2 flex items-center gap-1.5">
              <Bot className="w-4 h-4 text-violet-600" /> Plan narrative
              <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-700 font-medium ml-1">
                Claude Sonnet 4.6
              </span>
            </h3>
            <div className="text-sm text-gray-800 whitespace-pre-wrap leading-relaxed">
              {narrative}
            </div>
          </section>

          {/* Plan summary */}
          <section className="grid grid-cols-2 md:grid-cols-5 gap-3">
            <SummaryTile label="Total variants"         value={plan.length.toString()} />
            <SummaryTile label="Feature-subset probes"  value={plan.filter(v => v.category === 'feature_subset').length.toString()} />
            <SummaryTile label="Interaction probes"     value={plan.filter(v => v.category === 'interactions').length.toString()} />
            <SummaryTile label="Banding probes"         value={plan.filter(v => v.category === 'banding').length.toString()} />
            <SummaryTile label="Distributional families" value={new Set(plan.map(v => v.glm?.family)).size.toString()} />
          </section>

          {/* Plan table */}
          <section className="bg-white border border-gray-200 rounded-lg overflow-hidden">
            <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
              <h3 className="text-sm font-semibold text-gray-800">
                Proposed variants — actuary review before training
              </h3>
              <span className="text-xs text-gray-500">{plan.length} rows</span>
            </div>
            <div className="max-h-[440px] overflow-y-auto">
              <table className="w-full text-sm">
                <thead className="sticky top-0 bg-gray-50">
                  <tr className="text-xs text-gray-500 border-b">
                    <th className="text-left px-3 py-2 font-medium">ID</th>
                    <th className="text-left px-3 py-2 font-medium">Name</th>
                    <th className="text-left px-3 py-2 font-medium">Category</th>
                    <th className="text-right px-3 py-2 font-medium">Features</th>
                    <th className="text-left px-3 py-2 font-medium">Interactions</th>
                    <th className="text-left px-3 py-2 font-medium">Banding</th>
                    <th className="text-left px-3 py-2 font-medium">GLM</th>
                  </tr>
                </thead>
                <tbody>
                  {plan.map(v => (
                    <tr key={v.variant_id} className="border-b last:border-0 hover:bg-gray-50">
                      <td className="px-3 py-1.5 font-mono text-xs">{v.variant_id}</td>
                      <td className="px-3 py-1.5 text-xs text-gray-800">{v.name}</td>
                      <td className="px-3 py-1.5">
                        <CategoryChip cat={v.category} />
                      </td>
                      <td className="px-3 py-1.5 text-right font-mono text-xs">{v.features?.length || 0}</td>
                      <td className="px-3 py-1.5 text-xs text-gray-600">
                        {(v.interactions || []).map(p => p.join(' × ')).join(', ') || '—'}
                      </td>
                      <td className="px-3 py-1.5 text-xs text-gray-600 font-mono">{v.banding}</td>
                      <td className="px-3 py-1.5 text-xs text-gray-600">{v.glm?.family}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </section>

          <div className="flex justify-end">
            <button onClick={approve}
                    disabled={approving}
                    className="inline-flex items-center gap-1.5 px-4 py-2 bg-emerald-600 text-white rounded text-sm font-medium hover:bg-emerald-700 disabled:opacity-60">
              {approving
                ? <><Loader2 className="w-4 h-4 animate-spin" /> Starting training run…</>
                : <><Check className="w-4 h-4" /> Approve plan &amp; train <ArrowRight className="w-4 h-4" /></>}
            </button>
          </div>
        </>
      )}
    </div>
  );
}

function SummaryTile({ label, value }: { label: string; value: string }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-3">
      <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className="text-2xl font-semibold text-gray-900 mt-0.5">{value}</div>
    </div>
  );
}

function CategoryChip({ cat }: { cat: Variant['category'] }) {
  const styles = {
    feature_subset: 'bg-blue-100 text-blue-700',
    interactions:   'bg-purple-100 text-purple-700',
    banding:        'bg-amber-100 text-amber-700',
  }[cat];
  return <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${styles}`}>
    {cat.replace('_', ' ')}
  </span>;
}

// ---------------------------------------------------------------------------
// Step 2 — Train (virtual)
// ---------------------------------------------------------------------------

function StepTrain({ runId, runStatus, mode = 'demo' }: { runId: string; runStatus: any; mode?: 'demo' | 'real' }) {
  const total = runStatus?.variant_count || 50;
  const complete = runStatus?.n_complete || 0;
  const pct = runStatus?.progress ? Math.round(runStatus.progress * 100) : 0;

  return (
    <div className="space-y-4">
      <section className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="flex items-center gap-3 mb-3">
          {runStatus?.status === 'COMPLETED'
            ? <Check className="w-5 h-5 text-emerald-600" />
            : <Loader2 className="w-5 h-5 animate-spin text-indigo-600" />}
          <div>
            <h3 className="text-sm font-semibold text-gray-800">
              {runStatus?.status === 'COMPLETED' ? 'Training complete' : 'Training in progress…'}
            </h3>
            <p className="text-xs text-gray-500">Run {runId} · {complete} of {total} variants trained</p>
          </div>
          <div className="ml-auto text-right text-xs text-gray-500">
            <div>Elapsed: {runStatus?.elapsed_seconds?.toFixed(1) || 0}s</div>
            <div className={`text-[10px] ${mode === 'real' ? 'text-emerald-700' : 'text-amber-700'}`}>
              {mode === 'real' ? 'Real fit on serverless' : 'Virtual — synthesised metrics'}
            </div>
          </div>
        </div>
        <div className="w-full h-3 bg-gray-100 rounded overflow-hidden">
          <div className={`h-full transition-all duration-700 ${mode === 'real' ? 'bg-emerald-500' : 'bg-indigo-500'}`}
               style={{ width: `${pct}%` }} />
        </div>
        <div className="text-[11px] text-gray-500 mt-2">
          {mode === 'real'
            ? <>Each variant is being fitted via the <code>v1 — Factory training (real)</code> Databricks
               Job on serverless. Real MLflow runs are logged to <code>/Users/[you]/pricing_workbench_factory</code>,
               and each variant registers as a UC model <code>factory_freq_glm_&lt;id&gt;</code>.</>
            : <>In the real training flow this is a Databricks Job over serverless compute, one task per variant,
               logging each to MLflow under <code>/Shared/pricing_demo/factory/</code>.</>}
        </div>
      </section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step 3 — Review (leaderboard / shortlist / portfolio / agent chat)
// ---------------------------------------------------------------------------

function StepReview({ runId, selected, setSelected, onPackFor, atPackStep, mode = 'demo', apiSet }: {
  runId: string;
  selected: Set<string>;
  setSelected: (s: Set<string>) => void;
  onPackFor: (variantId: string) => void;
  atPackStep: boolean;
  mode?: 'demo' | 'real';
  apiSet: typeof API_BY_MODE['demo'];
}) {
  const [tier, setTier]           = useState<'leaderboard' | 'shortlist' | 'portfolio'>('leaderboard');
  const [lb, setLb]               = useState<Variant[]>([]);
  const [short, setShort]         = useState<Variant[]>([]);
  const [portfolio, setPortfolio] = useState<any>(null);

  useEffect(() => {
    apiSet.leaderboard(runId).then((d: any) => setLb(d.variants || []));
    apiSet.shortlist(runId).then((d: any) => setShort(d.shortlist || []));
    apiSet.portfolio(runId).then(setPortfolio);
  }, [runId, mode]);

  const toggleSelected = (vid: string) => {
    const next = new Set(selected);
    if (next.has(vid)) next.delete(vid);
    else next.add(vid);
    setSelected(next);
  };

  return (
    <div className="space-y-4">
      {/* Tier tabs — leaderboard → portfolio → shortlist (shortlist last so
          the actuary lands on it after exploring the leaderboard + what-if). */}
      <div className="bg-white rounded-lg border border-gray-200 p-1 inline-flex gap-1">
        <TierButton active={tier === 'leaderboard'} onClick={() => setTier('leaderboard')}
                    icon={<Trophy className="w-3.5 h-3.5" />} label="Leaderboard" />
        {mode === 'demo' && (
          <TierButton active={tier === 'portfolio'}   onClick={() => setTier('portfolio')}
                      icon={<Layers className="w-3.5 h-3.5" />} label="Portfolio what-if" />
        )}
        <TierButton active={tier === 'shortlist'}   onClick={() => setTier('shortlist')}
                    icon={<Target className="w-3.5 h-3.5" />} label="Shortlist (top 5)" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-4">
          {tier === 'leaderboard' && <LeaderboardTable variants={lb} />}
          {tier === 'shortlist'   && (
            <ShortlistTable shortlist={short} selected={selected} onToggle={toggleSelected} onPackFor={onPackFor} mode={mode} />
          )}
          {tier === 'portfolio' && mode === 'demo' && portfolio && <PortfolioCards portfolio={portfolio} />}
        </div>
        <div className="lg:col-span-1">
          <ChatPane runId={runId} apiSet={apiSet} />
        </div>
      </div>

      {/* Selective-packaging footer */}
      {tier === 'shortlist' && selected.size > 0 && (
        <section className="bg-indigo-50 border border-indigo-200 rounded-lg p-4">
          <div className="flex items-center justify-between">
            <div>
              <h4 className="text-sm font-semibold text-indigo-900 flex items-center gap-1.5">
                <PackageCheck className="w-4 h-4" /> {selected.size} variant{selected.size > 1 ? 's' : ''} selected for packaging
              </h4>
              <p className="text-xs text-indigo-800 mt-0.5">
                Triggers governance pack generation per variant. Packaged variants become available in the
                Promote tab of Model Development.
              </p>
            </div>
            <button onClick={() => Array.from(selected).forEach(onPackFor)}
                    className="inline-flex items-center gap-1.5 px-3 py-2 bg-indigo-600 text-white rounded text-sm font-medium hover:bg-indigo-700">
              <PackageCheck className="w-4 h-4" /> Generate pack{selected.size > 1 ? 's' : ''}
            </button>
          </div>
        </section>
      )}

      {atPackStep && (
        <section className="bg-emerald-50 border border-emerald-200 rounded-lg p-4 text-sm text-emerald-900">
          <div className="flex items-start gap-2">
            <Check className="w-4 h-4 mt-0.5 shrink-0" />
            <div>
              <strong>Pack generation queued.</strong> In MVP this logs to <code>audit_log</code> but doesn't
              run the pack notebook — factory candidates are virtual. When real training lands, this button
              calls the existing <code>governance_pack_generation</code> job per selected variant and the
              resulting packs show up on the Model Governance tab.
            </div>
          </div>
        </section>
      )}
    </div>
  );
}

function TierButton({ active, onClick, icon, label }:
  { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button onClick={onClick}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition ${
              active ? 'bg-indigo-600 text-white' : 'text-gray-700 hover:bg-gray-100'
            }`}>
      {icon} {label}
    </button>
  );
}

function LeaderboardTable({ variants }: { variants: Variant[] }) {
  const [sortKey, setSortKey] = useState<'gini' | 'aic' | 'bic' | 'deviance_explained' | 'mae'>('gini');

  const sorted = useMemo(() => {
    const arr = [...variants];
    const dir = (sortKey === 'aic' || sortKey === 'bic' || sortKey === 'mae') ? 1 : -1;
    arr.sort((a, b) => dir * ((a.metrics?.[sortKey] ?? 0) - (b.metrics?.[sortKey] ?? 0)));
    return arr;
  }, [variants, sortKey]);

  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-1.5">
          <Trophy className="w-4 h-4" /> Leaderboard — {variants.length} variants
        </h3>
        <div className="text-[11px] text-gray-500">Click a column to sort</div>
      </div>
      <div className="max-h-[520px] overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-gray-50">
            <tr className="text-xs text-gray-500 border-b">
              <th className="text-left px-3 py-2 font-medium">ID</th>
              <th className="text-left px-3 py-2 font-medium">Name</th>
              <th className="text-left px-3 py-2 font-medium">Cat.</th>
              <th className="text-right px-3 py-2 font-medium">Feat.</th>
              <SortHeader current={sortKey} k="gini"               label="Gini" onClick={setSortKey} />
              <SortHeader current={sortKey} k="aic"                label="AIC"  onClick={setSortKey} />
              <SortHeader current={sortKey} k="bic"                label="BIC"  onClick={setSortKey} />
              <SortHeader current={sortKey} k="deviance_explained" label="Dev. expl." onClick={setSortKey} />
              <SortHeader current={sortKey} k="mae"                label="MAE"  onClick={setSortKey} />
            </tr>
          </thead>
          <tbody>
            {sorted.map((v, i) => {
              const m = v.metrics || {};
              return (
                <tr key={v.variant_id} className={`border-b last:border-0 hover:bg-gray-50 ${i < 5 ? 'bg-emerald-50/40' : ''}`}>
                  <td className="px-3 py-1.5 font-mono text-xs">{i + 1}. {v.variant_id}</td>
                  <td className="px-3 py-1.5 text-xs text-gray-800 truncate max-w-[260px]" title={v.name}>{v.name}</td>
                  <td className="px-3 py-1.5"><CategoryChip cat={v.category} /></td>
                  <td className="px-3 py-1.5 text-right text-xs font-mono">{v.n_features}</td>
                  <td className="px-3 py-1.5 text-right text-xs font-mono font-medium text-indigo-800">{m.gini?.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right text-xs font-mono text-gray-700">{Number(m.aic ?? 0).toFixed(0)}</td>
                  <td className="px-3 py-1.5 text-right text-xs font-mono text-gray-700">{Number(m.bic ?? 0).toFixed(0)}</td>
                  <td className="px-3 py-1.5 text-right text-xs font-mono text-gray-700">{m.deviance_explained?.toFixed(4)}</td>
                  <td className="px-3 py-1.5 text-right text-xs font-mono text-gray-700">{m.mae?.toFixed(4)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </section>
  );
}

function SortHeader({ current, k, label, onClick }:
  { current: string; k: any; label: string; onClick: (k: any) => void }) {
  return (
    <th onClick={() => onClick(k)}
        className={`text-right px-3 py-2 font-medium cursor-pointer hover:text-indigo-700 ${current === k ? 'text-indigo-700' : ''}`}>
      {label}{current === k && ' ▼'}
    </th>
  );
}

function ShortlistTable({ shortlist, selected, onToggle, onPackFor, mode = 'demo' }:
  { shortlist: Variant[]; selected: Set<string>; onToggle: (v: string) => void; onPackFor: (v: string) => void; mode?: 'demo' | 'real' }) {
  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b">
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-1.5">
          <Target className="w-4 h-4" /> Shortlist — top 5 by Gini · auto-selected from the leaderboard
        </h3>
      </div>
      <div className="divide-y">
        {shortlist.map(v => {
          const m = v.metrics || {};
          const cv = v.cv;
          const isSelected = selected.has(v.variant_id);
          return (
            <div key={v.variant_id} className={`p-4 ${isSelected ? 'bg-indigo-50/40' : ''}`}>
              <div className="flex items-start gap-3">
                <input type="checkbox" checked={isSelected} onChange={() => onToggle(v.variant_id)}
                       className="mt-1 shrink-0" />
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 flex-wrap">
                    <span className="font-mono text-sm font-semibold">{v.variant_id}</span>
                    <span className="text-sm text-gray-900">{v.name}</span>
                    <CategoryChip cat={v.category} />
                    {cv && <span className={`text-[10px] px-1.5 py-0.5 rounded font-medium ${
                      cv.stability === 'stable' ? 'bg-emerald-100 text-emerald-700' : 'bg-amber-100 text-amber-700'
                    }`}>CV {cv.stability}</span>}
                  </div>

                  <div className="mt-2 grid grid-cols-2 md:grid-cols-4 gap-2">
                    <MiniMetric label="Gini"          value={m.gini?.toFixed(4)} highlight />
                    <MiniMetric label="AIC / BIC"     value={`${Number(m.aic ?? 0).toFixed(0)} / ${Number(m.bic ?? 0).toFixed(0)}`} />
                    <MiniMetric label="Dev. explained" value={m.deviance_explained?.toFixed(4)} />
                    <MiniMetric label="CV Gini ± σ"   value={cv ? `${cv.cv_gini_mean.toFixed(4)} ± ${cv.cv_gini_std.toFixed(4)}` : '—'} />
                  </div>

                  <div className="mt-3 text-xs text-gray-700">
                    <strong>Config:</strong> {v.config?.features?.length ?? '?'} features,{' '}
                    {(v.config?.interactions?.length ?? 0)} interaction{v.config?.interactions?.length === 1 ? '' : 's'},{' '}
                    banding <code className="text-[11px]">{v.config?.banding}</code>,{' '}
                    family <code className="text-[11px]">{v.config?.glm?.family}</code>
                  </div>

                  {v.sign_checks && (
                    <div className="mt-2 flex flex-wrap gap-1">
                      {Object.entries(v.sign_checks).map(([feat, sign]) => (
                        <span key={feat} className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-50 border border-emerald-200 text-emerald-800 font-mono">
                          {feat}: {sign}
                        </span>
                      ))}
                    </div>
                  )}
                </div>
                <button onClick={() => onPackFor(v.variant_id)}
                        title={mode === 'real' ? 'Triggers governance_pack_generation for this factory candidate' : 'Records the pack request in audit log (virtual)'}
                        className={`shrink-0 inline-flex items-center gap-1 px-3 py-1.5 rounded text-xs font-medium text-white ${
                          mode === 'real' ? 'bg-emerald-600 hover:bg-emerald-700' : 'bg-indigo-600 hover:bg-indigo-700'
                        }`}>
                  <PackageCheck className="w-3.5 h-3.5" /> {mode === 'real' ? 'Generate real pack' : 'Pack'}
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </section>
  );
}

function MiniMetric({ label, value, highlight }:
  { label: string; value: string | undefined; highlight?: boolean }) {
  return (
    <div className={`border border-gray-200 rounded p-2 ${highlight ? 'bg-indigo-50 border-indigo-200' : 'bg-gray-50'}`}>
      <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`text-sm font-mono mt-0.5 ${highlight ? 'text-indigo-800 font-semibold' : 'text-gray-800'}`}>
        {value ?? '—'}
      </div>
    </div>
  );
}

function PortfolioCards({ portfolio }: { portfolio: any }) {
  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-1.5">
          <Layers className="w-4 h-4" /> Portfolio what-if · top 5 scored on full portfolio
        </h3>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
          Synthesised
        </span>
      </div>
      <div className="p-4 space-y-3">
        {portfolio.results.map((r: any) => (
          <div key={r.variant_id} className="border border-gray-200 rounded p-3">
            <div className="flex items-center justify-between gap-3">
              <div className="flex items-center gap-2 flex-wrap">
                <span className="font-mono text-sm font-semibold">{r.variant_id}</span>
                <span className="text-sm text-gray-900">{r.name}</span>
                <span className="text-[10px] text-gray-500">Gini {r.gini.toFixed(4)}</span>
              </div>
              <div className={`text-sm font-semibold inline-flex items-center gap-1 ${r.premium_shift_pct >= 0 ? 'text-emerald-700' : 'text-red-700'}`}>
                {r.premium_shift_pct >= 0 ? <TrendingUp className="w-3.5 h-3.5" /> : <TrendingDown className="w-3.5 h-3.5" />}
                {r.premium_shift_pct >= 0 ? '+' : ''}{r.premium_shift_pct.toFixed(2)}% vs champion
              </div>
            </div>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 mt-2">
              <MiniMetric label="Sampled policies" value={r.n_policies_sampled.toLocaleString()} />
              <MiniMetric label="Shifted > 10%" value={r.n_shift_gt_10pct.toString()} />
              <MiniMetric label="Shifted > 25%" value={r.n_shift_gt_25pct.toString()} />
              <MiniMetric label="Loss-ratio deciles" value="10 shown below" />
            </div>
            <div className="mt-3 flex items-end gap-1 h-14">
              {r.loss_ratio_deciles.map((d: any) => {
                const h1 = Math.min(56, d.champion_lr * 40);
                const h2 = Math.min(56, d.candidate_lr * 40);
                return (
                  <div key={d.decile} className="flex-1 flex flex-col items-center gap-0.5">
                    <div className="w-full flex gap-0.5 items-end h-12">
                      <div className="flex-1 bg-gray-300 rounded-t" style={{ height: h1 }} title={`Champion LR ${d.champion_lr}`} />
                      <div className="flex-1 bg-indigo-500 rounded-t" style={{ height: h2 }} title={`Candidate LR ${d.candidate_lr}`} />
                    </div>
                    <div className="text-[9px] text-gray-500">{d.decile}</div>
                  </div>
                );
              })}
            </div>
            <div className="text-[10px] text-gray-500 mt-1 flex items-center gap-3">
              <span className="inline-flex items-center gap-1"><span className="w-2 h-2 bg-gray-300 rounded-sm"/> Champion</span>
              <span className="inline-flex items-center gap-1"><span className="w-2 h-2 bg-indigo-500 rounded-sm"/> Candidate</span>
              <span className="ml-auto italic">Loss ratio by decile (flatter = better)</span>
            </div>
          </div>
        ))}
        <div className="text-[11px] text-gray-500 italic">{portfolio.notes}</div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Agent chat panel — governance pattern, grounded in the factory run
// ---------------------------------------------------------------------------

const SUGGESTED_QUESTIONS = [
  "Which variants look most stable by CV?",
  "Does adding interactions help here?",
  "Compare the top 3 — what's different about them?",
  "Which distributional family performs best?",
  "Any red flags in the shortlist?",
];

type Turn = {
  question: string;
  answer?: string; loading?: boolean;
  model?: string; cited?: string[];
  usage?: any; error?: string | null;
};

function ChatPane({ runId, apiSet }: { runId: string; apiSet: typeof API_BY_MODE['demo'] }) {
  const [input, setInput]   = useState('');
  const [turns, setTurns]   = useState<Turn[]>([]);
  const [busy, setBusy]     = useState(false);

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setInput('');
    setTurns(t => [...t, { question, loading: true }]);
    try {
      const r = await apiSet.chat(runId, question);
      setTurns(t => t.map((x, i) =>
        i === t.length - 1
          ? { ...x, loading: false, answer: r.answer, model: r.model, cited: r.cited_variants,
              usage: r.usage, error: r.error }
          : x));
    } catch (e: any) {
      setTurns(t => t.map((x, i) =>
        i === t.length - 1 ? { ...x, loading: false, error: e.message } : x));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden sticky top-4">
      <div className="px-4 py-2.5 bg-violet-50 border-b border-violet-200">
        <h3 className="text-sm font-semibold text-violet-900 flex items-center gap-1.5 flex-wrap">
          <Sparkles className="w-4 h-4 text-violet-600" />
          Factory assistant
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-800 font-medium">
            Claude Sonnet 4.6 · grounded in this run
          </span>
        </h3>
      </div>
      <div className="flex flex-col" style={{ height: '560px' }}>
        <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-gray-50">
          {turns.length === 0 ? (
            <div className="py-2">
              <div className="text-xs text-gray-500 mb-2 text-center">Try one of these</div>
              <div className="flex flex-col gap-1.5">
                {SUGGESTED_QUESTIONS.map(s => (
                  <button key={s} onClick={() => ask(s)}
                          className="text-xs text-left px-3 py-1.5 rounded bg-white border border-gray-200 hover:border-violet-300 hover:bg-violet-50 text-gray-800">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          ) : turns.map((t, i) => (
            <div key={i} className="space-y-1">
              <div className="flex gap-2">
                <UserCircle2 className="w-4 h-4 text-gray-500 shrink-0 mt-0.5" />
                <div className="text-sm text-gray-800">{t.question}</div>
              </div>
              <div className="flex gap-2 pl-1">
                <Bot className="w-4 h-4 text-violet-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  {t.loading ? (
                    <div className="text-xs text-gray-500 italic inline-flex items-center gap-1">
                      <Loader2 className="w-3 h-3 animate-spin" /> Reading leaderboard…
                    </div>
                  ) : t.error ? (
                    <div className="text-xs text-red-700">
                      <AlertTriangle className="w-3 h-3 inline mr-1" /> {t.error}
                    </div>
                  ) : (
                    <>
                      <div className="text-sm text-gray-900 whitespace-pre-wrap">{t.answer}</div>
                      <div className="mt-1 flex items-center gap-3 text-[10px] text-gray-500 flex-wrap">
                        {t.cited && t.cited.length > 0 &&
                          <span>cites: {t.cited.map(c => `#${c}`).join(', ')}</span>}
                        {t.usage?.total_tokens != null && <span>tokens: {t.usage.total_tokens}</span>}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>
        <form onSubmit={(e: FormEvent) => { e.preventDefault(); ask(input); }}
              className="border-t bg-white p-2 flex gap-2">
          <input value={input} onChange={e => setInput(e.target.value)}
                 placeholder="Ask about the run…"
                 disabled={busy}
                 className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-sm" />
          <button type="submit" disabled={busy || !input.trim()}
                  className="inline-flex items-center gap-1 px-3 py-1.5 bg-violet-600 text-white rounded text-sm font-medium hover:bg-violet-700 disabled:opacity-50">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />}
          </button>
        </form>
      </div>
    </section>
  );
}
