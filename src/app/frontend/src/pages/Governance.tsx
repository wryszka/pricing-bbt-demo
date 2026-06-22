import { useEffect, useMemo, useState, FormEvent } from 'react';
import {
  Shield, Layers, Calendar, Search, Loader2, MessageSquare,
  Bot, Send, ChevronDown, Sparkles, ExternalLink, Eye, EyeOff,
  AlertTriangle, UserCircle2, BookOpen, Scale, Activity, TrendingUp,
  PieChart, Zap, Clock, Wand2, Database, ArrowRight, FileText,
} from 'lucide-react';
import { api } from '../lib/api';

type Pack = {
  pack_id: string;
  model_family: string;
  model_version: string;
  story?: string;
  simulated?: boolean | null;
  primary_metric?: string;
  primary_value?: number | null;
  pdf_path?: string;
  size_bytes?: number;
  generated_by?: string;
  generated_at?: string;
};
type FamilyPacks = { key: string; label: string; packs: Pack[] };
type Mode = 'by-model' | 'by-date' | 'by-policy';
type TopTab = 'monitor' | 'search' | 'agent' | 'data';

export default function Governance() {
  const [tab, setTab] = useState<TopTab>('monitor');

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-4">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Shield className="w-6 h-6 text-amber-600" /> Model Governance
        </h2>
        <p className="text-gray-500 mt-1">
          Defend production models to regulators and auditors. Three lenses: continuous monitors,
          targeted search across packs and policies, and a free-form agent.
        </p>
      </div>

      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4 mb-6">
        <div className="flex flex-wrap gap-1.5">
          {[
            'Unity Catalog lineage',
            'MLflow model registry',
            'Governance pack generation',
            'Fairness & risk register',
            'Agent-assisted review · Claude Sonnet 4.6 (Foundation Model API)',
            'Immutable audit trail',
          ].map(f => (
            <span key={f} className="px-2 py-0.5 rounded text-[11px] font-medium bg-amber-100 text-amber-800">{f}</span>
          ))}
        </div>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-1 mb-6 inline-flex gap-1">
        <TopTabButton active={tab === 'monitor'} onClick={() => setTab('monitor')}
                      icon={<Activity className="w-3.5 h-3.5" />} label="Monitor"
                      sub="bias · drift · adequacy" />
        <TopTabButton active={tab === 'search'} onClick={() => setTab('search')}
                      icon={<Search className="w-3.5 h-3.5" />} label="Search"
                      sub="by model · date · policy" />
        <TopTabButton active={tab === 'agent'} onClick={() => setTab('agent')}
                      icon={<Bot className="w-3.5 h-3.5" />} label="Agent"
                      sub="free-form Q&A" />
        <TopTabButton active={tab === 'data'} onClick={() => setTab('data')}
                      icon={<Database className="w-3.5 h-3.5" />} label="What's collected"
                      sub="data inputs · how used" />
      </div>

      {tab === 'monitor' && <MonitorTab />}
      {tab === 'search'  && <SearchTab />}
      {tab === 'agent'   && <GovernanceAgentChat />}
      {tab === 'data'    && <DataInfoTab />}
    </div>
  );
}

function TopTabButton({ active, onClick, icon, label, sub }:
  { active: boolean; onClick: () => void; icon: React.ReactNode; label: string; sub: string }) {
  return (
    <button onClick={onClick}
            className={`px-3.5 py-2 rounded-md text-sm font-medium transition flex items-center gap-2 ${
              active ? 'bg-amber-600 text-white' : 'text-gray-700 hover:bg-gray-100'
            }`}>
      {icon}
      <span className="flex flex-col items-start leading-tight">
        <span>{label}</span>
        <span className={`text-[10px] ${active ? 'text-amber-100' : 'text-gray-500'}`}>{sub}</span>
      </span>
    </button>
  );
}

// ---------------------------------------------------------------------------
// Monitor tab — continuous quality checks (bias live, others proposed)
// ---------------------------------------------------------------------------

function MonitorTab() {
  return (
    <div className="space-y-6">
      <BiasMonitor />
      <PremiumAdequacyMonitor />

      <div>
        <div className="mb-2 flex items-baseline justify-between">
          <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">More monitors</h3>
          <span className="text-[11px] text-gray-500 italic">
            same pattern as bias and adequacy — continuous scan + grounded agent investigation
          </span>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ProposedMonitorCard
            icon={<Activity className="w-5 h-5 text-blue-600" />}
            title="Model drift"
            tagline="Has the world moved since the last refit?"
            description="Compare the live feature distribution to the training data of the current champion. Flag features whose KL-divergence breaches a threshold. Investigate with an agent that can read the pack's training-data section and check the next refit window."
            chips={['feature drift', 'PSI / KL-divergence', 'refit-window check']}
          />
          <ProposedMonitorCard
            icon={<PieChart className="w-5 h-5 text-purple-600" />}
            title="Concentration risk"
            tagline="How exposed are we to the top decile?"
            description="Pareto check across geography, industry, and broker channel. Flags when the top 10% of policies generate &gt;X% of GWP — the kind of question the actuary gets asked at year-end."
            chips={['Pareto / Gini', 'top-decile exposure', 'reinsurance lens']}
          />
        </div>
      </div>
    </div>
  );
}

function ProposedMonitorCard({ icon, title, tagline, description, chips }: {
  icon: React.ReactNode; title: string; tagline: string; description: string; chips: string[];
}) {
  return (
    <div className="rounded-lg border border-dashed border-gray-300 bg-gray-50/50 p-4 relative">
      <span className="absolute top-2 right-2 text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-gray-200 text-gray-700">
        Coming next
      </span>
      <div className="flex items-center gap-2 mb-1.5">
        {icon}
        <h4 className="font-semibold text-gray-900">{title}</h4>
      </div>
      <p className="text-xs font-medium text-gray-700 italic mb-2">{tagline}</p>
      <p className="text-xs text-gray-600 leading-relaxed mb-3">{description}</p>
      <div className="flex flex-wrap gap-1">
        {chips.map(c => (
          <span key={c} className="text-[10px] px-1.5 py-0.5 rounded bg-white border border-gray-200 text-gray-600">{c}</span>
        ))}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Search tab — by-model / by-date / by-policy
// ---------------------------------------------------------------------------

function SearchTab() {
  const [mode, setMode]         = useState<Mode>('by-model');
  const [familyPacks, setFam]   = useState<FamilyPacks[]>([]);
  const [selectedPack, setPack] = useState<Pack | null>(null);
  const [policyContext, setPolicyContext] = useState<{ policy_id: string } | null>(null);

  useEffect(() => {
    api.listAllPacks().then(d => setFam(d.families || []));
  }, []);

  return (
    <div>
      <div className="bg-white rounded-lg border border-gray-200 p-1 mb-5 inline-flex gap-1">
        <SegButton active={mode === 'by-model'}  onClick={() => { setMode('by-model');  setPack(null); setPolicyContext(null); }}
                   icon={<Layers className="w-3.5 h-3.5" />} label="By model" />
        <SegButton active={mode === 'by-date'}   onClick={() => { setMode('by-date');   setPack(null); setPolicyContext(null); }}
                   icon={<Calendar className="w-3.5 h-3.5" />} label="By date" />
        <SegButton active={mode === 'by-policy'} onClick={() => { setMode('by-policy'); setPack(null); setPolicyContext(null); }}
                   icon={<Search className="w-3.5 h-3.5" />} label="By policy" />
      </div>

      <div className="grid gap-5">
        {!selectedPack && mode === 'by-model' &&
          <ByModel familyPacks={familyPacks} onPick={setPack} />}
        {!selectedPack && mode === 'by-date' &&
          <ByDate onPick={setPack} />}
        {!selectedPack && mode === 'by-policy' &&
          <ByPolicy onPick={(p, ctx) => { setPack(p); setPolicyContext(ctx); }} />}

        {selectedPack && (
          <PackViewer
            pack={selectedPack}
            policyContext={policyContext}
            onBack={() => { setPack(null); setPolicyContext(null); }}
          />
        )}
      </div>
    </div>
  );
}

function SegButton({ active, onClick, icon, label }:
  { active: boolean; onClick: () => void; icon: React.ReactNode; label: string }) {
  return (
    <button onClick={onClick}
            className={`inline-flex items-center gap-1.5 px-3 py-1.5 rounded text-sm font-medium transition ${
              active ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-100'
            }`}>
      {icon} {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// By model
// ---------------------------------------------------------------------------

function ByModel({ familyPacks, onPick }: { familyPacks: FamilyPacks[]; onPick: (p: Pack) => void }) {
  const [expanded, setExpanded] = useState<string | null>(familyPacks[0]?.key || null);
  useEffect(() => {
    if (!expanded && familyPacks.length > 0) setExpanded(familyPacks[0].key);
  }, [familyPacks.length]);

  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">Production champions</h3>
        <span className="text-xs text-gray-500">{familyPacks.length} model families</span>
      </div>
      <div>
        {familyPacks.map(fam => {
          const isOpen = expanded === fam.key;
          const champion = fam.packs[0];
          return (
            <div key={fam.key} className="border-b last:border-b-0">
              <button onClick={() => setExpanded(isOpen ? null : fam.key)}
                      className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-50">
                <div className="flex items-center gap-3">
                  <ChevronDown className={`w-4 h-4 text-gray-500 transition-transform ${isOpen ? '' : '-rotate-90'}`} />
                  <div className="text-left">
                    <div className="font-medium text-gray-900">{fam.label}</div>
                    <div className="text-xs text-gray-500 font-mono">{fam.key}</div>
                  </div>
                </div>
                <div className="text-xs text-gray-600 text-right">
                  {champion ? (
                    <>
                      <div>Current champion: <span className="font-mono text-gray-900">v{champion.model_version}</span></div>
                      <div className="text-[11px] text-gray-500">Pack {formatDate(champion.generated_at)} · {fam.packs.length} total</div>
                    </>
                  ) : <span className="italic text-gray-400">no packs yet</span>}
                </div>
              </button>
              {isOpen && <PackTimeline packs={fam.packs} onPick={onPick} />}
            </div>
          );
        })}
      </div>
    </section>
  );
}

function PackTimeline({ packs, onPick }: { packs: Pack[]; onPick: (p: Pack) => void }) {
  if (packs.length === 0) {
    return <div className="px-4 pb-4 text-xs text-gray-500 italic">No packs generated for this family yet.</div>;
  }
  return (
    <div className="px-4 pb-4">
      <div className="relative pl-6">
        <div className="absolute left-2 top-1 bottom-1 w-px bg-gray-200" />
        {packs.map((p, i) => {
          const isChampion = i === 0;
          return (
            <button key={p.pack_id} onClick={() => onPick(p)}
                    className="relative block w-full text-left py-2 px-3 hover:bg-gray-50 rounded group">
              <div className={`absolute -left-5 top-3 w-3 h-3 rounded-full border-2 ${
                isChampion ? 'bg-emerald-500 border-emerald-600' : 'bg-white border-gray-400'
              }`} />
              <div className="flex items-center justify-between gap-4">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-sm text-gray-900">v{p.model_version}</span>
                    {isChampion && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-700 font-medium">current</span>
                    )}
                    {p.simulated && (
                      <span className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-600">simulated</span>
                    )}
                    <span className="text-xs text-gray-600">{p.story}</span>
                  </div>
                  <div className="text-[11px] text-gray-500 mt-0.5">
                    Pack {formatDate(p.generated_at)} · generated by {(p.generated_by || '').split('@')[0]}
                    {p.primary_metric && p.primary_value != null && (
                      <span> · <span className="font-mono">{p.primary_metric}={Number(p.primary_value).toFixed(4)}</span></span>
                    )}
                  </div>
                </div>
                <span className="text-xs text-blue-600 group-hover:underline">Open pack →</span>
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// By date
// ---------------------------------------------------------------------------

function ByDate({ onPick }: { onPick: (p: Pack) => void }) {
  const today = new Date().toISOString().substring(0, 10);
  const [date, setDate] = useState(today);
  const [packs, setPacks] = useState<Pack[]>([]);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    api.getPacksOnDate(date).then(d => setPacks(d.packs || []))
      .catch(() => setPacks([])).finally(() => setLoading(false));
  }, [date]);

  const examples = [
    { label: 'Feb 2026 — initial baseline', date: '2026-02-25' },
    { label: 'Mar 2026 — mid-cycle refit',  date: '2026-03-20' },
    { label: 'Apr 2026 — current champions', date: '2026-04-25' },
  ];

  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between gap-3">
        <h3 className="text-sm font-semibold text-gray-800">Models in production on</h3>
        <input type="date" value={date} onChange={e => setDate(e.target.value)}
               className="border border-gray-300 rounded px-2 py-1 text-sm" />
      </div>
      <div className="px-4 py-2 bg-white border-b flex flex-wrap items-center gap-2">
        <span className="text-[11px] uppercase tracking-wider font-semibold text-gray-500">Try a date</span>
        {examples.map(e => (
          <button key={e.date}
                  onClick={() => setDate(e.date)}
                  className={`text-xs px-2.5 py-1 rounded-full border transition ${
                    date === e.date
                      ? 'bg-blue-600 text-white border-blue-600'
                      : 'bg-white text-gray-700 border-gray-300 hover:border-blue-400 hover:bg-blue-50'
                  }`}>
            {e.label}
          </button>
        ))}
        <span className="ml-auto text-[11px] text-gray-500 italic">
          For each family, shows the pack that was the most-recent-at-or-before the chosen date.
        </span>
      </div>
      {loading ? (
        <div className="py-8 text-center text-sm text-gray-500">
          <Loader2 className="w-4 h-4 inline animate-spin mr-1" /> Looking up packs…
        </div>
      ) : packs.length === 0 ? (
        <div className="py-8 text-center text-sm text-gray-500 italic">
          No packs had been generated on or before this date.
        </div>
      ) : (
        <table className="w-full text-sm">
          <thead>
            <tr className="text-xs text-gray-500 border-b bg-gray-50">
              <th className="text-left px-3 py-2 font-medium">Family</th>
              <th className="text-left px-3 py-2 font-medium">Version</th>
              <th className="text-left px-3 py-2 font-medium">Story</th>
              <th className="text-right px-3 py-2 font-medium">Primary metric</th>
              <th className="text-left px-3 py-2 font-medium">Pack generated</th>
              <th className="text-right px-3 py-2 font-medium">&nbsp;</th>
            </tr>
          </thead>
          <tbody>
            {packs.map(p => (
              <tr key={p.pack_id} className="border-b last:border-0 hover:bg-gray-50">
                <td className="px-3 py-2 font-medium text-gray-900">{p.model_family}</td>
                <td className="px-3 py-2 font-mono text-xs">v{p.model_version}</td>
                <td className="px-3 py-2 text-xs text-gray-600">{p.story || '—'}</td>
                <td className="px-3 py-2 text-right text-xs font-mono">
                  {p.primary_metric}={p.primary_value != null ? Number(p.primary_value).toFixed(4) : '—'}
                </td>
                <td className="px-3 py-2 text-xs text-gray-600">{formatDate(p.generated_at)}</td>
                <td className="px-3 py-2 text-right">
                  <button onClick={() => onPick(p)}
                          className="text-blue-600 hover:text-blue-800 text-xs font-medium">
                    Open pack →
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  );
}

// ---------------------------------------------------------------------------
// By policy — flagship flow
// ---------------------------------------------------------------------------

function ByPolicy({ onPick }: { onPick: (p: Pack, ctx: { policy_id: string }) => void }) {
  const [policyId, setPolicyId] = useState('');
  const [scoring, setScoring]   = useState<any>(null);
  const [loading, setLoading]   = useState(false);
  const [err, setErr]           = useState<string | null>(null);

  const run = async (id: string) => {
    if (!id.trim()) return;
    setLoading(true); setErr(null); setScoring(null);
    try {
      const d = await api.getPolicyScoring(id.trim().toUpperCase());
      setScoring(d);
    } catch (e: any) {
      setErr(e.message);
    } finally {
      setLoading(false);
    }
  };

  const submit = (e?: FormEvent) => { e?.preventDefault(); run(policyId); };

  const openPackFor = async (fam: any) => {
    if (!fam.pack_id) return;
    const pack = await api.getPackDetail(fam.pack_id);
    onPick(pack, { policy_id: scoring.policy_id });
  };

  return (
    <>
      <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="px-4 py-2.5 bg-gray-50 border-b">
          <h3 className="text-sm font-semibold text-gray-800 flex items-center gap-2">
            <UserCircle2 className="w-4 h-4 text-gray-500" /> Why was this customer priced £X?
          </h3>
        </div>
        <form onSubmit={submit} className="p-4 flex items-center gap-2">
          <input value={policyId}
                 onChange={e => setPolicyId(e.target.value.toUpperCase())}
                 placeholder="Policy ID (e.g. POL-100042)"
                 className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-sm font-mono" />
          <button type="submit" disabled={loading || !policyId.trim()}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
            {loading ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Search className="w-3.5 h-3.5" />}
            Look up
          </button>
          <button type="button"
                  onClick={() => { setPolicyId('POL-100042'); run('POL-100042'); }}
                  className="text-xs text-gray-500 hover:text-gray-700 ml-2">
            try POL-100042
          </button>
        </form>
        {err && <div className="px-4 pb-4 text-xs text-red-700"><AlertTriangle className="w-3 h-3 inline mr-1" /> {err}</div>}
      </section>

      {scoring && <ScoringStory scoring={scoring} openPackFor={openPackFor} />}
    </>
  );
}

function ScoringStory({ scoring, openPackFor }: { scoring: any; openPackFor: (fam: any) => void }) {
  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-800">
          Scoring story — {scoring.policy_id}
        </h3>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-amber-100 text-amber-800 font-medium">
          Simulated — no real inference log
        </span>
      </div>
      <div className="p-4 grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div>
          <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">Policy features at quote time</h4>
          <div className="bg-gray-50 border border-gray-200 rounded p-3 space-y-0.5 text-xs">
            {Object.entries(scoring.policy).map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2">
                <span className="text-gray-600">{k}</span>
                <span className="font-mono text-gray-900 text-right truncate">{String(v ?? '—')}</span>
              </div>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">Each model's prediction</h4>
          <div className="space-y-2">
            {scoring.models.map((m: any) => (
              <button key={m.family}
                      onClick={() => openPackFor(m)}
                      disabled={!m.pack_id}
                      className="w-full bg-white border border-gray-200 rounded p-2.5 hover:border-blue-300 hover:shadow-sm disabled:opacity-50 text-left group">
                <div className="flex items-center justify-between">
                  <div className="text-xs font-semibold text-gray-900">{m.label}</div>
                  <div className="text-xs font-mono text-gray-900">
                    {m.unit === 'GBP' ? `£${Number(m.prediction).toLocaleString()}` : Number(m.prediction).toFixed(3)}
                  </div>
                </div>
                <div className="text-[10px] text-gray-500 mt-0.5 flex items-center justify-between">
                  <span>v{m.model_version || '—'} · {m.unit}</span>
                  {m.pack_id && <span className="text-blue-600 group-hover:underline">open pack →</span>}
                </div>
              </button>
            ))}
          </div>
        </div>

        <div>
          <h4 className="text-xs font-semibold text-gray-700 uppercase tracking-wide mb-2">Price build-up</h4>
          <div className="bg-gray-50 border border-gray-200 rounded p-3 text-xs space-y-1">
            {scoring.price_build_up.map((step: any, i: number) => (
              <div key={i}
                   className={`flex justify-between ${step.emphasis ? 'border-t border-gray-300 pt-2 mt-1 font-semibold text-gray-900' : ''}`}>
                <span className="text-gray-700">{step.label}</span>
                <span className="font-mono">£{Number(step.amount).toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
              </div>
            ))}
          </div>
          <div className="text-[11px] text-gray-500 italic mt-2">
            {scoring.note}
          </div>
        </div>
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Pack Viewer
// ---------------------------------------------------------------------------

function PackViewer({ pack, policyContext, onBack }:
  { pack: Pack; policyContext: { policy_id: string } | null; onBack: () => void }) {
  const [showPdf, setShowPdf]   = useState(true);
  const [showChat, setShowChat] = useState(true);
  const pdfUrl = api.packPdfUrl(pack.pack_id);

  return (
    <section className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-2.5 bg-gray-50 border-b flex items-center justify-between gap-3">
        <div>
          <h3 className="text-sm font-semibold text-gray-800">
            Pack · {pack.model_family} v{pack.model_version}
          </h3>
          <div className="text-[11px] text-gray-500 font-mono">{pack.pack_id}</div>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={() => setShowPdf(!showPdf)}
                  className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1 border ${
                    showPdf ? 'bg-blue-50 text-blue-700 border-blue-200'
                            : 'bg-gray-50 text-gray-600 border-gray-200'
                  }`}>
            {showPdf ? <Eye className="w-3 h-3" /> : <EyeOff className="w-3 h-3" />} PDF
          </button>
          <button onClick={() => setShowChat(!showChat)}
                  className={`text-xs px-2 py-1 rounded inline-flex items-center gap-1 border ${
                    showChat ? 'bg-blue-50 text-blue-700 border-blue-200'
                             : 'bg-gray-50 text-gray-600 border-gray-200'
                  }`}>
            <Bot className="w-3 h-3" /> Agent
          </button>
          <a href={pdfUrl} target="_blank" rel="noopener noreferrer"
             className="text-xs text-gray-600 hover:text-gray-900 inline-flex items-center gap-1">
            <ExternalLink className="w-3 h-3" /> Download
          </a>
          <button onClick={onBack}
                  className="text-xs text-gray-500 hover:text-gray-800">← Back</button>
        </div>
      </div>

      <div className={`grid gap-4 p-4 ${showPdf && showChat ? 'lg:grid-cols-2' : 'grid-cols-1'}`}>
        {showPdf  && <PdfPane pack={pack} pdfUrl={pdfUrl} />}
        {showChat && <ChatPane pack={pack} policyContext={policyContext} />}
      </div>
    </section>
  );
}

function PdfPane({ pack, pdfUrl }: { pack: Pack; pdfUrl: string }) {
  const sections = [
    { id: 1,  label: 'Executive summary' },
    { id: 2,  label: 'Business context & intended use' },
    { id: 3,  label: 'Data lineage & sources' },
    { id: 4,  label: 'Model specification' },
    { id: 5,  label: 'Performance evidence' },
    { id: 6,  label: 'Feature behaviour' },
    { id: 7,  label: 'Stability & version history' },
    { id: 8,  label: 'Fairness & ethical considerations' },
    { id: 9,  label: 'Risks & controls' },
    { id: 10, label: 'Regulatory coverage' },
    { id: 11, label: 'Audit trail' },
    { id: 12, label: 'Committee sign-off' },
  ];

  return (
    <div>
      <div className="flex items-center gap-2 text-xs text-gray-600 mb-2">
        <BookOpen className="w-3.5 h-3.5" />
        Section bookmarks
      </div>
      <div className="flex flex-wrap gap-1 mb-2">
        {sections.map(s => (
          <a key={s.id}
             href={`${pdfUrl}#page=${s.id + 1}`} target="_blank" rel="noopener noreferrer"
             className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-700 hover:bg-blue-100 hover:text-blue-700"
             title={s.label}>
            {s.id}. {s.label}
          </a>
        ))}
      </div>
      <div className="border border-gray-200 rounded overflow-hidden bg-gray-50" style={{ height: '70vh' }}>
        <object data={pdfUrl} type="application/pdf" className="w-full h-full">
          <div className="flex flex-col items-center justify-center h-full text-sm text-gray-600 p-6">
            Your browser can't preview this PDF inline.
            <a href={pdfUrl} target="_blank" rel="noreferrer"
               className="mt-2 inline-flex items-center gap-1.5 px-3 py-1.5 rounded bg-blue-600 text-white hover:bg-blue-700">
              Open the pack in a new tab <ExternalLink className="w-3.5 h-3.5" />
            </a>
          </div>
        </object>
      </div>
      <div className="mt-2 flex justify-end">
        <a href={pdfUrl} target="_blank" rel="noreferrer"
           className="text-[11px] text-blue-600 hover:underline inline-flex items-center gap-1">
          Open full-size in a new tab <ExternalLink className="w-3 h-3" />
        </a>
      </div>
      <div className="text-[11px] text-gray-500 mt-2">
        Generated {formatDate(pack.generated_at)} by {(pack.generated_by || '').split('@')[0]} · {formatBytes(pack.size_bytes)}
      </div>
    </div>
  );
}

type ToolStep = { hop: number; tool: string; arguments: any; result_summary: string };

type ChatTurn = {
  question: string;
  answer?: string;
  loading?: boolean;
  model?: string;
  endpoint?: string;
  source?: string;
  cited_sections?: string[];
  tool_trace?: ToolStep[];
  usage?: { prompt_tokens?: number; completion_tokens?: number; total_tokens?: number };
  error?: string;
  fallback_reason?: string;
};

const SUGGESTED_QUESTIONS_GENERAL = [
  "Why was this model promoted over the previous version?",
  "Is this model fair across protected characteristics?",
  "What are the top 5 drivers of predictions?",
  "Has this model drifted since the last version?",
  "Draft a regulator response: customer complains the pricing is unfair",
];
const SUGGESTED_QUESTIONS_POLICY = [
  "Draft a formal response to the customer explaining this price",
  "Are there any fairness concerns specific to this policy?",
  "Which single factor contributed most to this price?",
];

function ChatPane({ pack, policyContext }:
  { pack: Pack; policyContext: { policy_id: string } | null }) {
  const [input, setInput]   = useState('');
  const [turns, setTurns]   = useState<ChatTurn[]>([]);
  const [busy, setBusy]     = useState(false);
  const [showTrace, setShowTrace] = useState(false);

  const suggestions = useMemo(() => [
    ...(policyContext ? SUGGESTED_QUESTIONS_POLICY : []),
    ...SUGGESTED_QUESTIONS_GENERAL,
  ], [policyContext]);

  const ask = async (q: string) => {
    const question = q.trim();
    if (!question || busy) return;
    setBusy(true);
    setInput('');
    setTurns(t => [...t, { question, loading: true }]);
    try {
      const r = await api.chatWithPack(pack.pack_id, question, policyContext?.policy_id);
      setTurns(t => t.map((x, i) =>
        i === t.length - 1
          ? { ...x, loading: false, answer: r.answer, model: r.model, endpoint: r.endpoint,
              source: r.source, fallback_reason: r.fallback_reason,
              cited_sections: r.cited_sections, tool_trace: r.tool_trace,
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
    <div>
      <div className="flex items-center gap-2 text-xs text-gray-600 mb-2 flex-wrap">
        <Sparkles className="w-3.5 h-3.5 text-violet-500" />
        <span>Ask a question — grounded in this pack's content</span>
        <span className="text-[10px] px-1.5 py-0.5 rounded bg-violet-100 text-violet-700 font-medium">
          Agent Framework · pricing_governance_agent
        </span>
      </div>

      <div className="border border-gray-200 rounded flex flex-col" style={{ height: '70vh' }}>
        <div className="flex-1 overflow-y-auto p-3 space-y-3 bg-gray-50">
          {turns.length === 0 && (
            <div className="py-4">
              <div className="text-xs text-gray-500 mb-3 text-center">
                Pre-populated questions{policyContext && <> · <code className="bg-white px-1 rounded">{policyContext.policy_id}</code></>}
              </div>
              <div className="flex flex-col gap-1.5">
                {suggestions.map(s => (
                  <button key={s} onClick={() => ask(s)}
                          className="text-xs text-left px-3 py-1.5 rounded bg-white border border-gray-200 hover:border-blue-300 hover:bg-blue-50 text-gray-800">
                    {s}
                  </button>
                ))}
              </div>
            </div>
          )}

          {turns.map((t, i) => (
            <div key={i} className="space-y-1">
              <div className="flex gap-2">
                <UserCircle2 className="w-4 h-4 text-gray-500 shrink-0 mt-0.5" />
                <div className="text-sm text-gray-800 flex-1">{t.question}</div>
              </div>
              <div className="flex gap-2 pl-1">
                <Bot className="w-4 h-4 text-violet-600 shrink-0 mt-0.5" />
                <div className="flex-1">
                  {t.loading ? (
                    <div className="text-xs text-gray-500 italic inline-flex items-center gap-1">
                      <Loader2 className="w-3 h-3 animate-spin" /> Grounding in pack…
                    </div>
                  ) : t.error ? (
                    <div className="text-xs text-red-700">
                      <AlertTriangle className="w-3 h-3 inline mr-1" /> {t.error}
                    </div>
                  ) : (
                    <>
                      {t.tool_trace && t.tool_trace.length > 0 && (
                        <div className="mb-2 space-y-0.5">
                          {t.tool_trace.map((s, idx) => (
                            <div key={idx}
                                 className="text-[11px] text-violet-900 bg-violet-50 border border-violet-200 rounded px-2 py-1 inline-flex items-center gap-1.5 mr-1">
                              <Bot className="w-3 h-3" />
                              <span className="font-mono">{s.tool}</span>
                              <span className="text-violet-700 font-mono">({summariseArgs(s.arguments)})</span>
                              <span className="text-violet-600">→ {s.result_summary}</span>
                            </div>
                          ))}
                        </div>
                      )}
                      <div className="text-sm text-gray-900 whitespace-pre-wrap">{t.answer}</div>
                      <div className="mt-1 flex items-center gap-3 flex-wrap text-[10px] text-gray-500">
                        {t.endpoint && <span>endpoint: {t.endpoint}</span>}
                        {t.source === 'fm_api_fallback' && (
                          <span className="text-amber-700" title={t.fallback_reason || ''}>
                            fallback: FM API (agent unavailable)
                          </span>
                        )}
                        {t.cited_sections && t.cited_sections.length > 0 && (
                          <span>cited: {t.cited_sections.map(s => `§${s}`).join(', ')}</span>
                        )}
                        {t.usage?.total_tokens != null && <span>tokens: {t.usage.total_tokens}</span>}
                      </div>
                    </>
                  )}
                </div>
              </div>
            </div>
          ))}
        </div>

        <form onSubmit={(e) => { e.preventDefault(); ask(input); }}
              className="border-t bg-white p-2 flex gap-2">
          <input value={input} onChange={e => setInput(e.target.value)}
                 placeholder="Ask about this pack…"
                 disabled={busy}
                 className="flex-1 border border-gray-300 rounded px-3 py-1.5 text-sm" />
          <button type="submit" disabled={busy || !input.trim()}
                  className="inline-flex items-center gap-1 px-3 py-1.5 bg-blue-600 text-white rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
            {busy ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : <Send className="w-3.5 h-3.5" />} Ask
          </button>
        </form>
      </div>

      <button onClick={() => setShowTrace(!showTrace)}
              className="text-[11px] text-gray-500 hover:text-gray-700 mt-2 inline-flex items-center gap-1">
        <MessageSquare className="w-3 h-3" /> {showTrace ? 'Hide' : 'Show'} full LLM interaction
      </button>
      {showTrace && turns.length > 0 && (
        <div className="bg-gray-900 text-gray-100 text-[11px] font-mono rounded p-3 mt-2 whitespace-pre-wrap max-h-96 overflow-auto">
          {turns.map((t, i) => (
            <div key={i} className="mb-3">
              <div className="text-violet-300">user: {t.question}</div>
              {(t.tool_trace || []).map((s, idx) => (
                <div key={idx} className="text-amber-300 mt-1">
                  tool[{s.hop}]: {s.tool}({summariseArgs(s.arguments)}) → {s.result_summary}
                </div>
              ))}
              <div className="text-emerald-300 mt-1">assistant: {t.answer || t.error || '(loading…)'}</div>
              {t.usage && <div className="text-gray-400 mt-1 text-[10px]">tokens: {JSON.stringify(t.usage)}</div>}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso?: string | null) {
  if (!iso) return '—';
  const d = new Date(iso);
  if (isNaN(d.getTime())) return String(iso).substring(0, 10);
  return d.toISOString().substring(0, 10);
}

function formatBytes(n?: number) {
  if (!n) return '—';
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

function summariseArgs(args: any): string {
  if (!args || typeof args !== 'object') return '';
  const parts: string[] = [];
  for (const [k, v] of Object.entries(args)) {
    if (v == null || v === '') continue;
    let s = typeof v === 'string' ? v : JSON.stringify(v);
    if (s.length > 40) s = s.slice(0, 37) + '…';
    parts.push(`${k}=${s}`);
  }
  return parts.join(' ');
}

// ---------------------------------------------------------------------------
// Bias Monitor — the Governance tab's flagship panel
// ---------------------------------------------------------------------------

type Cohort = {
  cohort: string; n: number;
  avg_premium?: number | null;
  metric?: number | null;
  freq_pred?: number | null; sev_pred?: number | null;
  demand_pred?: number | null; fraud_pred?: number | null;
};
type BiasMonitorData = {
  protected_attribute: string;
  family: string;
  cohorts: Cohort[];
  headline: { max_premium: number; min_premium: number; gap_abs: number; gap_pct: number | null } | null;
  scan_timestamp: string;
};

type BiasAttr = 'director_gender' | 'postcode_demographic' | 'ethnicity_proxy' | 'director_age_band';

function BiasMonitor() {
  const [attr, setAttr]       = useState<BiasAttr>('director_gender');
  const [data, setData]       = useState<BiasMonitorData | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr]         = useState<string | null>(null);
  const [scanMs, setScanMs]   = useState<number | null>(null);

  // Agent investigation state
  const [investigating, setInvestigating]   = useState(false);
  const [investigation, setInvestigation]   = useState<any>(null);
  const [investMs, setInvestMs]             = useState<number | null>(null);
  const [question, setQuestion]             = useState('');

  const loadMonitor = async () => {
    setLoading(true); setErr(null); setScanMs(null);
    const t0 = performance.now();
    try {
      const d = await api.getBiasMonitor(attr);
      setData(d);
      setScanMs(Math.round(performance.now() - t0));
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadMonitor(); /* eslint-disable-next-line */ }, [attr]);

  const runInvestigate = async (customQuestion?: string) => {
    setInvestigating(true); setInvestigation(null); setInvestMs(null);
    const t0 = performance.now();
    try {
      const q = customQuestion || question.trim() ||
        `Scan ${attr.replace('_', ' ')} across all production families. Can we defend the gap to a regulator?`;
      const r = await api.biasInvestigate(q, attr);
      setInvestigation(r);
      setInvestMs(Math.round(performance.now() - t0));
    } catch (e: any) {
      setInvestigation({ ok: false, error: e.message || String(e) });
    } finally {
      setInvestigating(false);
    }
  };

  const gapPct    = data?.headline?.gap_pct ?? null;
  const severity  = gapPct == null ? 'unknown'
                   : gapPct >= 15 ? 'red'
                   : gapPct >= 5  ? 'amber'
                   : 'green';
  const sevHero  = severity === 'red'    ? 'from-red-50 to-red-100 border-red-300'
                 : severity === 'amber'  ? 'from-amber-50 to-amber-100 border-amber-300'
                 : severity === 'green'  ? 'from-emerald-50 to-emerald-100 border-emerald-300'
                 : 'from-gray-50 to-gray-100 border-gray-300';
  const sevText  = severity === 'red'    ? 'text-red-900'
                 : severity === 'amber'  ? 'text-amber-900'
                 : severity === 'green'  ? 'text-emerald-900'
                 : 'text-gray-700';
  const sevLabel = severity === 'red'    ? 'Material disparity'
                 : severity === 'amber'  ? 'Notable disparity'
                 : severity === 'green'  ? 'Within tolerance'
                 : 'No signal';

  // Suggested investigation prompts the actuary can fire one-click
  const suggestions = [
    'Is the gap risk-justified by actual loss experience?',
    'Identify any proxy features driving the disparity',
    'Draft a Consumer Duty fair-value statement for the FCA',
  ];

  return (
    <div className="rounded-xl border border-indigo-200 bg-white overflow-hidden shadow-sm">
      {/* Header */}
      <div className="px-5 py-4 border-b border-indigo-200 bg-gradient-to-r from-indigo-50 to-white flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <Scale className="w-5 h-5 text-indigo-700 mt-0.5" />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-gray-900">Bias monitor</span>
              <span className="text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                Live
              </span>
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              Production predictions grouped by a protected attribute we do <span className="font-semibold">not</span> model.
              If a gap exists, the agent investigates whether we can defend it.
            </div>
          </div>
        </div>
        <div className="flex flex-wrap rounded-md border border-gray-300 bg-white overflow-hidden shrink-0">
          {([
            { k: 'director_gender',      label: 'Gender' },
            { k: 'ethnicity_proxy',      label: 'Ethnicity (proxy)' },
            { k: 'director_age_band',    label: 'Age band' },
            { k: 'postcode_demographic', label: 'Postcode demographic' },
          ] as const).map(opt => (
            <button key={opt.k}
                    onClick={() => setAttr(opt.k)}
                    className={`px-3 py-1.5 text-xs font-medium transition border-l first:border-l-0 border-gray-200 ${
                      attr === opt.k ? 'bg-indigo-600 text-white border-indigo-600' : 'text-gray-700 hover:bg-gray-50'
                    }`}>
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Wow framing — traditional vs Databricks */}
      <div className="px-5 py-3 border-b border-indigo-100 bg-indigo-50/40">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div className="flex items-start gap-2 text-gray-600">
            <Clock className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-gray-700">Traditional approach:</span>{' '}
              data pull → SQL by analyst → pivot → fairness memo → legal review. Typically a 2–3 week project per attribute.
            </div>
          </div>
          <div className="flex items-start gap-2 text-indigo-900">
            <Zap className="w-3.5 h-3.5 mt-0.5 shrink-0 text-indigo-700" />
            <div>
              <span className="font-semibold">On Databricks:</span>{' '}
              one click — scan in <span className="font-mono">{scanMs != null ? `${scanMs} ms` : '…'}</span>,
              regulator-ready narrative in <span className="font-mono">{investMs != null ? `${(investMs/1000).toFixed(1)} s` : '~10 s'}</span>,
              audit-logged with the full tool trace.
            </div>
          </div>
        </div>
      </div>

      {/* Body */}
      <div className="px-5 py-4">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" /> Scanning portfolio…
          </div>
        )}
        {err && (
          <div className="text-sm text-red-600">Failed to load monitor: {err}</div>
        )}
        {data && data.cohorts.length > 0 && (
          <>
            {/* Severity hero */}
            <div className={`rounded-lg border bg-gradient-to-r ${sevHero} px-5 py-4 mb-4`}>
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <div className={`text-[11px] uppercase tracking-wider font-bold ${sevText} opacity-80`}>
                    Verdict
                  </div>
                  <div className={`text-2xl font-bold ${sevText}`}>{sevLabel}</div>
                  {data.headline && (
                    <div className="text-sm text-gray-700 mt-1">
                      <span className="font-semibold">£{data.headline.max_premium.toLocaleString()}</span>
                      {' '} top cohort vs{' '}
                      <span className="font-semibold">£{data.headline.min_premium.toLocaleString()}</span> bottom
                    </div>
                  )}
                </div>
                {gapPct != null && (
                  <div className="text-right">
                    <div className={`text-4xl font-extrabold ${sevText} leading-none`}>{gapPct.toFixed(1)}%</div>
                    <div className={`text-xs ${sevText} opacity-80 mt-1`}>premium gap</div>
                  </div>
                )}
              </div>
              <div className={`text-[11px] ${sevText} opacity-70 mt-3 flex items-center gap-1`}>
                <Clock className="w-3 h-3" />
                Scanned {(data.cohorts.reduce((s, c) => s + (c.n || 0), 0)).toLocaleString()} policies
                across {data.cohorts.length} cohorts
                {scanMs != null && <> in <span className="font-mono">{scanMs} ms</span></>}
              </div>
            </div>

            {/* Cohort table */}
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-gray-200 bg-gray-50">
                  <tr className="text-left text-gray-600">
                    <th className="py-1.5 px-3">Cohort</th>
                    <th className="py-1.5 px-3 text-right">Policies</th>
                    <th className="py-1.5 px-3 text-right">Freq</th>
                    <th className="py-1.5 px-3 text-right">Sev</th>
                    <th className="py-1.5 px-3 text-right">Demand</th>
                    <th className="py-1.5 px-3 text-right">Fraud</th>
                    <th className="py-1.5 px-3 text-right">Avg premium</th>
                  </tr>
                </thead>
                <tbody>
                  {data.cohorts.map(c => (
                    <tr key={c.cohort} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-1.5 px-3 font-medium text-gray-900">{c.cohort}</td>
                      <td className="py-1.5 px-3 text-right text-gray-600">{(c.n || 0).toLocaleString()}</td>
                      <td className="py-1.5 px-3 text-right">{c.freq_pred?.toFixed(3) ?? '—'}</td>
                      <td className="py-1.5 px-3 text-right">{c.sev_pred ? `£${c.sev_pred.toLocaleString()}` : '—'}</td>
                      <td className="py-1.5 px-3 text-right">{c.demand_pred?.toFixed(3) ?? '—'}</td>
                      <td className="py-1.5 px-3 text-right">{c.fraud_pred?.toFixed(3) ?? '—'}</td>
                      <td className="py-1.5 px-3 text-right font-semibold text-gray-900">£{(c.avg_premium || 0).toLocaleString()}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      {/* Agent investigation */}
      <div className="px-5 py-4 border-t border-indigo-100 bg-gradient-to-b from-indigo-50/30 to-white">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="text-sm font-medium text-gray-800 flex items-center gap-1.5">
            <Wand2 className="w-4 h-4 text-indigo-600" /> One-click investigation — grounded agent
          </div>
          {investigation?.usage?.total_tokens && (
            <div className="text-[11px] text-gray-500 flex items-center gap-2">
              <span>{investigation.trace?.length || 0} tool calls</span>
              <span>·</span>
              <span>{investigation.usage.total_tokens.toLocaleString()} tokens</span>
              {investMs != null && <><span>·</span><span className="font-mono">{(investMs/1000).toFixed(1)}s wall</span></>}
            </div>
          )}
        </div>

        {/* Suggestion chips — pick a question, run instantly */}
        {!investigating && !investigation && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {suggestions.map(s => (
              <button key={s}
                      onClick={() => runInvestigate(s)}
                      className="text-[11px] px-2.5 py-1 rounded-full border border-indigo-300 bg-white hover:bg-indigo-50 text-indigo-800">
                {s}
              </button>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask e.g. 'Is this gap risk-justified?' (or leave blank for a full scan)"
            className="flex-1 px-3 py-1.5 rounded-md border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-indigo-400"
            disabled={investigating}
            onKeyDown={e => { if (e.key === 'Enter') runInvestigate(); }}
          />
          <button
            onClick={() => runInvestigate()}
            disabled={investigating}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-indigo-600 text-white text-sm font-medium hover:bg-indigo-700 disabled:opacity-50"
          >
            {investigating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            {investigating ? 'Agent working…' : 'Run investigation'}
          </button>
        </div>

        {/* Live "agent thinking" placeholder while running */}
        {investigating && (
          <div className="mt-3 rounded-md border border-indigo-200 bg-white p-3 text-xs text-indigo-800 flex items-center gap-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Agent is calling its tools — reading the bias monitor, joining actual loss experience,
            inspecting proxy features, and citing the governance pack…
          </div>
        )}

        {investigation && (
          <div className="mt-4">
            {investigation.ok === false && investigation.error && (
              <div className="text-sm text-red-600">Agent failed: {investigation.error}</div>
            )}
            {investigation.ok && (
              <>
                {/* Tool trace as a numbered timeline */}
                {(investigation.trace?.length || 0) > 0 && (
                  <div className="mb-3 rounded-md border border-indigo-100 bg-indigo-50/40 p-3">
                    <div className="text-[11px] uppercase tracking-wider font-semibold text-indigo-800 mb-2 flex items-center gap-1">
                      <Clock className="w-3 h-3" /> Tool trace · executed in order
                    </div>
                    <ol className="space-y-1">
                      {(investigation.trace || []).map((t: any, i: number) => {
                        const errored = String(t.result_summary || '').startsWith('error');
                        return (
                          <li key={i} className="text-xs flex items-start gap-2">
                            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-white border border-indigo-200 text-indigo-700 shrink-0">
                              {String(i + 1).padStart(2, '0')}
                            </span>
                            <div className="min-w-0">
                              <code className={`font-semibold ${errored ? 'text-red-700' : 'text-indigo-900'}`}>{t.tool}</code>
                              {t.result_summary && (
                                <span className="text-gray-600 ml-2">{String(t.result_summary).slice(0, 140)}</span>
                              )}
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                )}
                {/* Structured answer */}
                <div className="rounded-md border border-gray-200 bg-white p-4 text-sm whitespace-pre-wrap leading-relaxed">
                  {investigation.answer}
                </div>
                <div className="mt-2 text-[11px] text-gray-500 italic">
                  Logged to <code>audit_log</code> as a <code>bias_investigation</code> event with the full tool trace
                  — every step is reproducible.
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Premium Adequacy Monitor — predicted vs actual loss ratio per cohort
// ---------------------------------------------------------------------------

type AdequacyDim = 'industry_risk_tier' | 'region' | 'construction_type';
type AdequacyCohort = {
  cohort: string; n: number;
  avg_premium: number | null;
  avg_annual_loss: number | null;
  loss_ratio: number | null;
};
type AdequacyData = {
  cohort_dimension: AdequacyDim;
  cohorts: AdequacyCohort[];
  headline: { max_loss_ratio: number; min_loss_ratio: number; spread_pp: number; underpriced_cohorts: number } | null;
  scan_timestamp: string;
};

function PremiumAdequacyMonitor() {
  const [dim, setDim]         = useState<AdequacyDim>('industry_risk_tier');
  const [data, setData]       = useState<AdequacyData | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr]         = useState<string | null>(null);
  const [scanMs, setScanMs]   = useState<number | null>(null);

  const [investigating, setInvestigating] = useState(false);
  const [investigation, setInvestigation] = useState<any>(null);
  const [investMs, setInvestMs]           = useState<number | null>(null);
  const [question, setQuestion]           = useState('');

  const load = async () => {
    setLoading(true); setErr(null); setScanMs(null);
    const t0 = performance.now();
    try {
      const d = await api.getPremiumAdequacy(dim);
      setData(d);
      setScanMs(Math.round(performance.now() - t0));
    } catch (e: any) {
      setErr(e.message || String(e));
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { load(); /* eslint-disable-next-line */ }, [dim]);

  const runInvestigate = async (customQuestion?: string) => {
    setInvestigating(true); setInvestigation(null); setInvestMs(null);
    const t0 = performance.now();
    try {
      const q = customQuestion || question.trim() ||
        `Loss ratio spread is material across ${dim.replace(/_/g, ' ')}. Which features could the rating engine adopt to close the gap? Defend the recommendation against actual loss experience.`;
      const r = await api.adequacyInvestigate(q, dim);
      setInvestigation(r);
      setInvestMs(Math.round(performance.now() - t0));
    } catch (e: any) {
      setInvestigation({ ok: false, error: e.message || String(e) });
    } finally {
      setInvestigating(false);
    }
  };

  // Verdict: ratio of max to min cohort loss ratio. Wide spread = pricing
  // inadequacy in some cohort.
  const ratio = data?.headline && data.headline.min_loss_ratio > 0
    ? data.headline.max_loss_ratio / data.headline.min_loss_ratio
    : null;
  const severity  = ratio == null ? 'unknown'
                   : ratio >= 2.0 ? 'red'
                   : ratio >= 1.5 ? 'amber'
                   : 'green';
  const sevHero  = severity === 'red'    ? 'from-red-50 to-red-100 border-red-300'
                 : severity === 'amber'  ? 'from-amber-50 to-amber-100 border-amber-300'
                 : severity === 'green'  ? 'from-emerald-50 to-emerald-100 border-emerald-300'
                 : 'from-gray-50 to-gray-100 border-gray-300';
  const sevText  = severity === 'red'    ? 'text-red-900'
                 : severity === 'amber'  ? 'text-amber-900'
                 : severity === 'green'  ? 'text-emerald-900'
                 : 'text-gray-700';
  const sevLabel = severity === 'red'    ? 'Material spread — pricing inadequacy likely'
                 : severity === 'amber'  ? 'Notable spread'
                 : severity === 'green'  ? 'Within tolerance'
                 : 'No signal';

  const suggestions = [
    'Which cohorts are mis-priced relative to the rest of the book?',
    'What features could the rating engine adopt to close the spread?',
    'Should we trigger an out-of-cycle refit for the worst cohort?',
  ];

  // Sort cohorts so worst (highest loss ratio) is first
  const sortedCohorts = data ? [...data.cohorts].sort(
    (a, b) => (b.loss_ratio || 0) - (a.loss_ratio || 0)
  ) : [];
  const maxLR = data?.headline?.max_loss_ratio ?? 1;

  return (
    <div className="rounded-xl border border-emerald-200 bg-white overflow-hidden shadow-sm">
      <div className="px-5 py-4 border-b border-emerald-200 bg-gradient-to-r from-emerald-50 to-white flex items-start justify-between gap-4">
        <div className="flex items-start gap-3">
          <TrendingUp className="w-5 h-5 text-emerald-700 mt-0.5" />
          <div>
            <div className="flex items-center gap-2">
              <span className="font-semibold text-gray-900">Premium adequacy</span>
              <span className="text-[10px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-emerald-100 text-emerald-800 border border-emerald-200">
                Live
              </span>
            </div>
            <div className="text-xs text-gray-500 mt-0.5">
              Loss ratio per cohort — predicted technical premium against the book's
              actual incurred experience. Wide spread implies under- or over-pricing in some cohort.
            </div>
          </div>
        </div>
        <div className="flex flex-wrap rounded-md border border-gray-300 bg-white overflow-hidden shrink-0">
          {([
            { k: 'industry_risk_tier', label: 'Industry tier' },
            { k: 'region',             label: 'Region' },
            { k: 'construction_type',  label: 'Construction' },
          ] as const).map(opt => (
            <button key={opt.k}
                    onClick={() => setDim(opt.k)}
                    className={`px-3 py-1.5 text-xs font-medium transition border-l first:border-l-0 border-gray-200 ${
                      dim === opt.k ? 'bg-emerald-600 text-white border-emerald-600' : 'text-gray-700 hover:bg-gray-50'
                    }`}>
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      <div className="px-5 py-3 border-b border-emerald-100 bg-emerald-50/40">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
          <div className="flex items-start gap-2 text-gray-600">
            <Clock className="w-3.5 h-3.5 mt-0.5 shrink-0" />
            <div>
              <span className="font-semibold text-gray-700">Traditional approach:</span>{' '}
              actuary builds a quarterly experience study — claims data pull, premium reconciliation, cohort grids in Excel. Days of work per dimension.
            </div>
          </div>
          <div className="flex items-start gap-2 text-emerald-900">
            <Zap className="w-3.5 h-3.5 mt-0.5 shrink-0 text-emerald-700" />
            <div>
              <span className="font-semibold">On Databricks:</span>{' '}
              one click — scan in <span className="font-mono">{scanMs != null ? `${scanMs} ms` : '…'}</span>,
              recommendation in <span className="font-mono">{investMs != null ? `${(investMs/1000).toFixed(1)} s` : '~10 s'}</span>,
              ties straight back to the rating factor catalogue.
            </div>
          </div>
        </div>
      </div>

      <div className="px-5 py-4">
        {loading && (
          <div className="flex items-center gap-2 text-sm text-gray-500">
            <Loader2 className="w-4 h-4 animate-spin" /> Scanning cohorts…
          </div>
        )}
        {err && <div className="text-sm text-red-600">Failed to load monitor: {err}</div>}
        {data && data.cohorts.length > 0 && (
          <>
            <div className={`rounded-lg border bg-gradient-to-r ${sevHero} px-5 py-4 mb-4`}>
              <div className="flex items-center justify-between gap-4 flex-wrap">
                <div>
                  <div className={`text-[11px] uppercase tracking-wider font-bold ${sevText} opacity-80`}>
                    Verdict
                  </div>
                  <div className={`text-2xl font-bold ${sevText}`}>{sevLabel}</div>
                  {data.headline && (
                    <div className="text-sm text-gray-700 mt-1">
                      Worst cohort loss ratio:{' '}
                      <span className="font-semibold">{data.headline.max_loss_ratio.toFixed(2)}</span> ·
                      best:{' '}
                      <span className="font-semibold">{data.headline.min_loss_ratio.toFixed(2)}</span>
                      {' '}·{' '}
                      <span className="font-semibold">{data.headline.underpriced_cohorts}</span>{' '}
                      cohort{data.headline.underpriced_cohorts === 1 ? '' : 's'} with LR &gt; 1.0
                    </div>
                  )}
                </div>
                {ratio != null && (
                  <div className="text-right">
                    <div className={`text-4xl font-extrabold ${sevText} leading-none`}>{ratio.toFixed(1)}×</div>
                    <div className={`text-xs ${sevText} opacity-80 mt-1`}>worst-vs-best spread</div>
                  </div>
                )}
              </div>
              <div className={`text-[11px] ${sevText} opacity-70 mt-3 flex items-center gap-1`}>
                <Clock className="w-3 h-3" />
                Scanned {(data.cohorts.reduce((s, c) => s + (c.n || 0), 0)).toLocaleString()} policies
                across {data.cohorts.length} cohorts
                {scanMs != null && <> in <span className="font-mono">{scanMs} ms</span></>}
                {' '}· loss ratio = average annual incurred / average technical premium
              </div>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead className="border-b border-gray-200 bg-gray-50">
                  <tr className="text-left text-gray-600">
                    <th className="py-1.5 px-3">Cohort</th>
                    <th className="py-1.5 px-3 text-right">Policies</th>
                    <th className="py-1.5 px-3 text-right">Avg premium</th>
                    <th className="py-1.5 px-3 text-right">Avg annual loss</th>
                    <th className="py-1.5 px-3 text-right">Loss ratio</th>
                    <th className="py-1.5 px-3" style={{minWidth: '180px'}}>Relative</th>
                  </tr>
                </thead>
                <tbody>
                  {sortedCohorts.map(c => {
                    const lr = c.loss_ratio || 0;
                    const pct = maxLR > 0 ? Math.min(100, (lr / maxLR) * 100) : 0;
                    const barTone = lr >= maxLR * 0.95 ? 'bg-red-500'
                                  : lr >= maxLR * 0.7  ? 'bg-amber-500'
                                  :                      'bg-emerald-500';
                    return (
                      <tr key={c.cohort} className="border-b border-gray-100 hover:bg-gray-50">
                        <td className="py-1.5 px-3 font-medium text-gray-900">{c.cohort}</td>
                        <td className="py-1.5 px-3 text-right text-gray-600">{(c.n || 0).toLocaleString()}</td>
                        <td className="py-1.5 px-3 text-right">£{(c.avg_premium || 0).toLocaleString()}</td>
                        <td className="py-1.5 px-3 text-right">£{(c.avg_annual_loss || 0).toLocaleString()}</td>
                        <td className={`py-1.5 px-3 text-right font-semibold ${
                          lr >= maxLR * 0.95 ? 'text-red-700'
                          : lr >= maxLR * 0.7 ? 'text-amber-700'
                          : 'text-gray-900'
                        }`}>{lr.toFixed(2)}</td>
                        <td className="py-1.5 px-3">
                          <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                            <div className={`h-full ${barTone}`} style={{ width: `${pct}%` }} />
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}
      </div>

      <div className="px-5 py-4 border-t border-emerald-100 bg-gradient-to-b from-emerald-50/30 to-white">
        <div className="flex items-center justify-between mb-3 flex-wrap gap-2">
          <div className="text-sm font-medium text-gray-800 flex items-center gap-1.5">
            <Wand2 className="w-4 h-4 text-emerald-600" /> One-click investigation — grounded agent
          </div>
          {investigation?.usage?.total_tokens && (
            <div className="text-[11px] text-gray-500 flex items-center gap-2">
              <span>{investigation.trace?.length || 0} tool calls</span>
              <span>·</span>
              <span>{investigation.usage.total_tokens.toLocaleString()} tokens</span>
              {investMs != null && <><span>·</span><span className="font-mono">{(investMs/1000).toFixed(1)}s wall</span></>}
            </div>
          )}
        </div>

        {!investigating && !investigation && (
          <div className="flex flex-wrap gap-1.5 mb-3">
            {suggestions.map(s => (
              <button key={s}
                      onClick={() => runInvestigate(s)}
                      className="text-[11px] px-2.5 py-1 rounded-full border border-emerald-300 bg-white hover:bg-emerald-50 text-emerald-800">
                {s}
              </button>
            ))}
          </div>
        )}

        <div className="flex gap-2">
          <input
            value={question}
            onChange={e => setQuestion(e.target.value)}
            placeholder="Ask e.g. 'Why is the Midlands cohort under-priced?' (or leave blank for a full scan)"
            className="flex-1 px-3 py-1.5 rounded-md border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
            disabled={investigating}
            onKeyDown={e => { if (e.key === 'Enter') runInvestigate(); }}
          />
          <button onClick={() => runInvestigate()}
                  disabled={investigating}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-emerald-600 text-white text-sm font-medium hover:bg-emerald-700 disabled:opacity-50">
            {investigating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            {investigating ? 'Agent working…' : 'Run investigation'}
          </button>
        </div>

        {investigating && (
          <div className="mt-3 rounded-md border border-emerald-200 bg-white p-3 text-xs text-emerald-800 flex items-center gap-2">
            <Loader2 className="w-3.5 h-3.5 animate-spin" />
            Agent is reading the rating factor catalogue, joining cohort experience, and checking
            recent refits…
          </div>
        )}

        {investigation && (
          <div className="mt-4">
            {investigation.ok === false && investigation.error && (
              <div className="text-sm text-red-600">Agent failed: {investigation.error}</div>
            )}
            {investigation.ok && (
              <>
                {(investigation.trace?.length || 0) > 0 && (
                  <div className="mb-3 rounded-md border border-emerald-100 bg-emerald-50/40 p-3">
                    <div className="text-[11px] uppercase tracking-wider font-semibold text-emerald-800 mb-2 flex items-center gap-1">
                      <Clock className="w-3 h-3" /> Tool trace · executed in order
                    </div>
                    <ol className="space-y-1">
                      {(investigation.trace || []).map((t: any, i: number) => {
                        const errored = String(t.result_summary || '').startsWith('error');
                        return (
                          <li key={i} className="text-xs flex items-start gap-2">
                            <span className="font-mono text-[10px] px-1.5 py-0.5 rounded bg-white border border-emerald-200 text-emerald-700 shrink-0">
                              {String(i + 1).padStart(2, '0')}
                            </span>
                            <div className="min-w-0">
                              <code className={`font-semibold ${errored ? 'text-red-700' : 'text-emerald-900'}`}>{t.tool}</code>
                              {t.result_summary && (
                                <span className="text-gray-600 ml-2">{String(t.result_summary).slice(0, 140)}</span>
                              )}
                            </div>
                          </li>
                        );
                      })}
                    </ol>
                  </div>
                )}
                <div className="rounded-md border border-gray-200 bg-white p-4 text-sm whitespace-pre-wrap leading-relaxed">
                  {investigation.answer}
                </div>
                <div className="mt-2 text-[11px] text-gray-500 italic">
                  Logged to <code>audit_log</code> as a <code>premium_adequacy_investigation</code> event with the full tool trace.
                </div>
              </>
            )}
          </div>
        )}
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// Free-form Governance Agent — main-window chat for open questions
// ---------------------------------------------------------------------------

type AgentChatTurn = { role: 'user' | 'assistant'; text: string; trace?: any[]; usage?: any; model?: string };

function GovernanceAgentChat() {
  const [q, setQ]             = useState('');
  const [turns, setTurns]     = useState<AgentChatTurn[]>([]);
  const [busy, setBusy]       = useState(false);

  const examples = [
    'What governance packs have we generated in the last 30 days?',
    'Which models have not been retrained recently?',
    'Summarise the approvals for fraud_gbm.',
    'What does the fairness section say for the latest freq_glm pack?',
  ];

  const ask = async (question: string) => {
    if (!question.trim() || busy) return;
    const userTurn: AgentChatTurn = { role: 'user', text: question };
    setTurns(t => [...t, userTurn]);
    setQ('');
    setBusy(true);
    try {
      const r = await api.askGovernanceAgent(question);
      setTurns(t => [...t, {
        role: 'assistant',
        text:  r.answer || (r.error ? `[agent error: ${r.error}]` : ''),
        trace: r.trace,
        usage: r.usage,
        model: r.model,
      }]);
    } catch (e: any) {
      setTurns(t => [...t, { role: 'assistant', text: `[request failed: ${e.message || e}]` }]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mb-6 rounded-xl border border-gray-200 bg-white overflow-hidden">
      <div className="px-5 py-4 border-b border-gray-200 bg-gradient-to-r from-amber-50 to-white flex items-center gap-3">
        <Bot className="w-5 h-5 text-amber-700" />
        <div>
          <div className="font-semibold text-gray-900">Ask the governance agent</div>
          <div className="text-xs text-gray-500">
            Free-form Q&A across every pack, audit event, and model artefact.
            Tools: <code>query_pack_index</code> · <code>read_pack_artefact</code> · <code>query_audit_log</code>.
          </div>
        </div>
      </div>

      <div className="px-5 py-4 max-h-[520px] overflow-y-auto space-y-3">
        {turns.length === 0 && !busy && (
          <div>
            <div className="text-xs text-gray-500 mb-2">Try one of these, or ask anything governance-related:</div>
            <div className="flex flex-wrap gap-1.5">
              {examples.map((ex, i) => (
                <button key={i}
                        onClick={() => ask(ex)}
                        className="text-[11px] px-2 py-1 rounded-full border border-amber-300 bg-amber-50 hover:bg-amber-100 text-amber-900">
                  {ex}
                </button>
              ))}
            </div>
          </div>
        )}
        {turns.map((t, i) => (
          t.role === 'user' ? (
            <div key={i} className="flex justify-end">
              <div className="max-w-[80%] rounded-lg bg-blue-600 text-white text-sm px-3 py-2 whitespace-pre-wrap">
                {t.text}
              </div>
            </div>
          ) : (
            <div key={i} className="flex flex-col">
              {t.trace && t.trace.length > 0 && (
                <div className="flex flex-wrap gap-1 mb-1.5">
                  {t.trace.map((step: any, j: number) => (
                    <span key={j}
                          className="text-[10px] px-1.5 py-0.5 rounded border border-amber-200 bg-amber-50 text-amber-800">
                      {step.tool}{String(step.result_summary || '').startsWith('error') ? ' ⚠' : ''}
                    </span>
                  ))}
                </div>
              )}
              <div className="max-w-[92%] rounded-lg bg-gray-50 border border-gray-200 text-sm px-3 py-2 whitespace-pre-wrap">
                {t.text}
              </div>
              {t.usage?.total_tokens && (
                <div className="text-[10px] text-gray-500 mt-0.5">
                  {t.trace?.length || 0} tool calls · {t.usage.total_tokens.toLocaleString()} tokens · {t.model}
                </div>
              )}
            </div>
          )
        ))}
        {busy && (
          <div className="flex items-center gap-2 text-xs text-gray-500 italic">
            <Loader2 className="w-3.5 h-3.5 animate-spin" /> Agent is calling tools…
          </div>
        )}
      </div>

      <div className="px-5 py-3 border-t border-gray-200 bg-gray-50">
        <div className="flex gap-2">
          <input
            value={q}
            onChange={e => setQ(e.target.value)}
            placeholder="Ask anything about packs, audit history, lineage, approvals…"
            className="flex-1 px-3 py-1.5 rounded-md border border-gray-300 text-sm focus:outline-none focus:ring-2 focus:ring-amber-400"
            disabled={busy}
            onKeyDown={e => { if (e.key === 'Enter') ask(q); }}
          />
          <button
            onClick={() => ask(q)}
            disabled={busy || !q.trim()}
            className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md bg-amber-600 text-white text-sm font-medium hover:bg-amber-700 disabled:opacity-50">
            {busy ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
            {busy ? 'Thinking…' : 'Ask'}
          </button>
        </div>
      </div>
    </div>
  );
}


// ---------------------------------------------------------------------------
// What's collected — pure static catalogue.
//
// Reframes the page from "look at our row counts" to "we collect everything
// end-to-end — what you see on the other tabs is a curated set of examples;
// any other slicing is a config change away." No SQL, no live counts.
// ---------------------------------------------------------------------------

type DataInput = {
  key: string; table: string; label: string; purpose: string; grain: string;
  fields: string[];
  examples_shown: string[];
  extensible_to: string;
};
type Surface = {
  key: string; label: string; uses: string[]; summary: string;
  currently_showing: string;
  extensible_to: string;
};
type DataSummary = { inputs: DataInput[]; surfaces: Surface[]; narrative: string };

function DataInfoTab() {
  const [data, setData] = useState<DataSummary | null>(null);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.getGovernanceDataSummary()
      .then(setData)
      .catch((e: any) => setErr(e.message || String(e)))
      .finally(() => setLoading(false));
  }, []);

  if (loading) return (
    <div className="py-12 text-center text-sm text-gray-500">
      <Loader2 className="w-4 h-4 inline animate-spin mr-1" /> Loading…
    </div>
  );
  if (err) return (
    <div className="bg-red-50 border border-red-200 rounded p-4 text-sm text-red-700">{err}</div>
  );
  if (!data) return null;

  const inputByKey = Object.fromEntries(data.inputs.map(i => [i.key, i]));

  return (
    <div className="space-y-6">
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
        <p className="text-sm text-amber-900">
          {data.narrative ||
            "Everything is collected end-to-end in Unity Catalog. What we *show* on the other tabs is a curated set of examples — any other slicing is a config change away."}
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <section className="lg:col-span-7 space-y-3">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">Inputs · what we collect</h3>
            <span className="text-[11px] text-gray-500">7 Delta tables in Unity Catalog</span>
          </div>
          {data.inputs.map(i => (
            <DataInputCard key={i.key} i={i} />
          ))}
        </section>

        <section className="lg:col-span-5 space-y-3">
          <div className="flex items-baseline justify-between">
            <h3 className="text-sm font-semibold text-gray-900 uppercase tracking-wide">Example surfaces</h3>
            <span className="text-[11px] text-gray-500">currently showing → extensible to</span>
          </div>
          {data.surfaces.map(s => (
            <SurfaceCard key={s.key} s={s} inputByKey={inputByKey} />
          ))}
        </section>
      </div>
    </div>
  );
}

function DataInputCard({ i }: { i: DataInput }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-baseline justify-between gap-3 mb-1">
        <div className="flex items-center gap-2">
          <FileText className="w-4 h-4 text-gray-500" />
          <h4 className="font-semibold text-gray-900">{i.label}</h4>
        </div>
        <code className="text-[10px] text-gray-500 font-mono truncate max-w-[280px]">{i.table}</code>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed mb-2">{i.purpose}</p>

      {i.fields?.length > 0 && (
        <div className="mb-2">
          <div className="text-[10px] uppercase tracking-wider font-semibold text-gray-500 mb-1">Fields</div>
          <div className="flex flex-wrap gap-1">
            {i.fields.map((f, k) => (
              <span key={k} className="text-[10px] px-1.5 py-0.5 rounded bg-gray-100 text-gray-700 font-mono">
                {f}
              </span>
            ))}
          </div>
        </div>
      )}

      {i.examples_shown?.length > 0 && (
        <div className="mb-2">
          <div className="text-[10px] uppercase tracking-wider font-semibold text-emerald-700 mb-1">Currently surfaced as</div>
          <ul className="text-xs text-gray-700 space-y-0.5">
            {i.examples_shown.map((ex, k) => (
              <li key={k} className="flex items-start gap-1.5">
                <span className="text-emerald-600 mt-0.5">•</span>
                <span>{ex}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {i.extensible_to && (
        <div className="text-[11px] text-gray-600 bg-amber-50/50 border-l-2 border-amber-300 pl-2 py-1 italic">
          <span className="not-italic font-semibold text-amber-900">Extensible to: </span>
          {i.extensible_to}
        </div>
      )}
    </div>
  );
}

function SurfaceCard({ s, inputByKey }: { s: Surface; inputByKey: Record<string, DataInput> }) {
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-1">
        <ArrowRight className="w-4 h-4 text-amber-600" />
        <h4 className="font-semibold text-gray-900">{s.label}</h4>
      </div>
      <p className="text-xs text-gray-600 leading-relaxed mb-2">{s.summary}</p>

      {s.currently_showing && (
        <div className="mb-2">
          <div className="text-[10px] uppercase tracking-wider font-semibold text-emerald-700 mb-0.5">Currently showing</div>
          <p className="text-xs text-gray-800">{s.currently_showing}</p>
        </div>
      )}
      {s.extensible_to && (
        <div className="text-[11px] text-gray-600 bg-amber-50/50 border-l-2 border-amber-300 pl-2 py-1 italic mb-2">
          <span className="not-italic font-semibold text-amber-900">Extensible to: </span>
          {s.extensible_to}
        </div>
      )}

      <div className="flex flex-wrap gap-1">
        {s.uses.map(u => {
          const i = inputByKey[u];
          return (
            <span key={u}
                  title={i?.table || u}
                  className="text-[10px] px-1.5 py-0.5 rounded bg-amber-50 border border-amber-200 text-amber-900">
              {i?.label || u}
            </span>
          );
        })}
      </div>
    </div>
  );
}
