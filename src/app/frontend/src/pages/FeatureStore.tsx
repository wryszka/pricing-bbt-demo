import { useEffect, useMemo, useState } from 'react';
import {
  Database, Zap, ExternalLink, AlertTriangle, Tag,
  BookOpen, Shield, Loader2, PlayCircle, PauseCircle, CheckCircle2, XCircle,
  FileInput, Briefcase, Globe2, ArrowRight,
} from 'lucide-react';
import { api } from '../lib/api';
import GenieChat from '../components/GenieChat';

type Feature = {
  feature_name: string;
  feature_group: string;
  data_type: string;
  description: string;
  source_tables: string[] | string;
  source_columns: string[] | string;
  transformation: string;
  owner: string;
  regulatory_sensitive: boolean | string;
  pii: boolean | string;
};

const GROUP_COLORS: Record<string, string> = {
  rating_factor: 'bg-blue-100 text-blue-700 border-blue-200',
  enrichment:    'bg-indigo-100 text-indigo-700 border-indigo-200',
  claim_derived: 'bg-amber-100 text-amber-700 border-amber-200',
  quote_derived: 'bg-red-100 text-red-700 border-red-200',
  derived:       'bg-purple-100 text-purple-700 border-purple-200',
  key:           'bg-gray-100 text-gray-700 border-gray-200',
  audit:         'bg-gray-100 text-gray-500 border-gray-200',
};

export default function FeatureStore() {
  const [data, setData] = useState<any>(null);
  const [config, setConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  const [catalog, setCatalog] = useState<{ features: Feature[]; counts_by_group: Record<string, number>; total: number; error?: string } | null>(null);
  const [sources, setSources] = useState<any>(null);

  const [promoting, setPromoting] = useState(false);
  const [pausing, setPausing] = useState(false);
  const [lifecycleMsg, setLifecycleMsg] = useState<string | null>(null);
  const [lifecycleTone, setLifecycleTone] = useState<'ok' | 'err'>('ok');

  useEffect(() => {
    Promise.all([
      api.getFeatureStoreStatus(),
      api.getConfig(),
      api.getFeatureCatalog().catch(() => ({ features: [], counts_by_group: {}, total: 0 })),
      api.getFeatureSources().catch(() => null),
    ]).then(([d, c, cat, src]) => {
      setData(d); setConfig(c); setCatalog(cat); setSources(src);
    }).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-center text-gray-500">Loading feature store…</div>;
  if (!data)   return <div className="p-8 text-center text-red-500">Failed to load feature store status</div>;

  const upt = data.upt || {};
  const os = data.online_store || {};
  const tags = upt.tags || {};
  const storeActive = asBool(os.state === 'AVAILABLE' || os.state === 'ACTIVE');

  const refreshStatus = async () => {
    const d = await api.getFeatureStoreStatus();
    setData(d);
  };

  const promote = async () => {
    setPromoting(true); setLifecycleMsg(null);
    try {
      const r = await api.promoteOnline();
      setLifecycleTone('ok');
      setLifecycleMsg(`${r.message || 'Promoted to online serving.'}\n${(r.steps || []).join('\n')}`);
      await refreshStatus();
    } catch (e: any) {
      setLifecycleTone('err');
      setLifecycleMsg(`Promote failed: ${e?.message || e}`);
    } finally {
      setPromoting(false);
    }
  };

  const pause = async () => {
    setPausing(true); setLifecycleMsg(null);
    try {
      const r = await api.pauseOnline();
      setLifecycleTone('ok');
      setLifecycleMsg(r.message || 'Online store paused.');
      await refreshStatus();
    } catch (e: any) {
      setLifecycleTone('err');
      setLifecycleMsg(`Pause failed: ${e?.message || e}`);
    } finally {
      setPausing(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-5">
        <h2 className="text-2xl font-bold text-gray-900">Modelling Mart</h2>
        <p className="text-gray-500 mt-1">
          The modelling dataset — every approved feed joined onto the active book: policies, claims,
          market benchmarks, geospatial hazard, credit bureau, and real UK postcode enrichment.
          <code className="text-xs bg-gray-100 px-1 rounded">policy_id</code> is the <em>grain</em>
          (one row per policy), not the identity. Ask questions in plain English via the AI/BI Genie
          panel below, or browse the factor catalog for provenance.
        </p>
      </div>

      {/* Role-in-the-flow banner */}
      <div className="mb-6 bg-gradient-to-r from-green-50 to-teal-50 border border-green-200 rounded-lg p-5 text-sm text-green-900">
        <div className="flex items-start gap-3">
          <Database className="w-5 h-5 text-green-700 shrink-0 mt-0.5" />
          <div>
            <p>
              <strong>Training:</strong> this offline Delta table feeds every GLM and GBM run on the Model Factory.
              Each row is a labelled example — the feature vector at policy inception plus the observed claim outcomes.
            </p>
            <p className="mt-1.5">
              <strong>Serving:</strong> promote the same feature table to the online store (Lakebase) and models logged with
              FeatureLookup automatically resolve the feature vector by <code className="bg-white px-1 rounded">policy_id</code> in under 10ms. For fresh new-business quotes,
              the front-end sends the feature vector directly — the same shape, no lookup required.
            </p>
          </div>
        </div>
      </div>

      {/* Sources — every upstream that contributes */}
      <SourcesPanel sources={sources} targetLabel="Modelling Mart" />


      {/* Offline + Online status */}
      <div className="grid grid-cols-2 gap-5 mb-6">
        {/* Offline */}
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-5 py-3 bg-gray-50 border-b flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Database className="w-4 h-4 text-blue-600" />
              <h3 className="font-semibold text-gray-800">Offline (Delta Lake)</h3>
            </div>
            <span className="px-2 py-0.5 rounded text-xs font-medium bg-green-50 text-green-700 border border-green-200">Active</span>
          </div>
          <div className="p-5 space-y-3">
            <div className="grid grid-cols-2 gap-3">
              <Stat label="Rows"          value={Number(upt.row_count || 0).toLocaleString()} />
              <Stat label="Columns"       value={String(upt.column_count || 0)} />
              <Stat label="Delta version" value={`v${upt.delta_version}`} />
              <Stat label="Primary key"   value={upt.primary_key || 'policy_id'} />
            </div>
            <div className="text-xs text-gray-500">Last modified: {upt.last_modified || '—'}</div>
            {upt.catalog_url && (
              <a href={upt.catalog_url} target="_blank" rel="noopener noreferrer"
                className="inline-flex items-center gap-1 text-xs text-blue-600 hover:text-blue-800 font-medium">
                <ExternalLink className="w-3 h-3" /> View in Catalog Explorer
              </a>
            )}
          </div>
        </div>

        {/* Online */}
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-5 py-3 bg-gray-50 border-b flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Zap className="w-4 h-4 text-amber-500" />
              <h3 className="font-semibold text-gray-800">Online (Lakebase)</h3>
            </div>
            <StoreStateBadge state={os.state} />
          </div>
          <div className="p-5 space-y-3">
            {storeActive ? (
              <div className="grid grid-cols-2 gap-3">
                <Stat label="Store name" value={os.name} />
                <Stat label="Capacity"   value={os.capacity || '—'} />
              </div>
            ) : (
              <div className="flex items-center gap-2 text-amber-600 text-sm">
                <AlertTriangle className="w-4 h-4" />
                {os.state === 'NOT_CREATED' ? 'Online serving disabled. Click Promote to provision Lakebase.' : `State: ${os.state}`}
              </div>
            )}
            <div className="flex items-center gap-2 pt-2">
              <button onClick={promote} disabled={promoting || pausing}
                className="flex items-center gap-1.5 px-3 py-1.5 bg-emerald-600 text-white rounded text-sm hover:bg-emerald-700 disabled:opacity-50">
                {promoting ? <Loader2 className="w-4 h-4 animate-spin" /> : <PlayCircle className="w-4 h-4" />}
                {storeActive ? 'Re-publish' : 'Promote to online'}
              </button>
              {storeActive && (
                <button onClick={pause} disabled={promoting || pausing}
                  className="flex items-center gap-1.5 px-3 py-1.5 bg-white border border-gray-300 rounded text-sm text-gray-700 hover:bg-gray-50 disabled:opacity-50">
                  {pausing ? <Loader2 className="w-4 h-4 animate-spin" /> : <PauseCircle className="w-4 h-4" />}
                  Pause (drop)
                </button>
              )}
            </div>
            {lifecycleMsg && (
              <div className={`rounded px-3 py-2 text-xs whitespace-pre-line ${
                lifecycleTone === 'ok' ? 'bg-green-50 text-green-700 border border-green-200' :
                                         'bg-red-50 text-red-700 border border-red-200'}`}>
                {lifecycleMsg}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Feature catalog */}
      <FeatureCatalogPanel catalog={catalog} />

      {/* Tags */}
      {Object.keys(tags).length > 0 && (
        <div className="mt-6 bg-white rounded-lg border border-gray-200 overflow-hidden">
          <div className="px-5 py-3 bg-gray-50 border-b flex items-center gap-2">
            <Tag className="w-4 h-4 text-gray-600" />
            <h3 className="font-semibold text-gray-800 text-sm">Feature table tags</h3>
          </div>
          <div className="p-5 flex flex-wrap gap-2">
            {Object.entries(tags).map(([k, v]) => (
              <span key={k} className="px-3 py-1 bg-gray-100 rounded-full text-xs text-gray-700">
                <strong>{k}:</strong> {v as string}
              </span>
            ))}
          </div>
        </div>
      )}

      {/* AI/BI Genie panel — uses the Conversation API, not an iframe, so the
          replies (SQL, charts, tables) land inline in the app. */}
      {config?.genie_space_id && (
        <div className="mt-6">
          <GenieChat
            spaceId={config.genie_space_id}
            fullScreenUrl={config.genie_url}
            variant="card"
            height={560}
            suggestions={[
              "What is the total gross written premium by industry risk tier?",
              "Show average 5-year claim count by construction type",
              "Which 10 postcode sectors generate the most premium?",
              "How many policies are in flood zones 7 and above?",
              "Compare average claim severity — London vs North East",
              "Which SIC codes have the highest 5-year loss ratio?",
            ]}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Feature Catalog — lineage + governance metadata for every UPT feature
// ---------------------------------------------------------------------------

function FeatureCatalogPanel({ catalog }: {
  catalog: { features: Feature[]; counts_by_group: Record<string, number>; total: number; error?: string } | null;
}) {
  const [filter, setFilter] = useState<string>('all');
  const [search, setSearch] = useState<string>('');
  const [selected, setSelected] = useState<Feature | null>(null);

  const features = catalog?.features || [];

  const filtered = useMemo(() => {
    return features.filter(f => {
      if (filter !== 'all' && f.feature_group !== filter) return false;
      if (!search) return true;
      const q = search.toLowerCase();
      return (
        f.feature_name.toLowerCase().includes(q) ||
        (f.description || '').toLowerCase().includes(q) ||
        joinList(f.source_tables).toLowerCase().includes(q)
      );
    });
  }, [features, filter, search]);

  if (!catalog || catalog.error) {
    return (
      <div className="bg-amber-50 border border-amber-200 rounded-lg p-5">
        <div className="flex items-center gap-2 mb-1">
          <BookOpen className="w-5 h-5 text-amber-600" />
          <h3 className="font-semibold text-amber-800">Feature catalog not available</h3>
        </div>
        <p className="text-sm text-amber-700">
          Run <code className="bg-white px-1 rounded">build_feature_catalog</code> (part of the{' '}
          <code className="bg-white px-1 rounded">build_upt</code> bundle job) to populate the{' '}
          <code className="bg-white px-1 rounded">feature_catalog</code> table that drives this panel.
        </p>
        {catalog?.error && <p className="mt-2 text-xs text-amber-600">{catalog.error}</p>}
      </div>
    );
  }

  const groups = Object.keys(catalog.counts_by_group).sort();

  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-5 py-3 bg-gray-50 border-b flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <BookOpen className="w-4 h-4 text-emerald-700" />
          <h3 className="font-semibold text-gray-800 text-sm">Feature catalog</h3>
          <span className="text-xs text-gray-500">{catalog.total} features · one row per UPT column, with provenance</span>
        </div>
        <div className="flex items-center gap-2">
          <input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search feature…"
            className="px-2 py-1 border border-gray-300 rounded text-xs font-mono w-52" />
          <select value={filter} onChange={e => setFilter(e.target.value)}
            className="px-2 py-1 border border-gray-300 rounded text-xs bg-white">
            <option value="all">All groups</option>
            {groups.map(g => (
              <option key={g} value={g}>{g} ({catalog.counts_by_group[g]})</option>
            ))}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-5 divide-x divide-gray-100 text-[10px] font-semibold text-gray-500 uppercase tracking-wide bg-gray-50">
        <div className="px-3 py-1.5">Feature</div>
        <div className="px-3 py-1.5">Group</div>
        <div className="px-3 py-1.5">Source tables</div>
        <div className="px-3 py-1.5">Owner</div>
        <div className="px-3 py-1.5">Flags</div>
      </div>

      <div className="max-h-[480px] overflow-y-auto divide-y text-xs">
        {filtered.map(f => (
          <button key={f.feature_name} onClick={() => setSelected(f)}
            className="w-full text-left grid grid-cols-5 divide-x divide-gray-100 hover:bg-emerald-50 transition-colors">
            <div className="px-3 py-1.5 font-mono font-medium text-gray-900">{f.feature_name}</div>
            <div className="px-3 py-1.5">
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium border ${GROUP_COLORS[f.feature_group] || 'bg-gray-100 text-gray-600 border-gray-200'}`}>
                {f.feature_group}
              </span>
            </div>
            <div className="px-3 py-1.5 text-gray-600 font-mono truncate">{joinList(f.source_tables) || '—'}</div>
            <div className="px-3 py-1.5 text-gray-600">{f.owner || '—'}</div>
            <div className="px-3 py-1.5 flex items-center gap-1">
              {asBool(f.regulatory_sensitive) && <span className="px-1.5 py-0.5 rounded text-[10px] bg-red-100 text-red-700">reg</span>}
              {asBool(f.pii) && <span className="px-1.5 py-0.5 rounded text-[10px] bg-orange-100 text-orange-700">pii</span>}
            </div>
          </button>
        ))}
        {filtered.length === 0 && (
          <div className="px-3 py-4 text-xs text-gray-400 text-center">No features match.</div>
        )}
      </div>

      {selected && <FeatureDetailDrawer feature={selected} onClose={() => setSelected(null)} />}
    </div>
  );
}

function FeatureDetailDrawer({ feature, onClose }: { feature: Feature; onClose: () => void }) {
  return (
    <div className="fixed inset-0 bg-black/30 z-50 flex items-end sm:items-center justify-center p-4" onClick={onClose}>
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[85vh] overflow-y-auto"
        onClick={e => e.stopPropagation()}>
        <div className="px-5 py-3 border-b flex items-center justify-between">
          <div>
            <div className="text-[10px] uppercase tracking-wide text-gray-500">{feature.feature_group} · {feature.data_type}</div>
            <h3 className="font-mono font-semibold text-gray-900">{feature.feature_name}</h3>
          </div>
          <button onClick={onClose} className="p-1 text-gray-400 hover:text-gray-700"><XCircle className="w-5 h-5" /></button>
        </div>
        <div className="p-5 space-y-4 text-sm">
          <div>
            <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">Description</div>
            <p className="text-gray-800">{feature.description || '—'}</p>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">Source tables</div>
              <div className="text-gray-800 font-mono text-xs space-y-0.5">
                {asArr(feature.source_tables).length
                  ? asArr(feature.source_tables).map(t => <div key={t}>{t}</div>)
                  : <span className="text-gray-400">—</span>}
              </div>
            </div>
            <div>
              <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">Source columns</div>
              <div className="text-gray-800 font-mono text-xs space-y-0.5">
                {asArr(feature.source_columns).length
                  ? asArr(feature.source_columns).map(t => <div key={t}>{t}</div>)
                  : <span className="text-gray-400">—</span>}
              </div>
            </div>
          </div>
          <div>
            <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">Transformation</div>
            <code className="block bg-gray-50 px-3 py-2 rounded text-xs text-gray-800">
              {feature.transformation || '—'}
            </code>
          </div>
          <div className="grid grid-cols-3 gap-3 text-xs">
            <div>
              <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">Owner</div>
              <div className="text-gray-800">{feature.owner || '—'}</div>
            </div>
            <div>
              <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">Regulatory</div>
              <div>{asBool(feature.regulatory_sensitive)
                ? <span className="inline-flex items-center gap-1 text-red-700"><Shield className="w-3 h-3" /> sensitive</span>
                : <span className="text-gray-500">not flagged</span>}</div>
            </div>
            <div>
              <div className="text-[11px] font-medium text-gray-500 uppercase tracking-wide mb-1">PII</div>
              <div>{asBool(feature.pii)
                ? <span className="text-orange-700">contains PII</span>
                : <span className="text-gray-500">no PII</span>}</div>
            </div>
          </div>
          <div className="bg-blue-50 border border-blue-200 rounded px-3 py-2 text-xs text-blue-700">
            <CheckCircle2 className="w-4 h-4 inline mr-1" />
            This catalog entry is what lets a regulator (or anyone) trace a feature back to its source.
            Future bolt-ons query this table to answer <em>"if we drop this feature, which models are affected?"</em>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Small components
// ---------------------------------------------------------------------------

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <div className="text-xs text-gray-500">{label}</div>
      <div className="text-sm font-medium text-gray-900 font-mono">{value}</div>
    </div>
  );
}

function StoreStateBadge({ state }: { state?: string }) {
  const s = String(state || 'UNKNOWN');
  const active = s === 'AVAILABLE' || s === 'ACTIVE';
  const provisioning = s === 'PROVISIONING' || s === 'CREATING';
  const cls = active       ? 'bg-green-50 text-green-700 border-green-200'
           : provisioning  ? 'bg-blue-50 text-blue-700 border-blue-200'
           : s === 'NOT_CREATED' ? 'bg-gray-50 text-gray-600 border-gray-200'
                                 : 'bg-amber-50 text-amber-700 border-amber-200';
  return <span className={`px-2 py-0.5 rounded text-xs font-medium border ${cls}`}>{s}</span>;
}

function joinList(v: any): string {
  if (Array.isArray(v)) return v.join(', ');
  if (typeof v === 'string') return v;
  return '';
}
function asArr(v: any): string[] {
  if (Array.isArray(v)) return v;
  if (typeof v === 'string' && v.length) {
    // Backends sometimes serialise arrays as "[a, b]" strings
    const trimmed = v.replace(/^\[|\]$/g, '');
    return trimmed ? trimmed.split(',').map(s => s.trim().replace(/^"|"$/g, '')) : [];
  }
  return [];
}
function asBool(v: any): boolean {
  if (typeof v === 'boolean') return v;
  if (typeof v === 'string')  return v.toLowerCase() === 'true';
  return !!v;
}

// ---------------------------------------------------------------------------
// Sources panel — every upstream that feeds the Pricing Feature Table
// ---------------------------------------------------------------------------

function SourcesPanel({ sources, targetLabel }: { sources: any; targetLabel: string }) {
  if (!sources || !sources.sources) {
    return (
      <div className="mb-6 bg-amber-50 border border-amber-200 rounded-lg p-4 text-xs text-amber-800">
        Sources panel unavailable — feature_catalog or dataset_approvals tables are empty.
      </div>
    );
  }
  const list: any[] = sources.sources || [];
  const ingested   = list.filter(s => s.kind === 'ingested');
  const internal   = list.filter(s => s.kind === 'internal');
  const enrichment = list.filter(s => s.kind === 'enrichment');
  return (
    <div className="mb-6 bg-white border border-gray-200 rounded-lg overflow-hidden">
      <div className="px-5 py-3 bg-gray-50 border-b flex items-center justify-between">
        <div className="flex items-center gap-2">
          <ArrowRight className="w-4 h-4 text-gray-600" />
          <h3 className="font-semibold text-gray-800 text-sm">Sources → {targetLabel}</h3>
        </div>
        <span className="text-[11px] text-gray-500">{list.length} contributing sources</span>
      </div>

      <div className="p-5 space-y-4">
        <SourcesColumn title="External vendor feeds (HITL-approved)" subtitle="Data Ingestion tab" icon={FileInput} items={ingested} tone="blue" showApproval />
        <SourcesColumn title="Internal systems of record"            subtitle="Authoritative transactional tables" icon={Briefcase} items={internal} tone="gray" />
        <SourcesColumn title="Reference / enrichment"                 subtitle="Real UK public data + derived factors" icon={Globe2} items={enrichment} tone="indigo" />

        <div className="flex items-center gap-3 pt-3 border-t border-gray-100 text-xs text-gray-600">
          <div className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-gray-900" />
            Joined + transformed by <code className="bg-gray-100 px-1.5 rounded">build_upt</code> pipeline
          </div>
          <ArrowRight className="w-3 h-3 text-gray-400" />
          <div className="font-medium text-gray-900">{targetLabel}</div>
          <span className="text-gray-500">({sources.target_table})</span>
        </div>

        <p className="text-[11px] text-gray-500 leading-snug">
          {sources.note}
        </p>
      </div>
    </div>
  );
}

function SourcesColumn({ title, subtitle, icon: Icon, items, tone, showApproval }: {
  title: string; subtitle: string; icon: any; items: any[]; tone: 'blue' | 'gray' | 'indigo';
  showApproval?: boolean;
}) {
  if (items.length === 0) return null;
  const toneMap = {
    blue:   'bg-blue-50 border-blue-200 text-blue-700',
    gray:   'bg-gray-50 border-gray-200 text-gray-700',
    indigo: 'bg-indigo-50 border-indigo-200 text-indigo-700',
  } as const;
  return (
    <div>
      <div className="flex items-center gap-2 mb-1.5">
        <Icon className="w-4 h-4 text-gray-500" />
        <span className="text-[11px] font-semibold text-gray-600 uppercase tracking-wide">{title}</span>
        <span className="text-[11px] text-gray-400">· {subtitle}</span>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {items.map(s => {
          const appr = s.approval || {};
          const approved = showApproval && String(appr.decision || '').toLowerCase() === 'approved';
          const pending  = showApproval && !appr.decision;
          const rejected = showApproval && String(appr.decision || '').toLowerCase() === 'rejected';
          return (
            <div key={s.id} className={`rounded-lg border p-3 ${toneMap[tone]}`}>
              <div className="flex items-center justify-between gap-2 mb-1">
                <div className="text-xs font-semibold text-gray-900 truncate">{s.title}</div>
                {showApproval && (
                  approved
                    ? <span className="inline-flex items-center gap-0.5 text-[10px] text-green-700"><CheckCircle2 className="w-3 h-3" /> approved</span>
                    : rejected
                      ? <span className="inline-flex items-center gap-0.5 text-[10px] text-red-700"><XCircle className="w-3 h-3" /> rejected</span>
                      : pending
                        ? <span className="inline-flex items-center gap-0.5 text-[10px] text-amber-700"><AlertTriangle className="w-3 h-3" /> pending</span>
                        : null
                )}
              </div>
              <div className="text-[10px] text-gray-500 font-mono truncate">{s.table}</div>
              <div className="text-[10px] text-gray-500 mt-0.5">
                {s.row_count != null ? `${s.row_count.toLocaleString()} rows` : 'row count unknown'}
              </div>
              {s.features_feed && s.features_feed.length > 0 && (
                <div className="mt-1.5 flex flex-wrap gap-1">
                  {s.features_feed.slice(0, 3).map((f: string) => (
                    <span key={f} className="px-1.5 py-0.5 text-[9px] bg-white/80 border border-gray-200 rounded text-gray-700 font-mono">{f}</span>
                  ))}
                  {s.features_feed.length > 3 && (
                    <span className="px-1.5 py-0.5 text-[9px] text-gray-500">+{s.features_feed.length - 3} more</span>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
