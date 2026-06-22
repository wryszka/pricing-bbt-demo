import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Database, ChevronRight, CheckCircle2, XCircle, Clock, Building2, Globe } from 'lucide-react';
import { api } from '../lib/api';

export default function DatasetList() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [statsLoading, setStatsLoading] = useState(true);

  useEffect(() => {
    // 1. Fast meta call — returns instantly so the cards draw immediately.
    // 2. Stats call in parallel — each card gets its row counts + approval
    //    state when it resolves a couple of seconds later.
    api.getDatasetsMeta().then(setDatasets).catch(() => {});
    api.getDatasets()
      .then(setDatasets)                      // overwrite with real stats
      .finally(() => setStatsLoading(false));
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6 flex items-baseline gap-3">
        <h2 className="text-2xl font-bold text-gray-900">Data Ingestion</h2>
        {statsLoading && datasets.length > 0 && (
          <span className="text-xs text-gray-500">loading stats…</span>
        )}
      </div>
      <p className="text-gray-500 -mt-5 mb-6">Every dataset that feeds pricing — the internal book (policies, claims), vendor feeds awaiting review, and real public reference data.</p>

      {/* Context panels */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-blue-800 uppercase tracking-wide mb-1">Databricks features demonstrated</h4>
          <div className="flex flex-wrap gap-1.5">
            {["Delta Live Tables expectations", "Unity Catalog governance", "Volumes for file ingestion", "Shadow pricing simulation", "Audit trail logging"].map(f => (
              <span key={f} className="px-2 py-0.5 rounded text-[10px] font-medium bg-blue-100 text-blue-700">{f}</span>
            ))}
          </div>
        </div>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">Why it matters for actuaries</h4>
          <p className="text-xs text-amber-700">
            Replaces manual spreadsheet comparison of data versions. Actuaries see the exact financial
            impact of new data on their portfolio <em>before</em> it enters the rating engine — with
            a single click to approve or reject.
          </p>
        </div>
      </div>

      {/* Skeleton shown only when even the meta hasn't arrived yet. */}
      {datasets.length === 0 && (
        <div className="grid gap-3 mb-6">
          {[0, 1, 2, 3, 4].map(i => (
            <div key={i} className="h-20 rounded-lg bg-gray-100 border border-gray-200 animate-pulse" />
          ))}
        </div>
      )}

      {/* Group by category so the internal book sits above the vendor feeds */}
      {['internal', 'external_vendor', 'reference_data'].map((cat) => {
        const rows = datasets.filter((d) => (d.category || 'external_vendor') === cat);
        if (rows.length === 0) return null;
        const heading =
          cat === 'internal'         ? 'Internal book'
        : cat === 'external_vendor'  ? 'External vendor feeds (review required)'
        :                              'Public reference data';
        const headingHelp =
          cat === 'internal'         ? 'Policy and claim records from our own systems. Shown for completeness — no actuary approval needed.'
        : cat === 'external_vendor'  ? 'Incoming vendor data needs an actuary review before it can feed pricing.'
        :                              'Freely-available reference data (e.g. ONS). One-shot builds.';

        return (
          <section key={cat} className="mb-6">
            <div className="flex items-baseline justify-between mb-2">
              <h3 className="text-sm font-semibold text-gray-700 uppercase tracking-wide">{heading}</h3>
              <span className="text-xs text-gray-400">{headingHelp}</span>
            </div>
            <div className="grid gap-3">
              {rows.map((ds) => (
                <DatasetCard key={ds.id} ds={ds} />
              ))}
            </div>
          </section>
        );
      })}
    </div>
  );
}

// Render an ingestion timestamp in relative form ("2 hours ago") with the ISO
// date as tooltip. Takes the shape Databricks SQL returns — either ISO string
// or epoch-like.
function formatIngested(iso?: string | null): string {
  if (!iso) return '—';
  const t = new Date(iso).getTime();
  if (isNaN(t)) return '—';
  const now = Date.now();
  const diff = Math.max(0, now - t);
  const mins = Math.floor(diff / 60_000);
  if (mins < 1)            return 'Just now';
  if (mins < 60)           return `${mins} min ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)            return `${hrs} hour${hrs === 1 ? '' : 's'} ago`;
  const days = Math.floor(hrs / 24);
  if (days < 30)           return `${days} day${days === 1 ? '' : 's'} ago`;
  return new Date(iso).toLocaleDateString();
}

function DatasetCard({ ds }: { ds: any }) {
  const category = ds.category || 'external_vendor';
  const isInternal = category === 'internal';
  const isReference = category === 'reference_data';

  // Icon + tile colour per category
  const tile =
    isInternal  ? { Icon: Building2, ring: 'bg-slate-100 text-slate-600' }
  : isReference ? { Icon: Globe,     ring: 'bg-indigo-50 text-indigo-600' }
  :               { Icon: Database,  ring: 'bg-blue-50 text-blue-600' };

  return (
    <Link
      to={`/dataset/${ds.id}`}
      className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all group block"
    >
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-4 min-w-0">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 ${tile.ring}`}>
            <tile.Icon className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
              {ds.display_name}
            </h3>
            <p className="text-sm text-gray-500 line-clamp-2">{ds.description}</p>
          </div>
        </div>
        <div className="flex items-center gap-6 shrink-0">
          <div className="text-right min-w-[8rem]">
            <div className="text-xs text-gray-500">Source</div>
            <div className="text-sm font-medium truncate max-w-[12rem]">{ds.source}</div>
          </div>
          {isInternal || isReference ? (
            <div className="text-right min-w-[7rem]">
              <div className="text-xs text-gray-500">Rows</div>
              <div className="text-sm font-medium">
                {ds.silver_row_count == null
                  ? <span className="inline-block w-16 h-3 rounded bg-gray-200 animate-pulse" />
                  : Number(ds.silver_row_count).toLocaleString()}
              </div>
            </div>
          ) : (
            <>
              <div className="text-right min-w-[8rem]">
                <div className="text-xs text-gray-500">Last ingested</div>
                <div className="text-sm font-medium">
                  {ds.last_ingested === undefined
                    ? <span className="inline-block w-24 h-3 rounded bg-gray-200 animate-pulse" />
                    : formatIngested(ds.last_ingested)}
                </div>
              </div>
              <div className="text-right min-w-[9rem]">
                <div className="text-xs text-gray-500">Pending / Approved</div>
                <div className="text-sm font-medium">
                  {ds.raw_row_count == null
                    ? <span className="inline-block w-24 h-3 rounded bg-gray-200 animate-pulse" />
                    : <>{Number(ds.raw_row_count).toLocaleString()} / {Number(ds.silver_row_count).toLocaleString()}</>}
                </div>
              </div>
              <div className="text-right min-w-[5rem]">
                <div className="text-xs text-gray-500">Rows blocked</div>
                <div className={`text-sm font-medium ${ds.rows_dropped_by_dq && ds.rows_dropped_by_dq > 0 ? 'text-amber-600' : 'text-green-600'}`}>
                  {ds.rows_dropped_by_dq == null
                    ? <span className="inline-block w-8 h-3 rounded bg-gray-200 animate-pulse" />
                    : ds.rows_dropped_by_dq}
                </div>
              </div>
            </>
          )}
          <StatusBadge dataset={ds} />
          <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500 shrink-0" />
        </div>
      </div>
    </Link>
  );
}

function StatusBadge({ dataset }: { dataset: any }) {
  const cat = dataset.category || 'external_vendor';
  // Internal and reference datasets skip approval entirely.
  if (cat === 'internal') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-slate-100 text-slate-700 border border-slate-200">
        <Building2 className="w-3.5 h-3.5" /> Internal source
      </span>
    );
  }
  if (cat === 'reference_data') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-indigo-50 text-indigo-700 border border-indigo-200">
        <Globe className="w-3.5 h-3.5" /> Reference
      </span>
    );
  }
  // External vendor feed — real approval workflow
  const status = dataset.approval?.decision || 'pending';
  if (status === 'approved') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">
        <CheckCircle2 className="w-3.5 h-3.5" /> Approved
      </span>
    );
  }
  if (status === 'rejected') {
    return (
      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">
        <XCircle className="w-3.5 h-3.5" /> Rejected
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
      <Clock className="w-3.5 h-3.5" /> Pending review
    </span>
  );
}
