import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, GitCompare, TrendingUp, ShieldCheck, CheckCircle2, XCircle, Loader2, Download, Upload, History, Bot, ChevronDown, ChevronUp } from 'lucide-react';
import { api } from '../lib/api';

type Tab = 'diff' | 'impact' | 'quality' | 'upload' | 'approval';

export default function DatasetDetail() {
  const { datasetId } = useParams<{ datasetId: string }>();
  const [tab, setTab] = useState<Tab>('diff');
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getDatasets().then(setDatasets).finally(() => setLoading(false));
  }, []);

  const ds = datasets.find((d) => d.id === datasetId);

  if (loading) return <div className="p-8 text-center text-gray-500">Loading...</div>;
  if (!ds) return <div className="p-8 text-center text-red-500">Dataset not found</div>;

  const tabs: { id: Tab; label: string; icon: any }[] = [
    { id: 'diff', label: 'Data Changes', icon: GitCompare },
    { id: 'impact', label: 'Impact Analysis', icon: TrendingUp },
    { id: 'quality', label: 'Data Quality', icon: ShieldCheck },
    { id: 'upload', label: 'Upload / Download', icon: Upload },
    { id: 'approval', label: 'Approve / Reject', icon: CheckCircle2 },
  ];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Link to="/" className="inline-flex items-center gap-1.5 text-sm text-gray-500 hover:text-blue-600 mb-4">
        <ArrowLeft className="w-4 h-4" /> Back to datasets
      </Link>

      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">{ds.display_name}</h2>
        <p className="text-gray-500 mt-1">{ds.description} &middot; Source: {ds.source} &middot; Join key: <code className="bg-gray-100 px-1.5 py-0.5 rounded text-xs">{ds.join_key}</code></p>
      </div>

      {/* Tab bar */}
      <div className="flex gap-1 border-b border-gray-200 mb-6">
        {tabs.map((t) => (
          <button
            key={t.id}
            onClick={() => setTab(t.id)}
            className={`flex items-center gap-1.5 px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              tab === t.id
                ? 'border-blue-600 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700 hover:border-gray-300'
            }`}
          >
            <t.icon className="w-4 h-4" />
            {t.label}
          </button>
        ))}
      </div>

      {tab === 'diff' && <DiffTab datasetId={datasetId!} />}
      {tab === 'impact' && <ImpactTab datasetId={datasetId!} />}
      {tab === 'quality' && <QualityTab datasetId={datasetId!} />}
      {tab === 'upload' && <UploadDownloadTab datasetId={datasetId!} datasetName={ds.display_name} />}
      {tab === 'approval' && <ApprovalTab datasetId={datasetId!} />}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 1: Data Changes (Diff)
// ---------------------------------------------------------------------------

function DiffTab({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getDatasetDiff(datasetId).then(setData).finally(() => setLoading(false));
  }, [datasetId]);

  if (loading) return <Spinner />;
  if (!data) return <ErrorMsg msg="Failed to load diff" />;

  const s = data.summary;

  return (
    <div className="space-y-6">
      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Raw (Pending)" value={Number(s.raw_total).toLocaleString()} color="blue" />
        <MetricCard label="Silver (Current)" value={Number(s.silver_total).toLocaleString()} color="gray" />
        <MetricCard label="New Rows" value={Number(s.new_rows).toLocaleString()} color="green" />
        <MetricCard label="Removed Rows" value={Number(s.removed_rows).toLocaleString()} color="red" />
      </div>

      {/* Changed rows */}
      {data.changed_rows.length > 0 && (
        <Section title={`Changed Records (${data.changed_rows.length} shown)`}>
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50">
                  <th className="px-3 py-2 text-left font-medium text-gray-600">{data.key_column}</th>
                  {data.compare_columns.map((col: string) => (
                    <th key={col} colSpan={2} className="px-3 py-2 text-center font-medium text-gray-600">{col}</th>
                  ))}
                </tr>
                <tr className="bg-gray-50 border-b">
                  <th></th>
                  {data.compare_columns.map((col: string) => (
                    <>
                      <th key={`old_${col}`} className="px-3 py-1 text-center text-xs text-gray-400">Old</th>
                      <th key={`new_${col}`} className="px-3 py-1 text-center text-xs text-blue-500">New</th>
                    </>
                  ))}
                </tr>
              </thead>
              <tbody>
                {data.changed_rows.slice(0, 20).map((row: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-blue-50/30">
                    <td className="px-3 py-2 font-mono text-xs">{row[data.key_column]}</td>
                    {data.compare_columns.map((col: string) => {
                      const oldVal = row[`old_${col}`];
                      const newVal = row[`new_${col}`];
                      const changed = String(oldVal) !== String(newVal);
                      return (
                        <>
                          <td key={`old_${col}_${i}`} className="px-3 py-2 text-center text-gray-500">{formatVal(oldVal)}</td>
                          <td key={`new_${col}_${i}`} className={`px-3 py-2 text-center font-medium ${changed ? 'text-blue-600 bg-blue-50' : ''}`}>
                            {formatVal(newVal)}
                          </td>
                        </>
                      );
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}

      {/* New rows */}
      {data.new_rows.length > 0 && (
        <Section title={`New Records (${data.new_rows.length} shown)`}>
          <SimpleTable rows={data.new_rows} />
        </Section>
      )}

      {/* Removed rows */}
      {data.removed_rows.length > 0 && (
        <Section title={`Removed Records (${data.removed_rows.length} shown)`}>
          <SimpleTable rows={data.removed_rows} />
        </Section>
      )}

      {data.changed_rows.length === 0 && data.new_rows.length === 0 && data.removed_rows.length === 0 && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
          <CheckCircle2 className="w-8 h-8 text-green-500 mx-auto mb-2" />
          <p className="text-green-700 font-medium">No differences detected between raw and silver versions.</p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 2: Impact Analysis
// ---------------------------------------------------------------------------

function ImpactTab({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [explainResult, setExplainResult] = useState<any>(null);
  const [explainLoading, setExplainLoading] = useState(false);
  const [explainExpanded, setExplainExpanded] = useState(false);

  useEffect(() => {
    api.getDatasetImpact(datasetId).then(setData).finally(() => setLoading(false));
  }, [datasetId]);

  const askExplain = async () => {
    setExplainLoading(true);
    try {
      setExplainResult(await api.runExplainability(
        `Why did premiums change for the ${datasetId.replace(/_/g, ' ')} dataset update?`
      ));
    } catch (e: any) { setExplainResult({ success: false, error: e.message }); }
    finally { setExplainLoading(false); }
  };

  if (loading) return <Spinner />;
  if (!data) return <ErrorMsg msg="Failed to load impact analysis" />;

  const dd = data.data_diff || {};
  const pi = data.portfolio_impact || {};
  const rs = data.risk_summary || {};

  return (
    <div className="space-y-8">
      {/* Header + download */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-bold text-gray-900">Shadow Pricing Impact Analysis</h3>
          <p className="text-sm text-gray-500 mt-1">
            Automatic re-rating of every affected policy using the current pricing model to show exact financial impact before approval.
          </p>
        </div>
        <a href={api.downloadImpactReport(datasetId)}
           className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors">
          <Download className="w-4 h-4" /> Download Report
        </a>
      </div>

      {/* ── Section 1: Data Diff Summary ── */}
      <Section title="Data Change Summary">
        <p className="text-sm text-gray-500 mb-4">
          <strong>What this shows:</strong> Statistical comparison between the incoming data (raw/bronze)
          and the current approved version (silver). Highlights magnitude of change in each column.
        </p>
        <div className="grid grid-cols-4 gap-3 mb-4">
          <MetricCard label="Incoming Rows" value={dd.raw_count?.toLocaleString() || '0'} color="blue" />
          <MetricCard label="Current Rows" value={dd.silver_count?.toLocaleString() || '0'} color="gray" />
          <MetricCard label="New Rows" value={dd.new_rows?.toLocaleString() || '0'} color="green" />
          <MetricCard label="Removed Rows" value={dd.removed_rows?.toLocaleString() || '0'} color="red" />
        </div>
        {dd.column_shifts?.length > 0 && (
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Column</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Old Mean</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">New Mean</th>
                  <th className="px-3 py-2 text-right font-medium text-gray-600">Shift</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Direction</th>
                </tr>
              </thead>
              <tbody>
                {dd.column_shifts.map((s: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="px-3 py-2 font-mono text-xs">{s.column}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{s.old_mean}</td>
                    <td className="px-3 py-2 text-right font-medium">{s.new_mean}</td>
                    <td className={`px-3 py-2 text-right font-semibold ${
                      s.severity === 'high' ? 'text-red-600' : s.severity === 'medium' ? 'text-amber-600' : 'text-green-600'
                    }`}>{s.shift_pct > 0 ? '+' : ''}{s.shift_pct}%</td>
                    <td className="px-3 py-2">
                      <div className="w-20 h-3 bg-gray-100 rounded-full overflow-hidden relative">
                        <div className={`absolute top-0 h-full rounded-full ${s.shift_pct >= 0 ? 'bg-red-400 left-1/2' : 'bg-green-400 right-1/2'}`}
                             style={{ width: `${Math.min(50, Math.abs(s.shift_pct) * 2.5)}%` }} />
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Section>

      {/* ── Section 2: Portfolio Impact ── */}
      {pi.affected_policies > 0 ? (
        <>
          <Section title="Portfolio Impact — Affected Policies">
            <p className="text-sm text-gray-500 mb-4">
              <strong>What this shows:</strong> The system joined the new data against the active policy book
              and re-rated every affected policy using the proxy pricing model. This is the "shadow run" —
              showing exact premium impact before any changes go live.
            </p>
            <div className="grid grid-cols-4 gap-3 mb-4">
              <MetricCard label="Affected Policies" value={`${pi.affected_policies?.toLocaleString()} of ${pi.total_policies?.toLocaleString()} (${pi.affected_pct}%)`} color="blue" />
              <MetricCard label="Total Premium Delta" value={`£${formatGwp(pi.premium_delta_total)}`} color={pi.premium_delta_total > 0 ? 'red' : 'green'} />
              <MetricCard label="Avg Delta / Policy" value={`£${pi.premium_delta_avg?.toLocaleString()}`} color="gray" />
              <MetricCard label="Flagged (>10% change)" value={pi.flagged_count?.toLocaleString() || '0'} color={pi.flagged_count > 0 ? 'red' : 'green'} />
            </div>
            <div className="grid grid-cols-3 gap-3">
              <div className="border rounded-lg p-3 text-center bg-red-50 border-red-200">
                <div className="text-2xl font-bold text-red-700">{pi.policies_increase?.toLocaleString()}</div>
                <div className="text-xs text-red-600 mt-1">Premium Increase</div>
              </div>
              <div className="border rounded-lg p-3 text-center bg-green-50 border-green-200">
                <div className="text-2xl font-bold text-green-700">{pi.policies_decrease?.toLocaleString()}</div>
                <div className="text-xs text-green-600 mt-1">Premium Decrease</div>
              </div>
              <div className="border rounded-lg p-3 text-center bg-gray-50 border-gray-200">
                <div className="text-2xl font-bold text-gray-700">{pi.policies_unchanged?.toLocaleString()}</div>
                <div className="text-xs text-gray-600 mt-1">Unchanged</div>
              </div>
            </div>
          </Section>

          {/* Histogram */}
          {pi.histogram?.length > 0 && (
            <Section title="Premium Change Distribution">
              <p className="text-sm text-gray-500 mb-4">
                <strong>What this shows:</strong> How premium changes are distributed across the portfolio.
                A concentration in the tails (&gt;10% or &lt;-10%) indicates significant pricing disruption.
              </p>
              <div className="flex items-end gap-1.5 h-40">
                {pi.histogram.map((h: any, i: number) => {
                  const maxCount = Math.max(...pi.histogram.map((x: any) => x.count || 0));
                  const heightPct = maxCount > 0 ? (h.count / maxCount) * 100 : 0;
                  const isNegative = h.bucket.includes('-');
                  const isExtreme = h.bucket.includes('10');
                  return (
                    <div key={i} className="flex-1 flex flex-col items-center">
                      <div className="text-xs font-medium text-gray-700 mb-1">{h.count}</div>
                      <div className={`w-full rounded-t transition-all ${
                        isExtreme ? (isNegative ? 'bg-green-500' : 'bg-red-500')
                        : isNegative ? 'bg-green-300' : h.bucket === '0%' ? 'bg-gray-300' : 'bg-red-300'
                      }`} style={{ height: `${heightPct}%`, minHeight: h.count > 0 ? '4px' : '0' }} />
                      <div className="text-[10px] text-gray-500 mt-1 text-center leading-tight">{h.bucket}</div>
                    </div>
                  );
                })}
              </div>
            </Section>
          )}

          {/* By Industry */}
          {pi.by_industry?.length > 0 && (
            <Section title="Impact by Class of Business">
              <p className="text-sm text-gray-500 mb-3">
                <strong>What this shows:</strong> How the premium impact breaks down by industry risk tier.
                Identifies which business lines face the most pricing pressure.
              </p>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b">
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Industry Tier</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Policies</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">GWP</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Total Delta</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Impact</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pi.by_industry.map((row: any, i: number) => {
                      const maxDelta = Math.max(...pi.by_industry.map((x: any) => Math.abs(x.total_delta || 0)));
                      const barPct = maxDelta > 0 ? Math.abs(row.total_delta) / maxDelta * 100 : 0;
                      return (
                        <tr key={i} className="border-b hover:bg-gray-50">
                          <td className="px-3 py-2 font-medium">{row.industry}</td>
                          <td className="px-3 py-2 text-right">{row.policies?.toLocaleString()}</td>
                          <td className="px-3 py-2 text-right">£{formatGwp(row.gwp)}</td>
                          <td className={`px-3 py-2 text-right font-semibold ${row.total_delta > 0 ? 'text-red-600' : 'text-green-600'}`}>
                            {row.total_delta > 0 ? '+' : ''}£{formatGwp(row.total_delta)}
                          </td>
                          <td className="px-3 py-2">
                            <div className="w-24 h-3 bg-gray-100 rounded-full overflow-hidden">
                              <div className={`h-full rounded-full ${row.total_delta > 0 ? 'bg-red-400' : 'bg-green-400'}`}
                                   style={{ width: `${barPct}%` }} />
                            </div>
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </Section>
          )}

          {/* By Region */}
          {pi.by_region?.length > 0 && (
            <Section title="Geographic Distribution of Impact">
              <p className="text-sm text-gray-500 mb-3">
                <strong>What this shows:</strong> Regional breakdown of premium changes. Identifies geographic
                concentration risk — if one region is disproportionately affected, it may indicate
                localised risk events (e.g. new flood mapping) that need specific underwriting attention.
              </p>
              <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-2">
                {pi.by_region.map((row: any, i: number) => (
                  <div key={i} className={`border rounded-lg p-3 text-center ${
                    row.total_delta > 0 ? 'bg-red-50 border-red-200' : row.total_delta < 0 ? 'bg-green-50 border-green-200' : 'bg-gray-50 border-gray-200'
                  }`}>
                    <div className="font-bold text-gray-800">{row.region}</div>
                    <div className="text-xs text-gray-500">{row.policies} policies</div>
                    <div className={`text-sm font-semibold mt-1 ${row.total_delta > 0 ? 'text-red-600' : 'text-green-600'}`}>
                      {row.total_delta > 0 ? '+' : ''}£{formatGwp(row.total_delta)}
                    </div>
                  </div>
                ))}
              </div>
            </Section>
          )}

          {/* Flagged Policies */}
          {pi.flagged_policies?.length > 0 && (
            <Section title={`Policies Requiring Attention (${pi.flagged_count} with >10% change)`}>
              <p className="text-sm text-gray-500 mb-3">
                <strong>What this shows:</strong> Individual policies where the premium impact exceeds 10%.
                These require senior underwriter review before the data merge is approved — a large rate
                change could trigger customer churn or regulatory scrutiny.
              </p>
              <div className="overflow-x-auto">
                <table className="min-w-full text-sm">
                  <thead>
                    <tr className="bg-gray-50 border-b">
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Policy ID</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Postcode</th>
                      <th className="px-3 py-2 text-left font-medium text-gray-600">Industry</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Current Premium</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Delta</th>
                      <th className="px-3 py-2 text-right font-medium text-gray-600">Change %</th>
                    </tr>
                  </thead>
                  <tbody>
                    {pi.flagged_policies.slice(0, 15).map((row: any, i: number) => (
                      <tr key={i} className="border-b hover:bg-red-50/30">
                        <td className="px-3 py-2 font-mono text-xs">{row.policy_id}</td>
                        <td className="px-3 py-2">{row.postcode}</td>
                        <td className="px-3 py-2">{row.industry}</td>
                        <td className="px-3 py-2 text-right">£{row.current_premium?.toLocaleString()}</td>
                        <td className={`px-3 py-2 text-right font-semibold ${row.premium_delta > 0 ? 'text-red-600' : 'text-green-600'}`}>
                          {row.premium_delta > 0 ? '+' : ''}£{row.premium_delta?.toLocaleString()}
                        </td>
                        <td className="px-3 py-2 text-right">
                          <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                            Math.abs(row.delta_pct) > 20 ? 'bg-red-100 text-red-700' : 'bg-amber-100 text-amber-700'
                          }`}>{row.delta_pct > 0 ? '+' : ''}{row.delta_pct}%</span>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Section>
          )}
        </>
      ) : (
        <div className="bg-green-50 border border-green-200 rounded-lg p-6 text-center">
          <CheckCircle2 className="w-8 h-8 text-green-500 mx-auto mb-2" />
          <p className="text-green-700 font-medium">No pricing impact detected — the new data matches current values.</p>
        </div>
      )}

      {/* ── Section 3: Risk Summary ── */}
      {rs.score_type && (
        <Section title={`Risk Score Analysis — ${rs.score_type}`}>
          <p className="text-sm text-gray-500 mb-4">
            <strong>What this shows:</strong> How risk scores are shifting between the old and new dataset version.
            Tier migration shows policies moving between risk categories — "Low to High" transitions are the
            most important to flag as they represent newly discovered risk.
          </p>
          {rs.score_shift && (
            <div className="grid grid-cols-5 gap-3 mb-4">
              <MetricCard label={`Old Avg ${rs.score_type}`} value={String(rs.score_shift.old_avg_score ?? '—')} color="gray" />
              <MetricCard label={`New Avg ${rs.score_type}`} value={String(rs.score_shift.new_avg_score ?? '—')} color="blue" />
              <MetricCard label="Worsened" value={String(rs.score_shift.worsened ?? 0)} color="red" />
              <MetricCard label="Improved" value={String(rs.score_shift.improved ?? 0)} color="green" />
              <MetricCard label="Unchanged" value={String(rs.score_shift.unchanged ?? 0)} color="gray" />
            </div>
          )}
          {rs.tier_migration?.length > 0 && (
            <div className="overflow-x-auto">
              <table className="min-w-full text-sm">
                <thead>
                  <tr className="bg-gray-50 border-b">
                    <th className="px-3 py-2 text-left font-medium text-gray-600">From Tier</th>
                    <th className="px-3 py-2 text-left font-medium text-gray-600">To Tier</th>
                    <th className="px-3 py-2 text-right font-medium text-gray-600">Count</th>
                    <th className="px-3 py-2 text-left font-medium text-gray-600">Direction</th>
                  </tr>
                </thead>
                <tbody>
                  {rs.tier_migration.map((row: any, i: number) => {
                    const from = row.old_tier;
                    const to = row.new_tier;
                    const key = Object.keys(row).find(k => k !== 'old_tier' && k !== 'new_tier') || '';
                    const count = row[key] || 0;
                    const worsened = (from === 'Low' && to !== 'Low') || (from === 'Medium' && to === 'High') ||
                                     (from === 'Prime' && to !== 'Prime') || (from === 'Standard' && (to === 'Sub-Standard' || to === 'High Risk'));
                    const improved = (to === 'Low' && from !== 'Low') || (to === 'Medium' && from === 'High') ||
                                     (to === 'Prime' && from !== 'Prime');
                    return (
                      <tr key={i} className={`border-b ${worsened ? 'bg-red-50/40' : improved ? 'bg-green-50/40' : ''}`}>
                        <td className="px-3 py-2 font-medium">{from}</td>
                        <td className="px-3 py-2 font-medium">{to}</td>
                        <td className="px-3 py-2 text-right">{Number(count).toLocaleString()}</td>
                        <td className="px-3 py-2">
                          {from === to ? <span className="text-gray-400 text-xs">No change</span> :
                           worsened ? <span className="text-red-600 text-xs font-medium">Risk increased</span> :
                           improved ? <span className="text-green-600 text-xs font-medium">Risk decreased</span> :
                           <span className="text-amber-600 text-xs">Shifted</span>}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Section>
      )}
      {/* AI Explainability */}
      <div className="bg-purple-50 border border-purple-200 rounded-lg p-5">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <Bot className="w-5 h-5 text-purple-600" />
            <h3 className="font-semibold text-purple-800">Ask AI: Why did this change?</h3>
            <span className="px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-100 text-purple-600 border border-purple-200">OPTIONAL</span>
          </div>
          <button onClick={askExplain} disabled={explainLoading}
            className="px-3 py-1.5 bg-purple-600 text-white rounded-lg text-xs font-medium hover:bg-purple-700 disabled:opacity-50 flex items-center gap-1.5">
            {explainLoading ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Analysing...</> : 'Explain This'}
          </button>
        </div>
        <p className="text-xs text-purple-600 mb-3">
          Uses AI to trace the causal chain from data changes to premium impact. Produces a
          plain-English explanation suitable for regulatory submissions.
        </p>
        {explainResult?.success && explainResult.explanation && (
          <div className="bg-white border border-purple-200 rounded-lg p-4 space-y-3">
            <h4 className="font-semibold text-gray-900">{explainResult.explanation.headline}</h4>
            <p className="text-sm text-gray-700 whitespace-pre-line">{explainResult.explanation.explanation}</p>
            {explainResult.explanation.key_drivers?.length > 0 && (
              <div>
                <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">Key Drivers</h5>
                {explainResult.explanation.key_drivers.map((d: any, i: number) => (
                  <div key={i} className="text-sm text-gray-600 mb-1">
                    <strong>{d.factor}</strong> ({d.contribution}): {d.detail}
                  </div>
                ))}
              </div>
            )}
            {explainResult.explanation.regulatory_statement && (
              <div className="bg-gray-50 border rounded p-3">
                <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">Regulatory Statement</h5>
                <p className="text-sm text-gray-700 italic">{explainResult.explanation.regulatory_statement}</p>
              </div>
            )}
            <button onClick={() => setExplainExpanded(!explainExpanded)} className="text-xs text-purple-500 flex items-center gap-1">
              {explainExpanded ? <ChevronUp className="w-3 h-3" /> : <ChevronDown className="w-3 h-3" />}
              {explainExpanded ? 'Hide' : 'Show'} full LLM interaction
            </button>
            {explainExpanded && (
              <pre className="text-[10px] bg-gray-50 border rounded p-2 max-h-40 overflow-auto whitespace-pre-wrap">
                {explainResult.transparency?.raw_response}
              </pre>
            )}
          </div>
        )}
        {explainResult && !explainResult.success && (
          <p className="text-red-500 text-sm">Agent error: {explainResult.error || 'Unknown'}</p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 3: Data Quality
// ---------------------------------------------------------------------------

function QualityTab({ datasetId }: { datasetId: string }) {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getDatasetQuality(datasetId).then(setData).finally(() => setLoading(false));
  }, [datasetId]);

  if (loading) return <Spinner />;
  if (!data) return <ErrorMsg msg="Failed to load quality metrics" />;

  return (
    <div className="space-y-6">
      {/* Top-line metrics */}
      <div className="grid grid-cols-4 gap-4">
        <MetricCard label="Raw Rows" value={Number(data.raw_row_count).toLocaleString()} color="blue" />
        <MetricCard label="Silver Rows (passed DQ)" value={Number(data.silver_row_count).toLocaleString()} color="green" />
        <MetricCard label="Rows Dropped" value={Number(data.rows_dropped).toLocaleString()} color="red" />
        <MetricCard
          label="DQ Pass Rate"
          value={`${data.dq_pass_rate}%`}
          color={data.dq_pass_rate >= 95 ? 'green' : data.dq_pass_rate >= 85 ? 'amber' : 'red'}
        />
      </div>

      {/* Freshness */}
      <div className={`rounded-lg p-4 border ${data.freshness_status === 'fresh' ? 'bg-green-50 border-green-200' : 'bg-amber-50 border-amber-200'}`}>
        <div className="flex items-center justify-between">
          <div>
            <h4 className="font-semibold text-gray-800">Data Freshness</h4>
            <p className="text-sm text-gray-600">Last ingested: {data.last_ingested || 'Never'}</p>
          </div>
          <span className={`px-3 py-1 rounded-full text-xs font-medium ${
            data.freshness_status === 'fresh' ? 'bg-green-100 text-green-700' : 'bg-amber-100 text-amber-700'
          }`}>
            {data.freshness_status === 'fresh' ? 'Fresh' : 'Stale'}
          </span>
        </div>
      </div>

      {/* DQ Expectations */}
      <Section title="Data Quality Expectations">
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="bg-gray-50 border-b">
                <th className="px-3 py-2 text-left font-medium text-gray-600">Expectation</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">Rule</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">Action</th>
                <th className="px-3 py-2 text-left font-medium text-gray-600">Status</th>
              </tr>
            </thead>
            <tbody>
              {data.expectations?.map((exp: any, i: number) => (
                <tr key={i} className="border-b hover:bg-gray-50">
                  <td className="px-3 py-2 font-mono text-xs">{exp.name}</td>
                  <td className="px-3 py-2 text-gray-700">{exp.rule}</td>
                  <td className="px-3 py-2">
                    <span className="px-2 py-0.5 rounded text-xs bg-red-50 text-red-600 border border-red-200">{exp.action}</span>
                  </td>
                  <td className="px-3 py-2">
                    <span className="px-2 py-0.5 rounded text-xs bg-green-50 text-green-600 border border-green-200">{exp.status}</span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Section>

      {/* Column completeness */}
      <Section title="Column Completeness (% non-null in Silver)">
        <div className="grid grid-cols-3 gap-3">
          {Object.entries(data.completeness || {}).map(([col, pct]: [string, any]) => (
            <div key={col} className="flex items-center justify-between bg-white border rounded-lg px-3 py-2">
              <span className="text-sm font-mono text-gray-700">{col}</span>
              <div className="flex items-center gap-2">
                <div className="w-24 h-2 bg-gray-100 rounded-full overflow-hidden">
                  <div
                    className={`h-full rounded-full ${Number(pct) >= 99 ? 'bg-green-500' : Number(pct) >= 90 ? 'bg-amber-400' : 'bg-red-400'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className={`text-xs font-medium w-12 text-right ${Number(pct) >= 99 ? 'text-green-600' : Number(pct) >= 90 ? 'text-amber-600' : 'text-red-600'}`}>
                  {pct}%
                </span>
              </div>
            </div>
          ))}
        </div>
      </Section>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 4: Approve / Reject
// ---------------------------------------------------------------------------

function ApprovalTab({ datasetId }: { datasetId: string }) {
  const [notes, setNotes] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [result, setResult] = useState<any>(null);
  const [history, setHistory] = useState<any[]>([]);

  useEffect(() => {
    api.getApprovalHistory(datasetId).then(setHistory).catch(() => {});
  }, [datasetId, result]);

  const handleDecision = async (decision: string) => {
    setSubmitting(true);
    try {
      const res = await api.approveDataset(datasetId, decision, notes);
      setResult(res);
      setNotes('');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="space-y-6">
      {result && (
        <div className={`rounded-lg p-4 border ${result.decision === 'approved' ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
          <p className={`font-semibold ${result.decision === 'approved' ? 'text-green-800' : 'text-red-800'}`}>
            {result.message}
          </p>
          <p className="text-sm text-gray-600 mt-1">Reviewer: {result.reviewer}</p>
        </div>
      )}

      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Review Decision</h3>
        <p className="text-sm text-gray-600 mb-4">
          Confirm that this dataset version has been reviewed and is suitable for merging into the Unified Pricing Table,
          or reject it with notes explaining why.
        </p>

        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-1">Reviewer Notes</label>
          <textarea
            value={notes}
            onChange={(e) => setNotes(e.target.value)}
            rows={3}
            placeholder="Optional: add notes about this review decision..."
            className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
          />
        </div>

        <div className="flex gap-3">
          <button
            onClick={() => handleDecision('approved')}
            disabled={submitting}
            className="flex items-center gap-2 px-6 py-2.5 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
          >
            {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <CheckCircle2 className="w-4 h-4" />}
            Approve & Merge
          </button>
          <button
            onClick={() => handleDecision('rejected')}
            disabled={submitting}
            className="flex items-center gap-2 px-6 py-2.5 bg-red-600 text-white rounded-lg font-medium hover:bg-red-700 disabled:opacity-50 transition-colors"
          >
            {submitting ? <Loader2 className="w-4 h-4 animate-spin" /> : <XCircle className="w-4 h-4" />}
            Reject
          </button>
        </div>
      </div>

      {/* Approval history */}
      {history.length > 0 && (
        <Section title="Approval History">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Date</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Decision</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Reviewer</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Notes</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Raw/Silver</th>
                </tr>
              </thead>
              <tbody>
                {history.map((h: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="px-3 py-2 text-gray-600">{h.reviewed_at}</td>
                    <td className="px-3 py-2">
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        h.decision === 'approved' ? 'bg-green-50 text-green-700' : 'bg-red-50 text-red-700'
                      }`}>
                        {h.decision}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-gray-700">{h.reviewer}</td>
                    <td className="px-3 py-2 text-gray-500">{h.reviewer_notes || '—'}</td>
                    <td className="px-3 py-2 text-gray-600">{h.raw_row_count} / {h.silver_row_count}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab 5: Upload / Download
// ---------------------------------------------------------------------------

function UploadDownloadTab({ datasetId, datasetName }: { datasetId: string; datasetName: string }) {
  const [file, setFile] = useState<File | null>(null);
  const [validation, setValidation] = useState<any>(null);
  const [validating, setValidating] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadResult, setUploadResult] = useState<any>(null);
  const [mode, setMode] = useState<'replace' | 'append'>('replace');
  const [uploadHistory, setUploadHistory] = useState<any[]>([]);

  useEffect(() => {
    api.getUploadHistory(datasetId).then(setUploadHistory).catch(() => {});
  }, [datasetId, uploadResult]);

  const handleFileSelect = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setValidation(null);
    setUploadResult(null);
    setValidating(true);
    try {
      const result = await api.validateUpload(datasetId, f);
      setValidation(result);
    } catch (err: any) {
      setValidation({ valid: false, error: err.message });
    } finally {
      setValidating(false);
    }
  };

  const handleConfirm = async () => {
    if (!file) return;
    setUploading(true);
    try {
      const result = await api.confirmUpload(datasetId, file, mode);
      setUploadResult(result);
      setFile(null);
      setValidation(null);
    } catch (err: any) {
      setUploadResult({ error: err.message });
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Download section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Download Current Data</h3>
        <p className="text-sm text-gray-600 mb-4">
          Export the current version of this dataset as CSV for offline review or editing.
          Downloads are logged in the audit trail.
        </p>
        <div className="flex gap-3">
          <a
            href={api.downloadDataset(datasetId, 'silver')}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg text-sm font-medium hover:bg-blue-700 transition-colors"
          >
            <Download className="w-4 h-4" /> Download Silver (Cleansed)
          </a>
          <a
            href={api.downloadDataset(datasetId, 'raw')}
            className="flex items-center gap-2 px-4 py-2 bg-gray-600 text-white rounded-lg text-sm font-medium hover:bg-gray-700 transition-colors"
          >
            <Download className="w-4 h-4" /> Download Raw (Bronze)
          </a>
        </div>
      </div>

      {/* Upload section */}
      <div className="bg-white rounded-lg border border-gray-200 p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-2">Upload New Data</h3>
        <p className="text-sm text-gray-600 mb-4">
          Upload a CSV file to replace or append data in the bronze layer. The file must match
          the expected column schema. After upload, run the ingestion pipeline to promote to silver.
        </p>

        {/* Upload result banner */}
        {uploadResult && !uploadResult.error && (
          <div className="mb-4 bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="font-semibold text-green-800">Upload successful</p>
            <p className="text-sm text-green-600 mt-1">
              {uploadResult.row_count} rows written to {uploadResult.target_table} ({uploadResult.mode} mode).
              File hash: <code className="text-xs">{uploadResult.file_hash?.slice(0, 16)}...</code>
            </p>
          </div>
        )}
        {uploadResult?.error && (
          <div className="mb-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="font-semibold text-red-800">Upload failed: {uploadResult.error}</p>
          </div>
        )}

        {/* Mode selector */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">Upload Mode</label>
          <div className="flex gap-4">
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" checked={mode === 'replace'} onChange={() => setMode('replace')}
                className="text-blue-600" />
              <span className="text-sm"><strong>Replace</strong> — overwrite all existing rows</span>
            </label>
            <label className="flex items-center gap-2 cursor-pointer">
              <input type="radio" checked={mode === 'append'} onChange={() => setMode('append')}
                className="text-blue-600" />
              <span className="text-sm"><strong>Append</strong> — add rows to existing data</span>
            </label>
          </div>
        </div>

        {/* File input */}
        <div className="mb-4">
          <input
            type="file"
            accept=".csv"
            onChange={handleFileSelect}
            className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border-0 file:text-sm file:font-medium file:bg-blue-50 file:text-blue-700 hover:file:bg-blue-100"
          />
        </div>

        {/* Validation loading */}
        {validating && (
          <div className="flex items-center gap-2 text-blue-600 text-sm mb-4">
            <Loader2 className="w-4 h-4 animate-spin" /> Validating schema...
          </div>
        )}

        {/* Validation results */}
        {validation && !validation.error && (
          <div className="space-y-4">
            <div className={`rounded-lg p-4 border ${validation.valid ? 'bg-green-50 border-green-200' : 'bg-red-50 border-red-200'}`}>
              <div className="flex items-center gap-2 mb-2">
                {validation.valid
                  ? <CheckCircle2 className="w-5 h-5 text-green-600" />
                  : <XCircle className="w-5 h-5 text-red-600" />}
                <span className={`font-semibold ${validation.valid ? 'text-green-800' : 'text-red-800'}`}>
                  {validation.valid ? 'Schema validated' : 'Schema mismatch'}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-4 text-sm mt-2">
                <div><span className="text-gray-500">Filename:</span> {validation.filename}</div>
                <div><span className="text-gray-500">Rows:</span> {validation.row_count?.toLocaleString()}</div>
                <div><span className="text-gray-500">Hash:</span> <code className="text-xs">{validation.file_hash?.slice(0, 16)}...</code></div>
              </div>
              {validation.missing_columns?.length > 0 && (
                <div className="mt-2 text-sm text-red-600">
                  Missing columns: <strong>{validation.missing_columns.join(', ')}</strong>
                </div>
              )}
              {validation.extra_columns?.length > 0 && (
                <div className="mt-2 text-sm text-amber-600">
                  Extra columns (will be ignored): {validation.extra_columns.join(', ')}
                </div>
              )}
            </div>

            {/* Preview */}
            {validation.preview?.length > 0 && (
              <Section title={`Preview (first ${validation.preview.length} of ${validation.row_count} rows)`}>
                <SimpleTable rows={validation.preview} />
              </Section>
            )}

            {/* Confirm button */}
            {validation.valid && (
              <button
                onClick={handleConfirm}
                disabled={uploading}
                className="flex items-center gap-2 px-6 py-2.5 bg-green-600 text-white rounded-lg font-medium hover:bg-green-700 disabled:opacity-50 transition-colors"
              >
                {uploading
                  ? <><Loader2 className="w-4 h-4 animate-spin" /> Uploading...</>
                  : <><Upload className="w-4 h-4" /> Confirm Upload ({mode})</>
                }
              </button>
            )}
          </div>
        )}
        {validation?.error && (
          <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
            {validation.error}
          </div>
        )}
      </div>

      {/* Upload history */}
      {uploadHistory.length > 0 && (
        <Section title="Upload History">
          <div className="overflow-x-auto">
            <table className="min-w-full text-sm">
              <thead>
                <tr className="bg-gray-50 border-b">
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Date</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">User</th>
                  <th className="px-3 py-2 text-left font-medium text-gray-600">Details</th>
                </tr>
              </thead>
              <tbody>
                {uploadHistory.map((h: any, i: number) => {
                  let details: any = {};
                  try { details = typeof h.details === 'string' ? JSON.parse(h.details) : h.details; } catch {}
                  return (
                    <tr key={i} className="border-b hover:bg-gray-50">
                      <td className="px-3 py-2 text-gray-600 whitespace-nowrap">{h.timestamp}</td>
                      <td className="px-3 py-2 text-gray-700">{h.user_id}</td>
                      <td className="px-3 py-2 text-gray-500 text-xs">
                        {details.original_filename} — {details.row_count} rows ({details.upload_mode}) — <code>{details.file_hash?.slice(0, 12)}...</code>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Section>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared components
// ---------------------------------------------------------------------------

function MetricCard({ label, value, color }: { label: string; value: string; color: string }) {
  const colorMap: Record<string, string> = {
    blue: 'border-blue-200 bg-blue-50',
    green: 'border-green-200 bg-green-50',
    red: 'border-red-200 bg-red-50',
    amber: 'border-amber-200 bg-amber-50',
    gray: 'border-gray-200 bg-gray-50',
  };
  const textMap: Record<string, string> = {
    blue: 'text-blue-700',
    green: 'text-green-700',
    red: 'text-red-700',
    amber: 'text-amber-700',
    gray: 'text-gray-700',
  };

  return (
    <div className={`rounded-lg border p-4 ${colorMap[color] || colorMap.gray}`}>
      <div className="text-xs font-medium text-gray-500 uppercase tracking-wide">{label}</div>
      <div className={`text-2xl font-bold mt-1 ${textMap[color] || textMap.gray}`}>{value}</div>
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b">
        <h3 className="font-semibold text-gray-800">{title}</h3>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function SimpleTable({ rows }: { rows: any[] }) {
  if (!rows.length) return <p className="text-gray-500 text-sm">No data</p>;
  const cols = Object.keys(rows[0]).filter((c) => !c.startsWith('_'));
  return (
    <div className="overflow-x-auto">
      <table className="min-w-full text-sm">
        <thead>
          <tr className="bg-gray-50 border-b">
            {cols.map((c) => (
              <th key={c} className="px-3 py-2 text-left font-medium text-gray-600 whitespace-nowrap">{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.slice(0, 20).map((row, i) => (
            <tr key={i} className="border-b hover:bg-gray-50">
              {cols.map((c) => (
                <td key={c} className="px-3 py-2 text-gray-700 whitespace-nowrap">{formatVal(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Spinner() {
  return (
    <div className="flex items-center justify-center p-12">
      <Loader2 className="w-8 h-8 animate-spin text-blue-500" />
    </div>
  );
}

function ErrorMsg({ msg }: { msg: string }) {
  return <div className="p-8 text-center text-red-500">{msg}</div>;
}

function formatVal(v: any): string {
  if (v === null || v === undefined) return '—';
  if (typeof v === 'number') return v.toLocaleString();
  return String(v);
}

function formatGwp(v: any): string {
  const num = Number(v);
  if (isNaN(num)) return String(v);
  if (num >= 1_000_000) return `${(num / 1_000_000).toFixed(1)}M`;
  if (num >= 1_000) return `${(num / 1_000).toFixed(0)}K`;
  return num.toLocaleString();
}
