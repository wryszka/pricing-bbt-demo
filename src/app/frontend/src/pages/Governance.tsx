import { useEffect, useState } from 'react';
import { Shield, Clock, Database, Activity, Bot, CheckCircle2, AlertTriangle } from 'lucide-react';
import { api } from '../lib/api';

export default function Governance() {
  const [data, setData] = useState<any>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getGovernanceSummary().then(setData).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-center text-gray-500">Loading governance data...</div>;
  if (!data) return <div className="p-8 text-center text-red-500">Failed to load</div>;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Governance & Audit</h2>
        <p className="text-gray-500 mt-1">End-to-end traceability — from raw data to live pricing</p>
      </div>

      {/* What this demonstrates */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-5 mb-6">
        <h3 className="font-semibold text-blue-800 mb-2">What this demonstrates</h3>
        <p className="text-sm text-blue-600">
          Everything that happened — from data ingestion to model approval to live serving —
          is recorded, versioned, and auditable on one platform. A regulatory auditor can
          reconstruct the exact state of any model, its training data, and the human decisions
          that approved it. Unity Catalog provides lineage automatically; the audit_log adds
          the human governance layer on top.
        </p>
      </div>

      {/* Event summary cards */}
      <div className="grid grid-cols-4 gap-4 mb-6">
        <StatCard icon={Activity} label="Total Audit Events"
          value={data.events_by_type?.reduce((s: number, e: any) => s + Number(e.event_count || 0), 0) || 0} color="blue" />
        <StatCard icon={Shield} label="Event Types"
          value={data.events_by_type?.length || 0} color="purple" />
        <StatCard icon={Database} label="Datasets Tracked"
          value={data.data_quality?.length || 0} color="green" />
        <StatCard icon={Bot} label="AI Agent Calls"
          value={data.agent_usage?.length || 0} color="amber" />
      </div>

      <div className="grid grid-cols-2 gap-6">
        {/* Audit Events by Type */}
        <Section title="Audit Events by Type" icon={Activity}>
          <p className="text-xs text-gray-500 mb-3">
            <strong>Why this matters:</strong> Regulators require evidence that every decision
            in the pricing chain was made by a named individual with a timestamp.
          </p>
          {data.events_by_type?.length > 0 ? (
            <table className="min-w-full text-sm">
              <thead><tr className="bg-gray-50 border-b">
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Event</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Entity</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">Count</th>
                <th className="px-3 py-2 text-right text-xs font-medium text-gray-500">Users</th>
              </tr></thead>
              <tbody>
                {data.events_by_type.map((e: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="px-3 py-2 font-mono text-xs">{e.event_type}</td>
                    <td className="px-3 py-2 text-gray-600 text-xs">{e.entity_type}</td>
                    <td className="px-3 py-2 text-right font-medium">{e.event_count}</td>
                    <td className="px-3 py-2 text-right text-gray-500">{e.unique_users}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="text-gray-400 text-sm">No events yet</p>}
        </Section>

        {/* Data Quality */}
        <Section title="Data Quality — Pipeline Pass Rates" icon={CheckCircle2}>
          <p className="text-xs text-gray-500 mb-3">
            <strong>Why this matters:</strong> DLT expectations enforce data contracts at ingestion.
            Rows that fail are dropped — this table shows how many and for which datasets.
          </p>
          {data.data_quality?.map((dq: any, i: number) => (
            <div key={i} className="mb-3">
              <div className="flex items-center justify-between mb-1">
                <span className="text-sm font-medium">{dq.dataset}</span>
                <span className={`text-xs font-medium ${Number(dq.pass_rate) >= 95 ? 'text-green-600' : 'text-amber-600'}`}>
                  {dq.pass_rate}%
                </span>
              </div>
              <div className="w-full h-2 bg-gray-100 rounded-full overflow-hidden">
                <div className={`h-full rounded-full ${Number(dq.pass_rate) >= 95 ? 'bg-green-500' : 'bg-amber-400'}`}
                     style={{ width: `${dq.pass_rate}%` }} />
              </div>
              <div className="text-[10px] text-gray-400 mt-0.5">
                {dq.raw_rows} raw → {dq.silver_rows} silver ({dq.dropped} dropped)
              </div>
            </div>
          ))}
        </Section>

        {/* Delta Lineage */}
        <Section title="Feature Table Version History" icon={Clock}>
          <p className="text-xs text-gray-500 mb-3">
            <strong>Why this matters:</strong> Delta Time Travel lets you reconstruct the exact
            feature table state used for any historical model or pricing decision.
          </p>
          {data.delta_lineage?.length > 0 ? (
            <table className="min-w-full text-sm">
              <thead><tr className="bg-gray-50 border-b">
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Version</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Operation</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">Timestamp</th>
                <th className="px-3 py-2 text-left text-xs font-medium text-gray-500">User</th>
              </tr></thead>
              <tbody>
                {data.delta_lineage.map((h: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="px-3 py-2 font-mono font-bold text-xs">v{h.version}</td>
                    <td className="px-3 py-2 text-xs">{h.operation}</td>
                    <td className="px-3 py-2 text-xs text-gray-500">{h.timestamp}</td>
                    <td className="px-3 py-2 text-xs text-gray-500">{h.userName?.split('@')[0]}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <p className="text-gray-400 text-sm">No history</p>}
        </Section>

        {/* AI Agent Transparency */}
        <Section title="AI Agent Interactions" icon={Bot}>
          <p className="text-xs text-gray-500 mb-3">
            <strong>Why this matters:</strong> Every LLM call is logged with the full prompt
            and response. Unlike black-box SaaS, a regulator can inspect exactly what the
            AI recommended and verify the human approved or rejected it.
          </p>
          {data.agent_usage?.length > 0 ? (
            data.agent_usage.map((e: any, i: number) => {
              let details: any = {};
              try { details = typeof e.details === 'string' ? JSON.parse(e.details) : e.details; } catch {}
              return (
                <div key={i} className="border rounded-lg p-3 mb-2">
                  <div className="flex items-center justify-between text-xs">
                    <span className="font-medium">{e.user_id?.split('@')[0]}</span>
                    <span className="text-gray-400">{e.timestamp}</span>
                  </div>
                  <div className="text-xs text-gray-500 mt-1">
                    Endpoint: {details.model_endpoint || '—'} |
                    Recommendations: {details.recommendations_count || 0} |
                    Success: {details.llm_success ? '✓' : '✗'}
                  </div>
                </div>
              );
            })
          ) : (
            <p className="text-gray-400 text-sm">No agent interactions — AI assistant is optional</p>
          )}
        </Section>
      </div>

      {/* Recent Activity Timeline */}
      <div className="mt-6">
        <Section title="Recent Activity Timeline" icon={Activity}>
          <p className="text-xs text-gray-500 mb-3">
            Complete chronological log of all governance events across data, models, and serving.
          </p>
          <div className="space-y-2">
            {data.recent_activity?.map((e: any, i: number) => (
              <div key={i} className="flex items-center gap-3 py-1.5 border-b border-gray-100 last:border-0">
                <div className={`w-2 h-2 rounded-full shrink-0 ${
                  e.event_type?.includes('approved') ? 'bg-green-500' :
                  e.event_type?.includes('rejected') ? 'bg-red-500' :
                  e.event_type?.includes('agent') ? 'bg-purple-500' :
                  'bg-blue-500'
                }`} />
                <span className="text-xs text-gray-400 w-36 shrink-0">{e.timestamp?.slice(0, 19)}</span>
                <span className="text-xs font-mono text-gray-700 w-36 shrink-0">{e.event_type}</span>
                <span className="text-xs text-gray-500">{e.entity_type}/{e.entity_id}</span>
                <span className="text-xs text-gray-400 ml-auto">{e.user_id?.split('@')[0]} ({e.source})</span>
              </div>
            ))}
          </div>
        </Section>
      </div>
    </div>
  );
}

function StatCard({ icon: Icon, label, value, color }: { icon: any; label: string; value: number; color: string }) {
  const colorMap: Record<string, string> = {
    blue: 'bg-blue-50 border-blue-200 text-blue-700',
    purple: 'bg-purple-50 border-purple-200 text-purple-700',
    green: 'bg-green-50 border-green-200 text-green-700',
    amber: 'bg-amber-50 border-amber-200 text-amber-700',
  };
  return (
    <div className={`rounded-lg border p-4 ${colorMap[color]}`}>
      <div className="flex items-center gap-2 mb-1">
        <Icon className="w-4 h-4" />
        <span className="text-xs font-medium uppercase tracking-wide opacity-70">{label}</span>
      </div>
      <div className="text-2xl font-bold">{value}</div>
    </div>
  );
}

function Section({ title, icon: Icon, children }: { title: string; icon: any; children: React.ReactNode }) {
  return (
    <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
      <div className="px-4 py-3 bg-gray-50 border-b flex items-center gap-2">
        <Icon className="w-4 h-4 text-gray-600" />
        <h3 className="font-semibold text-gray-800">{title}</h3>
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}
