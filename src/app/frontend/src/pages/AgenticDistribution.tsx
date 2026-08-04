import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Network, MessageSquare, Plug, Bot, ArrowRight, Info, ChevronDown, ChevronUp,
  RefreshCw, Terminal, Copy, Check,
} from 'lucide-react';
import { api } from '../lib/api';

/**
 * Agentic Distribution — the carrier's presence in the agent channel.
 *
 * Three surfaces, one backend: the conversational journey, the MCP server for
 * outside agents, and this telemetry view showing who called what and what
 * converted. Every number here comes from `mcp_tool_calls`.
 */
export default function AgenticDistribution() {
  const [tel, setTel]           = useState<any>(null);
  const [manifest, setManifest] = useState<any>(null);
  const [loading, setLoading]   = useState(true);
  const [showWhat, setShowWhat] = useState(false);
  const [copied, setCopied]     = useState(false);

  const load = async () => {
    setLoading(true);
    try {
      const [t, m] = await Promise.all([
        api.distributionTelemetry(24).catch(() => null),
        api.mcpManifest().catch(() => null),
      ]);
      setTel(t); setManifest(m);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const mcpUrl = `${window.location.origin}/api/mcp`;
  const copy = () => {
    navigator.clipboard.writeText(mcpUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  const f = tel?.funnel || {};
  const p = tel?.premiums || {};

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
          <Network className="w-6 h-6 text-violet-600" /> Agentic Distribution
        </h2>
        <p className="text-gray-500 mt-1">
          Being present and priceable when the customer arrives through an AI agent
          rather than a website. Three channels, one pricing engine.
        </p>
      </div>

      {/* What am I seeing */}
      <div className="mb-6 rounded-lg border border-gray-200 bg-white">
        <button onClick={() => setShowWhat(s => !s)}
                className="w-full flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-gray-700">
          <Info className="w-4 h-4 text-violet-600" />
          What am I seeing?
          {showWhat ? <ChevronUp className="w-4 h-4 ml-auto text-gray-400" />
                    : <ChevronDown className="w-4 h-4 ml-auto text-gray-400" />}
        </button>
        {showWhat && (
          <div className="px-4 pb-4 text-sm text-gray-600 space-y-2 border-t border-gray-100 pt-3">
            <p>
              When a customer asks an AI agent to sort their insurance, the carrier that
              can be <em>called</em> wins the risk. An aggregator feed publishes raw data;
              this publishes <strong>services</strong> — discover what we need, get a real
              price, ask why it is that price, read the terms.
            </p>
            <p>
              The distinction matters commercially: an aggregator cannot explain a premium.
              A carrier can, because the explanation comes from the models that set it.
            </p>
            <p>
              Every panel below reads <code>mcp_tool_calls</code> — a real Delta table
              written on every call from either channel. Empty numbers before the first
              run are expected.
            </p>
          </div>
        )}
      </div>

      {/* The three channels */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-5 mb-8">
        <a href="/quote-chat" target="_blank" rel="noreferrer"
           className="group rounded-lg border border-blue-200 bg-blue-50 p-5 hover:shadow-md transition-all block">
          <div className="flex items-center gap-3 mb-2">
            <MessageSquare className="w-5 h-5 text-blue-600" />
            <h3 className="font-semibold text-gray-900 group-hover:text-blue-700">
              Direct &amp; broker chat
            </h3>
            <ArrowRight className="w-4 h-4 text-gray-400 ml-auto group-hover:translate-x-1 transition-transform" />
          </div>
          <p className="text-sm text-gray-700 mb-2">
            A customer buys by conversation. Claude runs the journey; the premium comes
            from the pricing engine, never from the model.
          </p>
          <div className="text-[11px] text-gray-500 italic">Live — opens the customer journey</div>
        </a>

        <div className="rounded-lg border border-violet-200 bg-violet-50 p-5">
          <div className="flex items-center gap-3 mb-2">
            <Plug className="w-5 h-5 text-violet-600" />
            <h3 className="font-semibold text-gray-900">Embedded &amp; external agents</h3>
          </div>
          <p className="text-sm text-gray-700 mb-2">
            The same tools over MCP. A partner's assistant — or any general agent that has
            never seen an insurance form — discovers what we need and gets a real price.
          </p>
          <div className="text-[11px] text-gray-500 italic">
            {manifest?.tools?.length ?? 0} tools published
          </div>
        </div>

        <div className="rounded-lg border border-gray-200 bg-white p-5">
          <div className="flex items-center gap-3 mb-2">
            <Bot className="w-5 h-5 text-gray-600" />
            <h3 className="font-semibold text-gray-900">Channel telemetry</h3>
          </div>
          <p className="text-sm text-gray-700 mb-2">
            Which agents call us, where journeys stall, what converts — the levers a
            distribution team needs to tune presence in the agent channel.
          </p>
          <div className="text-[11px] text-gray-500 italic">Below on this page</div>
        </div>
      </div>

      {/* Connect an agent */}
      <div className="mb-8 rounded-lg border border-gray-200 bg-white p-5">
        <h3 className="font-semibold text-gray-900 mb-1 flex items-center gap-2">
          <Terminal className="w-4 h-4 text-gray-600" /> Connect an outside agent
        </h3>
        <p className="text-sm text-gray-600 mb-3">
          Point any MCP client at this URL. It will discover the tools, learn the question
          set, and be able to price a risk without any insurance knowledge baked in.
        </p>
        <div className="flex items-center gap-2 mb-4">
          <code className="flex-1 px-3 py-2 rounded bg-gray-50 border border-gray-200 text-xs text-gray-800 overflow-x-auto">
            {mcpUrl}
          </code>
          <button onClick={copy}
                  className="px-3 py-2 rounded border border-gray-300 text-xs font-medium
                             text-gray-700 hover:bg-gray-50 inline-flex items-center gap-1.5">
            {copied ? <><Check className="w-3.5 h-3.5 text-emerald-600" /> Copied</>
                    : <><Copy className="w-3.5 h-3.5" /> Copy</>}
          </button>
        </div>
        {manifest?.tools && (
          <div className="grid sm:grid-cols-2 gap-2">
            {manifest.tools.map((t: any) => (
              <div key={t.name} className="rounded border border-gray-100 bg-gray-50 p-2.5">
                <code className="text-[11px] font-semibold text-violet-700">{t.name}</code>
                <p className="text-[11px] text-gray-600 mt-0.5 line-clamp-3">{t.description}</p>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Telemetry */}
      <div className="flex items-center justify-between mb-3">
        <h3 className="font-semibold text-gray-900">Last 24 hours</h3>
        <button onClick={load} disabled={loading}
                className="text-xs text-gray-500 hover:text-gray-800 inline-flex items-center gap-1.5 disabled:opacity-50">
          <RefreshCw className={`w-3.5 h-3.5 ${loading ? 'animate-spin' : ''}`} /> Refresh
        </button>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
        <Stat label="Sessions"      value={f.sessions} />
        <Stat label="Priced"        value={f.priced} accent="emerald" />
        <Stat label="Asked why"     value={f.explained} accent="violet" />
        <Stat label="Avg premium"   value={p.avg_premium != null ? `£${Number(p.avg_premium).toLocaleString('en-GB')}` : null} />
        <Stat label="Avg answers given" value={p.avg_fields_supplied} />
      </div>

      <div className="grid lg:grid-cols-2 gap-6 mb-6">
        {/* By tool */}
        <Panel title="Calls by service">
          {tel?.by_tool?.length ? (
            <table className="w-full text-sm">
              <thead>
                <tr className="text-[11px] uppercase tracking-wide text-gray-400 border-b border-gray-100">
                  <th className="text-left py-1.5 font-medium">Tool</th>
                  <th className="text-right py-1.5 font-medium">Calls</th>
                  <th className="text-right py-1.5 font-medium">OK</th>
                  <th className="text-right py-1.5 font-medium">p50</th>
                </tr>
              </thead>
              <tbody>
                {tel.by_tool.map((r: any) => (
                  <tr key={r.tool} className="border-b border-gray-50 last:border-0">
                    <td className="py-1.5"><code className="text-[11px] text-gray-700">{r.tool}</code></td>
                    <td className="py-1.5 text-right text-gray-800">{r.calls}</td>
                    <td className="py-1.5 text-right text-gray-500">{r.ok_calls}</td>
                    <td className="py-1.5 text-right text-gray-500">
                      {r.p50_latency_ms != null ? `${Math.round(r.p50_latency_ms)}ms` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : <Empty />}
        </Panel>

        {/* Agents */}
        <Panel title="Agents calling us">
          {tel?.agents?.length ? (
            <div className="space-y-1.5">
              {tel.agents.map((a: any) => (
                <div key={a.agent_id} className="flex items-center gap-2 text-sm">
                  <Bot className="w-3.5 h-3.5 text-gray-400 shrink-0" />
                  <span className="truncate text-gray-700 text-[12px]">{a.agent_id}</span>
                  <span className="ml-auto text-[11px] text-gray-400 shrink-0">
                    {a.sessions} sessions · {a.calls} calls
                  </span>
                </div>
              ))}
            </div>
          ) : <Empty />}
        </Panel>
      </div>

      {/* Recent */}
      <Panel title="Recent calls">
        {tel?.recent?.length ? (
          <div className="overflow-x-auto max-h-72 overflow-y-auto">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-white">
                <tr className="text-[11px] uppercase tracking-wide text-gray-400 border-b border-gray-100">
                  <th className="text-left py-1.5 font-medium">When</th>
                  <th className="text-left py-1.5 font-medium">Channel</th>
                  <th className="text-left py-1.5 font-medium">Tool</th>
                  <th className="text-right py-1.5 font-medium">Latency</th>
                  <th className="text-right py-1.5 font-medium">Premium</th>
                </tr>
              </thead>
              <tbody>
                {tel.recent.map((r: any, i: number) => (
                  <tr key={i} className="border-b border-gray-50 last:border-0">
                    <td className="py-1.5 text-[11px] text-gray-500">
                      {String(r.ts).slice(11, 19)}
                    </td>
                    <td className="py-1.5 text-[11px] text-gray-600">{r.surface}</td>
                    <td className="py-1.5">
                      <code className={`text-[11px] ${r.ok ? 'text-gray-700' : 'text-amber-700'}`}>
                        {r.tool}
                      </code>
                    </td>
                    <td className="py-1.5 text-right text-[11px] text-gray-500">
                      {r.latency_ms != null ? `${Math.round(r.latency_ms)}ms` : '—'}
                    </td>
                    <td className="py-1.5 text-right text-[11px] text-gray-700">
                      {r.annual_premium != null
                        ? `£${Number(r.annual_premium).toLocaleString('en-GB')}` : '—'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <Empty />}
      </Panel>
    </div>
  );
}

function Stat({ label, value, accent }: { label: string; value: any; accent?: 'emerald' | 'violet' }) {
  const color = accent === 'emerald' ? 'text-emerald-700'
              : accent === 'violet'  ? 'text-violet-700' : 'text-gray-900';
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-4">
      <div className="text-[11px] uppercase tracking-wide text-gray-400 font-medium mb-1">{label}</div>
      <div className={`text-2xl font-bold ${color}`}>
        {value === null || value === undefined ? '—' : value}
      </div>
    </div>
  );
}

function Panel({ title, children }: { title: string; children: any }) {
  return (
    <div className="rounded-lg border border-gray-200 bg-white p-5">
      <h4 className="text-sm font-semibold text-gray-900 mb-3">{title}</h4>
      {children}
    </div>
  );
}

function Empty() {
  return (
    <div className="text-xs text-gray-400 py-4 text-center">
      No calls yet. Run the chat journey or point an MCP client at the URL above.
    </div>
  );
}
