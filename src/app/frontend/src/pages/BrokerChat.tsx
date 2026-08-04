import { useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Car, Lock, Send, Loader2, Wrench, CheckCircle2, AlertTriangle,
  ArrowRight, Gauge, Info, ChevronDown, ChevronUp,
} from 'lucide-react';
import { api } from '../lib/api';

/**
 * Conversational motor quote — the direct/broker chatbot surface.
 *
 * The customer talks; Claude runs the journey; the premium comes from the
 * carrier's pricing engine, never from the model. The right-hand panel makes
 * that visible: every tool call, its latency, and which endpoint answered.
 */

const OPENER =
  "Hello — I can sort your car insurance quote in a couple of minutes. " +
  "To get started, how old are you and what's the car worth roughly?";

type Msg = { role: 'user' | 'assistant'; content: string };

export default function BrokerChat() {
  const [msgs, setMsgs]       = useState<Msg[]>([{ role: 'assistant', content: OPENER }]);
  const [input, setInput]     = useState('');
  const [busy, setBusy]       = useState(false);
  const [history, setHistory] = useState<any[]>([]);
  const [answers, setAnswers] = useState<Record<string, any>>({});
  const [session, setSession] = useState<string | null>(null);
  const [toolLog, setToolLog] = useState<any[]>([]);
  const [quote, setQuote]     = useState<any>(null);
  const [breakdown, setBreakdown] = useState<any>(null);
  const [prov, setProv]       = useState<any>(null);
  const [progress, setProgress] = useState<any>(null);
  const [tools, setTools]     = useState<any[]>([]);
  const [model, setModel]     = useState<string>('');
  const [showWhat, setShowWhat] = useState(false);
  const endRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    api.brokerTools().then(r => { setTools(r.tools || []); setModel(r.model || ''); })
      .catch(() => {});
  }, []);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [msgs, busy]);

  const send = async () => {
    const text = input.trim();
    if (!text || busy) return;
    setInput('');
    setMsgs(m => [...m, { role: 'user', content: text }]);
    setBusy(true);
    try {
      const r = await api.brokerChat({
        message: text, history, answers, session_id: session, breakdown,
      });
      setMsgs(m => [...m, { role: 'assistant', content: r.reply || '…' }]);
      if (r.ok) {
        setHistory(r.history || []);
        setAnswers(r.answers || {});
        setSession(r.session_id || null);
        setProgress(r.progress || null);
        if (r.quote) setQuote(r.quote);
        if (r.breakdown) setBreakdown(r.breakdown);
        if (r.provenance) setProv(r.provenance);
      }
      if (r.tool_log?.length) setToolLog(l => [...l, ...r.tool_log]);
    } catch {
      setMsgs(m => [...m, {
        role: 'assistant',
        content: 'Sorry — the quote assistant is briefly unavailable. Please try again.',
      }]);
    } finally {
      setBusy(false);
    }
  };

  const collected = progress?.collected?.length ?? 0;
  const required  = progress?.required?.length ?? 9;

  return (
    <div className="min-h-screen bg-slate-50 text-slate-900 flex flex-col">
      <header className="bg-white border-b border-slate-200">
        <div className="max-w-6xl mx-auto px-6 h-16 flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-blue-600 flex items-center justify-center">
            <Car className="w-5 h-5 text-white" />
          </div>
          <div className="font-bold text-lg tracking-tight">
            Bricksurance<span className="text-blue-600"> Motor</span>
          </div>
          <span className="text-[11px] px-2 py-0.5 rounded bg-blue-50 text-blue-700 font-medium">
            Quote assistant
          </span>
          <div className="ml-auto flex items-center gap-4 text-xs text-slate-500">
            <Link to="/quote" className="hover:text-slate-800 inline-flex items-center gap-1">
              form journey <ArrowRight className="w-3 h-3" />
            </Link>
            <span className="inline-flex items-center gap-1.5">
              <Lock className="w-3.5 h-3.5" /> Secure
            </span>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl w-full mx-auto px-6 py-6">
        {/* What am I seeing */}
        <div className="mb-4 rounded-lg border border-slate-200 bg-white">
          <button onClick={() => setShowWhat(s => !s)}
                  className="w-full flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-slate-700">
            <Info className="w-4 h-4 text-blue-600" />
            What am I seeing?
            {showWhat ? <ChevronUp className="w-4 h-4 ml-auto text-slate-400" />
                      : <ChevronDown className="w-4 h-4 ml-auto text-slate-400" />}
          </button>
          {showWhat && (
            <div className="px-4 pb-4 text-sm text-slate-600 space-y-2 border-t border-slate-100 pt-3">
              <p>
                A customer buying insurance by conversation instead of filling in a form.
                Claude ({model || 'Foundation Model API'}) runs the journey and decides what to
                ask next — but it <strong>cannot price a risk</strong>. When it has enough
                answers it calls the carrier's pricing engine, the same deployed models and
                the same route-optimized serving endpoint the Live Pricing System uses.
              </p>
              <p>
                The panel on the right shows each tool call and how long the engine took, so
                you can see the premium is computed, not generated. The same tools are
                published over MCP for outside agents — one backend, three channels.
              </p>
            </div>
          )}
        </div>

        <div className="grid lg:grid-cols-[1fr_340px] gap-6 items-start">
          {/* Conversation */}
          <div className="bg-white rounded-lg border border-slate-200 flex flex-col h-[560px]">
            <div className="flex-1 overflow-y-auto p-5 space-y-4">
              {msgs.map((m, i) => (
                <div key={i} className={`flex ${m.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                  <div className={`max-w-[80%] rounded-2xl px-4 py-2.5 text-sm whitespace-pre-wrap ${
                    m.role === 'user'
                      ? 'bg-blue-600 text-white rounded-br-sm'
                      : 'bg-slate-100 text-slate-800 rounded-bl-sm'}`}>
                    {m.content}
                  </div>
                </div>
              ))}
              {busy && (
                <div className="flex justify-start">
                  <div className="bg-slate-100 rounded-2xl rounded-bl-sm px-4 py-2.5">
                    <Loader2 className="w-4 h-4 animate-spin text-slate-400" />
                  </div>
                </div>
              )}
              <div ref={endRef} />
            </div>

            <div className="border-t border-slate-200 p-3 flex gap-2">
              <input
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); } }}
                placeholder="Type your answer…"
                disabled={busy}
                className="flex-1 px-3 py-2 rounded-lg border border-slate-300 text-sm
                           focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-slate-50"
              />
              <button onClick={send} disabled={busy || !input.trim()}
                      className="px-4 py-2 rounded-lg bg-blue-600 text-white text-sm font-medium
                                 hover:bg-blue-700 disabled:opacity-40 inline-flex items-center gap-1.5">
                <Send className="w-4 h-4" /> Send
              </button>
            </div>
          </div>

          {/* Engine panel */}
          <div className="space-y-4">
            {/* Quote */}
            {quote?.annual_premium != null && (
              <div className="rounded-lg border border-emerald-200 bg-emerald-50 p-4">
                <div className="text-[11px] uppercase tracking-wide text-emerald-700 font-semibold mb-1">
                  Priced by the engine
                </div>
                <div className="text-3xl font-bold text-slate-900">
                  £{Number(quote.annual_premium).toLocaleString('en-GB',
                      { minimumFractionDigits: 2, maximumFractionDigits: 2 })}
                </div>
                <div className="text-xs text-slate-600 mt-0.5">
                  per year · or £{Number(quote.monthly_premium).toFixed(2)}/month
                </div>
                <div className="mt-2 pt-2 border-t border-emerald-200 text-[11px] text-emerald-800 flex items-center gap-1.5">
                  <Gauge className="w-3.5 h-3.5" />
                  {quote.engine} · {quote.latency_ms}ms
                </div>
              </div>
            )}

            {/* Progress */}
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">
                Answers collected
              </div>
              <div className="flex items-center gap-2 mb-2">
                <div className="flex-1 h-1.5 rounded-full bg-slate-100 overflow-hidden">
                  <div className="h-full bg-blue-600 transition-all"
                       style={{ width: `${Math.min(100, (collected / required) * 100)}%` }} />
                </div>
                <span className="text-xs font-medium text-slate-600">{collected}/{required}</span>
              </div>
              {Object.keys(answers).length > 0 && (
                <div className="text-[11px] text-slate-500 space-y-0.5 max-h-32 overflow-y-auto">
                  {Object.entries(answers).map(([k, v]) => (
                    <div key={k} className="flex justify-between gap-2">
                      <span className="text-slate-400">{k}</span>
                      <span className="font-medium text-slate-700 truncate">{String(v)}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Provenance — the honest bit */}
            {prov && (
              <div className="rounded-lg border border-amber-200 bg-amber-50 p-4">
                <div className="text-[11px] uppercase tracking-wide text-amber-800 font-semibold mb-2">
                  Where the inputs came from
                </div>
                <div className="text-[11px] text-amber-900 space-y-1.5">
                  <div><strong>{prov.customer_supplied?.length || 0}</strong> from the customer</div>
                  <div><strong>{prov.journey_default?.length || 0}</strong> journey defaults / derived</div>
                  <div><strong>{prov.book_mean_fallback?.length || 0}</strong> at book mean —
                    telematics and behaviour history a new customer cannot have yet.
                    These tighten once real driving data arrives.
                  </div>
                </div>
              </div>
            )}

            {/* Tool calls */}
            <div className="rounded-lg border border-slate-200 bg-white p-4">
              <div className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2 flex items-center gap-1.5">
                <Wrench className="w-3.5 h-3.5" /> Tool calls
              </div>
              {toolLog.length === 0 ? (
                <div className="text-[11px] text-slate-400">
                  None yet — the assistant calls the carrier's services as the
                  conversation progresses.
                </div>
              ) : (
                <div className="space-y-1.5 max-h-48 overflow-y-auto">
                  {toolLog.map((t, i) => (
                    <div key={i} className="flex items-center gap-2 text-[11px]">
                      {t.ok ? <CheckCircle2 className="w-3.5 h-3.5 text-emerald-600 shrink-0" />
                            : <AlertTriangle className="w-3.5 h-3.5 text-amber-600 shrink-0" />}
                      <code className="text-slate-700">{t.tool}</code>
                      {t.latency_ms != null && (
                        <span className="ml-auto text-slate-400">{Math.round(t.latency_ms)}ms</span>
                      )}
                      {!t.ok && t.detail && <span className="ml-auto text-amber-700">{t.detail}</span>}
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Tools available */}
            {tools.length > 0 && (
              <div className="rounded-lg border border-slate-200 bg-white p-4">
                <div className="text-[11px] uppercase tracking-wide text-slate-500 font-semibold mb-2">
                  Services published
                </div>
                <div className="space-y-1">
                  {tools.map((t: any) => (
                    <code key={t.name} className="block text-[11px] text-slate-600">{t.name}</code>
                  ))}
                </div>
                <a href="/add-ons/agentic-distribution"
                   className="mt-3 text-[11px] text-blue-600 hover:underline inline-flex items-center gap-1">
                  same tools over MCP <ArrowRight className="w-3 h-3" />
                </a>
              </div>
            )}
          </div>
        </div>
      </main>

      <footer className="border-t border-slate-200 bg-white py-3">
        <div className="max-w-6xl mx-auto px-6 text-[11px] text-slate-400">
          About this demo — Bricksurance SE is a fictional insurer. The pricing models,
          serving endpoints, governance and data are real Databricks components; the
          portfolio is synthetic. Nothing here reflects a real insurer's rates or book.
        </div>
      </footer>
    </div>
  );
}
