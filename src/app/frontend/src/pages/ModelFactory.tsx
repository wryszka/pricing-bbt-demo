import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FlaskConical, ChevronRight, CheckCircle2, Clock, AlertTriangle, Bot, Loader2, ChevronDown, ChevronUp, Shield } from 'lucide-react';
import { api } from '../lib/api';

export default function ModelFactory() {
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  // AI Agent state (optional)
  const [agentEnabled, setAgentEnabled] = useState(false);
  const [agentStatus, setAgentStatus] = useState<any>(null);
  const [agentResult, setAgentResult] = useState<any>(null);
  const [agentLoading, setAgentLoading] = useState(false);
  const [showTransparency, setShowTransparency] = useState(false);

  useEffect(() => {
    api.getFactoryRuns().then(setRuns).finally(() => setLoading(false));
    api.getAgentStatus().then(setAgentStatus).catch(() => {});
  }, []);

  const runAgent = async () => {
    setAgentLoading(true);
    try {
      const result = await api.runAgentAnalysis();
      setAgentResult(result);
    } catch (err: any) {
      setAgentResult({ success: false, error: err.message });
    } finally {
      setAgentLoading(false);
    }
  };

  if (loading) return <div className="p-8 text-center text-gray-500">Loading factory runs...</div>;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Model Factory</h2>
          <p className="text-gray-500 mt-1">Review model factory runs and approve models for production</p>
        </div>
      </div>

      {/* Context panels */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-purple-800 uppercase tracking-wide mb-1">Databricks features demonstrated</h4>
          <div className="flex flex-wrap gap-1.5">
            {["MLflow experiment tracking", "UC model registry", "Automated model evaluation", "Regulatory suitability scoring", "PDF model reports", "Foundation Model API (optional)"].map(f => (
              <span key={f} className="px-2 py-0.5 rounded text-[10px] font-medium bg-purple-100 text-purple-700">{f}</span>
            ))}
          </div>
        </div>
        <div className="bg-amber-50 border border-amber-200 rounded-lg p-4">
          <h4 className="text-xs font-semibold text-amber-800 uppercase tracking-wide mb-1">Why it matters for actuaries</h4>
          <p className="text-xs text-amber-700">
            Replaces the manual model comparison process. The factory trains 20+ model configurations,
            ranks them on insurance-specific metrics (Gini, PSI, regulatory suitability), and presents
            a leaderboard for actuarial sign-off — with a one-click PDF report for regulators.
          </p>
        </div>
      </div>

      {/* ── AI Assistant Toggle (OPTIONAL) ── */}
      <div className="mb-6 bg-white rounded-lg border border-gray-200 p-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <Bot className={`w-5 h-5 ${agentEnabled ? 'text-purple-600' : 'text-gray-400'}`} />
            <div>
              <span className="text-sm font-medium text-gray-800">AI Model Selection Assistant</span>
              <span className="ml-2 px-1.5 py-0.5 rounded text-[10px] font-medium bg-purple-50 text-purple-600 border border-purple-200">
                OPTIONAL
              </span>
            </div>
          </div>
          <label className="relative inline-flex items-center cursor-pointer">
            <input type="checkbox" checked={agentEnabled} onChange={(e) => setAgentEnabled(e.target.checked)}
              className="sr-only peer" />
            <div className="w-9 h-5 bg-gray-200 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full rtl:peer-checked:after:-translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:start-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-purple-600"></div>
          </label>
        </div>
        {agentEnabled && (
          <div className="mt-3 text-xs text-gray-500 border-t pt-3">
            Uses Foundation Model API ({agentStatus?.endpoint || 'loading...'}) to analyse the Unified Pricing
            Table and recommend which models to train. The AI recommends — you decide.
            {agentStatus?.available === false && (
              <span className="block mt-1 text-amber-600">Agent unavailable: {agentStatus?.message}</span>
            )}
          </div>
        )}
      </div>

      {/* ── AI Analysis Panel ── */}
      {agentEnabled && (
        <div className="mb-6 space-y-4">
          {!agentResult && (
            <div className="bg-purple-50 border border-purple-200 rounded-lg p-6 text-center">
              <Bot className="w-10 h-10 text-purple-400 mx-auto mb-3" />
              <p className="text-purple-700 font-medium mb-3">
                Run the AI assistant to analyse your feature table and recommend models
              </p>
              <button onClick={runAgent} disabled={agentLoading || !agentStatus?.available}
                className="px-5 py-2 bg-purple-600 text-white rounded-lg text-sm font-medium hover:bg-purple-700 disabled:opacity-50 transition-colors inline-flex items-center gap-2">
                {agentLoading ? <><Loader2 className="w-4 h-4 animate-spin" /> Analysing...</>
                 : <><Bot className="w-4 h-4" /> Run Analysis</>}
              </button>
            </div>
          )}

          {agentResult && agentResult.success && (
            <>
              {/* Disclaimer */}
              <div className="bg-amber-50 border border-amber-200 rounded-lg px-4 py-2 flex items-center gap-2">
                <Shield className="w-4 h-4 text-amber-600 shrink-0" />
                <span className="text-xs text-amber-700">
                  AI-assisted recommendation — review before approving. All LLM interactions are logged and auditable.
                </span>
              </div>

              {/* Overall strategy */}
              <div className="bg-white rounded-lg border border-gray-200 p-5">
                <h3 className="font-semibold text-gray-900 mb-2 flex items-center gap-2">
                  <Bot className="w-4 h-4 text-purple-600" /> Recommended Strategy
                </h3>
                <p className="text-sm text-gray-700">{agentResult.recommendations?.overall_strategy}</p>

                {/* Data quality observations */}
                {agentResult.recommendations?.data_quality_observations?.length > 0 && (
                  <div className="mt-3 border-t pt-3">
                    <h4 className="text-xs font-semibold text-gray-500 uppercase mb-1">Data Quality Observations</h4>
                    <ul className="text-sm text-gray-600 space-y-1">
                      {agentResult.recommendations.data_quality_observations.map((obs: string, i: number) => (
                        <li key={i} className="flex items-start gap-1.5">
                          <span className="text-amber-500 mt-1">-</span> {obs}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>

              {/* Individual recommendations */}
              <div className="grid gap-3">
                {agentResult.recommendations?.recommendations?.map((rec: any, i: number) => (
                  <div key={i} className="bg-white rounded-lg border border-gray-200 p-4">
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                          rec.priority === 'high' ? 'bg-red-50 text-red-700 border border-red-200' :
                          rec.priority === 'medium' ? 'bg-amber-50 text-amber-700 border border-amber-200' :
                          'bg-blue-50 text-blue-700 border border-blue-200'
                        }`}>{rec.priority}</span>
                        <h4 className="font-semibold text-gray-900">{rec.model_name}</h4>
                      </div>
                      <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                        rec.model_type.startsWith('GLM') ? 'bg-blue-100 text-blue-700' : 'bg-green-100 text-green-700'
                      }`}>{rec.model_type}</span>
                    </div>
                    <p className="text-sm text-gray-600 mb-2">{rec.purpose}</p>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div>
                        <span className="text-gray-500 font-medium">Target:</span>{' '}
                        <code className="bg-gray-100 px-1 py-0.5 rounded">{rec.target_variable}</code>
                      </div>
                      <div>
                        <span className="text-gray-500 font-medium">Features:</span>{' '}
                        {rec.recommended_features?.length || 0} selected
                      </div>
                    </div>
                    <p className="text-xs text-gray-500 mt-2">{rec.feature_rationale}</p>
                    {rec.regulatory_notes && (
                      <p className="text-xs text-amber-600 mt-1 flex items-center gap-1">
                        <Shield className="w-3 h-3" /> {rec.regulatory_notes}
                      </p>
                    )}
                  </div>
                ))}
              </div>

              {/* Transparency panel */}
              <div className="bg-gray-50 rounded-lg border border-gray-200">
                <button onClick={() => setShowTransparency(!showTransparency)}
                  className="w-full px-4 py-3 text-left flex items-center justify-between text-sm font-medium text-gray-700 hover:bg-gray-100 transition-colors">
                  <span className="flex items-center gap-2">
                    <Shield className="w-4 h-4 text-gray-500" />
                    Transparency — Full LLM Interaction
                    {agentResult.token_usage?.total_tokens && (
                      <span className="text-xs text-gray-400">({agentResult.token_usage.total_tokens} tokens)</span>
                    )}
                  </span>
                  {showTransparency ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
                </button>
                {showTransparency && (
                  <div className="px-4 pb-4 space-y-3 border-t">
                    <div className="mt-3">
                      <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">Model Endpoint</h5>
                      <code className="text-xs bg-white border rounded px-2 py-1">{agentResult.endpoint}</code>
                    </div>
                    <div>
                      <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">System Prompt</h5>
                      <pre className="text-xs bg-white border rounded p-2 overflow-x-auto max-h-32 whitespace-pre-wrap">
                        {agentResult.transparency?.system_prompt}
                      </pre>
                    </div>
                    <div>
                      <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">User Prompt (Data Profile)</h5>
                      <pre className="text-xs bg-white border rounded p-2 overflow-x-auto max-h-40 whitespace-pre-wrap">
                        {agentResult.transparency?.user_prompt}
                      </pre>
                    </div>
                    <div>
                      <h5 className="text-xs font-semibold text-gray-500 uppercase mb-1">Raw LLM Response</h5>
                      <pre className="text-xs bg-white border rounded p-2 overflow-x-auto max-h-60 whitespace-pre-wrap">
                        {agentResult.transparency?.raw_response}
                      </pre>
                    </div>
                    <div className="bg-blue-50 border border-blue-200 rounded p-3 text-xs text-blue-700">
                      <strong>Governance note:</strong> This interaction is logged in the audit trail (event_type: agent_recommendation).
                      A regulatory auditor can reconstruct exactly what the AI recommended, what data it saw,
                      and which human approved or rejected the recommendation.
                      Unlike black-box SaaS tools, every LLM call in Databricks is logged via AI Gateway
                      with full prompt/response capture, token usage, and user identity.
                    </div>
                  </div>
                )}
              </div>

              {/* Re-run button */}
              <div className="text-center">
                <button onClick={runAgent} disabled={agentLoading}
                  className="text-sm text-purple-600 hover:text-purple-800 font-medium">
                  Re-run analysis
                </button>
              </div>
            </>
          )}

          {agentResult && !agentResult.success && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
              AI analysis failed: {agentResult.error || 'Unknown error'}. The demo works normally without it.
            </div>
          )}
        </div>
      )}

      {/* ── Factory Runs List ── */}
      {runs.length === 0 ? (
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <FlaskConical className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-700 mb-2">No factory runs yet</h3>
          <p className="text-gray-500 max-w-md mx-auto">
            Run the Model Factory pipeline from the Databricks workflow to generate model candidates.
          </p>
        </div>
      ) : (
        <div className="grid gap-4">
          {runs.map((run) => {
            const allDecided = Number(run.models_decided || 0) >= Number(run.models_succeeded || 0) && Number(run.models_succeeded || 0) > 0;
            const hasApprovals = Number(run.models_approved || 0) > 0;

            return (
              <Link
                key={run.factory_run_id}
                to={`/models/${encodeURIComponent(run.factory_run_id)}`}
                className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all group"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-4">
                    <div className="w-10 h-10 bg-purple-50 rounded-lg flex items-center justify-center">
                      <FlaskConical className="w-5 h-5 text-purple-600" />
                    </div>
                    <div>
                      <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors font-mono">
                        {run.factory_run_id}
                      </h3>
                      <p className="text-sm text-gray-500">
                        Started {run.started_at ? new Date(run.started_at).toLocaleString() : 'Unknown'}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-6">
                    <div className="text-right">
                      <div className="text-sm text-gray-500">Planned</div>
                      <div className="text-sm font-medium">{run.models_planned}</div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-500">Trained</div>
                      <div className="text-sm font-medium text-green-600">
                        {run.models_succeeded}
                        {Number(run.models_failed) > 0 && (
                          <span className="text-red-500 ml-1">({run.models_failed} failed)</span>
                        )}
                      </div>
                    </div>
                    <div className="text-right">
                      <div className="text-sm text-gray-500">Approved</div>
                      <div className="text-sm font-medium">{run.models_approved || 0}</div>
                    </div>
                    {allDecided ? (
                      hasApprovals ? (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-green-50 text-green-700 border border-green-200">
                          <CheckCircle2 className="w-3.5 h-3.5" /> Complete
                        </span>
                      ) : (
                        <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-red-50 text-red-700 border border-red-200">
                          <AlertTriangle className="w-3.5 h-3.5" /> All Rejected
                        </span>
                      )
                    ) : (
                      <span className="inline-flex items-center gap-1 px-2.5 py-1 rounded-full text-xs font-medium bg-amber-50 text-amber-700 border border-amber-200">
                        <Clock className="w-3.5 h-3.5" /> Review Needed
                      </span>
                    )}
                    <ChevronRight className="w-5 h-5 text-gray-400 group-hover:text-blue-500" />
                  </div>
                </div>
              </Link>
            );
          })}
        </div>
      )}
    </div>
  );
}
