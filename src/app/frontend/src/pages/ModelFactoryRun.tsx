import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, CheckCircle2, XCircle, Clock, Shield, TrendingUp, AlertTriangle, FileText } from 'lucide-react';
import { api } from '../lib/api';

export default function ModelFactoryRun() {
  const { runId } = useParams<{ runId: string }>();
  const [leaderboard, setLeaderboard] = useState<any[]>([]);
  const [audit, setAudit] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);
  const [activeTab, setActiveTab] = useState<'leaderboard' | 'audit'>('leaderboard');
  const [selectedModel, setSelectedModel] = useState<string | null>(null);
  const [decisionNotes, setDecisionNotes] = useState('');
  const [decisionConditions, setDecisionConditions] = useState('');
  const [submitting, setSubmitting] = useState(false);

  const decodedRunId = runId ? decodeURIComponent(runId) : '';

  useEffect(() => {
    if (!decodedRunId) return;
    Promise.all([
      api.getLeaderboard(decodedRunId),
      api.getAuditTrail(decodedRunId),
    ]).then(([lb, au]) => {
      setLeaderboard(lb);
      setAudit(au);
    }).finally(() => setLoading(false));
  }, [decodedRunId]);

  const handleDecision = async (configId: string, decision: string) => {
    if (!decodedRunId) return;
    setSubmitting(true);
    try {
      await api.decideModel(decodedRunId, configId, decision, decisionNotes, decisionConditions);
      const lb = await api.getLeaderboard(decodedRunId);
      setLeaderboard(lb);
      setSelectedModel(null);
      setDecisionNotes('');
      setDecisionConditions('');
    } catch (e) {
      alert('Failed to record decision');
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <div className="p-8 text-center text-gray-500">Loading leaderboard...</div>;

  // Group by target
  const targets = [...new Set(leaderboard.map(m => m.target_column))];

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <Link to="/models" className="inline-flex items-center gap-1 text-sm text-blue-600 hover:text-blue-800 mb-4">
        <ArrowLeft className="w-4 h-4" /> Back to Factory Runs
      </Link>

      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Factory Run: <span className="font-mono">{decodedRunId}</span></h2>
        <p className="text-gray-500 mt-1">{leaderboard.length} models evaluated across {targets.length} target{targets.length !== 1 ? 's' : ''}</p>
      </div>

      {/* Tabs */}
      <div className="border-b border-gray-200 mb-6">
        <nav className="flex gap-6">
          {(['leaderboard', 'audit'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-3 text-sm font-medium border-b-2 transition-colors ${
                activeTab === tab
                  ? 'border-blue-500 text-blue-600'
                  : 'border-transparent text-gray-500 hover:text-gray-700'
              }`}
            >
              {tab === 'leaderboard' ? 'Leaderboard' : 'Audit Trail'}
            </button>
          ))}
        </nav>
      </div>

      {activeTab === 'leaderboard' && (
        <>
          {targets.map(target => {
            const models = leaderboard.filter(m => m.target_column === target);
            return (
              <div key={target} className="mb-8">
                <h3 className="text-lg font-semibold text-gray-800 mb-3 flex items-center gap-2">
                  <TrendingUp className="w-5 h-5 text-purple-500" />
                  {target === 'claim_count_5y' ? 'Frequency Models' :
                   target === 'total_incurred_5y' ? 'Severity Models' :
                   'Demand / Conversion Models'}
                  <span className="text-sm font-normal text-gray-500">({target})</span>
                </h3>

                <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
                  <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Rank</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Model</th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Family</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Features</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Gini</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">RMSE</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">PSI</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Reg Score</th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">Composite</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Status</th>
                        <th className="px-4 py-3 text-center text-xs font-medium text-gray-500 uppercase">Action</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100">
                      {models.map(m => (
                        <tr key={m.model_config_id} className={`hover:bg-gray-50 ${selectedModel === m.model_config_id ? 'bg-blue-50' : ''}`}>
                          <td className="px-4 py-3 text-sm font-bold text-gray-900">#{m.rank}</td>
                          <td className="px-4 py-3 text-sm font-mono text-gray-700">{m.model_config_id}</td>
                          <td className="px-4 py-3 text-sm">
                            <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                              m.model_family === 'GLM' ? 'bg-blue-100 text-blue-700' :
                              m.model_family === 'GBM' ? 'bg-green-100 text-green-700' :
                              'bg-purple-100 text-purple-700'
                            }`}>
                              {m.model_family}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-sm text-right text-gray-600">{m.feature_count}</td>
                          <td className="px-4 py-3 text-sm text-right font-medium">{Number(m.gini || 0).toFixed(4)}</td>
                          <td className="px-4 py-3 text-sm text-right text-gray-600">{m.rmse ? Number(m.rmse).toFixed(4) : m.roc_auc ? `AUC ${Number(m.roc_auc).toFixed(4)}` : '-'}</td>
                          <td className="px-4 py-3 text-sm text-right">
                            <span className={Number(m.psi || 0) < 0.1 ? 'text-green-600' : Number(m.psi || 0) < 0.25 ? 'text-amber-600' : 'text-red-600'}>
                              {Number(m.psi || 0).toFixed(4)}
                            </span>
                          </td>
                          <td className="px-4 py-3 text-sm text-right">
                            <div className="flex items-center justify-end gap-1">
                              <Shield className={`w-3.5 h-3.5 ${Number(m.regulatory_suitability_score || 0) >= 50 ? 'text-green-500' : 'text-amber-500'}`} />
                              {Number(m.regulatory_suitability_score || 0).toFixed(0)}
                            </div>
                          </td>
                          <td className="px-4 py-3 text-sm text-right font-bold text-gray-900">{Number(m.composite_score || 0).toFixed(4)}</td>
                          <td className="px-4 py-3 text-center">
                            {m.decision === 'APPROVED' ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-green-50 text-green-700">
                                <CheckCircle2 className="w-3 h-3" /> Approved
                              </span>
                            ) : m.decision === 'REJECTED' ? (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-red-50 text-red-700">
                                <XCircle className="w-3 h-3" /> Rejected
                              </span>
                            ) : (
                              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-50 text-amber-700">
                                <Clock className="w-3 h-3" /> Pending
                              </span>
                            )}
                          </td>
                          <td className="px-4 py-3 text-center">
                            {!m.decision && (
                              <button
                                onClick={() => setSelectedModel(selectedModel === m.model_config_id ? null : m.model_config_id)}
                                className="text-xs text-blue-600 hover:text-blue-800 font-medium"
                              >
                                Review
                              </button>
                            )}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>

                {/* Decision panel */}
                {selectedModel && models.find(m => m.model_config_id === selectedModel) && (
                  <div className="mt-4 bg-white rounded-lg border border-blue-200 p-6">
                    <h4 className="font-semibold text-gray-900 mb-4">
                      Review: <span className="font-mono">{selectedModel}</span>
                    </h4>
                    <div className="space-y-4">
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Notes</label>
                        <textarea
                          value={decisionNotes}
                          onChange={e => setDecisionNotes(e.target.value)}
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          rows={2}
                          placeholder="Actuarial review notes..."
                        />
                      </div>
                      <div>
                        <label className="block text-sm font-medium text-gray-700 mb-1">Conditions (optional)</label>
                        <input
                          value={decisionConditions}
                          onChange={e => setDecisionConditions(e.target.value)}
                          className="w-full border border-gray-300 rounded-lg px-3 py-2 text-sm"
                          placeholder="e.g., Subject to 6-month monitoring period"
                        />
                      </div>
                      <div className="flex gap-3">
                        <button
                          onClick={() => handleDecision(selectedModel, 'APPROVED')}
                          disabled={submitting}
                          className="px-4 py-2 bg-green-600 text-white text-sm font-medium rounded-lg hover:bg-green-700 disabled:opacity-50"
                        >
                          <CheckCircle2 className="w-4 h-4 inline mr-1" /> Approve
                        </button>
                        <button
                          onClick={() => handleDecision(selectedModel, 'REJECTED')}
                          disabled={submitting}
                          className="px-4 py-2 bg-red-600 text-white text-sm font-medium rounded-lg hover:bg-red-700 disabled:opacity-50"
                        >
                          <XCircle className="w-4 h-4 inline mr-1" /> Reject
                        </button>
                        <button
                          onClick={() => handleDecision(selectedModel, 'DEFERRED')}
                          disabled={submitting}
                          className="px-4 py-2 bg-gray-200 text-gray-700 text-sm font-medium rounded-lg hover:bg-gray-300 disabled:opacity-50"
                        >
                          Defer
                        </button>
                        <button
                          onClick={() => setSelectedModel(null)}
                          className="px-4 py-2 text-gray-500 text-sm hover:text-gray-700"
                        >
                          Cancel
                        </button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </>
      )}

      {activeTab === 'audit' && (
        <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Timestamp</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Event</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Actor</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">MLflow Run</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Details</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              {audit.map((event, i) => (
                <tr key={i} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-xs font-mono text-gray-600">
                    {event.event_timestamp ? new Date(event.event_timestamp).toLocaleString() : '-'}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <span className={`px-2 py-0.5 rounded text-xs font-medium ${
                      event.event_type?.includes('COMPLETE') || event.event_type?.includes('APPROVED') ? 'bg-green-100 text-green-700' :
                      event.event_type?.includes('FAILED') || event.event_type?.includes('REJECTED') ? 'bg-red-100 text-red-700' :
                      event.event_type?.includes('STARTED') ? 'bg-blue-100 text-blue-700' :
                      'bg-gray-100 text-gray-700'
                    }`}>
                      {event.event_type}
                    </span>
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-600">{event.actor}</td>
                  <td className="px-4 py-3 text-xs font-mono text-gray-500">
                    {event.mlflow_run_id ? event.mlflow_run_id.substring(0, 12) + '...' : '-'}
                  </td>
                  <td className="px-4 py-3 text-xs text-gray-500 max-w-md truncate">
                    {event.details_json ? event.details_json.substring(0, 100) + (event.details_json.length > 100 ? '...' : '') : '-'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
