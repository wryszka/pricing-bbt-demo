import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { FlaskConical, ChevronRight, CheckCircle2, Clock, AlertTriangle } from 'lucide-react';
import { api } from '../lib/api';

export default function ModelFactory() {
  const [runs, setRuns] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getFactoryRuns().then(setRuns).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-center text-gray-500">Loading factory runs...</div>;

  if (runs.length === 0) {
    return (
      <div className="max-w-7xl mx-auto px-6 py-8">
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Model Factory</h2>
        <p className="text-gray-500 mb-8">Automated model training, evaluation, and actuary approval</p>
        <div className="bg-white rounded-lg border border-gray-200 p-12 text-center">
          <FlaskConical className="w-12 h-12 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-700 mb-2">No factory runs yet</h3>
          <p className="text-gray-500 max-w-md mx-auto">
            Run the Model Factory pipeline from the Databricks workflow to generate model candidates.
            The feature inspector will profile your UPT, train ~20 models, and present them here for review.
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">Model Factory</h2>
        <p className="text-gray-500 mt-1">Review model factory runs and approve models for production</p>
      </div>

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
    </div>
  );
}
