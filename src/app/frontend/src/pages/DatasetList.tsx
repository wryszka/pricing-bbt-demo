import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Database, ChevronRight, CheckCircle2, XCircle, Clock } from 'lucide-react';
import { api } from '../lib/api';

export default function DatasetList() {
  const [datasets, setDatasets] = useState<any[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    api.getDatasets().then(setDatasets).finally(() => setLoading(false));
  }, []);

  if (loading) return <div className="p-8 text-center text-gray-500">Loading datasets...</div>;

  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-gray-900">External Data Sources</h2>
        <p className="text-gray-500 mt-1">Review, validate and approve external datasets before they merge into the Unified Pricing Table</p>
      </div>

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

      <div className="grid gap-4">
        {datasets.map((ds) => {
          const approval = ds.approval;
          const status = approval?.decision || 'pending';

          return (
            <Link
              key={ds.id}
              to={`/dataset/${ds.id}`}
              className="bg-white rounded-lg border border-gray-200 p-5 hover:border-blue-300 hover:shadow-md transition-all group"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <div className="w-10 h-10 bg-blue-50 rounded-lg flex items-center justify-center">
                    <Database className="w-5 h-5 text-blue-600" />
                  </div>
                  <div>
                    <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">
                      {ds.display_name}
                    </h3>
                    <p className="text-sm text-gray-500">{ds.description}</p>
                  </div>
                </div>
                <div className="flex items-center gap-6">
                  <div className="text-right">
                    <div className="text-sm text-gray-500">Source</div>
                    <div className="text-sm font-medium">{ds.source}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-500">Raw / Silver</div>
                    <div className="text-sm font-medium">
                      {Number(ds.raw_row_count).toLocaleString()} / {Number(ds.silver_row_count).toLocaleString()}
                    </div>
                  </div>
                  <div className="text-right">
                    <div className="text-sm text-gray-500">DQ Dropped</div>
                    <div className={`text-sm font-medium ${ds.rows_dropped_by_dq > 0 ? 'text-amber-600' : 'text-green-600'}`}>
                      {ds.rows_dropped_by_dq}
                    </div>
                  </div>
                  <StatusBadge status={status} />
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

function StatusBadge({ status }: { status: string }) {
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
      <Clock className="w-3.5 h-3.5" /> Pending
    </span>
  );
}
