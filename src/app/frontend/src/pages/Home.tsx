import { Link } from 'react-router-dom';
import { Database, FlaskConical, Zap, Shield, ArrowRight } from 'lucide-react';

export default function Home() {
  return (
    <div className="max-w-7xl mx-auto px-6 py-8">
      {/* Hero */}
      <div className="text-center mb-10">
        <h1 className="text-3xl font-bold text-gray-900 mb-2">P&C Insurance Pricing</h1>
        <p className="text-lg text-blue-600 font-medium">Databricks Accelerator</p>
        <p className="text-gray-500 mt-3 max-w-2xl mx-auto">
          End-to-end pricing data transformation on a single platform: from raw vendor data
          to live pricing decisions, with full governance, human-in-the-loop approval,
          and regulatory-grade auditability.
        </p>
      </div>

      {/* Section cards */}
      <div className="grid grid-cols-2 gap-5 mb-8">
        <SectionCard
          to="/datasets"
          icon={Database}
          color="blue"
          title="Data Ingestion"
          description="Review, validate and approve external datasets. Upload/download data, analyse pricing impact, and track data quality before merging into the Unified Pricing Table."
          features={["Delta Live Tables (DLT) expectations", "Shadow pricing simulation", "Manual upload with audit trail"]}
        />
        <SectionCard
          to="/models"
          icon={FlaskConical}
          color="purple"
          title="Model Factory"
          description="Train, evaluate, and approve pricing models. GLMs for frequency/severity, GBMs for demand and fraud, with regulatory-grade PDF reports."
          features={["MLflow experiment tracking", "Unity Catalog model registry", "AI-assisted model selection (optional)"]}
        />
        <SectionCard
          to="/pricing-table"
          icon={Zap}
          color="green"
          title="Pricing Table"
          description="The Unified Pricing Table — single wide table with all features for model training and real-time serving. Query with natural language via Genie."
          features={["UC Feature Engineering", "Lakebase online store", "Genie natural language queries"]}
        />
        <SectionCard
          to="/governance"
          icon={Shield}
          color="amber"
          title="Governance & Audit"
          description="Complete audit trail from raw data to live pricing. Every decision tracked, every version reproducible, every LLM call logged."
          features={["Unity Catalog lineage", "Immutable audit log", "Regulatory export (PDF + JSON)"]}
        />
      </div>

      {/* About */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-5">
        <h3 className="font-semibold text-gray-800 mb-2">About this demo</h3>
        <p className="text-sm text-gray-600 mb-3">
          <strong>This is not a Databricks product.</strong> It is an example of what can be built
          on the Databricks platform using standard capabilities (Unity Catalog, Delta Lake, MLflow,
          Mosaic AI, Databricks Apps). The full source code is available on GitHub — feel free to
          fork it, adapt it to your own data, and use it as a starting point.
        </p>
        <p className="text-sm text-gray-600">
          All company names (Bricksurance SE), policy data, and financial figures are entirely
          fictional and generated for illustrative purposes. No real customer data is used.
        </p>
      </div>
    </div>
  );
}

function SectionCard({ to, icon: Icon, color, title, description, features }: {
  to: string; icon: any; color: string; title: string; description: string; features: string[];
}) {
  const colorMap: Record<string, { bg: string; border: string; icon: string; badge: string }> = {
    blue:   { bg: 'bg-blue-50',   border: 'border-blue-200',   icon: 'text-blue-600',   badge: 'bg-blue-100 text-blue-700' },
    purple: { bg: 'bg-purple-50', border: 'border-purple-200', icon: 'text-purple-600', badge: 'bg-purple-100 text-purple-700' },
    green:  { bg: 'bg-green-50',  border: 'border-green-200',  icon: 'text-green-600',  badge: 'bg-green-100 text-green-700' },
    amber:  { bg: 'bg-amber-50',  border: 'border-amber-200',  icon: 'text-amber-600',  badge: 'bg-amber-100 text-amber-700' },
  };
  const c = colorMap[color] || colorMap.blue;

  return (
    <Link to={to} className={`block ${c.bg} border ${c.border} rounded-lg p-5 hover:shadow-md transition-all group`}>
      <div className="flex items-center gap-3 mb-2">
        <Icon className={`w-5 h-5 ${c.icon}`} />
        <h3 className="font-semibold text-gray-900 group-hover:text-blue-600 transition-colors">{title}</h3>
        <ArrowRight className="w-4 h-4 text-gray-400 ml-auto group-hover:translate-x-1 transition-transform" />
      </div>
      <p className="text-sm text-gray-600 mb-3">{description}</p>
      <div className="flex flex-wrap gap-1.5">
        {features.map((f, i) => (
          <span key={i} className={`px-2 py-0.5 rounded text-[10px] font-medium ${c.badge}`}>{f}</span>
        ))}
      </div>
    </Link>
  );
}
