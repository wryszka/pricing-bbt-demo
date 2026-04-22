import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Database, FlaskConical, Shield, Code, Rocket, Activity, Home as HomeIcon, Table2, Receipt } from 'lucide-react';
import Home from './pages/Home';
import DatasetList from './pages/DatasetList';
import DatasetDetail from './pages/DatasetDetail';
import FeatureStore from './pages/FeatureStore';
import ModelDevelopment from './pages/ModelDevelopment';
import ModelFactory from './pages/ModelFactory';
import ModelFactoryRun from './pages/ModelFactoryRun';
import ModelDeployment from './pages/ModelDeployment';
import Monitoring from './pages/Monitoring';
import Governance from './pages/Governance';
import QuoteReview from './pages/QuoteReview';

const NAV_ITEMS = [
  { to: '/', label: 'Home', icon: HomeIcon, match: (p: string) => p === '/' },
  { to: '/datasets', label: 'Data Ingestion', icon: Database, match: (p: string) => p.startsWith('/dataset') },
  { to: '/pricing-table', label: 'Pricing Table', icon: Table2, match: (p: string) => p.startsWith('/pricing-table') },
  { to: '/development', label: 'Model Development', icon: Code, match: (p: string) => p.startsWith('/development') },
  { to: '/models', label: 'Model Factory', icon: FlaskConical, match: (p: string) => p.startsWith('/models') },
  { to: '/deployment', label: 'Model Deployment', icon: Rocket, match: (p: string) => p.startsWith('/deployment') },
  { to: '/quote-review', label: 'Quote Review', icon: Receipt, match: (p: string) => p.startsWith('/quote-review') },
  { to: '/monitoring', label: 'Monitoring', icon: Activity, match: (p: string) => p.startsWith('/monitoring') },
  { to: '/governance', label: 'Governance', icon: Shield, match: (p: string) => p.startsWith('/governance') },
];

function Sidebar() {
  const { pathname } = useLocation();

  return (
    <aside className="w-56 bg-[#1e293b] text-white min-h-screen flex flex-col shrink-0">
      {/* Brand */}
      <Link to="/" className="px-4 py-5 flex items-center gap-3 hover:opacity-90 transition-opacity border-b border-white/10">
        <Database className="w-7 h-7 text-blue-400" />
        <div>
          <h1 className="text-sm font-bold tracking-tight leading-tight">Pricing Workbench</h1>
          <p className="text-[10px] text-gray-400">Bricksurance SE</p>
        </div>
      </Link>

      {/* Nav items */}
      <nav className="flex-1 px-2 py-3 space-y-0.5">
        {NAV_ITEMS.map(({ to, label, icon: Icon, match }) => (
          <Link key={to} to={to}
            className={`flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm transition-colors ${
              match(pathname)
                ? 'bg-blue-600/20 text-white font-medium'
                : 'text-gray-400 hover:text-white hover:bg-white/5'
            }`}
          >
            <Icon className={`w-4 h-4 shrink-0 ${match(pathname) ? 'text-blue-400' : ''}`} />
            {label}
          </Link>
        ))}
      </nav>

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/10 text-[10px] text-gray-500">
        Demo accelerator — not a Databricks product
      </div>
    </aside>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100 font-[system-ui] flex">
        <Sidebar />
        <main className="flex-1 overflow-auto">
          <Routes>
            <Route path="/" element={<Home />} />
            <Route path="/datasets" element={<DatasetList />} />
            <Route path="/dataset/:datasetId" element={<DatasetDetail />} />
            <Route path="/pricing-table" element={<FeatureStore />} />
            <Route path="/development" element={<ModelDevelopment />} />
            <Route path="/models" element={<ModelFactory />} />
            <Route path="/models/:runId" element={<ModelFactoryRun />} />
            <Route path="/deployment" element={<ModelDeployment />} />
            <Route path="/quote-review" element={<QuoteReview />} />
            <Route path="/monitoring" element={<Monitoring />} />
            <Route path="/governance" element={<Governance />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
