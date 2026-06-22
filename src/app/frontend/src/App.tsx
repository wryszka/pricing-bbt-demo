import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { useEffect, useState } from 'react';
import { Database, FlaskConical, Shield, Code, Rocket, Home as HomeIcon, Table2, Package, Sparkles, Calculator, Zap, Archive } from 'lucide-react';
import Home from './pages/Home';
import DatasetList from './pages/DatasetList';
import DatasetDetail from './pages/DatasetDetail';
import FeatureStore from './pages/FeatureStore';
import ModelDevelopment from './pages/ModelDevelopment';
import ModelFactory from './pages/ModelFactory';
import ModelDeployment from './pages/ModelDeployment';
import Governance from './pages/Governance';
import QuoteReview from './pages/QuoteReview';
import Addons from './pages/Addons';
import NewDataImpact from './pages/NewDataImpact';
import RatingEngineIntegration from './pages/RatingEngineIntegration';
import Supervisor from './pages/Supervisor';
import PricingEngine from './pages/PricingEngine';

const NAV_ITEMS = [
  { to: '/',              label: 'Home',              icon: HomeIcon,     match: (p: string) => p === '/' },
  { to: '/datasets',      label: 'Data Ingestion',         icon: Database,     match: (p: string) => p.startsWith('/dataset') },
  { to: '/pricing-table', label: 'Modelling Mart',    icon: Table2,       match: (p: string) => p.startsWith('/pricing-table') },
  { to: '/development',   label: 'Model Development', icon: Code,         match: (p: string) => p.startsWith('/development') },
  { to: '/deployment',    label: 'Model Deployment',  icon: Rocket,       match: (p: string) => p.startsWith('/deployment') },
  { to: '/pricing-engine',label: 'Pricing Engine',    icon: Calculator,   match: (p: string) => p.startsWith('/pricing-engine') },
  { to: '/governance',    label: 'Model Governance',  icon: Shield,       match: (p: string) => p.startsWith('/governance') },
  { to: '/pricing-ai',    label: 'Pricing AI',        icon: Sparkles,     match: (p: string) => p.startsWith('/pricing-ai') || p.startsWith('/supervisor') || p.startsWith('/regulatory-ai') },
  { to: '/models',        label: 'Model Factory',     icon: FlaskConical, match: (p: string) => p.startsWith('/models') },
  { to: '/add-ons',       label: 'Add-ons',           icon: Package,      match: (p: string) => p.startsWith('/add-ons') || p.startsWith('/quote-review') },
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

      <AiModeBadge />

      {/* Footer */}
      <div className="px-4 py-3 border-t border-white/10 text-[10px] text-gray-500">
        Demo accelerator — not a Databricks product
      </div>
    </aside>
  );
}

function AiModeBadge() {
  const [mode, setMode] = useState<'live' | 'cached' | null>(null);
  const [busy, setBusy] = useState(false);
  const [entries, setEntries] = useState<number>(0);

  useEffect(() => {
    fetch('/api/admin/ai-mode')
      .then((r) => r.json())
      .then((d) => { setMode(d.mode); setEntries(d.entries ?? 0); })
      .catch(() => setMode('live'));
  }, []);

  async function flip() {
    if (busy || !mode) return;
    setBusy(true);
    const next = mode === 'live' ? 'cached' : 'live';
    try {
      const r = await fetch('/api/admin/ai-mode', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ mode: next }),
      });
      const d = await r.json();
      setMode(d.mode);
      setEntries(d.entries ?? 0);
    } finally {
      setBusy(false);
    }
  }

  const isCached = mode === 'cached';
  const Icon = isCached ? Archive : Zap;
  const colour = isCached ? 'bg-amber-500/15 text-amber-300 hover:bg-amber-500/25 border-amber-400/30'
                          : 'bg-emerald-500/15 text-emerald-300 hover:bg-emerald-500/25 border-emerald-400/30';
  return (
    <div className="px-3 py-2 border-t border-white/10">
      <button
        type="button"
        onClick={flip}
        disabled={!mode || busy}
        title={isCached
          ? `Serving cached AI responses (${entries} stored). Click to switch to live.`
          : 'Calling real serving endpoints. Click to switch to cached / consistent / fast.'}
        className={`w-full flex items-center gap-2 px-2.5 py-1.5 rounded-md border text-[11px] font-medium transition-colors disabled:opacity-50 ${colour}`}
      >
        <Icon className="w-3.5 h-3.5 shrink-0" />
        <span className="flex-1 text-left">AI: {mode ?? '…'}</span>
        {isCached && entries > 0 && (
          <span className="text-[10px] opacity-70">{entries}</span>
        )}
      </button>
    </div>
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
            <Route path="/deployment" element={<ModelDeployment />} />
            <Route path="/pricing-engine" element={<PricingEngine />} />
            <Route path="/governance" element={<Governance />} />
            <Route path="/pricing-ai"   element={<Supervisor />} />
            <Route path="/supervisor"   element={<Supervisor />} />  {/* legacy URL */}
            <Route path="/regulatory-ai" element={<Supervisor />} />  {/* legacy URL */}
            <Route path="/add-ons" element={<Addons />} />
            <Route path="/add-ons/quote-review" element={<QuoteReview />} />
            <Route path="/add-ons/new-data-impact" element={<NewDataImpact />} />
            <Route path="/add-ons/rating-engine" element={<RatingEngineIntegration />} />
            {/* Legacy redirect */}
            <Route path="/quote-review" element={<QuoteReview />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
