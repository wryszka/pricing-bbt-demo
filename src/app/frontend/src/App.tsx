import { BrowserRouter, Routes, Route, Link, useLocation } from 'react-router-dom';
import { Database, Building2, FlaskConical } from 'lucide-react';
import DatasetList from './pages/DatasetList';
import DatasetDetail from './pages/DatasetDetail';
import ModelFactory from './pages/ModelFactory';
import ModelFactoryRun from './pages/ModelFactoryRun';

function Nav() {
  const location = useLocation();
  const isModels = location.pathname.startsWith('/models');

  return (
    <header className="bg-[#1e293b] text-white">
      <div className="max-w-7xl mx-auto px-6 py-3 flex items-center justify-between">
        <div className="flex items-center gap-8">
          <Link to="/" className="flex items-center gap-3 hover:opacity-90 transition-opacity">
            <Database className="w-6 h-6 text-blue-400" />
            <div>
              <h1 className="text-lg font-bold tracking-tight">Pricing Governance</h1>
              <p className="text-xs text-gray-400">Data & Model Review</p>
            </div>
          </Link>
          <nav className="flex items-center gap-1">
            <Link
              to="/"
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                !isModels ? 'bg-white/10 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              <Database className="w-3.5 h-3.5 inline mr-1.5" />
              Data Ingestion
            </Link>
            <Link
              to="/models"
              className={`px-3 py-1.5 rounded text-sm font-medium transition-colors ${
                isModels ? 'bg-white/10 text-white' : 'text-gray-400 hover:text-white'
              }`}
            >
              <FlaskConical className="w-3.5 h-3.5 inline mr-1.5" />
              Model Factory
            </Link>
          </nav>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <Building2 className="w-4 h-4" />
          <span className="font-medium text-gray-300">Bricksurance SE</span>
        </div>
      </div>
    </header>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <div className="min-h-screen bg-gray-100 font-[system-ui]">
        <Nav />
        <main>
          <Routes>
            <Route path="/" element={<DatasetList />} />
            <Route path="/dataset/:datasetId" element={<DatasetDetail />} />
            <Route path="/models" element={<ModelFactory />} />
            <Route path="/models/:runId" element={<ModelFactoryRun />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}
