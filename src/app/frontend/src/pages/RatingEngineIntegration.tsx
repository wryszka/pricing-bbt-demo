import { useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Radar, ArrowLeft, Database, Table2, Shield, Lock, KeyRound,
  Network, FileCheck2, Workflow, Layers, Boxes, Activity,
  ScrollText, Zap, Cpu, GitBranch, Repeat,
} from 'lucide-react';

/**
 * Rating Engine Integration. Demo-grade explainer of three integration
 * patterns the workbench supports for Radar / Earnix:
 *
 *   1. Build — model construction lifecycle (data prep on Databricks → Radar
 *      builds → result back into Databricks for governance)
 *   2. Batch what-if — Databricks orchestrates parallel Radar runs to compare
 *      scenarios, gathers results back for analysis
 *   3. Live — two sub-patterns: Databricks as enrichment layer in front of
 *      Radar, or Radar calling Databricks at quote time for enriched features
 *
 * Each tab renders a clean SVG, a numbered step list, and a security &
 * governance side-panel anchored in Databricks features.
 */

type Pattern = 'build' | 'batch' | 'live' | 'standards';

export default function RatingEngineIntegration() {
  const [pattern, setPattern] = useState<Pattern>('build');

  return (
    <div className="max-w-6xl mx-auto px-6 py-8">
      <Link to="/add-ons"
            className="inline-flex items-center gap-1 text-sm text-gray-500 hover:text-gray-800 mb-2">
        <ArrowLeft className="w-3.5 h-3.5" /> Back to Add-ons
      </Link>

      <div className="mb-5">
        <div className="flex items-center gap-2 flex-wrap">
          <h2 className="text-2xl font-bold text-gray-900 flex items-center gap-2">
            <Radar className="w-6 h-6 text-emerald-600" /> Rating Engine Integration
          </h2>
          <span className="text-[10px] uppercase tracking-wider font-bold px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200">
            Reference architectures
          </span>
        </div>
        <p className="text-gray-600 mt-2 text-sm max-w-3xl">
          The workbench complements Willis Towers Watson Radar (and Earnix, or any in-house engine)
          across the model lifecycle. Three patterns, all governed end-to-end in Unity Catalog.
        </p>
      </div>

      <div className="bg-white rounded-lg border border-gray-200 p-1 mb-6 inline-flex gap-1 flex-wrap">
        <PatternTab active={pattern === 'build'} onClick={() => setPattern('build')}
                    icon={<Workflow className="w-3.5 h-3.5" />}
                    label="1 · Build" sub="model construction lifecycle" />
        <PatternTab active={pattern === 'batch'} onClick={() => setPattern('batch')}
                    icon={<GitBranch className="w-3.5 h-3.5" />}
                    label="2 · Batch what-if" sub="parallel Radar runs" />
        <PatternTab active={pattern === 'live'} onClick={() => setPattern('live')}
                    icon={<Zap className="w-3.5 h-3.5" />}
                    label="3 · Live" sub="quote-time enrichment" />
        <PatternTab active={pattern === 'standards'} onClick={() => setPattern('standards')}
                    icon={<FileCheck2 className="w-3.5 h-3.5" />}
                    label="Standards" sub="Radar integration standards" />
      </div>

      {pattern === 'build' && <BuildPattern />}
      {pattern === 'batch' && <BatchPattern />}
      {pattern === 'live'  && <LivePattern />}
      {pattern === 'standards' && <StandardsTab />}
    </div>
  );
}

function PatternTab({ active, onClick, icon, label, sub }:
  { active: boolean; onClick: () => void; icon: React.ReactNode; label: string; sub: string }) {
  return (
    <button onClick={onClick}
            className={`px-3.5 py-2 rounded-md text-sm font-medium transition flex items-center gap-2 ${
              active ? 'bg-emerald-600 text-white' : 'text-gray-700 hover:bg-gray-100'
            }`}>
      {icon}
      <span className="flex flex-col items-start leading-tight">
        <span>{label}</span>
        <span className={`text-[10px] ${active ? 'text-emerald-100' : 'text-gray-500'}`}>{sub}</span>
      </span>
    </button>
  );
}

// ===========================================================================
// PATTERN 1 — Build (model construction)
// ===========================================================================

function BuildPattern() {
  return (
    <PatternFrame
      title="Pattern 1 · Build the model"
      subtitle="Databricks prepares the modelling dataset, Radar 5 reads it via the supported ODBC connector against a Databricks SQL warehouse, builds the GLMs, and writes results straight back through the same connector."
      svg={<BuildDiagram />}
      steps={[
        { n: 1, label: 'Data ingestion', body: 'External vendor feeds + internal book land in UC Volumes / Delta. DLT pipelines apply expectations. Approved silvers feed the Modelling Mart.' },
        { n: 2, label: 'Modelling Mart',  body: 'Single Delta table — every approved feed joined onto the active book at policy_id grain. Per-LOB by design. ~50 vetted factors.' },
        { n: 3, label: 'Radar 5 ODBC connector → DBSQL', body: 'Radar 5\'s built-in Databricks ODBC connector reads the mart over a Databricks SQL warehouse. Bidirectional: the same DSN handles both inbound (Radar pulls modelling data) and outbound (Radar writes factor tables / model objects back to UC).' },
        { n: 4, label: 'Radar build',     body: 'Actuaries build GLMs / GBMs in Radar with the workbench-prepared data. Familiar UX, no migration required.' },
        { n: 5, label: 'Results back to Delta', body: 'Radar writes factor tables, model objects, and fit metrics back over the same ODBC connection. Land in a UC schema with full lineage; no file shuffling.' },
        { n: 6, label: 'Governance pack', body: 'On import, the workbench generates a pack PDF + sidecar — fit metrics, lineage, approval flow, audit event. Tied to the Radar model version.' },
      ]}
      governance={[
        { icon: <Lock className="w-4 h-4 text-amber-700" />, label: 'UC ACLs on the warehouse', body: 'The ODBC DSN authenticates as a scoped service principal with grants only on the modelling mart and the result schema — never broader. Row/column filters and PII masks are enforced by UC at query time.' },
        { icon: <Network className="w-4 h-4 text-amber-700" />, label: 'Network isolation', body: 'DBSQL warehouse exposed via PrivateLink (or front-door allowlist). No public endpoint. Radar 5 connects from inside the customer VPC.' },
        { icon: <ScrollText className="w-4 h-4 text-amber-700" />, label: 'Lineage end-to-end', body: 'Every column flows source feed → mart → ODBC read → Radar build → ODBC write → result table. UC tracks the lineage graph; regulator reaches source data in two clicks.' },
        { icon: <FileCheck2 className="w-4 h-4 text-amber-700" />, label: 'Governance pack', body: 'Every Radar model that gets re-imported is bound to a pack_id with approval, fit metrics, fairness scan, and audit trail.' },
      ]}
    />
  );
}

function BuildDiagram() {
  return (
    <svg viewBox="0 0 920 340" className="w-full max-w-5xl" aria-label="Build-pattern architecture">
      <defs>
        <marker id="b-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#475569" />
        </marker>
        <marker id="b-arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#1e40af" />
        </marker>
        <marker id="b-arr-emerald" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#047857" />
        </marker>
        <filter id="b-shd" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.08"/>
        </filter>
      </defs>

      {/* Pricing Workbench lane (left) */}
      <rect x="20" y="20" width="280" height="300" rx="12" fill="#ecfdf5" stroke="#10b981" strokeWidth="1.5" filter="url(#b-shd)"/>
      <text x="160" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#064e3b">Pricing Workbench (Databricks)</text>

      <rect x="40" y="60" width="240" height="44" rx="6" fill="white" stroke="#34d399"/>
      <text x="160" y="80" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">1 · Data ingestion</text>
      <text x="160" y="95" textAnchor="middle" fontSize="9" fill="#047857">DLT · UC Volumes · expectations</text>

      <rect x="40" y="115" width="240" height="44" rx="6" fill="white" stroke="#34d399"/>
      <text x="160" y="135" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">2 · Modelling Mart</text>
      <text x="160" y="150" textAnchor="middle" fontSize="9" fill="#047857">Delta · per-LOB · ~50 factors</text>

      <rect x="40" y="170" width="240" height="44" rx="6" fill="white" stroke="#34d399"/>
      <text x="160" y="190" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">5 · Result schema</text>
      <text x="160" y="205" textAnchor="middle" fontSize="9" fill="#047857">factor tables · fit metrics · model objects</text>

      <rect x="40" y="245" width="240" height="44" rx="6" fill="white" stroke="#34d399"/>
      <text x="160" y="265" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">6 · Governance pack</text>
      <text x="160" y="280" textAnchor="middle" fontSize="9" fill="#047857">PDF + sidecar · pack_id binding</text>

      {/* DBSQL endpoint (middle) */}
      <rect x="320" y="20" width="220" height="300" rx="12" fill="#fefce8" stroke="#eab308" strokeWidth="1.5" filter="url(#b-shd)"/>
      <text x="430" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#713f12">Databricks SQL warehouse</text>
      <text x="430" y="58" textAnchor="middle" fontSize="9" fill="#a16207">single endpoint · UC governed</text>

      <rect x="340" y="80" width="180" height="60" rx="6" fill="white" stroke="#facc15"/>
      <text x="430" y="100" textAnchor="middle" fontSize="11" fontWeight="600" fill="#854d0e">3 · Radar 5 ODBC</text>
      <text x="430" y="116" textAnchor="middle" fontSize="9" fill="#a16207">supported native connector</text>
      <text x="430" y="130" textAnchor="middle" fontSize="9" fill="#a16207">DSN · service principal · OAuth</text>

      {/* Bidirectional indication */}
      <rect x="340" y="155" width="180" height="42" rx="6" fill="#fff7ed" stroke="#fb923c"/>
      <text x="430" y="175" textAnchor="middle" fontSize="11" fontWeight="600" fill="#9a3412">bidirectional</text>
      <text x="430" y="189" textAnchor="middle" fontSize="9" fill="#9a3412">read modelling data · write results</text>

      <rect x="340" y="215" width="180" height="60" rx="6" fill="white" stroke="#facc15"/>
      <text x="430" y="235" textAnchor="middle" fontSize="11" fontWeight="600" fill="#854d0e">UC enforcement</text>
      <text x="430" y="251" textAnchor="middle" fontSize="9" fill="#a16207">column masks · row filters</text>
      <text x="430" y="265" textAnchor="middle" fontSize="9" fill="#a16207">audit log · lineage</text>

      {/* Radar lane (right) */}
      <rect x="560" y="20" width="340" height="300" rx="12" fill="#eff6ff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#b-shd)"/>
      <text x="730" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#1e3a8a">Radar 5</text>
      <text x="730" y="58" textAnchor="middle" fontSize="9" fill="#2563eb">actuary's existing workflow</text>

      <rect x="580" y="80" width="300" height="220" rx="6" fill="white" stroke="#60a5fa"/>
      <text x="730" y="103" textAnchor="middle" fontSize="11" fontWeight="600" fill="#1e3a8a">4 · Radar build</text>
      <text x="730" y="123" textAnchor="middle" fontSize="9" fill="#2563eb">reads modelling data via ODBC DSN</text>
      <text x="730" y="143" textAnchor="middle" fontSize="9" fill="#2563eb">GLMs · GBMs · interaction tables</text>
      <text x="730" y="163" textAnchor="middle" fontSize="9" fill="#2563eb">Radar UI for fit + diagnostics</text>
      <text x="730" y="183" textAnchor="middle" fontSize="9" fill="#2563eb">factor relativities · model objects</text>
      <text x="730" y="218" textAnchor="middle" fontSize="9" fill="#2563eb">writes results via the same DSN</text>
      <text x="730" y="238" textAnchor="middle" fontSize="9" fill="#2563eb">no file shuffling · no migration</text>
      <text x="730" y="270" textAnchor="middle" fontSize="11" fontWeight="600" fill="#1e3a8a">single integration surface</text>

      {/* Arrows: read path (mart → DBSQL → Radar) */}
      <line x1="280" y1="137" x2="338" y2="105" stroke="#047857" strokeWidth="2.2" markerEnd="url(#b-arr-emerald)"/>
      <line x1="520" y1="105" x2="578" y2="135" stroke="#047857" strokeWidth="2.2" markerEnd="url(#b-arr-emerald)"/>
      <text x="280" y="80" textAnchor="middle" fontSize="9" fontWeight="600" fill="#047857">read · SELECT</text>

      {/* Arrows: write path (Radar → DBSQL → result schema) */}
      <line x1="578" y1="245" x2="520" y2="245" stroke="#1e40af" strokeWidth="2.2" markerEnd="url(#b-arr-blue)"/>
      <line x1="338" y1="245" x2="282" y2="195" stroke="#1e40af" strokeWidth="2.2" markerEnd="url(#b-arr-blue)"/>
      <text x="430" y="295" textAnchor="middle" fontSize="9" fontWeight="600" fill="#1e40af">write · INSERT (factor tables)</text>

      {/* Arrow: result → governance pack */}
      <line x1="160" y1="214" x2="160" y2="244" stroke="#475569" strokeWidth="1.5" markerEnd="url(#b-arr)"/>
    </svg>
  );
}

// ===========================================================================
// PATTERN 2 — Batch what-if
// ===========================================================================

function BatchPattern() {
  return (
    <div className="space-y-5">
      <PatternFrame
        title="Pattern 2 · Batch what-if (hybrid default)"
        subtitle="Split the portfolio into N partitions, fan out to N parallel Radar licences. Read via ODBC against DBSQL (governed, fast), write back as files to a UC Volume picked up by Autoloader. Hybrid is the sweet spot — but the right pattern depends on volume, licences, and Radar config (see the matrix below)."
        svg={<BatchDiagram />}
        steps={[
          { n: 1, label: 'Partition the book',   body: 'Workbench splits the policy table into N partitions (round-robin, postcode hash, or LOB cut — whatever balances load). Each partition becomes a separate slice readable by one Radar licence.' },
          { n: 2, label: 'ODBC read in', body: 'Each Radar 5 instance reads its partition over the same DBSQL ODBC DSN as Pattern 1. UC enforces ACLs — no broader access than the scoped service principal allows. ~5-20 s per 100k rows on a Medium warehouse.' },
          { n: 3, label: 'Parallel Radar runs',  body: 'N Radar licences process their partitions independently — same scenario across all, or one scenario per licence depending on the what-if matrix. No DBSQL contention; the warehouse already returned the data.' },
          { n: 4, label: 'Files out to UC Volume', body: 'Each Radar instance writes its result file (Parquet / CSV) to a UC Volume directory. Files-out avoids the ODBC writeback bottleneck — bulk results land in seconds, not minutes.' },
          { n: 5, label: 'Autoloader → Delta', body: 'Workbench Autoloader stream picks up new files exactly-once, handles schema evolution, lands a single Delta results table keyed by scenario_id × partition_id. No polling job to maintain.' },
          { n: 6, label: 'Compare + governance', body: 'Dashboards + Genie over the results: premium delta, loss-ratio impact, fairness across cohorts. The chosen scenario gets a governance pack with full inputs / outputs / audit trail.' },
        ]}
        governance={[
          { icon: <Lock className="w-4 h-4 text-amber-700" />, label: 'UC governs both legs', body: 'ODBC read enforces row/column ACLs at the warehouse. UC Volume writeback uses the same scoped service principal — no separate IAM for the file path.' },
          { icon: <Boxes className="w-4 h-4 text-amber-700" />, label: 'Per-partition isolation', body: 'Each Radar licence sees only its partition\'s data. A misconfigured licence can\'t read another partition; results land in scoped Volume sub-paths.' },
          { icon: <Activity className="w-4 h-4 text-amber-700" />, label: 'Audit log + Autoloader checkpoints', body: 'Every Radar invocation logged with partition_id + scenario_id. Autoloader\'s rocksdb checkpoint guarantees exactly-once ingestion — replay is safe.' },
          { icon: <Repeat className="w-4 h-4 text-amber-700" />, label: 'Reproducibility', body: 'Inputs are Delta with versions, partitioning is deterministic, output files are immutable. Re-running the same scenario hash gives the same answer — regulators love this.' },
        ]}
      />
      <RegimeMatrix />
    </div>
  );
}

function RegimeMatrix() {
  const regimes = [
    {
      name: 'ODBC in & out',
      tone: 'blue',
      when: '~10k rows / partition · few scenarios · narrow output',
      pros: ['Single integration pattern', 'Simplest ops', 'No file plumbing'],
      cons: ['Writeback contention on the warehouse', 'Slow above ~50k rows', 'Radar row-by-row writeback amplifies cost'],
      verdict: 'Good for small portfolios or quick exploratory runs',
    },
    {
      name: 'Hybrid (ODBC in · files out)',
      tone: 'emerald',
      when: '~100k rows / partition · 4-10 parallel licences · standard scenario count',
      pros: ['Fast ODBC reads stay governed by UC', 'Files+Autoloader scales writes linearly', 'Exactly-once + schema evolution free'],
      cons: ['Two integration surfaces to explain', 'Output volume needs to be sized'],
      verdict: 'Default. Sweet spot for typical commercial what-if work',
      primary: true,
    },
    {
      name: 'Files in & out',
      tone: 'purple',
      when: '~1M rows / partition · 10+ licences · wide output / large factor tables',
      pros: ['No warehouse load at all', 'Maximum parallelism', 'Cheap on DBSQL credits'],
      cons: ['Two file legs to orchestrate', 'Loses real-time UC enforcement on read (relies on Volume ACLs)'],
      verdict: 'Use when partition size or licence count makes ODBC the bottleneck',
    },
  ];
  const toneClasses: Record<string, { border: string; bg: string; head: string }> = {
    blue:    { border: 'border-blue-200',    bg: 'bg-blue-50',    head: 'text-blue-900' },
    emerald: { border: 'border-emerald-300', bg: 'bg-emerald-50', head: 'text-emerald-900' },
    purple:  { border: 'border-purple-200',  bg: 'bg-purple-50',  head: 'text-purple-900' },
  };

  return (
    <section className="bg-white border border-gray-200 rounded-lg p-5">
      <div className="mb-4">
        <h3 className="text-base font-semibold text-gray-900 flex items-center gap-2">
          <Cpu className="w-5 h-5 text-emerald-600" /> When to use which pattern
        </h3>
        <p className="text-sm text-gray-600 mt-1">
          No universal answer — the right approach depends on partition size, parallel licences, and how the
          Radar model writes back. Three regimes:
        </p>
      </div>
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
        {regimes.map(r => {
          const c = toneClasses[r.tone];
          return (
            <div key={r.name}
                 className={`rounded-lg border ${c.border} ${c.bg} p-4 relative ${r.primary ? 'ring-2 ring-emerald-400' : ''}`}>
              {r.primary && (
                <span className="absolute -top-2 left-3 text-[9px] uppercase tracking-wider font-bold px-1.5 py-0.5 rounded bg-emerald-600 text-white">
                  Default
                </span>
              )}
              <h4 className={`font-semibold ${c.head} mb-1`}>{r.name}</h4>
              <div className="text-[11px] text-gray-600 mb-3 italic">{r.when}</div>
              <div className="text-[11px] font-semibold text-gray-700 mb-1 uppercase tracking-wide">Pros</div>
              <ul className="space-y-0.5 mb-2">
                {r.pros.map((p, i) => (
                  <li key={i} className="text-xs text-gray-700 flex gap-1.5">
                    <span className="text-emerald-600 shrink-0">+</span><span>{p}</span>
                  </li>
                ))}
              </ul>
              <div className="text-[11px] font-semibold text-gray-700 mb-1 uppercase tracking-wide">Cons</div>
              <ul className="space-y-0.5 mb-2">
                {r.cons.map((p, i) => (
                  <li key={i} className="text-xs text-gray-700 flex gap-1.5">
                    <span className="text-red-600 shrink-0">−</span><span>{p}</span>
                  </li>
                ))}
              </ul>
              <div className={`text-xs ${c.head} font-medium border-t border-gray-200 pt-2 mt-2`}>
                {r.verdict}
              </div>
            </div>
          );
        })}
      </div>
      <div className="mt-4 text-[11px] text-gray-500 italic">
        Two more dimensions skew the choice: <span className="font-semibold">output cardinality</span> (50-column results vs 200-column factor tables) makes file-out faster much sooner; and{' '}
        <span className="font-semibold">Radar writeback config</span> — some installs stream rows, others batch on completion — flips the ODBC-out economics. Architecture review for any deployment beyond a small POC.
      </div>
    </section>
  );
}

function BatchDiagram() {
  return (
    <svg viewBox="0 0 940 480" className="w-full max-w-5xl" aria-label="Batch what-if hybrid architecture">
      <defs>
        <marker id="bt-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#475569" />
        </marker>
        <marker id="bt-arr-emerald" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#047857" />
        </marker>
        <marker id="bt-arr-blue" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#1e40af" />
        </marker>
        <filter id="bt-shd" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.08"/>
        </filter>
      </defs>

      {/* TOP HALF — read path: Mart → DBSQL ODBC → parallel Radar */}

      {/* Workbench: Mart with partitions (top-left) */}
      <rect x="20" y="20" width="240" height="180" rx="12" fill="#ecfdf5" stroke="#10b981" strokeWidth="1.5" filter="url(#bt-shd)"/>
      <text x="140" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#064e3b">Pricing Workbench</text>
      <text x="140" y="58" textAnchor="middle" fontSize="9" fill="#047857">1 · Modelling Mart partitioned</text>
      {[
        { y: 75, label: 'partition A · 100k policies' },
        { y: 100, label: 'partition B · 100k policies' },
        { y: 125, label: 'partition C · 100k policies' },
        { y: 150, label: '… · partition J' },
      ].map(p => (
        <g key={p.y}>
          <rect x="40" y={p.y} width="200" height="20" rx="4" fill="white" stroke="#34d399"/>
          <text x="140" y={p.y + 14} textAnchor="middle" fontSize="9" fill="#065f46">{p.label}</text>
        </g>
      ))}
      <text x="140" y="187" textAnchor="middle" fontSize="9" fill="#047857" fontStyle="italic">deterministic split (hash · LOB · postcode)</text>

      {/* DBSQL warehouse (center-top) */}
      <rect x="320" y="50" width="220" height="120" rx="12" fill="#fefce8" stroke="#eab308" strokeWidth="1.5" filter="url(#bt-shd)"/>
      <text x="430" y="72" textAnchor="middle" fontSize="13" fontWeight="700" fill="#713f12">Databricks SQL warehouse</text>
      <text x="430" y="88" textAnchor="middle" fontSize="9" fill="#a16207">2 · Radar 5 ODBC reads partition slices</text>
      <rect x="340" y="100" width="180" height="55" rx="6" fill="white" stroke="#facc15"/>
      <text x="430" y="118" textAnchor="middle" fontSize="11" fontWeight="600" fill="#854d0e">UC governs the read</text>
      <text x="430" y="133" textAnchor="middle" fontSize="9" fill="#a16207">column masks · row filters</text>
      <text x="430" y="148" textAnchor="middle" fontSize="9" fill="#a16207">scoped service principal</text>

      {/* Radar runs (top-right, parallel) */}
      <rect x="600" y="20" width="320" height="200" rx="12" fill="#eff6ff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#bt-shd)"/>
      <text x="760" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#1e3a8a">Radar 5 · N parallel licences</text>
      <text x="760" y="58" textAnchor="middle" fontSize="9" fill="#2563eb">3 · each instance handles one partition</text>
      {[
        { y: 70,  id: 'A' },
        { y: 105, id: 'B' },
        { y: 140, id: 'C' },
        { y: 175, id: 'J' },
      ].map(r => (
        <g key={r.id}>
          <rect x="620" y={r.y} width="280" height="28" rx="4" fill="white" stroke="#60a5fa"/>
          <text x="760" y={r.y + 18} textAnchor="middle" fontSize="10" fill="#1e3a8a">
            Radar instance · partition {r.id}
          </text>
        </g>
      ))}

      {/* Read arrows: Mart → DBSQL → Radar instances */}
      <line x1="260" y1="105" x2="318" y2="105" stroke="#047857" strokeWidth="2.2" markerEnd="url(#bt-arr-emerald)"/>
      {[84, 119, 154, 189].map((y, i) => (
        <line key={i} x1="540" y1="120" x2="618" y2={y} stroke="#047857" strokeWidth="1.5" markerEnd="url(#bt-arr-emerald)"/>
      ))}
      <text x="290" y="93" textAnchor="middle" fontSize="9" fontWeight="600" fill="#047857">ODBC SELECT</text>
      <text x="578" y="50" textAnchor="middle" fontSize="9" fontWeight="600" fill="#047857">fan out</text>

      {/* Divider */}
      <line x1="40" y1="240" x2="900" y2="240" stroke="#cbd5e1" strokeDasharray="4 4"/>
      <text x="470" y="234" textAnchor="middle" fontSize="9" fill="#64748b" fontStyle="italic">read leg · ODBC against governed DBSQL</text>
      <text x="470" y="252" textAnchor="middle" fontSize="9" fill="#64748b" fontStyle="italic">write leg · files + Autoloader</text>

      {/* BOTTOM HALF — write path: Radar → UC Volume → Autoloader → Delta */}

      {/* Radar instances writing files (bottom-right reuse) */}
      <rect x="600" y="270" width="320" height="80" rx="12" fill="#eff6ff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#bt-shd)"/>
      <text x="760" y="292" textAnchor="middle" fontSize="13" fontWeight="700" fill="#1e3a8a">Radar 5 · result writers</text>
      <text x="760" y="308" textAnchor="middle" fontSize="9" fill="#2563eb">4 · each writes a Parquet/CSV file</text>
      <text x="760" y="334" textAnchor="middle" fontSize="9" fill="#2563eb">no warehouse contention · finishes independently</text>

      {/* UC Volume (center-bottom) */}
      <rect x="320" y="270" width="220" height="80" rx="12" fill="#fff7ed" stroke="#fb923c" strokeWidth="1.5" filter="url(#bt-shd)"/>
      <text x="430" y="292" textAnchor="middle" fontSize="13" fontWeight="700" fill="#9a3412">UC Volume</text>
      <text x="430" y="308" textAnchor="middle" fontSize="9" fill="#c2410c">/results/scenario_X/partition_*.parquet</text>
      <text x="430" y="334" textAnchor="middle" fontSize="9" fill="#c2410c">scoped path · UC Volume ACLs</text>

      {/* Autoloader → Delta (bottom-left) */}
      <rect x="20" y="270" width="240" height="180" rx="12" fill="#ecfdf5" stroke="#10b981" strokeWidth="1.5" filter="url(#bt-shd)"/>
      <text x="140" y="292" textAnchor="middle" fontSize="13" fontWeight="700" fill="#064e3b">Pricing Workbench</text>
      <text x="140" y="308" textAnchor="middle" fontSize="9" fill="#047857">5 · Autoloader stream → Delta</text>
      <rect x="40" y="320" width="200" height="36" rx="6" fill="white" stroke="#34d399"/>
      <text x="140" y="335" textAnchor="middle" fontSize="10" fontWeight="600" fill="#065f46">Autoloader</text>
      <text x="140" y="349" textAnchor="middle" fontSize="9" fill="#047857">exactly-once · schema evolution</text>
      <rect x="40" y="365" width="200" height="36" rx="6" fill="white" stroke="#34d399"/>
      <text x="140" y="380" textAnchor="middle" fontSize="10" fontWeight="600" fill="#065f46">Delta results table</text>
      <text x="140" y="394" textAnchor="middle" fontSize="9" fill="#047857">scenario_id × partition_id</text>
      <rect x="40" y="410" width="200" height="32" rx="6" fill="white" stroke="#34d399"/>
      <text x="140" y="424" textAnchor="middle" fontSize="10" fontWeight="600" fill="#065f46">6 · Compare · pack</text>
      <text x="140" y="437" textAnchor="middle" fontSize="9" fill="#047857">dashboards · Genie · governance</text>

      {/* Write arrows: Radar → UC Volume → Autoloader */}
      <line x1="600" y1="310" x2="542" y2="310" stroke="#1e40af" strokeWidth="2.2" markerEnd="url(#bt-arr-blue)"/>
      <text x="571" y="302" textAnchor="middle" fontSize="9" fontWeight="600" fill="#1e40af">files</text>
      <line x1="320" y1="310" x2="262" y2="335" stroke="#1e40af" strokeWidth="2.2" markerEnd="url(#bt-arr-blue)"/>
      <text x="290" y="304" textAnchor="middle" fontSize="9" fontWeight="600" fill="#1e40af">stream</text>
    </svg>
  );
}

// ===========================================================================
// PATTERN 3 — Live (with sub-tabs)
// ===========================================================================

type LiveSub = 'enrichment' | 'callback';

function LivePattern() {
  const [sub, setSub] = useState<LiveSub>('enrichment');
  return (
    <div>
      <div className="bg-white rounded-lg border border-gray-200 p-1 mb-4 inline-flex gap-1">
        <SubTab active={sub === 'enrichment'} onClick={() => setSub('enrichment')}
                label="3a · Workbench-fronts-Radar"
                sub="workbench enriches → Radar prices"/>
        <SubTab active={sub === 'callback'} onClick={() => setSub('callback')}
                label="3b · Radar-calls-Workbench"
                sub="Radar pulls enrichment at quote time"/>
      </div>
      {sub === 'enrichment' && <LiveEnrichmentPattern />}
      {sub === 'callback'   && <LiveCallbackPattern />}
    </div>
  );
}

function SubTab({ active, onClick, label, sub }: { active: boolean; onClick: () => void; label: string; sub: string }) {
  return (
    <button onClick={onClick}
            className={`px-3 py-1.5 rounded-md text-sm font-medium transition flex flex-col items-start leading-tight ${
              active ? 'bg-blue-600 text-white' : 'text-gray-700 hover:bg-gray-100'
            }`}>
      <span>{label}</span>
      <span className={`text-[10px] ${active ? 'text-blue-100' : 'text-gray-500'}`}>{sub}</span>
    </button>
  );
}

function LiveEnrichmentPattern() {
  return (
    <PatternFrame
      title="Pattern 3a · Workbench-fronts-Radar (live)"
      subtitle="Quote arrives at the workbench, gets enriched, the workbench then calls Radar for the final price. Single API surface to broker / aggregator."
      svg={<LiveEnrichmentDiagram />}
      steps={[
        { n: 1, label: 'Quote arrives', body: 'Broker / aggregator hits a single Workbench API endpoint with a sparse quote (policy attributes, sums insured, postcode).' },
        { n: 2, label: 'Online enrichment', body: 'Workbench enriches from the Online Feature Store: postcode IMD, flood zone, market benchmark, credit signals, ML-predicted fraud / demand. <50 ms p50.' },
        { n: 3, label: 'Call Radar', body: 'Workbench POSTs the enriched feature bundle to the Radar pricing endpoint over a private network link with a scoped credential.' },
        { n: 4, label: 'Radar prices', body: 'Radar applies its rating tables / GLMs against the enriched bundle and returns the technical premium.' },
        { n: 5, label: 'Workbench logs + returns', body: 'Workbench writes the request, enriched features, and Radar response to inference_logs (Delta) and returns the price to the caller.' },
        { n: 6, label: 'Governance', body: 'Every quote is auditable end-to-end. Pack_id of the model used (or rating release) is stamped on the response. Drift + bias monitors run on the same Delta tables.' },
      ]}
      governance={[
        { icon: <Lock className="w-4 h-4 text-amber-700" />, label: 'Single API surface', body: 'Brokers see one endpoint. Workbench is the choke-point — auth, rate-limit, audit, replay all happen here. Radar stays internal.' },
        { icon: <Activity className="w-4 h-4 text-amber-700" />, label: 'Inference table', body: 'Every request → response cycle written to a Delta inference_logs table. Replay any quote at any historical model version.' },
        { icon: <Network className="w-4 h-4 text-amber-700" />, label: 'Private link to Radar', body: 'Workbench-to-Radar call goes over PrivateLink / VPC peering. No public egress. Radar credential rotated via UC Secrets.' },
        { icon: <Shield className="w-4 h-4 text-amber-700" />, label: 'Bias + adequacy live', body: 'Same inference_logs feeds the governance Monitor tab. Bias and premium-adequacy scans run continuously over real production traffic.' },
      ]}
    />
  );
}

function LiveEnrichmentDiagram() {
  return (
    <svg viewBox="0 0 920 320" className="w-full max-w-5xl" aria-label="Live enrichment architecture">
      <defs>
        <marker id="le-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#475569" />
        </marker>
        <filter id="le-shd" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.08"/>
        </filter>
      </defs>

      {/* Broker (far left) */}
      <rect x="20" y="100" width="140" height="120" rx="10" fill="#fdf4ff" stroke="#a855f7" filter="url(#le-shd)"/>
      <text x="90" y="135" textAnchor="middle" fontSize="12" fontWeight="700" fill="#6b21a8">Broker / aggregator</text>
      <text x="90" y="155" textAnchor="middle" fontSize="10" fill="#7e22ce">single API call</text>
      <text x="90" y="195" textAnchor="middle" fontSize="9" fill="#7e22ce">final price returned</text>

      {/* Workbench (center) */}
      <rect x="200" y="20" width="380" height="280" rx="12" fill="#ecfdf5" stroke="#10b981" strokeWidth="1.5" filter="url(#le-shd)"/>
      <text x="390" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#064e3b">Pricing Workbench</text>
      <text x="390" y="58" textAnchor="middle" fontSize="9" fill="#047857">single API surface · auth · audit</text>

      <rect x="220" y="80" width="340" height="38" rx="6" fill="white" stroke="#34d399"/>
      <text x="390" y="98" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">1 · Quote endpoint (Mosaic AI Model Serving)</text>
      <text x="390" y="111" textAnchor="middle" fontSize="9" fill="#047857">FastAPI · OAuth · per-broker rate limit</text>

      <rect x="220" y="130" width="340" height="38" rx="6" fill="white" stroke="#34d399"/>
      <text x="390" y="148" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">2 · Online Feature Store lookup</text>
      <text x="390" y="161" textAnchor="middle" fontSize="9" fill="#047857">~50 features · p50 38 ms · p99 92 ms</text>

      <rect x="220" y="180" width="340" height="38" rx="6" fill="white" stroke="#34d399"/>
      <text x="390" y="198" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">2 · ML scores (freq · sev · demand · fraud)</text>
      <text x="390" y="211" textAnchor="middle" fontSize="9" fill="#047857">Mosaic AI Model Serving · champion alias</text>

      <rect x="220" y="230" width="340" height="38" rx="6" fill="white" stroke="#34d399"/>
      <text x="390" y="248" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">5 · Inference table (Delta)</text>
      <text x="390" y="261" textAnchor="middle" fontSize="9" fill="#047857">request · enrichment · response · pack_id</text>

      {/* Radar (right) */}
      <rect x="630" y="100" width="270" height="120" rx="10" fill="#eff6ff" stroke="#3b82f6" filter="url(#le-shd)"/>
      <text x="765" y="125" textAnchor="middle" fontSize="13" fontWeight="700" fill="#1e3a8a">Radar pricing endpoint</text>
      <text x="765" y="145" textAnchor="middle" fontSize="10" fill="#2563eb">4 · applies rating tables / GLMs</text>
      <text x="765" y="165" textAnchor="middle" fontSize="10" fill="#2563eb">returns technical premium</text>
      <text x="765" y="195" textAnchor="middle" fontSize="9" fill="#2563eb">private network · scoped cred</text>

      {/* Arrows */}
      <line x1="160" y1="160" x2="218" y2="160" stroke="#475569" strokeWidth="2" markerEnd="url(#le-arr)"/>
      <line x1="218" y1="160" x2="160" y2="180" stroke="#475569" strokeWidth="2" strokeDasharray="3 3" markerEnd="url(#le-arr)"/>

      {/* Workbench → Radar (step 3) */}
      <line x1="560" y1="160" x2="628" y2="160" stroke="#475569" strokeWidth="2" markerEnd="url(#le-arr)"/>
      <text x="594" y="152" textAnchor="middle" fontSize="9" fontWeight="600" fill="#475569">3 · enriched bundle</text>
      <line x1="628" y1="180" x2="560" y2="180" stroke="#475569" strokeWidth="2" strokeDasharray="3 3" markerEnd="url(#le-arr)"/>
      <text x="594" y="195" textAnchor="middle" fontSize="9" fontWeight="600" fill="#475569">premium</text>
    </svg>
  );
}

function LiveCallbackPattern() {
  return (
    <PatternFrame
      title="Pattern 3b · Radar-calls-Workbench (live)"
      subtitle="Radar stays the broker-facing entry point. During price generation Radar calls the workbench for ML scores and external enrichment, then prices."
      svg={<LiveCallbackDiagram />}
      steps={[
        { n: 1, label: 'Quote arrives at Radar', body: 'Existing Radar entry point — broker / aggregator API or in-engine quote flow. No change to Radar\'s surface.' },
        { n: 2, label: 'Radar requests enrichment', body: 'Mid-rating, Radar fires a REST call to the Workbench Model Serving endpoint with the policy bundle.' },
        { n: 3, label: 'Workbench enriches + scores', body: 'Online Feature Store lookup → Mosaic AI Model Serving → returns enriched features + ML predictions (freq/sev/demand/fraud) in <100 ms.' },
        { n: 4, label: 'Radar applies rating tables', body: 'Radar continues its rating cycle with the enriched inputs as additional rating factors, applies relativities, returns the price.' },
        { n: 5, label: 'Workbench logs the call', body: 'Every enrichment call recorded to inference_logs — same Delta surface as Pattern 3a. Bias / drift / adequacy monitors run on this.' },
        { n: 6, label: 'Governance', body: 'Each enrichment response carries a pack_id / model version. Radar audit log stores it; regulator query reaches the workbench pack from the engine\'s rate.' },
      ]}
      governance={[
        { icon: <Lock className="w-4 h-4 text-amber-700" />, label: 'Workbench as service', body: 'Workbench is a Model Serving endpoint behind OAuth + service-principal auth. Radar can\'t see beyond what\'s served — no row-level access to the mart.' },
        { icon: <Activity className="w-4 h-4 text-amber-700" />, label: 'Inference table', body: 'Same Delta inference_logs as Pattern 3a — replay, drift, bias, adequacy all work identically regardless of which side initiates the call.' },
        { icon: <Network className="w-4 h-4 text-amber-700" />, label: 'Network', body: 'PrivateLink between Radar and Databricks workspace. No internet exposure. Credential rotation via UC Secrets / Databricks-managed identity.' },
        { icon: <FileCheck2 className="w-4 h-4 text-amber-700" />, label: 'Pack-id binding', body: 'Each enrichment response embeds the pack_id of the model used. Radar logs it on the rate; auditors trace from Radar\'s rate-card back to the workbench in two clicks.' },
      ]}
    />
  );
}

function LiveCallbackDiagram() {
  return (
    <svg viewBox="0 0 920 320" className="w-full max-w-5xl" aria-label="Live callback architecture">
      <defs>
        <marker id="lc-arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
          <path d="M0,0 L10,5 L0,10 z" fill="#475569" />
        </marker>
        <filter id="lc-shd" x="-10%" y="-10%" width="120%" height="120%">
          <feDropShadow dx="0" dy="2" stdDeviation="2" floodOpacity="0.08"/>
        </filter>
      </defs>

      {/* Broker */}
      <rect x="20" y="100" width="140" height="120" rx="10" fill="#fdf4ff" stroke="#a855f7" filter="url(#lc-shd)"/>
      <text x="90" y="135" textAnchor="middle" fontSize="12" fontWeight="700" fill="#6b21a8">Broker / aggregator</text>
      <text x="90" y="155" textAnchor="middle" fontSize="10" fill="#7e22ce">existing Radar API</text>
      <text x="90" y="195" textAnchor="middle" fontSize="9" fill="#7e22ce">price returned</text>

      {/* Radar (center) */}
      <rect x="200" y="20" width="380" height="280" rx="12" fill="#eff6ff" stroke="#3b82f6" strokeWidth="1.5" filter="url(#lc-shd)"/>
      <text x="390" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#1e3a8a">Radar (broker-facing)</text>
      <text x="390" y="58" textAnchor="middle" fontSize="9" fill="#2563eb">pricing engine · rating tables</text>

      <rect x="220" y="80" width="340" height="38" rx="6" fill="white" stroke="#60a5fa"/>
      <text x="390" y="98" textAnchor="middle" fontSize="11" fontWeight="600" fill="#1e3a8a">1 · Quote arrives at Radar</text>
      <text x="390" y="111" textAnchor="middle" fontSize="9" fill="#2563eb">unchanged broker integration</text>

      <rect x="220" y="130" width="340" height="38" rx="6" fill="white" stroke="#60a5fa"/>
      <text x="390" y="148" textAnchor="middle" fontSize="11" fontWeight="600" fill="#1e3a8a">2 · Mid-rating REST call out</text>
      <text x="390" y="161" textAnchor="middle" fontSize="9" fill="#2563eb">scoped service principal · PrivateLink</text>

      <rect x="220" y="230" width="340" height="38" rx="6" fill="white" stroke="#60a5fa"/>
      <text x="390" y="248" textAnchor="middle" fontSize="11" fontWeight="600" fill="#1e3a8a">4 · Apply rating tables → price</text>
      <text x="390" y="261" textAnchor="middle" fontSize="9" fill="#2563eb">enriched factors as new inputs</text>

      {/* Workbench (right) */}
      <rect x="630" y="20" width="270" height="280" rx="12" fill="#ecfdf5" stroke="#10b981" strokeWidth="1.5" filter="url(#lc-shd)"/>
      <text x="765" y="42" textAnchor="middle" fontSize="13" fontWeight="700" fill="#064e3b">Workbench API</text>
      <text x="765" y="58" textAnchor="middle" fontSize="9" fill="#047857">Mosaic AI Model Serving</text>

      <rect x="650" y="80" width="230" height="42" rx="6" fill="white" stroke="#34d399"/>
      <text x="765" y="99" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">3 · Online enrichment</text>
      <text x="765" y="113" textAnchor="middle" fontSize="9" fill="#047857">Online FS · ~50 features</text>

      <rect x="650" y="135" width="230" height="42" rx="6" fill="white" stroke="#34d399"/>
      <text x="765" y="154" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">3 · ML scores</text>
      <text x="765" y="168" textAnchor="middle" fontSize="9" fill="#047857">freq · sev · demand · fraud</text>

      <rect x="650" y="190" width="230" height="42" rx="6" fill="white" stroke="#34d399"/>
      <text x="765" y="209" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">5 · Inference table</text>
      <text x="765" y="223" textAnchor="middle" fontSize="9" fill="#047857">Delta · pack_id stamped on each row</text>

      <rect x="650" y="245" width="230" height="42" rx="6" fill="white" stroke="#34d399"/>
      <text x="765" y="264" textAnchor="middle" fontSize="11" fontWeight="600" fill="#065f46">drift / bias / adequacy</text>
      <text x="765" y="278" textAnchor="middle" fontSize="9" fill="#047857">live monitors, same data</text>

      {/* Arrows */}
      <line x1="160" y1="160" x2="218" y2="160" stroke="#475569" strokeWidth="2" markerEnd="url(#lc-arr)"/>
      <line x1="218" y1="180" x2="160" y2="180" stroke="#475569" strokeWidth="2" strokeDasharray="3 3" markerEnd="url(#lc-arr)"/>

      {/* Radar → Workbench (step 2) */}
      <line x1="560" y1="148" x2="648" y2="148" stroke="#475569" strokeWidth="2" markerEnd="url(#lc-arr)"/>
      <text x="604" y="140" textAnchor="middle" fontSize="9" fontWeight="600" fill="#475569">2 · enrichment request</text>
      <line x1="648" y1="170" x2="560" y2="170" stroke="#475569" strokeWidth="2" strokeDasharray="3 3" markerEnd="url(#lc-arr)"/>
      <text x="604" y="186" textAnchor="middle" fontSize="9" fontWeight="600" fill="#475569">3 · features + scores</text>
    </svg>
  );
}

// ===========================================================================
// Shared frame: diagram + steps + governance side-panel
// ===========================================================================

type Step = { n: number; label: string; body: string };
type GovItem = { icon: React.ReactNode; label: string; body: string };

function PatternFrame({ title, subtitle, svg, steps, governance }: {
  title: string; subtitle: string;
  svg: React.ReactNode;
  steps: Step[]; governance: GovItem[];
}) {
  return (
    <div className="space-y-5">
      <section className="bg-white border border-gray-200 rounded-lg p-5">
        <div className="mb-3">
          <h3 className="text-base font-semibold text-gray-900">{title}</h3>
          <p className="text-sm text-gray-600 mt-0.5">{subtitle}</p>
        </div>
        <div className="flex justify-center overflow-x-auto">
          {svg}
        </div>
      </section>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-5">
        <section className="lg:col-span-7 bg-white border border-gray-200 rounded-lg p-5">
          <h4 className="text-sm font-semibold text-gray-900 uppercase tracking-wide mb-3 flex items-center gap-1.5">
            <Layers className="w-4 h-4 text-emerald-600" /> Steps
          </h4>
          <ol className="space-y-3">
            {steps.map(s => (
              <li key={s.n} className="flex items-start gap-3">
                <span className="font-mono text-[11px] w-6 h-6 rounded-full bg-emerald-100 text-emerald-800 border border-emerald-200 inline-flex items-center justify-center shrink-0 font-bold">
                  {s.n}
                </span>
                <div>
                  <div className="text-sm font-semibold text-gray-900">{s.label}</div>
                  <div className="text-xs text-gray-700 leading-relaxed">{s.body}</div>
                </div>
              </li>
            ))}
          </ol>
        </section>

        <section className="lg:col-span-5 bg-amber-50 border border-amber-200 rounded-lg p-5">
          <h4 className="text-sm font-semibold text-amber-900 uppercase tracking-wide mb-3 flex items-center gap-1.5">
            <Shield className="w-4 h-4 text-amber-700" /> Security &amp; governance
          </h4>
          <p className="text-xs text-amber-900/80 mb-3">
            All four governance pillars apply by default — Unity Catalog, audit log,
            governance packs, network controls. Specifics for this pattern:
          </p>
          <ul className="space-y-3">
            {governance.map((g, i) => (
              <li key={i} className="flex items-start gap-2.5">
                <div className="mt-0.5 shrink-0">{g.icon}</div>
                <div>
                  <div className="text-sm font-semibold text-amber-900">{g.label}</div>
                  <div className="text-xs text-amber-900/90 leading-relaxed">{g.body}</div>
                </div>
              </li>
            ))}
          </ul>
        </section>
      </div>
    </div>
  );
}

// ===========================================================================
// STANDARDS — Radar integration standards (connector, sizing, live-path SLOs)
// ===========================================================================

function StandardsTab() {
  return (
    <div className="space-y-6">
      <div>
        <h3 className="text-lg font-bold text-gray-900">Radar integration standards</h3>
        <p className="text-sm text-gray-600 mt-1 max-w-3xl">
          The engineering standards behind the three patterns: how the connector is set up,
          how batch topologies are sized, and what the live quote path must guarantee.
          Every standard maps to a Databricks-governed control — nothing here changes
          the actuary's Radar experience.
        </p>
      </div>

      {/* The connector */}
      <section className="bg-white rounded-lg border border-gray-200 p-5">
        <h4 className="text-sm font-bold text-gray-900 mb-3 flex items-center gap-2">
          <Cpu className="w-4 h-4 text-emerald-600" /> The connector — WTW-supported, not bespoke
        </h4>
        <div className="grid md:grid-cols-2 gap-3">
          {[
            { t: 'Native in Radar v5+', b: 'The Databricks connector is built into Radar version 5 and above, fully maintained and supported by WTW as part of the Radar product — not a Databricks add-on or a custom bridge. Setup documentation is on the WTW client portal; any Radar 5+ customer already has access.' },
            { t: 'One bidirectional ODBC DSN', b: 'Radar reads the modelling dataset from, and writes factor tables / model objects back to, a governed Databricks SQL warehouse over a single DSN. No file shuffling, no extract maze.' },
            { t: 'Scoped service principal', b: 'The DSN authenticates as a service principal with grants only on the modelling mart and the result schema. Unity Catalog enforces row filters and column/PII masks at query time — Radar never sees more than it is granted.' },
            { t: 'Private networking', b: 'The warehouse is reached over PrivateLink — no public endpoint; Radar connects from inside the customer VPC. Credentials rotate via UC Secrets.' },
          ].map((x) => (
            <div key={x.t} className="rounded-md border border-gray-200 bg-gray-50 p-3">
              <div className="text-sm font-semibold text-gray-900">{x.t}</div>
              <div className="text-xs text-gray-600 leading-relaxed mt-1">{x.b}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Choosing an entry point */}
      <section className="bg-white rounded-lg border border-gray-200 p-5">
        <h4 className="text-sm font-bold text-gray-900 mb-3 flex items-center gap-2">
          <Layers className="w-4 h-4 text-emerald-600" /> Choosing an entry point
        </h4>
        <p className="text-xs text-gray-600 mb-3 max-w-3xl">
          The three patterns are standalone — an estate rarely needs all three on day one.
          Each closes a specific gap in a typical Radar setup, and all are governed end-to-end
          in Unity Catalog:
        </p>
        <div className="grid md:grid-cols-3 gap-3">
          {[
            { t: '1 · Build', gap: 'Modelling data lives in spreadsheets and extracts; evidencing how a rate was built is "trust us."', std: 'Governed lineage + weeks-to-hours data prep, zero change to the Radar experience. Lowest-friction entry point.' },
            { t: '2 · Batch what-if', gap: 'Radar runs one scenario at a time; a full-book what-if takes days with nowhere governed to compare results.', std: 'Partition the book, fan out across parallel Radar licences, gather into one governed comparison surface. Ten scenarios overnight, not over a fortnight.' },
            { t: '3 · Enrich live', gap: 'Live quotes see only the rating tables — no external enrichment, no ML signals, no replayable record of why a price was set.', std: 'Highest-value pattern: live enrichment + ML scores in the quote path with a full audit trail. Natural endgame after 1–2 build trust.' },
          ].map((x) => (
            <div key={x.t} className="rounded-md border border-emerald-200 bg-emerald-50/50 p-3">
              <div className="text-sm font-semibold text-emerald-900">{x.t}</div>
              <div className="text-[11px] text-gray-600 mt-1"><span className="font-semibold">Gap it closes:</span> {x.gap}</div>
              <div className="text-[11px] text-emerald-900 mt-1.5"><span className="font-semibold">Standard:</span> {x.std}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Batch sizing regimes */}
      <section className="bg-white rounded-lg border border-gray-200 p-5">
        <h4 className="text-sm font-bold text-gray-900 mb-3 flex items-center gap-2">
          <GitBranch className="w-4 h-4 text-emerald-600" /> Batch what-if — topology sizing standard
        </h4>
        <p className="text-xs text-gray-600 mb-3 max-w-3xl">
          There is no single right topology — it depends on volume, licence count, and Radar's
          writeback configuration. Size against these three regimes; anything beyond a small POC
          gets an architecture review.
        </p>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs uppercase tracking-wider text-gray-500 border-b border-gray-200">
                <th className="py-2 pr-4">Regime</th>
                <th className="py-2 pr-4">When</th>
                <th className="py-2">Trade-off</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-100">
              <tr>
                <td className="py-2.5 pr-4 font-semibold text-gray-900 whitespace-nowrap">ODBC in &amp; out</td>
                <td className="py-2.5 pr-4 text-gray-700 whitespace-nowrap">~10k rows/partition · few scenarios</td>
                <td className="py-2.5 text-gray-600">Simplest ops; writeback contention slows it above ~50k rows.</td>
              </tr>
              <tr className="bg-emerald-50/50">
                <td className="py-2.5 pr-4 font-semibold text-emerald-900 whitespace-nowrap">Hybrid — ODBC in, files out <span className="text-[10px] font-bold uppercase text-emerald-700 ml-1">default</span></td>
                <td className="py-2.5 pr-4 text-gray-700 whitespace-nowrap">~100k rows/partition · 4–10 licences</td>
                <td className="py-2.5 text-gray-600">Fast governed reads + linear-scaling file writes with exactly-once ingestion (Autoloader). The sweet spot.</td>
              </tr>
              <tr>
                <td className="py-2.5 pr-4 font-semibold text-gray-900 whitespace-nowrap">Files in &amp; out</td>
                <td className="py-2.5 pr-4 text-gray-700 whitespace-nowrap">~1M rows/partition · 10+ licences</td>
                <td className="py-2.5 text-gray-600">Max parallelism, no warehouse load; two file legs, read governance via Volume ACLs.</td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="text-[11px] text-gray-500 mt-3">
          Results land keyed <code className="bg-gray-100 px-1 rounded">scenario_id × partition_id</code> in Delta;
          dashboards + Genie compare premium delta, loss-ratio impact and fairness; the chosen
          scenario gets a governance pack.
        </p>
      </section>

      {/* Live path standards */}
      <section className="bg-white rounded-lg border border-gray-200 p-5">
        <h4 className="text-sm font-bold text-gray-900 mb-3 flex items-center gap-2">
          <Activity className="w-4 h-4 text-emerald-600" /> Live quote path — what the integration must guarantee
        </h4>
        <div className="grid md:grid-cols-2 gap-3">
          {[
            { t: 'Latency budgets', b: 'Workbench-fronts-Radar (3a): enrichment + 4-model scoring in <50 ms p50 before the bundle is POSTed to Radar. Radar-calls-Workbench (3b): features + scores returned in <100 ms so the mid-rating callout never threatens the aggregator SLA.' },
            { t: 'Payload recording', b: 'Every request → enrichment → response is written to a Delta inference_logs table. Replay any historical quote at any historical model version — the durable "why was I charged this?" record fair-value regulation demands.' },
            { t: 'Pack-id binding', b: 'Each response carries the model version used, stamped on the row and echoed into Radar\'s audit log. An auditor traces a broker\'s rate card back to the governed model in two clicks.' },
            { t: 'Live monitoring of the rate', b: 'The same inference table feeds continuous drift, bias and premium-adequacy monitors on real production traffic — not offline samples.' },
          ].map((x) => (
            <div key={x.t} className="rounded-md border border-gray-200 bg-gray-50 p-3">
              <div className="text-sm font-semibold text-gray-900">{x.t}</div>
              <div className="text-xs text-gray-600 leading-relaxed mt-1">{x.b}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Positioning line */}
      <section className="rounded-lg border border-emerald-300 bg-emerald-600 p-5 text-white">
        <div className="text-[10px] uppercase tracking-wider font-bold text-emerald-100 mb-1">The standard in one line</div>
        <p className="text-sm font-medium leading-relaxed">
          Integrate Radar: bring it to where the data is, help it scale, wrap it in audit and
          governance, and complement it with modelling in Databricks — all over WTW's own
          supported v5+ connector, with zero change to the actuary's workflow.
        </p>
      </section>
    </div>
  );
}
