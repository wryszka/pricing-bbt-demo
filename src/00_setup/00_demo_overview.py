# Databricks notebook source
# MAGIC %md
# MAGIC # Pricing Data Transformation — Demo Overview
# MAGIC
# MAGIC **Bricksurance SE** — Commercial P&C Insurance Pricing on Databricks
# MAGIC
# MAGIC This accelerator demonstrates end-to-end pricing data transformation:
# MAGIC from raw vendor data to live pricing decisions, with full governance,
# MAGIC human-in-the-loop approval, and regulatory-grade auditability.
# MAGIC
# MAGIC **Target audience:** Heads of Pricing, Chief Actuaries, Big4 Consultants,
# MAGIC Pricing Engineers, Data Science teams.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Architecture Overview
# MAGIC
# MAGIC The diagram below shows the complete data and model lifecycle.
# MAGIC Every coloured layer is a Databricks-native capability — no third-party
# MAGIC tools or custom infrastructure required.

# COMMAND ----------

displayHTML("""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #f8fafc; padding: 20px; border-radius: 12px; border: 1px solid #e2e8f0;">

<h3 style="text-align:center; color:#1e293b; margin:0 0 8px 0; font-size:18px;">End-to-End Pricing Architecture</h3>
<p style="text-align:center; color:#64748b; margin:0 0 16px 0; font-size:12px;">
  Complete data flow from external sources to live pricing decisions. Every step is governed, versioned, and auditable.
</p>

<svg viewBox="0 0 960 520" xmlns="http://www.w3.org/2000/svg" style="width:100%; max-width:960px; display:block; margin:auto;">
  <defs>
    <marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#94a3b8"/>
    </marker>
    <marker id="ah-blue" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6" fill="#3b82f6"/>
    </marker>
    <filter id="shadow"><feDropShadow dx="1" dy="1" stdDeviation="2" flood-opacity="0.1"/></filter>
  </defs>

  <!-- ═══ LAYER LABELS (left side) ═══ -->
  <text x="12" y="65" font-size="10" fill="#94a3b8" font-weight="600" transform="rotate(-90,12,65)">INGEST</text>
  <text x="12" y="185" font-size="10" fill="#94a3b8" font-weight="600" transform="rotate(-90,12,185)">TRANSFORM</text>
  <text x="12" y="315" font-size="10" fill="#94a3b8" font-weight="600" transform="rotate(-90,12,315)">MODEL</text>
  <text x="12" y="445" font-size="10" fill="#94a3b8" font-weight="600" transform="rotate(-90,12,445)">SERVE</text>

  <!-- ═══ ROW 1: DATA INGESTION ═══ -->
  <!-- External Sources -->
  <rect x="30" y="20" width="130" height="65" rx="8" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="95" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#475569">External Sources</text>
  <text x="95" y="56" text-anchor="middle" font-size="8" fill="#94a3b8">PCW Feeds, Bureau,</text>
  <text x="95" y="66" text-anchor="middle" font-size="8" fill="#94a3b8">Geo Vendors, CSVs</text>

  <line x1="160" y1="52" x2="195" y2="52" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- Volume (Landing) -->
  <rect x="200" y="25" width="100" height="55" rx="8" fill="#fef3c7" stroke="#f59e0b" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="250" y="48" text-anchor="middle" font-size="10" font-weight="700" fill="#92400e">Volume</text>
  <text x="250" y="60" text-anchor="middle" font-size="8" fill="#b45309">Landing Zone</text>

  <line x1="300" y1="52" x2="335" y2="52" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- Bronze -->
  <rect x="340" y="25" width="100" height="55" rx="8" fill="#fde68a" stroke="#d97706" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="390" y="48" text-anchor="middle" font-size="10" font-weight="700" fill="#78350f">Bronze</text>
  <text x="390" y="60" text-anchor="middle" font-size="8" fill="#92400e">Raw (immutable)</text>

  <line x1="440" y1="52" x2="475" y2="52" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- DLT -->
  <rect x="480" y="20" width="90" height="65" rx="8" fill="#dbeafe" stroke="#3b82f6" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="525" y="42" text-anchor="middle" font-size="10" font-weight="700" fill="#1e40af">DLT</text>
  <text x="525" y="54" text-anchor="middle" font-size="8" fill="#3b82f6">Expectations</text>
  <text x="525" y="64" text-anchor="middle" font-size="8" fill="#3b82f6">& Cleansing</text>

  <line x1="570" y1="52" x2="605" y2="52" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- Silver -->
  <rect x="610" y="25" width="100" height="55" rx="8" fill="#e2e8f0" stroke="#94a3b8" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="660" y="48" text-anchor="middle" font-size="10" font-weight="700" fill="#475569">Silver</text>
  <text x="660" y="60" text-anchor="middle" font-size="8" fill="#64748b">Cleansed + DQ</text>

  <!-- Internal Systems -->
  <rect x="30" y="100" width="130" height="45" rx="8" fill="#f1f5f9" stroke="#cbd5e1" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="95" y="120" text-anchor="middle" font-size="10" font-weight="700" fill="#475569">Internal Systems</text>
  <text x="95" y="132" text-anchor="middle" font-size="8" fill="#94a3b8">Policies, Claims, Quotes</text>

  <line x1="160" y1="122" x2="605" y2="52" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4"/>

  <!-- ═══ HITL APPROVAL (between Silver and Gold) ═══ -->
  <rect x="730" y="15" width="110" height="75" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="785" y="36" text-anchor="middle" font-size="9" font-weight="700" fill="#991b1b">HITL Review</text>
  <text x="785" y="50" text-anchor="middle" font-size="8" fill="#dc2626">Approve / Reject</text>
  <text x="785" y="62" text-anchor="middle" font-size="8" fill="#dc2626">Impact Analysis</text>
  <text x="785" y="74" text-anchor="middle" font-size="8" fill="#dc2626">Upload / Download</text>

  <line x1="710" y1="52" x2="725" y2="52" stroke="#ef4444" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- ═══ ROW 2: GOLD (UPT) ═══ -->
  <rect x="340" y="160" width="260" height="65" rx="8" fill="#fef9c3" stroke="#ca8a04" stroke-width="2" filter="url(#shadow)"/>
  <text x="470" y="182" text-anchor="middle" font-size="12" font-weight="700" fill="#713f12">Unified Pricing Table (Gold)</text>
  <text x="470" y="196" text-anchor="middle" font-size="9" fill="#a16207">50K policies × 90+ features | PK: policy_id</text>
  <text x="470" y="208" text-anchor="middle" font-size="8" fill="#ca8a04">Feature Table in Unity Catalog</text>

  <line x1="660" y1="80" x2="470" y2="155" stroke="#94a3b8" stroke-width="1.5" marker-end="url(#ah)"/>
  <line x1="95" y1="145" x2="340" y2="185" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4"/>

  <!-- ═══ ROW 3: MODELS ═══ -->
  <!-- Training -->
  <rect x="80" y="270" width="120" height="55" rx="8" fill="#ede9fe" stroke="#8b5cf6" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="140" y="292" text-anchor="middle" font-size="10" font-weight="700" fill="#5b21b6">Model Training</text>
  <text x="140" y="304" text-anchor="middle" font-size="8" fill="#7c3aed">GLM, GBM, Fraud,</text>
  <text x="140" y="314" text-anchor="middle" font-size="8" fill="#7c3aed">Retention (6 models)</text>

  <line x1="470" y1="225" x2="140" y2="265" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#ah-blue)"/>

  <!-- MLflow -->
  <rect x="240" y="270" width="100" height="55" rx="8" fill="#ede9fe" stroke="#8b5cf6" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="290" y="292" text-anchor="middle" font-size="10" font-weight="700" fill="#5b21b6">MLflow</text>
  <text x="290" y="306" text-anchor="middle" font-size="8" fill="#7c3aed">Experiments,</text>
  <text x="290" y="316" text-anchor="middle" font-size="8" fill="#7c3aed">Tracking, Artifacts</text>

  <line x1="200" y1="297" x2="235" y2="297" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#ah-blue)"/>

  <!-- UC Model Registry -->
  <rect x="380" y="270" width="120" height="55" rx="8" fill="#ede9fe" stroke="#8b5cf6" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="440" y="290" text-anchor="middle" font-size="10" font-weight="700" fill="#5b21b6">UC Registry</text>
  <text x="440" y="304" text-anchor="middle" font-size="8" fill="#7c3aed">Versioned Models</text>
  <text x="440" y="314" text-anchor="middle" font-size="8" fill="#7c3aed">+ FeatureLookup</text>

  <line x1="340" y1="297" x2="375" y2="297" stroke="#8b5cf6" stroke-width="1.5" marker-end="url(#ah-blue)"/>

  <!-- Model HITL -->
  <rect x="540" y="270" width="110" height="55" rx="8" fill="#fef2f2" stroke="#ef4444" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="595" y="290" text-anchor="middle" font-size="9" font-weight="700" fill="#991b1b">Actuary Review</text>
  <text x="595" y="304" text-anchor="middle" font-size="8" fill="#dc2626">Approve / Reject</text>
  <text x="595" y="314" text-anchor="middle" font-size="8" fill="#dc2626">PDF Report</text>

  <line x1="500" y1="297" x2="535" y2="297" stroke="#ef4444" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- AI Agent (optional) -->
  <rect x="700" y="270" width="110" height="55" rx="8" fill="#f5f3ff" stroke="#a78bfa" stroke-width="1" stroke-dasharray="4" filter="url(#shadow)"/>
  <text x="755" y="290" text-anchor="middle" font-size="9" font-weight="600" fill="#7c3aed">AI Agent</text>
  <text x="755" y="302" text-anchor="middle" font-size="8" fill="#a78bfa">Model Selection</text>
  <text x="755" y="312" text-anchor="middle" font-size="7" fill="#a78bfa" font-style="italic">(optional)</text>

  <line x1="650" y1="297" x2="695" y2="297" stroke="#a78bfa" stroke-width="1" stroke-dasharray="4"/>

  <!-- ═══ ROW 4: SERVING ═══ -->
  <!-- Online Store -->
  <rect x="200" y="390" width="140" height="55" rx="8" fill="#dcfce7" stroke="#22c55e" stroke-width="1.5" filter="url(#shadow)"/>
  <text x="270" y="412" text-anchor="middle" font-size="10" font-weight="700" fill="#166534">Online Store</text>
  <text x="270" y="424" text-anchor="middle" font-size="8" fill="#16a34a">Lakebase (sub-10ms)</text>
  <text x="270" y="436" text-anchor="middle" font-size="8" fill="#16a34a">Auto-synced from Gold</text>

  <line x1="470" y1="225" x2="270" y2="385" stroke="#22c55e" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- Model Serving -->
  <rect x="400" y="390" width="160" height="55" rx="8" fill="#dcfce7" stroke="#22c55e" stroke-width="2" filter="url(#shadow)"/>
  <text x="480" y="410" text-anchor="middle" font-size="11" font-weight="700" fill="#166534">Model Serving</text>
  <text x="480" y="424" text-anchor="middle" font-size="8" fill="#16a34a">Auto Feature Lookup</text>
  <text x="480" y="436" text-anchor="middle" font-size="8" fill="#16a34a">Champion / Challenger</text>

  <line x1="440" y1="325" x2="480" y2="385" stroke="#22c55e" stroke-width="1.5" marker-end="url(#ah)"/>
  <line x1="340" y1="418" x2="395" y2="418" stroke="#22c55e" stroke-width="1" stroke-dasharray="4"/>

  <!-- External Platform (future) -->
  <rect x="620" y="390" width="150" height="55" rx="8" fill="#f8fafc" stroke="#94a3b8" stroke-width="1" stroke-dasharray="5" filter="url(#shadow)"/>
  <text x="695" y="410" text-anchor="middle" font-size="9" font-weight="600" fill="#64748b">External Platform</text>
  <text x="695" y="424" text-anchor="middle" font-size="8" fill="#94a3b8">Earnix / Radar / Custom</text>
  <text x="695" y="436" text-anchor="middle" font-size="7" fill="#94a3b8" font-style="italic">(optional integration)</text>

  <line x1="560" y1="418" x2="615" y2="418" stroke="#94a3b8" stroke-width="1" stroke-dasharray="4" marker-end="url(#ah)"/>

  <!-- REST API callout -->
  <rect x="400" y="460" width="160" height="30" rx="6" fill="#1e293b"/>
  <text x="480" y="479" text-anchor="middle" font-size="9" font-weight="600" fill="white">REST API: send policy_id → get price</text>

  <line x1="480" y1="445" x2="480" y2="455" stroke="#1e293b" stroke-width="1.5" marker-end="url(#ah)"/>

  <!-- ═══ GOVERNANCE BAR (bottom) ═══ -->
  <rect x="30" y="500" width="900" height="18" rx="4" fill="#1e293b"/>
  <text x="480" y="513" text-anchor="middle" font-size="9" font-weight="600" fill="white">
    GOVERNANCE: Unity Catalog Lineage │ Audit Log │ Delta Time Travel │ DQ Monitoring │ AI Gateway Logging
  </text>
</svg>

<!-- Legend -->
<div style="display:flex; gap:20px; justify-content:center; margin-top:12px; flex-wrap:wrap;">
  <span style="font-size:11px; color:#475569;"><span style="display:inline-block;width:12px;height:12px;background:#fde68a;border:1px solid #d97706;border-radius:3px;vertical-align:middle;margin-right:4px;"></span> Medallion (Bronze/Silver/Gold)</span>
  <span style="font-size:11px; color:#475569;"><span style="display:inline-block;width:12px;height:12px;background:#dbeafe;border:1px solid #3b82f6;border-radius:3px;vertical-align:middle;margin-right:4px;"></span> Databricks Platform</span>
  <span style="font-size:11px; color:#475569;"><span style="display:inline-block;width:12px;height:12px;background:#ede9fe;border:1px solid #8b5cf6;border-radius:3px;vertical-align:middle;margin-right:4px;"></span> ML / Models</span>
  <span style="font-size:11px; color:#475569;"><span style="display:inline-block;width:12px;height:12px;background:#fef2f2;border:1px solid #ef4444;border-radius:3px;vertical-align:middle;margin-right:4px;"></span> Human-in-the-Loop</span>
  <span style="font-size:11px; color:#475569;"><span style="display:inline-block;width:12px;height:12px;background:#dcfce7;border:1px solid #22c55e;border-radius:3px;vertical-align:middle;margin-right:4px;"></span> Real-time Serving</span>
  <span style="font-size:11px; color:#475569;"><span style="display:inline-block;width:12px;height:12px;background:#f8fafc;border:1px solid #94a3b8;border-radius:3px;vertical-align:middle;margin-right:4px;"></span> Optional / External</span>
</div>
</div>
""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Demo Components
# MAGIC
# MAGIC | # | Component | Notebooks | Status |
# MAGIC |---|---|---|---|
# MAGIC | 1 | **Data Ingestion** | `01_ingestion/` | External CSVs → Volume → Bronze |
# MAGIC | 2 | **Silver (DLT)** | `02_silver/` | DQ expectations, cleansing, derived features |
# MAGIC | 3 | **Gold (UPT)** | `03_gold/` | Wide join, synthetic features, PK, tags |
# MAGIC | 4 | **Model Training** | `04_models/` | GLM freq/sev, GBM demand/uplift/fraud/retention |
# MAGIC | 5 | **Use Cases** | `05_use_cases/` | Shadow pricing, PIT backtesting, new dataset, enriched pricing |
# MAGIC | 6 | **Model Factory** | `06_model_factory/` | Automated training, evaluation, actuary review |
# MAGIC | 7 | **Serving** | `07_serving/` | Online store, model endpoint, champion/challenger |
# MAGIC | 8 | **Governance** | `08_governance/` | Dashboard, regulatory export, audit trail |
# MAGIC | 9 | **HITL App** | `app/` | React + FastAPI: data review, model review, feature store, governance |
# MAGIC
# MAGIC ## How to run this demo
# MAGIC
# MAGIC 1. **Setup:** Run `src/00_setup/setup.py` — creates schema, tables, and test data
# MAGIC 2. **Ingest:** Run the `Ingest External Data → Raw → Silver` job
# MAGIC 3. **Build UPT:** Run the `Build Unified Pricing Table (Gold)` job
# MAGIC 4. **Train Models:** Run the `Train Pricing Models` job
# MAGIC 5. **Open App:** Navigate to the Databricks App for HITL review
# MAGIC 6. **Use Cases:** Run individual UC notebooks for shadow pricing, time travel, etc.
# MAGIC 7. **Serving:** Run `Setup Online Feature Store` then `Deploy Model Serving Endpoint`
# MAGIC
# MAGIC ## About this demo
# MAGIC
# MAGIC This is a synthetic demonstration built by the Databricks Field Engineering team.
# MAGIC All company names, policy data, and financial figures are entirely fictional.
# MAGIC No real customer data is used. The demo is designed to be redeployable on any
# MAGIC Databricks workspace with serverless compute enabled.
