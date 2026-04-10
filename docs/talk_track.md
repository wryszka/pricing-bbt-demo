# Demo Talk Track — P&C Insurance Pricing Accelerator

## Overview

**Company:** Bricksurance SE (fictional commercial property insurer)
**Platform:** Databricks with serverless compute
**Audience:** Heads of Pricing, Chief Actuaries, Pricing Engineers, Big4 Consultants

**Key message:** The full pricing lifecycle — from raw vendor data to live pricing
decisions — runs on one platform. Not just modelling, but ingestion, governance,
serving, and auditability.

---

## EXECUTIVE VERSION (30 minutes, app-focused)

### Opening (2 min)

**What to show:** Landing page of the app

**What to say:**
> "This is a pricing data transformation accelerator for commercial P&C insurance.
> What you're looking at is a single platform that handles everything from data
> ingestion to live model serving — with human-in-the-loop governance at every step.
> Let me walk you through what that looks like in practice."

**Talking point:** This isn't a prototype or a PowerPoint — every button works,
every table is real (synthetic) data, every model is trained and serving.

---

### Section 1: Data Ingestion & Impact Analysis (8 min)

**What to show:** Data Ingestion tab → click into Geospatial Hazard → Impact Analysis tab

**What to say:**
> "New flood risk data has arrived from a vendor. Before anyone touches the rating
> engine, the system has already joined this data to our 50,000 active policies
> and re-rated every single one. Watch this..."

**WOW MOMENT — Shadow Pricing:**
> "6,018 policies are affected. The total premium impact is £X. 342 policies
> face a premium increase of more than 10%. And the actuary can see all of this
> before clicking Approve."

**Expected questions:**
- "How long does this shadow pricing take?" → "Seconds. It runs automatically
  when new data arrives."
- "Can we override individual values?" → "Yes — the Upload tab lets you amend
  data, and the app tracks who changed what."
- "How does this compare to what we do now?" → "Most insurers spend weeks on
  this manually. This is the same analysis, automated, in seconds."

**Pause for questions.**

---

### Section 2: Model Factory & Approval (8 min)

**What to show:** Model Factory tab → leaderboard → click PDF for a model

**What to say:**
> "The system trained 20+ model configurations — GLMs for regulatory submission,
> gradient boosted machines for demand and fraud, plus an uplift model that
> captures what the GLM missed. They're ranked on insurance-specific metrics:
> Gini, PSI for stability, and a regulatory suitability score."

**WOW MOMENT — PDF Report:**
> "Click this PDF button — you get a regulatory-grade model validation report.
> Model identity, performance metrics, data lineage, approval chain. Ready for
> your regulator."

**Optional WOW — AI Agent:**
> "There's an optional AI assistant that can analyse the feature table and
> recommend which models to train. It's turned off by default — actuaries
> decide. But it shows what's possible."

**Expected questions:**
- "Can we use our own models?" → "Absolutely. The platform supports any Python
  model — statsmodels, sklearn, LightGBM, PyTorch."
- "How do you handle regulatory requirements?" → "GLMs provide transparent
  relativities. Everything is versioned and auditable."

**Pause for questions.**

---

### Section 3: Feature Store & Serving (5 min)

**What to show:** Feature Store tab → latency numbers → explain online vs offline

**What to say:**
> "The Unified Pricing Table serves two purposes. Offline, it's a Delta Lake
> table for model training. Online, it's a Lakebase store with sub-10ms lookups.
> Same data, different access pattern."

**WOW MOMENT — Auto Feature Lookup:**
> "When we deploy a model, you send just a policy_id and get a price back.
> The endpoint automatically looks up all 90 features from the online store.
> No custom integration code. The model knows which features it needs because
> we captured that lineage at training time."

**Expected questions:**
- "What's the latency?" → "Sub-100ms end-to-end including feature lookup."
- "How does this compare to Radar/Earnix?" → "Those tools are great at what
  they do. This extends the capability upstream — data preparation, feature
  engineering, model training — and provides a REST API they can call."

**Pause for questions.**

---

### Section 4: Governance (5 min)

**What to show:** Governance tab → audit trail → DQ pass rates

**What to say:**
> "Everything that happened — every data approval, every model decision, every
> LLM call — is recorded with who did it, when, and why. A regulatory auditor
> can reconstruct the exact state of any model, its training data, and the
> human decisions that approved it."

**WOW MOMENT — Full Audit Trail:**
> "This isn't a separate governance tool. It's the same platform. Unity Catalog
> tracks lineage automatically. The audit log adds the human decisions on top.
> Delta Time Travel lets you go back to any point in history."

**Closing:**
> "What you've seen is a complete pricing data transformation pipeline — from
> raw vendor data to live pricing — on one platform. Not six tools with six
> integration projects. One platform, fully governed, fully auditable."

---

### Closing & Future (2 min)

**Future roadmap talking points:**
- Earnix/Radar mock integration (REST API enrichment)
- ABAC/attribute-based access control across business units
- CI/CD with Databricks Asset Bundles
- Agentic data quality monitoring
- Real-time streaming (Delta CDF for continuous online store sync)

---

## TECHNICAL VERSION (60 minutes, notebooks + app)

### Opening (5 min)
Show the architecture diagram notebook (`00_demo_overview.py`). Walk through
the end-to-end flow.

### Section 1: Data Setup & Ingestion (10 min)
- Run `setup.py` — show the SCALE_FACTOR parameter
- Show the Volume with CSV files
- Run the ingestion pipeline — show parallel task execution
- Show the DLT pipeline — walk through expectations in `silver_*.sql`
- Show DQ pass rates in the app

### Section 2: Gold Layer / UPT (10 min)
- Run `build_upt.py` — walk through the joins
- Show the PK constraint, tags, and column comments
- Show the Features UI in Catalog Explorer
- Explain synthetic vs real columns

### Section 3: Model Training (10 min)
- Run `train_pricing_models` — walk through the 4 notebooks
- Show GLM relativities (audience loves this)
- Show GBM feature importance
- Show the model comparison: GLM only vs GLM + uplift
- Show `fe.log_model()` and explain auto feature lookup

### Section 4: Use Cases (10 min)
- UC1: Shadow pricing — show the flood risk v2 simulation
- UC2: Point-in-time — show Delta Time Travel
- UC5: Enriched pricing waterfall (tech × fraud × retention)

### Section 5: Serving (10 min)
- Show online store setup
- Deploy model endpoint
- Send a request with just `policy_id` — show the response
- Feature override demo ("what if flood score = 10?")

### Section 6: Governance & App (5 min)
- Walk through all 4 app tabs
- Generate a regulatory PDF
- Show the Genie Space

---

## Tips for Presenters

1. **Start with the app, not the notebooks.** The app tells the story; the
   notebooks prove it's real.
2. **Don't skip the impact analyser.** It's the strongest "aha" moment.
3. **Have a backup policy_id ready** for the serving demo (e.g. POL-100042).
4. **If Genie Space is configured,** ask one of the sample questions live.
5. **Adapt for your audience:** actuaries care about GLM relativities and
   regulatory compliance; engineers care about the serverless architecture
   and feature lookup; Big4 consultants care about the governance story.
6. **The AI agent is optional.** Only show it if the audience is receptive
   to AI in regulated processes. Always lead with "the human decides."
