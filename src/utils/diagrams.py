# Databricks notebook source
# MAGIC %md
# MAGIC # Shared diagram components for section notebooks.
# MAGIC Use `%run ../utils/diagrams` then call `show_section_diagram(section_name)`.

# COMMAND ----------

def _box(x, y, w, h, label, sublabel="", fill="#f1f5f9", stroke="#cbd5e1", text_fill="#475569"):
    sub = f'<text x="{x+w//2}" y="{y+h//2+12}" text-anchor="middle" font-size="8" fill="{text_fill}" opacity="0.7">{sublabel}</text>' if sublabel else ""
    return f'''<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>
    <text x="{x+w//2}" y="{y+h//2+4}" text-anchor="middle" font-size="10" font-weight="600" fill="{text_fill}">{label}</text>{sub}'''

def _arrow(x1, y1, x2, y2, dashed=False):
    dash = ' stroke-dasharray="4"' if dashed else ""
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#94a3b8" stroke-width="1.5"{dash} marker-end="url(#ah)"/>'

def _highlight(x, y, w, h, label, fill="#dbeafe", stroke="#3b82f6", text_fill="#1e40af"):
    return f'''<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="6" fill="{fill}" stroke="{stroke}" stroke-width="2.5"/>
    <text x="{x+w//2}" y="{y+h//2+4}" text-anchor="middle" font-size="11" font-weight="700" fill="{text_fill}">{label}</text>'''

_SVG_HEADER = '''<svg viewBox="0 0 720 100" xmlns="http://www.w3.org/2000/svg" style="width:100%;max-width:720px;">
<defs><marker id="ah" markerWidth="8" markerHeight="6" refX="8" refY="3" orient="auto"><path d="M0,0 L8,3 L0,6" fill="#94a3b8"/></marker></defs>'''

SECTIONS = {
    "ingestion": {
        "title": "Data Ingestion Pipeline",
        "subtitle": "External vendor data lands in Volumes, is ingested to Bronze (immutable raw archive), then flows to Silver via DLT with data quality expectations.",
        "highlight": 1,  # Which box to highlight (0-indexed)
        "boxes": [
            ("External\nCSVs", "#f1f5f9", "#cbd5e1"),
            ("Volume\n(Landing)", "#fef3c7", "#f59e0b"),
            ("Bronze\n(Raw)", "#fde68a", "#d97706"),
            ("DLT\nExpectations", "#dbeafe", "#3b82f6"),
            ("Silver\n(Cleansed)", "#e2e8f0", "#94a3b8"),
        ],
    },
    "silver": {
        "title": "Silver Layer — Data Quality & Cleansing",
        "subtitle": "DLT materialized views apply expectations (null checks, range validation, referential integrity). Failed rows are dropped. Derived features (risk tiers, composite scores) are added.",
        "highlight": 3,
        "boxes": [
            ("Bronze\n(Raw)", "#fde68a", "#d97706"),
            ("→", None, None),
            ("DLT\nExpectations", "#dbeafe", "#3b82f6"),
            ("Silver\n(Cleansed)", "#e2e8f0", "#94a3b8"),
            ("→ Gold\n(UPT)", "#fef9c3", "#ca8a04"),
        ],
    },
    "gold": {
        "title": "Gold Layer — Unified Pricing Table",
        "subtitle": "All 6 data sources joined into a single wide denormalized table. Primary key (policy_id) registered as UC feature table. Tags and column comments for discovery.",
        "highlight": 2,
        "boxes": [
            ("Silver\nTables (3)", "#e2e8f0", "#94a3b8"),
            ("Internal\nTables (3)", "#f1f5f9", "#cbd5e1"),
            ("UPT (Gold)\n90+ features", "#fef9c3", "#ca8a04"),
            ("Online\nStore", "#dcfce7", "#22c55e"),
            ("Model\nTraining", "#ede9fe", "#8b5cf6"),
        ],
    },
    "models": {
        "title": "Model Training Pipeline",
        "subtitle": "6 pricing models trained from the UPT using statsmodels (GLMs) and LightGBM (GBMs). All logged to MLflow with FeatureLookup for automatic feature resolution at serving time.",
        "highlight": 2,
        "boxes": [
            ("UPT\n(Gold)", "#fef9c3", "#ca8a04"),
            ("Feature\nLookup", "#dbeafe", "#3b82f6"),
            ("Train\n6 Models", "#ede9fe", "#8b5cf6"),
            ("MLflow\nRegistry", "#ede9fe", "#8b5cf6"),
            ("Actuary\nReview", "#fef2f2", "#ef4444"),
        ],
    },
    "serving": {
        "title": "Real-Time Model Serving",
        "subtitle": "Models deployed to Mosaic AI endpoints with auto feature lookup from Lakebase online store. Send just a policy_id → get a price in <100ms. Champion/challenger traffic splits for safe rollout.",
        "highlight": 3,
        "boxes": [
            ("UPT\n(Gold)", "#fef9c3", "#ca8a04"),
            ("Online\nStore", "#dcfce7", "#22c55e"),
            ("UC Model\nRegistry", "#ede9fe", "#8b5cf6"),
            ("Model\nServing", "#dcfce7", "#22c55e"),
            ("REST API\npolicy_id → price", "#1e293b", "#1e293b"),
        ],
    },
    "governance": {
        "title": "Governance & Audit Trail",
        "subtitle": "Every action — data approval, model training, LLM call, deployment — is recorded in the audit_log. Unity Catalog tracks lineage automatically. Delta Time Travel enables point-in-time reconstruction.",
        "highlight": 2,
        "boxes": [
            ("Data\nApprovals", "#fef2f2", "#ef4444"),
            ("Model\nDecisions", "#fef2f2", "#ef4444"),
            ("Audit Log\n(immutable)", "#1e293b", "#1e293b"),
            ("UC\nLineage", "#dbeafe", "#3b82f6"),
            ("Regulatory\nExport", "#f1f5f9", "#cbd5e1"),
        ],
    },
}


def show_section_diagram(section):
    """Render a focused section diagram via displayHTML."""
    cfg = SECTIONS.get(section)
    if not cfg:
        print(f"Unknown section: {section}. Available: {list(SECTIONS.keys())}")
        return

    boxes_svg = []
    x = 20
    for i, (label, fill, stroke) in enumerate(cfg["boxes"]):
        if fill is None:  # Arrow placeholder
            x += 15
            continue
        w = 120
        text_fill = "white" if fill == "#1e293b" else "#475569"
        if i == cfg["highlight"]:
            boxes_svg.append(_highlight(x, 20, w, 60, label.replace("\n", " "), fill, stroke, text_fill))
        else:
            boxes_svg.append(_box(x, 25, w, 50, label.replace("\n", " "), fill=fill, stroke=stroke, text_fill=text_fill))
        if i < len(cfg["boxes"]) - 1 and cfg["boxes"][i+1][1] is not None:
            boxes_svg.append(_arrow(x + w + 2, 50, x + w + 18, 50))
        x += w + 22

    svg = f'''{_SVG_HEADER}
    {"".join(boxes_svg)}
    </svg>'''

    html = f'''<div style="font-family:-apple-system,sans-serif;background:#f8fafc;padding:16px;border-radius:10px;border:1px solid #e2e8f0;">
    <h4 style="margin:0 0 4px 0;color:#1e293b;font-size:14px;">{cfg["title"]}</h4>
    <p style="margin:0 0 10px 0;color:#64748b;font-size:11px;">{cfg["subtitle"]}</p>
    {svg}
    </div>'''

    displayHTML(html)
