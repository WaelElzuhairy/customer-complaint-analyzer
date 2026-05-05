"""
Customer Complaint Analyzer — Streamlit Demo App.
Dark-themed, modern SaaS UI.
"""

import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({
    "figure.facecolor": "#0F1117",
    "axes.facecolor":   "#0F1117",
    "axes.edgecolor":   "#2D3748",
    "axes.labelcolor":  "#94A3B8",
    "xtick.color":      "#94A3B8",
    "ytick.color":      "#94A3B8",
    "text.color":       "#E2E8F0",
    "grid.color":       "#1E2533",
    "grid.alpha":       1.0,
})

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Complaint Analyzer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Base ── */
.stApp { background: #080B12; }
[data-testid="stAppViewContainer"] { background: #080B12; }
[data-testid="stHeader"] { background: #080B12 !important; }
[data-testid="stMain"] { background: #080B12; }

[data-testid="stSidebar"] {
    background: #0D1117;
    border-right: 1px solid rgba(255,255,255,0.06);
}
[data-testid="stSidebar"] * { color: #CBD5E1 !important; }
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #F1F5F9 !important; }
[data-testid="stSidebar"] .stDataFrame { border-radius: 8px; overflow: hidden; }

/* ── Typography ── */
h1, h2, h3, h4, h5, h6 { color: #F1F5F9 !important; }
p, span, label, div { color: #CBD5E1; }
.stMarkdown p { color: #CBD5E1; }

/* ── Text area ── */
.stTextArea textarea,
textarea,
div[data-baseweb="textarea"] textarea {
    background: #0D1117 !important;
    background-color: #0D1117 !important;
    border: 1px solid #1E2D3D !important;
    border-radius: 12px !important;
    color: #E2E8F0 !important;
    font-size: 0.95rem !important;
    padding: 14px !important;
    caret-color: #6366f1 !important;
}
div[data-baseweb="textarea"] {
    background: #0D1117 !important;
    border-radius: 12px !important;
}
.stTextArea textarea:focus {
    border-color: rgba(99,102,241,0.5) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.12) !important;
}
.stTextArea label { color: #94A3B8 !important; font-size: 0.85rem !important; }

/* ── Buttons ── */
.stButton button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.6rem 1.8rem !important;
    box-shadow: 0 0 24px rgba(99,102,241,0.35) !important;
    transition: all 0.2s !important;
}
.stButton button[kind="primary"]:hover {
    box-shadow: 0 0 36px rgba(99,102,241,0.55) !important;
    transform: translateY(-1px) !important;
}
.stButton button[kind="secondary"],
.stButton button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #94A3B8 !important;
    border-radius: 20px !important;
    font-size: 0.82rem !important;
    transition: all 0.2s !important;
}
.stButton button[kind="secondary"]:hover,
.stButton button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(255,255,255,0.18) !important;
    color: #E2E8F0 !important;
}

/* ── Progress bars ── */
.stProgress > div > div { background: rgba(255,255,255,0.06) !important; border-radius: 99px; }
.stProgress > div > div > div { border-radius: 99px !important; }

/* ── Spinner / alerts ── */
.stSpinner > div { border-color: #6366f1 transparent transparent transparent !important; }
.stInfo  { background: rgba(99,102,241,0.08) !important; border-left-color: #6366f1 !important; color: #A5B4FC !important; border-radius: 8px !important; }
.stSuccess { background: rgba(16,185,129,0.08) !important; border-left-color: #10B981 !important; color: #6EE7B7 !important; border-radius: 8px !important; }
.stWarning { background: rgba(245,158,11,0.08) !important; border-left-color: #F59E0B !important; color: #FCD34D !important; border-radius: 8px !important; }
.stError   { background: rgba(239,68,68,0.08) !important;  border-left-color: #EF4444 !important; color: #FCA5A5 !important; border-radius: 8px !important; }

/* ── Divider ── */
hr { border-color: rgba(255,255,255,0.06) !important; }

/* ── Selectbox / inputs ── */
.stSelectbox > div > div {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
}

/* ── DataFrames ── */
.stDataFrame { border-radius: 10px; overflow: hidden; }
[data-testid="stDataFrameResizable"] { background: #0D1117 !important; }

/* ── Caption ── */
.stCaption, .caption { color: #64748B !important; font-size: 0.8rem; }

/* ── Custom components ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #E2E8F0 0%, #6366f1 60%, #8b5cf6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15;
    margin-bottom: 0.3rem;
}
.hero-sub {
    font-size: 1rem;
    color: #64748B;
    margin-bottom: 1.8rem;
}
.section-label {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin-bottom: 0.5rem;
}
.glass-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.07);
    border-radius: 16px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 0.6rem;
    backdrop-filter: blur(10px);
}
.model-name {
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.model-pred {
    font-size: 1.2rem;
    font-weight: 700;
    margin-bottom: 0.35rem;
    line-height: 1.2;
}
.conf-text {
    font-size: 0.78rem;
    color: #64748B;
    margin-bottom: 0.6rem;
}
.winner-ring {
    border: 1px solid rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.12);
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}
.badge-critical { background: rgba(239,68,68,0.15);  color: #F87171; border: 1px solid rgba(239,68,68,0.3); }
.badge-high     { background: rgba(245,158,11,0.15); color: #FBBF24; border: 1px solid rgba(245,158,11,0.3); }
.badge-medium   { background: rgba(59,130,246,0.15); color: #60A5FA; border: 1px solid rgba(59,130,246,0.3); }
.badge-low      { background: rgba(16,185,129,0.15); color: #34D399; border: 1px solid rgba(16,185,129,0.3); }
.badge-neutral  { background: rgba(255,255,255,0.06); color: #94A3B8; border: 1px solid rgba(255,255,255,0.1); }
.winner-star {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    color: #818CF8;
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 99px;
    font-size: 0.7rem;
    font-weight: 700;
    padding: 2px 8px;
    margin-left: 6px;
    letter-spacing: 0.04em;
    vertical-align: middle;
}
.attn-wrap {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
.stat-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-bottom: 1rem;
}
.stat-pill {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px 16px;
    flex: 1;
    min-width: 120px;
}
.stat-pill .label { font-size: 0.7rem; color: #475569; text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
.stat-pill .value { font-size: 1.1rem; font-weight: 700; color: #E2E8F0; margin-top: 2px; }
</style>
""", unsafe_allow_html=True)

# ── Examples ──────────────────────────────────────────────────────────────────

EXAMPLE_COMPLAINTS = {
    "Credit Fraud":     ("I noticed three unauthorized charges on my credit card totaling $1,247. "
                         "I did not make these purchases and believe my card information was stolen online. "
                         "I called the bank immediately but they refused to issue a provisional credit while "
                         "investigating. It has been three weeks and the charges are still on my account."),
    "Mortgage Error":   ("My mortgage servicer has been misapplying my extra principal payments for eight months. "
                         "I pay $300 extra every month toward principal but my loan balance has barely moved. "
                         "When I request a payment history they send incorrect statements that do not match my bank records."),
    "Credit Report":    ("There is an account on my credit report that does not belong to me showing a $4,500 "
                         "collection balance. I have never done business with this company. I disputed it with all "
                         "three bureaus two months ago but it is still showing and my score dropped 80 points."),
    "Student Loan":     ("My student loan servicer transferred my account without notice and the new servicer is "
                         "charging me a higher monthly payment than my income-driven repayment plan allows. "
                         "My original payment was $87 per month but they are now demanding $412."),
    "Debt Harassment":  ("A debt collection agency calls me 12 to 15 times per day including before 8am and after 9pm. "
                         "They contacted my employer and told coworkers about the debt. I sent a cease and desist "
                         "letter by certified mail three weeks ago and the calls have not stopped. Clear FDCPA violation."),
}

# ── Model loader ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading models… (first run may take 2–3 min to download from HuggingFace)")
def load_analyzer():
    from app.inference import ComplaintAnalyzer
    return ComplaintAnalyzer()

# ── Helpers ───────────────────────────────────────────────────────────────────

MODEL_META = {
    "baseline":   {"label": "TF-IDF + LogReg",       "color": "#6366f1", "accent": "rgba(99,102,241,0.15)"},
    "bilstm":     {"label": "BiLSTM + Attention",     "color": "#8b5cf6", "accent": "rgba(139,92,246,0.15)"},
    "distilbert": {"label": "DistilBERT Fine-Tuned",  "color": "#06b6d4", "accent": "rgba(6,182,212,0.15)"},
}

PRIORITY_BADGE = {
    "Critical": "badge-critical",
    "High":     "badge-high",
    "Medium":   "badge-medium",
    "Low":      "badge-low",
}

def priority_html(p):
    cls = PRIORITY_BADGE.get(p, "badge-neutral")
    return f"<span class='badge {cls}'>{p}</span>"

def render_attention_html(tokens, weights):
    if not tokens or not weights:
        return ""
    weights = np.array(weights[:len(tokens)])
    weights = weights / (weights.max() + 1e-9)
    cmap = plt.cm.plasma
    parts = ["<div style='line-height:2.4; font-size:0.92rem; font-family:monospace;'>"]
    for tok, w in zip(tokens[:45], weights[:45]):
        rgba  = cmap(float(w) * 0.85 + 0.15)
        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        alpha = 0.18 + float(w) * 0.82
        tc    = "#fff" if w > 0.45 else "#94A3B8"
        parts.append(
            f"<span style='background:rgba({r},{g},{b},{alpha:.2f}); color:{tc}; "
            f"padding:3px 6px; margin:2px; border-radius:5px; display:inline-block;'>{tok}</span>"
        )
    parts.append("</div>")
    return "".join(parts)

def render_proba_html(all_probabilities, prediction, accent_color):
    """Render probability bars as pure HTML — no matplotlib needed."""
    items = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:8]
    rows  = []
    for label, val in items:
        is_pred  = label[:24] == prediction[:24]
        bar_color = accent_color if is_pred else "#1E2533"
        txt_color = "#E2E8F0"  if is_pred else "#64748B"
        pct       = f"{val:.1%}"
        width     = f"{val * 100:.1f}%"
        rows.append(
            f"<div style='margin-bottom:6px;'>"
            f"<div style='display:flex;justify-content:space-between;margin-bottom:3px;'>"
            f"<span style='font-size:0.72rem;color:{txt_color};'>{label[:26]}</span>"
            f"<span style='font-size:0.72rem;color:#475569;font-weight:600;'>{pct}</span>"
            f"</div>"
            f"<div style='background:#0D1117;border-radius:99px;height:5px;'>"
            f"<div style='background:{bar_color};width:{width};height:5px;border-radius:99px;transition:width 0.4s;'></div>"
            f"</div></div>"
        )
    return "<div style='margin-top:8px;'>" + "".join(rows) + "</div>"

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### Complaint Analyzer")
    st.markdown("<span style='color:#475569;font-size:0.8rem;'>Three models. One verdict.</span>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**Models**")
    for key, m in MODEL_META.items():
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
            f"<span style='width:8px;height:8px;border-radius:50%;background:{m['color']};display:inline-block;'></span>"
            f"<span style='font-size:0.85rem;color:#94A3B8;'>{m['label']}</span></div>",
            unsafe_allow_html=True
        )

    st.markdown("---")
    st.markdown("**Trained on**")
    st.markdown("<span style='color:#64748B;font-size:0.82rem;'>CFPB Consumer Complaint Database<br>107,000 complaints · 9 product classes</span>", unsafe_allow_html=True)

    st.markdown("---")
    results_path = RESULTS_DIR / "comparison_table.csv"
    if results_path.exists():
        import pandas as pd
        st.markdown("**Model Performance**")
        df = pd.read_csv(results_path, index_col=0)
        st.dataframe(df.round(3), use_container_width=True)
    else:
        st.info("Run notebooks to see metrics here.")

    st.markdown("---")
    cm_files = list(RESULTS_DIR.glob("confusion_matrix_*.png")) if RESULTS_DIR.exists() else []
    if cm_files:
        st.markdown("**Confusion Matrix**")
        selected_cm = st.selectbox("Model:", [f.stem.replace("confusion_matrix_", "") for f in cm_files], label_visibility="collapsed")
        cm_path = RESULTS_DIR / f"confusion_matrix_{selected_cm}.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)

    st.markdown("---")
    st.markdown(
        "<span style='color:#334155;font-size:0.75rem;'>"
        "<a href='https://github.com/WaelElzuhairy/customer-complaint-analyzer' "
        "style='color:#475569;text-decoration:none;'>GitHub</a> · "
        "<a href='https://huggingface.co/Wael-Elzuhairy/complaint-analyzer-models' "
        "style='color:#475569;text-decoration:none;'>HuggingFace</a></span>",
        unsafe_allow_html=True
    )

# ── Hero ──────────────────────────────────────────────────────────────────────

st.markdown('<div class="hero-title">Customer Complaint Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-sub">Classify CFPB consumer complaints using TF-IDF, BiLSTM+Attention, and DistilBERT — side by side.</div>',
    unsafe_allow_html=True,
)

# ── Example pills ─────────────────────────────────────────────────────────────

st.markdown('<div class="section-label">Quick Examples</div>', unsafe_allow_html=True)
cols_ex = st.columns(len(EXAMPLE_COMPLAINTS))
selected_example = None
for col, (name, text) in zip(cols_ex, EXAMPLE_COMPLAINTS.items()):
    if col.button(name, use_container_width=True):
        selected_example = text

# ── Input ─────────────────────────────────────────────────────────────────────

complaint_text = st.text_area(
    "Complaint narrative",
    value=selected_example or "",
    height=170,
    placeholder="Paste or type a consumer complaint narrative here…",
    label_visibility="collapsed",
)

col_btn, col_hint = st.columns([1, 5])
with col_btn:
    run_all = st.button("⚡ Analyze", type="primary", disabled=not complaint_text.strip(), use_container_width=True)
with col_hint:
    st.markdown("<span style='color:#334155;font-size:0.8rem;line-height:2.6;'>Minimum ~10 words for best results</span>", unsafe_allow_html=True)

# ── Results ───────────────────────────────────────────────────────────────────

if run_all and complaint_text.strip():
    if len(complaint_text.split()) < 10:
        st.warning("Please enter a longer complaint (at least 10 words).")
        st.stop()

    with st.spinner("Running all 3 models…"):
        analyzer = load_analyzer()
        results  = analyzer.predict(complaint_text)

    st.markdown("---")

    # ── Top stats row ──────────────────────────────────────────────────────
    priority   = results.get("priority", "Low")
    issue_pred = results.get("issue_group", {}).get("prediction", "Unknown")
    issue_conf = results.get("issue_group", {}).get("confidence", 0)
    summary    = results.get("summary", "")
    word_count = len(complaint_text.split())

    p_cls  = PRIORITY_BADGE.get(priority, "badge-neutral")
    p_icon = {"Critical": "🔴", "High": "🟠", "Medium": "🔵", "Low": "🟢"}.get(priority, "⚪")

    st.markdown(
        f"""
        <div class="stat-row">
            <div class="stat-pill">
                <div class="label">Priority</div>
                <div class="value">{p_icon} <span class='badge {p_cls}' style='font-size:0.95rem;padding:4px 14px;'>{priority}</span></div>
            </div>
            <div class="stat-pill">
                <div class="label">Issue Group</div>
                <div class="value" style="font-size:0.95rem;">{issue_pred} <span style='color:#475569;font-size:0.75rem;'>({issue_conf:.0%})</span></div>
            </div>
            <div class="stat-pill">
                <div class="label">Word Count</div>
                <div class="value">{word_count}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if summary:
        st.markdown(
            f"<div class='glass-card' style='margin-bottom:1.2rem;'>"
            f"<div class='section-label'>AI Summary</div>"
            f"<div style='color:#CBD5E1;font-size:0.92rem;line-height:1.65;'>{summary}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # ── Model predictions ──────────────────────────────────────────────────
    st.markdown('<div class="section-label">Model Predictions</div>', unsafe_allow_html=True)

    most_confident = results.get("most_confident_model", "")
    cols = st.columns(3)

    for col, model_key in zip(cols, ["baseline", "bilstm", "distilbert"]):
        r    = results.get(model_key, {})
        meta = MODEL_META[model_key]
        is_winner = model_key == most_confident

        with col:
            if "error" in r:
                st.markdown(
                    f"<div class='glass-card'>"
                    f"<div class='model-name' style='color:{meta['color']};'>{meta['label']}</div>"
                    f"<div style='color:#EF4444;font-size:0.85rem;'>Model unavailable</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                continue

            pred = r.get("prediction", "N/A")
            conf = r.get("confidence", 0.0)
            winner_tag = "<span class='winner-star'>BEST</span>" if is_winner else ""
            ring_cls   = "winner-ring" if is_winner else ""

            st.markdown(
                f"<div class='glass-card {ring_cls}'>"
                f"<div class='model-name' style='color:{meta['color']};'>{meta['label']}{winner_tag}</div>"
                f"<div class='model-pred' style='color:#F1F5F9;'>{pred}</div>"
                f"<div class='conf-text'>{conf:.1%} confidence</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

            st.progress(conf)

            if r.get("all_probabilities"):
                html = render_proba_html(r["all_probabilities"], pred, meta["color"])
                st.markdown(html, unsafe_allow_html=True)

    # ── Attention weights ──────────────────────────────────────────────────
    bilstm_r = results.get("bilstm", {})
    if bilstm_r.get("attention_tokens"):
        st.markdown("---")
        st.markdown('<div class="section-label">BiLSTM Attention Weights</div>', unsafe_allow_html=True)
        st.markdown("<span style='color:#475569;font-size:0.8rem;'>Brighter tokens received higher attention during classification.</span>", unsafe_allow_html=True)
        attn_html = render_attention_html(bilstm_r["attention_tokens"], bilstm_r["attention_weights"])
        st.markdown(f"<div class='attn-wrap'>{attn_html}</div>", unsafe_allow_html=True)

    # ── Issue group breakdown ──────────────────────────────────────────────
    iss = results.get("issue_group", {})
    if iss.get("all_probabilities"):
        st.markdown("---")
        st.markdown('<div class="section-label">Issue Group Breakdown</div>', unsafe_allow_html=True)
        html = render_proba_html(iss["all_probabilities"], iss["prediction"], "#6366f1")
        col_iss, _ = st.columns([2, 1])
        with col_iss:
            st.markdown(html, unsafe_allow_html=True)

    st.markdown(
        "<div style='margin-top:1.5rem;padding:10px 16px;background:rgba(16,185,129,0.07);"
        "border:1px solid rgba(16,185,129,0.2);border-radius:10px;color:#6EE7B7;"
        "font-size:0.85rem;'>✓ Analysis complete</div>",
        unsafe_allow_html=True,
    )
