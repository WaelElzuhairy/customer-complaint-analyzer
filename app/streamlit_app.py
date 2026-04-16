"""
Customer Complaint Analyzer — Streamlit Demo App.

Classifies consumer complaints using three models and displays:
  - Product category prediction (all 3 models)
  - Issue group prediction
  - Priority level (rule-based)
  - Extractive summary
  - Attention weight visualization (BiLSTM)

Run with:
    streamlit run app/streamlit_app.py
"""

import sys
import json
from pathlib import Path

import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# Ensure project root is importable
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Customer Complaint Analyzer",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1f3b6e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .model-card {
        background: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
    }
    .prediction-label {
        font-size: 1.1rem;
        font-weight: 600;
        color: #1f3b6e;
    }
    .priority-critical { color: white; background: #d62728; padding: 4px 12px; border-radius: 6px; font-weight: 700; }
    .priority-high     { color: white; background: #ff7f0e; padding: 4px 12px; border-radius: 6px; font-weight: 700; }
    .priority-medium   { color: #333;  background: #ffdd57; padding: 4px 12px; border-radius: 6px; font-weight: 700; }
    .priority-low      { color: white; background: #2ca02c; padding: 4px 12px; border-radius: 6px; font-weight: 700; }
    .winner-badge { color: white; background: #17a2b8; padding: 2px 8px; border-radius: 4px; font-size: 0.8rem; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Example complaints
# ---------------------------------------------------------------------------

EXAMPLE_COMPLAINTS = {
    "Credit Card Fraud": (
        "I noticed several unauthorized charges on my credit card statement totaling over $800. "
        "I did not make these purchases and believe my card information was stolen. "
        "I have tried calling the bank multiple times but they keep putting me on hold for over an hour "
        "and have not resolved the issue. I need these fraudulent charges reversed immediately."
    ),
    "Mortgage Payment Issue": (
        "My mortgage servicer has been incorrectly applying my payments. I have been paying $200 extra "
        "each month toward principal for the past year, but my balance has not decreased accordingly. "
        "When I request a payment history breakdown they send incorrect statements. "
        "I am concerned they are mismanaging my account and I could face issues at year end."
    ),
    "Credit Report Error": (
        "I discovered an account on my credit report that does not belong to me. "
        "The account shows a collection for $1,200 from a company I have never done business with. "
        "I submitted a dispute with all three credit bureaus six weeks ago and have not received "
        "any response. This incorrect information is hurting my credit score significantly."
    ),
    "Student Loan Servicer Problem": (
        "My student loan servicer transferred my loans to a new company without proper notice. "
        "The new servicer claims I owe more than my original balance due to capitalized interest "
        "that was never disclosed to me. I am in the income-driven repayment plan and my monthly "
        "payments should be based on my income but they are charging me the full standard amount."
    ),
    "Debt Collection Harassment": (
        "A debt collection company has been calling me multiple times per day, including before 8am "
        "and after 9pm. They have also contacted my employer and family members about the debt. "
        "I sent them a written cease and desist letter via certified mail three weeks ago "
        "but the calls continue. This appears to be a violation of the FDCPA."
    ),
}

# ---------------------------------------------------------------------------
# Model loading (cached)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading models… (first run may take 2-3 min to download from HuggingFace)")
def load_analyzer():
    """Load the ComplaintAnalyzer with all models cached.

    On first run (e.g. Streamlit Cloud), models are auto-downloaded from
    HuggingFace Hub (~280 MB total). Subsequent runs use the cached models.
    """
    from app.inference import ComplaintAnalyzer
    return ComplaintAnalyzer()


# ---------------------------------------------------------------------------
# Helper: attention visualization
# ---------------------------------------------------------------------------

def render_attention_html(tokens: list[str], weights: list[float]) -> str:
    """Render word-level attention as colored HTML spans."""
    if not tokens or not weights:
        return ""

    weights = np.array(weights[:len(tokens)])
    weights = weights / (weights.max() + 1e-9)

    cmap = plt.cm.YlOrRd
    html_parts = ["<div style='line-height:2.2; font-size:1rem; font-family:monospace;'>"]
    for tok, w in zip(tokens[:40], weights[:40]):
        rgba = cmap(float(w))
        r, g, b = int(rgba[0]*255), int(rgba[1]*255), int(rgba[2]*255)
        text_color = "black" if w < 0.55 else "white"
        html_parts.append(
            f"<span style='background:rgb({r},{g},{b}); color:{text_color}; "
            f"padding:2px 4px; margin:2px; border-radius:3px;'>{tok}</span>"
        )
    html_parts.append("</div>")
    return "".join(html_parts)


# ---------------------------------------------------------------------------
# Helper: probability bar chart
# ---------------------------------------------------------------------------

def plot_proba_chart(all_probabilities: dict, prediction: str, height: int = 250):
    """Render a horizontal bar chart of class probabilities."""
    sorted_items = sorted(all_probabilities.items(), key=lambda x: x[1], reverse=True)[:8]
    labels = [k[:22] for k, _ in sorted_items]
    values = [v for _, v in sorted_items]
    colors = ["#17a2b8" if lbl[:22] == prediction[:22] else "#dee2e6" for lbl in labels]

    fig, ax = plt.subplots(figsize=(3.5, height / 72))
    bars = ax.barh(labels[::-1], values[::-1], color=colors[::-1])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Probability", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, values[::-1]):
        ax.text(min(val + 0.01, 0.97), bar.get_y() + bar.get_height()/2,
                f"{val:.2f}", va="center", fontsize=7)
    plt.tight_layout(pad=0.5)
    return fig


# ---------------------------------------------------------------------------
# Helper: priority badge
# ---------------------------------------------------------------------------

def priority_badge(priority: str) -> str:
    cls_map = {
        "Critical": "priority-critical",
        "High": "priority-high",
        "Medium": "priority-medium",
        "Low": "priority-low",
    }
    css = cls_map.get(priority, "priority-low")
    return f"<span class='{css}'>{priority}</span>"


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("About")
    st.markdown("""
    This app demonstrates three complaint classification models trained on the
    **CFPB Consumer Complaint Database** (Aug 2023–present):

    | Model | Type |
    |-------|------|
    | TF-IDF + LR | Bag-of-words baseline |
    | BiLSTM + Attention | Sequential deep learning |
    | DistilBERT | Pretrained Transformer |

    **Outputs:**
    - Product category
    - Issue group (7 types)
    - Priority level
    - Extractive summary
    """)

    st.header("Model Comparison")
    results_path = RESULTS_DIR / "comparison_table.csv"
    if results_path.exists():
        import pandas as pd
        df = pd.read_csv(results_path, index_col=0)
        st.dataframe(df.round(4), use_container_width=True)
    else:
        st.info("Run all notebooks to see trained model metrics here.")

    st.header("Confusion Matrices")
    cm_files = list(RESULTS_DIR.glob("confusion_matrix_*.png")) if RESULTS_DIR.exists() else []
    if cm_files:
        selected_cm = st.selectbox("Select model:", [f.stem.replace("confusion_matrix_", "") for f in cm_files])
        cm_path = RESULTS_DIR / f"confusion_matrix_{selected_cm}.png"
        if cm_path.exists():
            st.image(str(cm_path), use_container_width=True)
    else:
        st.info("Confusion matrices will appear after training.")


# ---------------------------------------------------------------------------
# Main area
# ---------------------------------------------------------------------------

st.markdown('<div class="main-header">📋 Customer Complaint Analyzer</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Compare TF-IDF, BiLSTM+Attention, and DistilBERT for classifying '
    'CFPB consumer complaints</div>',
    unsafe_allow_html=True,
)

# Example buttons
st.markdown("**Quick examples:**")
example_cols = st.columns(len(EXAMPLE_COMPLAINTS))
selected_example = None
for col, (name, text) in zip(example_cols, EXAMPLE_COMPLAINTS.items()):
    if col.button(name, use_container_width=True):
        selected_example = text

# Text input
default_text = selected_example or ""
complaint_text = st.text_area(
    "Paste a consumer complaint narrative:",
    value=default_text,
    height=180,
    placeholder="Type or paste a complaint here…",
    help="Enter any financial consumer complaint text. Minimum ~20 words for best results.",
)

run_all = st.button("🔍 Analyze Complaint", type="primary", disabled=not complaint_text.strip())

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

if run_all and complaint_text.strip():
    word_count = len(complaint_text.split())
    if word_count < 10:
        st.warning("Please enter a longer complaint (at least 10 words).")
        st.stop()

    with st.spinner("Running all 3 models…"):
        analyzer = load_analyzer()
        results = analyzer.predict(complaint_text)

    # ---- Priority + Summary ----
    meta_col1, meta_col2 = st.columns([1, 3])
    with meta_col1:
        st.markdown("#### Priority Level")
        st.markdown(priority_badge(results["priority"]), unsafe_allow_html=True)
        st.caption(f"Issue Group: **{results['issue_group']['prediction']}** "
                   f"({results['issue_group']['confidence']:.0%})")

    with meta_col2:
        st.markdown("#### Extractive Summary")
        st.info(results.get("summary", "N/A"))

    st.divider()

    # ---- Model predictions ----
    st.markdown("### Model Predictions")

    MODEL_LABELS = {
        "baseline":   "TF-IDF + Logistic Regression",
        "bilstm":     "BiLSTM + Attention",
        "distilbert": "DistilBERT (Fine-Tuned)",
    }
    COLORS = {
        "baseline":   "#4C72B0",
        "bilstm":     "#DD8452",
        "distilbert": "#55A868",
    }
    most_confident = results.get("most_confident_model", "")
    model_order = ["baseline", "bilstm", "distilbert"]

    cols = st.columns(3)
    for col, model_key in zip(cols, model_order):
        with col:
            r = results.get(model_key, {})
            is_winner = model_key == most_confident

            label = MODEL_LABELS[model_key]
            if is_winner:
                label += " ⭐"

            st.markdown(f"**{label}**")

            if "error" in r:
                st.error(f"Model not available: {r['error']}")
                continue

            pred = r.get("prediction", "N/A")
            conf = r.get("confidence", 0.0)
            color = COLORS[model_key]

            st.markdown(
                f"<div style='font-size:1.1rem; font-weight:700; color:{color};'>{pred}</div>",
                unsafe_allow_html=True,
            )
            st.progress(conf, text=f"Confidence: {conf:.1%}")

            if r.get("all_probabilities"):
                fig = plot_proba_chart(r["all_probabilities"], pred)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

    # ---- Attention visualization ----
    bilstm_result = results.get("bilstm", {})
    if bilstm_result.get("attention_tokens"):
        st.divider()
        st.markdown("### BiLSTM Attention Weights")
        st.caption("Darker/redder tokens received higher attention during classification.")
        attn_html = render_attention_html(
            bilstm_result["attention_tokens"],
            bilstm_result["attention_weights"],
        )
        st.markdown(attn_html, unsafe_allow_html=True)

    # ---- Issue group probabilities ----
    iss = results.get("issue_group", {})
    if iss.get("all_probabilities"):
        st.divider()
        st.markdown("### Issue Group Probabilities")
        fig, ax = plt.subplots(figsize=(8, 3))
        items = sorted(iss["all_probabilities"].items(), key=lambda x: x[1], reverse=True)
        labels = [k for k, _ in items]
        values = [v for _, v in items]
        colors = ["#17a2b8" if k == iss["prediction"] else "#dee2e6" for k in labels]
        ax.barh(labels[::-1], values[::-1], color=colors[::-1])
        ax.set_xlim(0, 1)
        ax.set_xlabel("Probability")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.success("Analysis complete.")
