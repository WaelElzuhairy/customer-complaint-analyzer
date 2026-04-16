"""
Generate all academic charts for the Customer Complaint Analyzer report.
Covers all 5 notebook sections: EDA, Baseline, BiLSTM, DistilBERT, Error Analysis.
"""

import sys
import json
import warnings
import re
from pathlib import Path
from collections import Counter

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data" / "processed"
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", palette="muted")
matplotlib.rcParams.update({"figure.dpi": 130, "font.size": 10})

print("=" * 60)
print("GENERATING ACADEMIC CHARTS")
print("=" * 60)

# ── Load data ────────────────────────────────────────────────────────────────
print("\n[1/7] Loading datasets...")
full_df  = pd.read_csv(DATA_DIR / "full_filtered.csv")
train_df = pd.read_csv(DATA_DIR / "train.csv")
val_df   = pd.read_csv(DATA_DIR / "val.csv")
test_df  = pd.read_csv(DATA_DIR / "test.csv")

full_df["Date received"] = pd.to_datetime(full_df["Date received"], errors="coerce")
full_df["word_count"]    = full_df["Consumer complaint narrative"].str.split().str.len()

print(f"  Full filtered: {len(full_df):,} rows | Train: {len(train_df):,} | Val: {len(val_df):,} | Test: {len(test_df):,}")
print(f"  Products: {sorted(full_df['Product'].unique())}")

# ── Load metrics ─────────────────────────────────────────────────────────────
with open(RESULTS_DIR / "baseline_product_metrics.json")       as f: base_metrics = json.load(f)
with open(RESULTS_DIR / "bilstm_attention_product_metrics.json") as f: bilstm_metrics = json.load(f)
with open(RESULTS_DIR / "distilbert_product_metrics.json")     as f: distil_metrics = json.load(f)
with open(RESULTS_DIR / "bilstm_training_history.json")        as f: history = json.load(f)

comparison_df = pd.read_csv(RESULTS_DIR / "comparison_table.csv", index_col=0)
print(f"  Metrics loaded for {len(comparison_df)} models.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — EDA
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[2/7] EDA charts...")

# 1a. Class distribution (raw vs balanced)
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
raw_counts  = full_df["Product"].value_counts()
bal_counts  = train_df["Product"].value_counts()
pal = sns.color_palette("muted", len(raw_counts))

for ax, counts, title in zip(
    axes,
    [raw_counts, bal_counts],
    ["Before Balancing (Full Filtered)", "After Balancing (Train Split)"]
):
    bars = ax.barh(counts.index[::-1], counts.values[::-1], color=pal)
    for bar, val in zip(bars, counts.values[::-1]):
        ax.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
                f"{val:,}", va="center", fontsize=8)
    ax.set_xlabel("Number of Complaints")
    ax.set_title(title, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

plt.suptitle("Class Distribution: Before vs After Balancing", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_balance_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK eda_balance_comparison.png")

# 1b. Text length distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

wc = full_df["word_count"].dropna()
axes[0].hist(wc.clip(upper=600), bins=70, color="steelblue", edgecolor="white", alpha=0.85)
axes[0].axvline(128, color="red",    linestyle="--", linewidth=1.8, label="128 tokens")
axes[0].axvline(256, color="orange", linestyle="--", linewidth=1.8, label="256 tokens")
axes[0].set_xlabel("Word Count (clipped at 600)")
axes[0].set_ylabel("Frequency")
axes[0].set_title("Narrative Length Distribution", fontweight="bold")
axes[0].legend()
axes[0].grid(alpha=0.3)

plot_data    = full_df[full_df["word_count"] < 600].copy()
product_order = plot_data.groupby("Product")["word_count"].median().sort_values().index
axes[1].boxplot(
    [plot_data[plot_data["Product"] == p]["word_count"].values for p in product_order],
    labels=[p[:22] for p in product_order],
    vert=False, patch_artist=True,
    boxprops=dict(facecolor="lightblue", color="steelblue"),
    medianprops=dict(color="darkred", linewidth=2),
    showfliers=False,
)
axes[1].set_xlabel("Word Count")
axes[1].set_title("Narrative Length by Product", fontweight="bold")
axes[1].tick_params(axis="y", labelsize=8)
axes[1].grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_text_length.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK eda_text_length.png")

# Print token coverage stats
print("\n  Token coverage:")
for cutoff in [64, 128, 192, 256, 512]:
    pct = (wc <= cutoff).mean() * 100
    print(f"    max_length={cutoff:>4}: {pct:.1f}% of complaints fully captured")

# 1c. Issue group distribution
issue_counts = full_df["issue_group"].value_counts()
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
sns.barplot(y=issue_counts.index, x=issue_counts.values, ax=axes[0], palette="Set2", orient="h")
axes[0].set_xlabel("Count")
axes[0].set_title("Issue Group Distribution", fontweight="bold")
axes[0].grid(axis="x", alpha=0.3)

colors_pie = sns.color_palette("Set2", len(issue_counts))
axes[1].pie(issue_counts.values, labels=issue_counts.index,
            autopct="%1.1f%%", startangle=90, colors=colors_pie)
axes[1].set_title("Issue Group Share", fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_issue_groups.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK eda_issue_groups.png")

# 1d. Priority distribution
priority_order  = ["Critical", "High", "Medium", "Low"]
priority_counts = full_df["priority"].value_counts().reindex(priority_order, fill_value=0)
colors_p = ["#d62728", "#ff7f0e", "#ffdd57", "#2ca02c"]
fig, ax  = plt.subplots(figsize=(8, 5))
bars = ax.bar(priority_counts.index, priority_counts.values, color=colors_p)
for bar, val in zip(bars, priority_counts.values):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
            f"{val:,}\n({val / len(full_df) * 100:.1f}%)", ha="center", va="bottom", fontsize=9)
ax.set_title("Complaint Priority Distribution (Rule-Based)", fontweight="bold")
ax.set_ylabel("Count")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_priority_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK eda_priority_distribution.png")

# 1e. Train/val/test split stratification
fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
for ax, split_df, name in zip(
    axes,
    [train_df, val_df, test_df],
    ["Train (70%)", "Val (15%)", "Test (15%)"]
):
    counts = split_df["Product"].value_counts(normalize=True).sort_index()
    ax.barh(counts.index, counts.values * 100, color=sns.color_palette("muted", len(counts)))
    ax.set_xlabel("Percentage (%)")
    ax.set_title(name, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
plt.suptitle("Class Distribution Across Splits (Stratified)", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_split_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK eda_split_distribution.png")

# 1f. Word clouds
print("  Generating word clouds (this takes ~30s)...")
STOPWORDS = {
    "i","me","my","we","our","you","your","he","him","his","she","her",
    "it","its","they","them","their","what","which","who","this","that",
    "these","those","am","is","are","was","were","be","been","have","has",
    "had","do","does","did","a","an","the","and","but","if","or","as",
    "of","at","by","for","with","to","from","in","on","not","also","get",
    "got","one","two","three","said","told","called","would","could","make",
    "made","like","know","still","even","back","never","well","since","yet",
    "redacted","xxxx","xx","complaint","company","account","bank","financial",
    "consumer","per","us","however","upon","within","without",
}

def get_word_freq(text_series, n=150):
    all_words = []
    for text in text_series.dropna().sample(min(3000, len(text_series)), random_state=42):
        words = re.findall(r"\b[a-z]{3,}\b", text.lower())
        all_words.extend([w for w in words if w not in STOPWORDS])
    return dict(Counter(all_words).most_common(n))

top_products = full_df["Product"].value_counts().head(6).index.tolist()
fig, axes    = plt.subplots(2, 3, figsize=(18, 10))
for ax, product in zip(axes.flatten(), top_products):
    subset = full_df[full_df["Product"] == product]["Consumer complaint narrative"]
    freq   = get_word_freq(subset)
    wc_img = WordCloud(width=500, height=300, background_color="white",
                       colormap="viridis", max_words=60,
                       prefer_horizontal=0.8).generate_from_frequencies(freq)
    ax.imshow(wc_img, interpolation="bilinear")
    ax.set_title(product[:35], fontsize=10, fontweight="bold")
    ax.axis("off")
plt.suptitle("Most Frequent Words per Product Category", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "eda_word_clouds.png", dpi=120, bbox_inches="tight")
plt.close()
print("  OK eda_word_clouds.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Baseline analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[3/7] Baseline charts...")

# Top TF-IDF features per class
import joblib
pipeline   = joblib.load(MODELS_DIR / "baseline_pipeline_product.joblib")
le_product = joblib.load(MODELS_DIR / "label_encoder_product.joblib")
vectorizer = pipeline.named_steps["tfidf"]
clf        = pipeline.named_steps["clf"]
# clf.classes_ are integer label-encoded indices; map back to string names
class_names = list(le_product.classes_)   # sorted string names
feat_names  = vectorizer.get_feature_names_out()

fig, axes = plt.subplots(3, 3, figsize=(18, 15))
for ax, (cls_idx, cls_name) in zip(axes.flatten(), enumerate(class_names)):
    coefs     = clf.coef_[cls_idx]
    top_idx   = coefs.argsort()[-15:][::-1]
    top_feats = [feat_names[i] for i in top_idx]
    top_coefs = [coefs[i]      for i in top_idx]
    ax.barh(top_feats[::-1], top_coefs[::-1], color="steelblue")
    ax.set_title(cls_name[:30], fontsize=9, fontweight="bold")
    ax.tick_params(labelsize=7)
    ax.grid(axis="x", alpha=0.3)
plt.suptitle("Top TF-IDF Features per Class (Logistic Regression Coefficients)",
             fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "baseline_top_features.png", dpi=130, bbox_inches="tight")
plt.close()
print("  OK baseline_top_features.png")

# Per-class metrics — use per_class_f1 dict (classification_report is plain text)
base_pcf1   = base_metrics["per_class_f1"]
classes_all = sorted(base_pcf1.keys())

def per_class_bar(pcf1_dict, macro_f1, title, outfile):
    classes = sorted(pcf1_dict.keys())
    f1s     = [pcf1_dict[c] for c in classes]
    colors  = ["#2ca02c" if v >= 0.85 else "#ff7f0e" if v >= 0.70 else "#d62728" for v in f1s]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(classes, f1s, color=colors)
    ax.axhline(macro_f1, color="navy", linestyle="--", linewidth=1.5, label=f"Macro F1={macro_f1:.3f}")
    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_ylim(0, 1.05); ax.set_ylabel("F1-Score")
    ax.set_title(title, fontweight="bold")
    ax.tick_params(axis="x", rotation=30, labelsize=8)
    ax.legend(); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150, bbox_inches="tight")
    plt.close()

per_class_bar(base_pcf1, base_metrics["f1_macro"],
              "Baseline (TF-IDF + LR): Per-Class F1-Score on Test Set",
              RESULTS_DIR / "baseline_per_class_f1.png")
print("  OK baseline_per_class_f1.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — BiLSTM training curves
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[4/7] BiLSTM charts...")

train_loss  = history["train_loss"]
val_loss    = history["val_loss"]
val_f1      = history["val_f1_macro"]
val_acc     = history["val_accuracy"]
epochs      = list(range(1, len(train_loss) + 1))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(epochs, train_loss, "o-", color="#4C72B0", label="Train Loss", linewidth=2)
axes[0].plot(epochs, val_loss,   "s--", color="#DD8452", label="Val Loss",   linewidth=2)
best_ep = epochs[val_f1.index(max(val_f1))]
axes[0].axvline(best_ep, color="green", linestyle=":", linewidth=1.5, label=f"Best epoch ({best_ep})")
axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Loss")
axes[0].set_title("BiLSTM: Training & Validation Loss", fontweight="bold")
axes[0].legend(); axes[0].grid(alpha=0.3)
axes[0].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

ax2 = axes[1].twinx()
axes[1].plot(epochs, val_f1,  "o-",  color="#55A868", label="Val Macro F1", linewidth=2)
ax2.plot    (epochs, val_acc, "s--", color="#8172B2", label="Val Accuracy",  linewidth=2, alpha=0.7)
axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Macro F1", color="#55A868")
ax2.set_ylabel("Accuracy", color="#8172B2")
axes[1].set_title("BiLSTM: Validation F1 & Accuracy", fontweight="bold")
axes[1].set_ylim(0.5, 1.0); ax2.set_ylim(0.5, 1.0)
lines1, labels1 = axes[1].get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
axes[1].legend(lines1 + lines2, labels1 + labels2)
axes[1].grid(alpha=0.3)
axes[1].xaxis.set_major_locator(mticker.MaxNLocator(integer=True))

plt.suptitle("BiLSTM + Additive Attention — Training Dynamics", fontsize=13, fontweight="bold")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "bilstm_training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK bilstm_training_curves.png")

per_class_bar(bilstm_metrics["per_class_f1"], bilstm_metrics["f1_macro"],
              "BiLSTM + Attention: Per-Class F1-Score on Test Set",
              RESULTS_DIR / "bilstm_per_class_f1.png")
print("  OK bilstm_per_class_f1.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — DistilBERT
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[5/7] DistilBERT charts...")

# DistilBERT training metrics
try:
    with open(RESULTS_DIR / "transformer_product_train_metrics.json") as f:
        transformer_hist = json.load(f)

    # Extract epoch-level validation metrics from HF Trainer log_history
    log_history = transformer_hist if isinstance(transformer_hist, list) else transformer_hist.get("log_history", [])
    val_logs  = [x for x in log_history if "eval_loss" in x]
    train_logs = [x for x in log_history if "loss" in x and "eval_loss" not in x]

    if val_logs:
        db_epochs    = [x.get("epoch", i+1) for i, x in enumerate(val_logs)]
        db_val_loss  = [x["eval_loss"] for x in val_logs]
        db_val_acc   = [x.get("eval_accuracy", x.get("eval_acc", None)) for x in val_logs]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(db_epochs, db_val_loss, "o-", color="#DD8452", linewidth=2, label="Val Loss")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("DistilBERT Fine-Tuning: Validation Loss", fontweight="bold")
        ax.legend(); ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(RESULTS_DIR / "distilbert_training_curve.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  OK distilbert_training_curve.png")
    else:
        print("  WARN No epoch-level logs found in transformer metrics — skipping training curve")
except Exception as e:
    print(f"  WARN DistilBERT training curve skipped: {e}")

per_class_bar(distil_metrics["per_class_f1"], distil_metrics["f1_macro"],
              "DistilBERT (Fine-Tuned): Per-Class F1-Score on Test Set",
              RESULTS_DIR / "distilbert_per_class_f1.png")
print("  OK distilbert_per_class_f1.png")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Cross-model comparison & error analysis
# ═══════════════════════════════════════════════════════════════════════════════
print("\n[6/7] Comparison & error analysis charts...")

# Grouped bar chart: all metrics
metrics_data = {
    "TF-IDF + LR":    {"Accuracy": base_metrics["accuracy"],   "Macro F1": base_metrics["f1_macro"],
                        "Precision": base_metrics["precision_macro"], "Recall": base_metrics["recall_macro"]},
    "BiLSTM+Attention": {"Accuracy": bilstm_metrics["accuracy"], "Macro F1": bilstm_metrics["f1_macro"],
                          "Precision": bilstm_metrics["precision_macro"], "Recall": bilstm_metrics["recall_macro"]},
    "DistilBERT":      {"Accuracy": distil_metrics["accuracy"],  "Macro F1": distil_metrics["f1_macro"],
                         "Precision": distil_metrics["precision_macro"], "Recall": distil_metrics["recall_macro"]},
}
metrics_plot = pd.DataFrame(metrics_data).T
x  = np.arange(len(metrics_plot.columns))
w  = 0.25
fig, ax = plt.subplots(figsize=(11, 6))
for i, (model_name, row) in enumerate(metrics_plot.iterrows()):
    bars = ax.bar(x + i*w, row.values, width=w, label=model_name,
                  color=["#4C72B0","#DD8452","#55A868"][i], alpha=0.85)
    for bar, val in zip(bars, row.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f"{val:.3f}", ha="center", va="bottom", fontsize=7)

ax.set_xticks(x + w)
ax.set_xticklabels(metrics_plot.columns, fontsize=11)
ax.set_ylim(0.80, 0.92)
ax.set_ylabel("Score")
ax.set_title("Model Comparison — Accuracy, Precision, Recall, Macro F1", fontweight="bold", fontsize=13)
ax.legend(fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison_grouped_bar.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK comparison_grouped_bar.png")

# Per-class F1 heatmap across all models
all_classes  = sorted(set(base_metrics["per_class_f1"]) |
                      set(bilstm_metrics["per_class_f1"]) |
                      set(distil_metrics["per_class_f1"]))
heatmap_data = pd.DataFrame({
    "TF-IDF+LR":  [base_metrics["per_class_f1"].get(c, 0)   for c in all_classes],
    "BiLSTM":     [bilstm_metrics["per_class_f1"].get(c, 0) for c in all_classes],
    "DistilBERT": [distil_metrics["per_class_f1"].get(c, 0) for c in all_classes],
}, index=all_classes)

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="RdYlGn",
            vmin=0.5, vmax=1.0, ax=ax, linewidths=0.5, annot_kws={"size": 10})
ax.set_title("Per-Class F1-Score Heatmap (All Models, Test Set)", fontweight="bold", fontsize=13)
ax.set_xlabel("Model"); ax.set_ylabel("Product Category")
ax.tick_params(axis="x", labelsize=10)
ax.tick_params(axis="y", labelsize=9, rotation=0)
plt.tight_layout()
plt.savefig(RESULTS_DIR / "comparison_f1_heatmap.png", dpi=150, bbox_inches="tight")
plt.close()
print("  OK comparison_f1_heatmap.png")

# Error analysis: where do models disagree?
# Load test predictions if available, otherwise compute from saved metrics
# We'll use the per-class support & errors from classification reports
print("\n[7/7] Summary table...")

summary_rows = []
for model_name, overall in [
    ("TF-IDF + LR (Baseline)",  base_metrics),
    ("BiLSTM + Attention",      bilstm_metrics),
    ("DistilBERT (Fine-Tuned)", distil_metrics),
]:
    summary_rows.append({
        "Model":         model_name,
        "Accuracy":      f"{overall['accuracy']:.4f}",
        "Macro F1":      f"{overall['f1_macro']:.4f}",
        "Macro Prec.":   f"{overall['precision_macro']:.4f}",
        "Macro Recall":  f"{overall['recall_macro']:.4f}",
        "Weighted F1":   f"{overall['f1_weighted']:.4f}",
    })

summary_df = pd.DataFrame(summary_rows).set_index("Model")
print("\n" + summary_df.to_string())
summary_df.to_csv(RESULTS_DIR / "final_comparison_table.csv")
print("\n  OK final_comparison_table.csv")

# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL CHARTS GENERATED SUCCESSFULLY")
print("=" * 60)
print(f"\nOutput directory: {RESULTS_DIR}")
for f in sorted(RESULTS_DIR.glob("*.png")):
    print(f"  {f.name}")
