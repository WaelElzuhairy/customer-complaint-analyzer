"""
Evaluation Module for Customer Complaint Analyzer.

Provides unified evaluation metrics and visualization utilities for
comparing all three models (Baseline, BiLSTM+Attention, DistilBERT).

Usage:
    from src.evaluate import evaluate_model, compare_models, plot_confusion_matrix
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "results"


# ===========================================================================
# Core Evaluation
# ===========================================================================

def evaluate_model(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    model_name: str,
) -> dict:
    """Compute and save evaluation metrics for a single model.

    Args:
        y_true: Ground truth integer labels.
        y_pred: Predicted integer labels.
        y_proba: Probability matrix of shape (n_samples, n_classes).
        label_names: Ordered list of class names.
        model_name: Name string used for saving results.

    Returns:
        Dictionary containing all computed metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)

    metrics = {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class_f1": dict(zip(label_names, per_class_f1.tolist())),
        "classification_report": classification_report(
            y_true, y_pred, target_names=label_names, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    results_path = RESULTS_DIR / f"{model_name}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(
        "[%s] Accuracy: %.4f | Macro F1: %.4f | Weighted F1: %.4f",
        model_name, metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"],
    )

    return metrics


# ===========================================================================
# Confusion Matrix Visualization
# ===========================================================================

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: list[str],
    model_name: str,
    normalize: bool = True,
    figsize: tuple = (12, 10),
) -> None:
    """Plot and save a confusion matrix heatmap.

    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        label_names: Class names in label index order.
        model_name: Name for the plot title and filename.
        normalize: If True, normalize each row to show proportions.
        figsize: Figure size in inches.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm_display = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        fmt = ".2f"
        title = f"Normalized Confusion Matrix — {model_name}"
    else:
        cm_display = cm
        fmt = "d"
        title = f"Confusion Matrix — {model_name}"

    # Shorten long label names for display
    short_names = [n[:20] for n in label_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=short_names,
        yticklabels=short_names,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=9)
    plt.yticks(rotation=0, fontsize=9)
    plt.tight_layout()

    save_path = RESULTS_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Confusion matrix saved to %s", save_path)


# ===========================================================================
# Per-Class F1 Visualization
# ===========================================================================

def plot_per_class_f1(
    metrics_dict: dict[str, dict],
    label_names: list[str],
    figsize: tuple = (14, 7),
) -> None:
    """Plot per-class F1 scores grouped by model.

    Args:
        metrics_dict: Mapping of model_name → metrics dict (output of evaluate_model).
        label_names: Class names.
        figsize: Figure size.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    models = list(metrics_dict.keys())
    n_classes = len(label_names)
    x = np.arange(n_classes)
    width = 0.8 / len(models)

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    for i, (model_name, metrics) in enumerate(metrics_dict.items()):
        f1_values = [metrics["per_class_f1"].get(name, 0.0) for name in label_names]
        offset = (i - len(models) / 2 + 0.5) * width
        bars = ax.bar(x + offset, f1_values, width, label=model_name, color=colors[i % len(colors)])

    short_names = [n[:18] for n in label_names]
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("F1 Score", fontsize=12)
    ax.set_title("Per-Class F1 Score by Model", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    save_path = RESULTS_DIR / "per_class_f1_comparison.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Per-class F1 comparison saved to %s", save_path)


# ===========================================================================
# Model Comparison
# ===========================================================================

def compare_models(
    metrics_dict: dict[str, dict],
    label_names: list[str],
) -> pd.DataFrame:
    """Build a comparison table and bar chart for all models.

    Args:
        metrics_dict: Mapping of model_name → metrics dict.
        label_names: Class names (for per-class plot).

    Returns:
        DataFrame with one row per model showing key metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name, m in metrics_dict.items():
        rows.append({
            "Model": model_name,
            "Accuracy": round(m["accuracy"], 4),
            "Precision (Macro)": round(m["precision_macro"], 4),
            "Recall (Macro)": round(m["recall_macro"], 4),
            "F1 (Macro)": round(m["f1_macro"], 4),
            "F1 (Weighted)": round(m["f1_weighted"], 4),
        })

    comparison_df = pd.DataFrame(rows).set_index("Model")
    comparison_df.to_csv(RESULTS_DIR / "comparison_table.csv")
    logger.info("Comparison table:\n%s", comparison_df.to_string())

    # Bar chart of main metrics
    metrics_to_plot = ["Accuracy", "F1 (Macro)", "F1 (Weighted)"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    colors = ["#4C72B0", "#DD8452", "#55A868"]
    for ax, metric in zip(axes, metrics_to_plot):
        values = comparison_df[metric].values
        bars = ax.bar(comparison_df.index, values, color=colors[: len(values)])
        ax.set_title(metric, fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Score")
        ax.set_xticklabels(comparison_df.index, rotation=20, ha="right", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.suptitle("Model Comparison", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "model_comparison_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Model comparison chart saved.")

    # Per-class F1
    plot_per_class_f1(metrics_dict, label_names)

    return comparison_df


# ===========================================================================
# Training Curve Visualization
# ===========================================================================

def plot_training_curves(
    history: dict,
    model_name: str,
) -> None:
    """Plot and save training loss and validation metric curves.

    Args:
        history: Dict with keys like 'train_loss', 'val_loss', 'val_f1_macro'.
        model_name: Name for the plot title and filename.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], "b-o", label="Train Loss", markersize=4)
    axes[0].plot(epochs, history["val_loss"], "r-o", label="Val Loss", markersize=4)
    axes[0].set_title("Loss Curves", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # F1 curve
    if "val_f1_macro" in history:
        axes[1].plot(epochs, history["val_f1_macro"], "g-o", label="Val Macro F1", markersize=4)
        axes[1].set_title("Validation Macro F1", fontsize=13, fontweight="bold")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_ylim(0, 1.05)
        axes[1].legend()
        axes[1].grid(alpha=0.3)

    plt.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()

    save_path = RESULTS_DIR / f"training_curves_{model_name}.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Training curves saved to %s", save_path)


# ===========================================================================
# Error Analysis
# ===========================================================================

def collect_errors(
    texts: list[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    label_names: list[str],
    model_name: str,
    n_samples: int = 30,
) -> pd.DataFrame:
    """Collect and categorize misclassified samples for error analysis.

    Error categories:
        - short_text: word count < 30
        - long_noisy: word count > 200
        - low_confidence: max predicted probability < 0.5
        - other: everything else

    Args:
        texts: List of complaint narratives.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_proba: Probability matrix.
        label_names: Class names.
        model_name: Model identifier for saving.
        n_samples: Number of error samples to return.

    Returns:
        DataFrame with misclassified samples and their error categories.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    mask = y_true != y_pred
    error_indices = np.where(mask)[0]
    logger.info("[%s] Misclassifications: %d / %d (%.1f%%)",
                model_name, len(error_indices), len(y_true),
                100 * len(error_indices) / len(y_true))

    rows = []
    for idx in error_indices:
        text = texts[idx]
        words = text.split()
        confidence = float(y_proba[idx].max())

        # Categorize error type
        if len(words) < 30:
            error_type = "short_text"
        elif len(words) > 200:
            error_type = "long_noisy"
        elif confidence < 0.50:
            error_type = "low_confidence"
        else:
            error_type = "class_overlap"

        rows.append({
            "index": idx,
            "text_excerpt": " ".join(words[:60]) + ("…" if len(words) > 60 else ""),
            "word_count": len(words),
            "true_label": label_names[y_true[idx]],
            "predicted_label": label_names[y_pred[idx]],
            "confidence": confidence,
            "error_type": error_type,
        })

    errors_df = pd.DataFrame(rows)

    # Sample up to n_samples
    if len(errors_df) > n_samples:
        # Stratified sample by error type
        sampled = (
            errors_df.groupby("error_type", group_keys=False)
            .apply(lambda g: g.sample(min(len(g), max(1, n_samples // errors_df["error_type"].nunique()))))
        )
        errors_df = sampled.head(n_samples).reset_index(drop=True)

    save_path = RESULTS_DIR / f"error_analysis_{model_name}.csv"
    errors_df.to_csv(save_path, index=False)
    logger.info("Error analysis saved to %s", save_path)

    return errors_df
