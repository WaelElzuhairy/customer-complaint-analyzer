"""
Baseline Model: TF-IDF + Logistic Regression.

Provides a simple but effective baseline for complaint classification using
bag-of-words features and a linear classifier.

Usage:
    from src.baseline import train_baseline, predict_baseline
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix,
)

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ===========================================================================
# Model Building
# ===========================================================================

def build_pipeline(max_features: int = 50000, C: float = 1.0) -> Pipeline:
    """Build a TF-IDF + Logistic Regression pipeline.

    Args:
        max_features: Maximum number of TF-IDF features.
        C: Regularization strength (inverse).

    Returns:
        Scikit-learn Pipeline object.
    """
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95,
            sublinear_tf=True,
            strip_accents="unicode",
            token_pattern=r"(?u)\b\w+\b",
        )),
        ("clf", LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=1000,
            solver="lbfgs",
            multi_class="multinomial",
            n_jobs=-1,
            random_state=42,
        )),
    ])
    return pipeline


def train_baseline(
    X_train: pd.Series,
    y_train: pd.Series,
    task: str = "product",
    max_features: int = 50000,
    C: float = 1.0,
) -> Pipeline:
    """Train the baseline TF-IDF + LR pipeline.

    Args:
        X_train: Training text data.
        y_train: Training labels.
        task: Task name ('product' or 'issue_group') for saving.
        max_features: Max TF-IDF features.
        C: Regularization parameter.

    Returns:
        Trained Pipeline.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Training baseline model for task='%s' …", task)
    pipeline = build_pipeline(max_features=max_features, C=C)
    pipeline.fit(X_train, y_train)

    model_path = MODELS_DIR / f"baseline_pipeline_{task}.joblib"
    joblib.dump(pipeline, model_path)
    logger.info("Saved baseline pipeline to %s", model_path)

    return pipeline


def predict_baseline(
    pipeline: Pipeline,
    X: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities from the baseline model.

    Args:
        pipeline: Trained Pipeline.
        X: Text data to predict.

    Returns:
        Tuple of (predictions, probability_matrix).
    """
    preds = pipeline.predict(X)
    probas = pipeline.predict_proba(X)
    return preds, probas


# ===========================================================================
# Evaluation
# ===========================================================================

def evaluate_baseline(
    pipeline: Pipeline,
    X_test: pd.Series,
    y_test: pd.Series,
    label_names: list[str],
    task: str = "product",
) -> dict:
    """Evaluate the baseline model and save results.

    Args:
        pipeline: Trained Pipeline.
        X_test: Test text data.
        y_test: True labels.
        label_names: List of class names.
        task: Task name for saving results.

    Returns:
        Dictionary of evaluation metrics.
    """
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    preds, probas = predict_baseline(pipeline, X_test)

    metrics = {
        "model": f"baseline_{task}",
        "accuracy": float(accuracy_score(y_test, preds)),
        "precision_macro": float(precision_score(y_test, preds, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, preds, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, preds, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, preds, average="weighted", zero_division=0)),
        "per_class_f1": dict(zip(
            label_names,
            f1_score(y_test, preds, average=None, zero_division=0).tolist(),
        )),
        "classification_report": classification_report(
            y_test, preds, target_names=label_names, zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
    }

    results_path = RESULTS_DIR / f"baseline_{task}_metrics.json"
    with open(results_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Saved baseline metrics to %s", results_path)
    logger.info("Baseline %s — Accuracy: %.4f, Macro F1: %.4f, Weighted F1: %.4f",
                task, metrics["accuracy"], metrics["f1_macro"], metrics["f1_weighted"])

    return metrics


def get_top_features(
    pipeline: Pipeline,
    label_names: list[str],
    top_n: int = 20,
) -> dict[str, list[str]]:
    """Extract top TF-IDF features per class from the trained pipeline.

    Args:
        pipeline: Trained Pipeline containing TfidfVectorizer and LogisticRegression.
        label_names: Class label names.
        top_n: Number of top features to return per class.

    Returns:
        Dictionary mapping class name → list of top feature words.
    """
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())

    top_features = {}
    for i, label in enumerate(label_names):
        coef = clf.coef_[i] if clf.coef_.shape[0] > 1 else clf.coef_[0]
        top_idx = np.argsort(coef)[-top_n:][::-1]
        top_features[label] = feature_names[top_idx].tolist()

    return top_features
