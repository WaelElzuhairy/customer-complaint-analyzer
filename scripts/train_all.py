"""
Training orchestrator: runs all three models sequentially.

    python scripts/train_all.py [--skip-baseline] [--skip-bilstm] [--skip-distilbert]

Saves all models and metrics to models/ and results/.
"""

import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import joblib
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.baseline import train_baseline, predict_baseline, evaluate_baseline
from src.attention_model import build_vocab, save_vocab, load_vocab, train_attention_model, predict_with_attention
from src.transformer_model import train_transformer, predict_transformer
from src.evaluate import evaluate_model, plot_confusion_matrix, compare_models, plot_training_curves
from src.data_prep import compute_class_weights

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DATA_DIR   = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_splits():
    log.info("Loading train/val/test splits …")
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df   = pd.read_csv(DATA_DIR / "val.csv")
    test_df  = pd.read_csv(DATA_DIR / "test.csv")

    le_product = joblib.load(MODELS_DIR / "label_encoder_product.joblib")
    label_names = list(le_product.classes_)

    log.info("Train: %d | Val: %d | Test: %d | Classes: %d",
             len(train_df), len(val_df), len(test_df), len(label_names))
    log.info("Products: %s", label_names)
    return train_df, val_df, test_df, label_names


def run_baseline(train_df, val_df, test_df, label_names):
    log.info("=" * 60)
    log.info("PHASE 1: Baseline — TF-IDF + Logistic Regression")
    log.info("=" * 60)

    X_train = train_df["narrative_clean"]
    y_train = train_df["product_encoded"]
    X_val   = val_df["narrative_clean"]
    y_val   = val_df["product_encoded"]
    X_test  = test_df["narrative_clean"]
    y_test  = test_df["product_encoded"]

    # Primary task: product
    pipeline = train_baseline(X_train, y_train, task="product")
    preds, probas = predict_baseline(pipeline, X_test)
    metrics = evaluate_model(y_test.values, preds, probas, label_names, "baseline_product")
    plot_confusion_matrix(y_test.values, preds, label_names, "baseline_product")
    log.info("Baseline product — Acc: %.4f | Macro F1: %.4f", metrics["accuracy"], metrics["f1_macro"])

    # Secondary task: issue group
    le_issue = joblib.load(MODELS_DIR / "label_encoder_issue_group.joblib")
    issue_names = list(le_issue.classes_)
    pipe_iss = train_baseline(train_df["narrative_clean"], train_df["issue_group_encoded"], task="issue_group")
    preds_iss, probas_iss = predict_baseline(pipe_iss, test_df["narrative_clean"])
    metrics_iss = evaluate_model(test_df["issue_group_encoded"].values, preds_iss, probas_iss,
                                  issue_names, "baseline_issue_group")
    log.info("Baseline issue_group — Acc: %.4f | Macro F1: %.4f",
             metrics_iss["accuracy"], metrics_iss["f1_macro"])

    return metrics


def run_bilstm(train_df, val_df, test_df, label_names):
    log.info("=" * 60)
    log.info("PHASE 2: BiLSTM + Attention")
    log.info("=" * 60)

    vocab_path = MODELS_DIR / "vocab.json"
    if vocab_path.exists():
        vocab = load_vocab(vocab_path)
        log.info("Loaded vocab: %d words.", len(vocab))
    else:
        vocab = build_vocab(train_df["narrative_clean"], max_vocab=50000, min_freq=3)
        save_vocab(vocab, vocab_path)

    class_weights = compute_class_weights(train_df["product_encoded"].values)

    model, history = train_attention_model(
        train_texts=train_df["narrative_clean"].tolist(),
        train_labels=train_df["product_encoded"].tolist(),
        val_texts=val_df["narrative_clean"].tolist(),
        val_labels=val_df["product_encoded"].tolist(),
        vocab=vocab,
        num_classes=len(label_names),
        class_weights=class_weights,
        max_len=256, embed_dim=128, hidden_dim=128,
        batch_size=64, learning_rate=1e-3,
        num_epochs=15, patience=3, device=DEVICE,
    )

    plot_training_curves(history, "BiLSTM_Attention")

    preds, probas, _ = predict_with_attention(
        model, test_df["narrative_clean"].tolist(), vocab,
        max_len=256, batch_size=64, device=DEVICE,
    )
    metrics = evaluate_model(test_df["product_encoded"].values, preds, probas,
                             label_names, "bilstm_attention_product")
    plot_confusion_matrix(test_df["product_encoded"].values, preds, label_names, "bilstm_attention_product")
    log.info("BiLSTM — Acc: %.4f | Macro F1: %.4f", metrics["accuracy"], metrics["f1_macro"])
    return metrics


def run_distilbert(train_df, val_df, test_df, label_names):
    log.info("=" * 60)
    log.info("PHASE 3: DistilBERT Fine-Tuning")
    log.info("=" * 60)

    class_weights = compute_class_weights(train_df["product_encoded"].values)

    model, tokenizer, trainer = train_transformer(
        train_texts=train_df["Consumer complaint narrative"].fillna("").tolist(),
        train_labels=train_df["product_encoded"].tolist(),
        val_texts=val_df["Consumer complaint narrative"].fillna("").tolist(),
        val_labels=val_df["product_encoded"].tolist(),
        num_classes=len(label_names),
        label_names=label_names,
        class_weights=class_weights,
        max_length=256, batch_size=16, learning_rate=2e-5, num_epochs=3,
        task="product",
    )

    preds, probas = predict_transformer(
        model, tokenizer,
        test_df["Consumer complaint narrative"].fillna("").tolist(),
        max_length=256, batch_size=32, device=DEVICE,
    )
    metrics = evaluate_model(test_df["product_encoded"].values, preds, probas,
                             label_names, "distilbert_product")
    plot_confusion_matrix(test_df["product_encoded"].values, preds, label_names, "distilbert_product")
    log.info("DistilBERT — Acc: %.4f | Macro F1: %.4f", metrics["accuracy"], metrics["f1_macro"])
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-baseline",   action="store_true")
    parser.add_argument("--skip-bilstm",     action="store_true")
    parser.add_argument("--skip-distilbert", action="store_true")
    args = parser.parse_args()

    log.info("Device: %s", DEVICE)
    if DEVICE == "cuda":
        log.info("GPU: %s | VRAM: %.1f GB",
                 torch.cuda.get_device_name(0),
                 torch.cuda.get_device_properties(0).total_memory / 1e9)

    train_df, val_df, test_df, label_names = load_splits()
    all_metrics = {}

    if not args.skip_baseline:
        all_metrics["TF-IDF + LR (Baseline)"] = run_baseline(train_df, val_df, test_df, label_names)

    if not args.skip_bilstm:
        all_metrics["BiLSTM + Attention"] = run_bilstm(train_df, val_df, test_df, label_names)

    if not args.skip_distilbert:
        all_metrics["DistilBERT (Fine-Tuned)"] = run_distilbert(train_df, val_df, test_df, label_names)

    if len(all_metrics) > 1:
        log.info("=" * 60)
        log.info("FINAL COMPARISON")
        log.info("=" * 60)
        df = compare_models(all_metrics, label_names)
        log.info("\n%s", df.to_string())

    log.info("All training complete. Models and results saved.")


if __name__ == "__main__":
    main()
