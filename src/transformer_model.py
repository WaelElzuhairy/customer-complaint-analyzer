"""
Transformer Model: DistilBERT Fine-tuning for Complaint Classification.

Uses HuggingFace Transformers to fine-tune DistilBERT for multi-class
text classification on consumer complaint narratives.

Usage:
    from src.transformer_model import train_transformer, predict_transformer
"""

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from datasets import Dataset as HFDataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"


# ===========================================================================
# Custom Trainer with class weights
# ===========================================================================

class WeightedTrainer(Trainer):
    """Custom Trainer that uses class weights in the loss function.

    This addresses class imbalance by penalizing misclassification of
    minority classes more heavily during training.
    """

    def __init__(self, class_weights: torch.Tensor = None, **kwargs):
        """
        Args:
            class_weights: Tensor of shape (num_classes,) with per-class weights.
            **kwargs: Arguments passed to the base Trainer.
        """
        super().__init__(**kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Compute weighted cross-entropy loss.

        Args:
            model: The model being trained.
            inputs: Dict of input tensors.
            return_outputs: Whether to also return model outputs.

        Returns:
            Loss tensor, and optionally model outputs.
        """
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        if self.class_weights is not None:
            weight = self.class_weights.to(logits.device)
            loss = F.cross_entropy(logits, labels, weight=weight)
        else:
            loss = F.cross_entropy(logits, labels)

        return (loss, outputs) if return_outputs else loss


# ===========================================================================
# Metrics
# ===========================================================================

def compute_metrics(eval_pred) -> dict:
    """Compute classification metrics for the HuggingFace Trainer.

    Args:
        eval_pred: EvalPrediction object with predictions and label_ids.

    Returns:
        Dictionary of metric names to values.
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro", zero_division=0),
        "f1_weighted": f1_score(labels, preds, average="weighted", zero_division=0),
        "precision_macro": precision_score(labels, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(labels, preds, average="macro", zero_division=0),
    }


# ===========================================================================
# Tokenization
# ===========================================================================

def tokenize_data(
    texts: list[str],
    labels: list[int],
    tokenizer: DistilBertTokenizerFast,
    max_length: int = 256,
) -> HFDataset:
    """Tokenize texts and create a HuggingFace Dataset.

    Args:
        texts: List of text strings.
        labels: List of integer labels.
        tokenizer: DistilBERT tokenizer.
        max_length: Maximum sequence length.

    Returns:
        HuggingFace Dataset with tokenized inputs.
    """
    dataset = HFDataset.from_dict({"text": texts, "labels": labels})

    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

    dataset = dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
    dataset.set_format("torch")
    return dataset


# ===========================================================================
# Training
# ===========================================================================

def train_transformer(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    num_classes: int,
    label_names: list[str] = None,
    class_weights: np.ndarray = None,
    max_length: int = 256,
    batch_size: int = 16,
    learning_rate: float = 2e-5,
    num_epochs: int = 3,
    task: str = "product",
) -> tuple:
    """Fine-tune DistilBERT for classification.

    Args:
        train_texts: Training texts.
        train_labels: Training integer labels.
        val_texts: Validation texts.
        val_labels: Validation integer labels.
        num_classes: Number of output classes.
        label_names: Optional list of class names.
        class_weights: Optional class weight array.
        max_length: Maximum token length.
        batch_size: Per-device batch size.
        learning_rate: Learning rate.
        num_epochs: Number of training epochs.
        task: Task name for saving.

    Returns:
        Tuple of (model, tokenizer, trainer).
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    output_dir = MODELS_DIR / f"distilbert_checkpoints_{task}"
    best_model_dir = MODELS_DIR / f"distilbert_best_{task}"

    # Load tokenizer and model
    logger.info("Loading DistilBERT tokenizer and model …")
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased",
        num_labels=num_classes,
    )

    if label_names:
        model.config.id2label = {i: name for i, name in enumerate(label_names)}
        model.config.label2id = {name: i for i, name in enumerate(label_names)}

    # Tokenize datasets
    logger.info("Tokenizing datasets …")
    train_dataset = tokenize_data(train_texts, train_labels, tokenizer, max_length)
    val_dataset = tokenize_data(val_texts, val_labels, tokenizer, max_length)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        warmup_steps=500,
        weight_decay=0.01,
        learning_rate=learning_rate,
        eval_strategy="steps",
        eval_steps=500,
        save_strategy="steps",
        save_steps=500,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        logging_steps=100,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
    )

    # Class weights tensor
    weight_tensor = None
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32)

    # Build trainer
    trainer = WeightedTrainer(
        class_weights=weight_tensor,
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train
    logger.info("Starting DistilBERT fine-tuning for task='%s' …", task)
    train_result = trainer.train()
    logger.info("Training complete. Metrics: %s", train_result.metrics)

    # Save best model
    trainer.save_model(str(best_model_dir))
    tokenizer.save_pretrained(str(best_model_dir))
    logger.info("Saved best model to %s", best_model_dir)

    # Save training metrics
    with open(RESULTS_DIR / f"transformer_{task}_train_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    return model, tokenizer, trainer


# ===========================================================================
# Inference
# ===========================================================================

def predict_transformer(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    texts: list[str],
    max_length: int = 256,
    batch_size: int = 32,
    device: str = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate predictions and probabilities from the fine-tuned model.

    Args:
        model: Trained DistilBERT model.
        tokenizer: DistilBERT tokenizer.
        texts: List of text strings.
        max_length: Maximum token length.
        batch_size: Batch size for inference.
        device: Device string.

    Returns:
        Tuple of (predictions, probability_matrix).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    all_preds = []
    all_probas = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            probas = F.softmax(logits, dim=-1)

        all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
        all_probas.extend(probas.cpu().numpy())

    return np.array(all_preds), np.array(all_probas)


def load_transformer(
    task: str = "product",
) -> tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
    """Load a saved fine-tuned DistilBERT model and tokenizer.

    Args:
        task: Task name ('product' or 'issue_group').

    Returns:
        Tuple of (model, tokenizer).
    """
    model_dir = MODELS_DIR / f"distilbert_best_{task}"
    logger.info("Loading DistilBERT from %s …", model_dir)

    tokenizer = DistilBertTokenizerFast.from_pretrained(str(model_dir))
    model = DistilBertForSequenceClassification.from_pretrained(str(model_dir))

    return model, tokenizer
