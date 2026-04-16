"""
Inference Module for the Complaint Analyzer Streamlit App.

Loads all three trained models and provides a unified predict() interface
for the demo application.

Usage:
    from app.inference import ComplaintAnalyzer
    analyzer = ComplaintAnalyzer()
    results = analyzer.predict("I was charged twice for the same transaction...")
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import joblib

# Add project root to path so src/ imports work from app/
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import clean_text, score_priority, map_issue_group
from src.attention_model import load_vocab, BiLSTMAttention, tokenize

logger = logging.getLogger(__name__)

MODELS_DIR  = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
HF_REPO     = "Wael-Elzuhairy/complaint-analyzer-models"


# ===========================================================================
# Auto-download models from HuggingFace Hub if not present locally
# ===========================================================================

def download_models_if_needed(models_dir: Path = None) -> None:
    """Check for missing model files and download them from HuggingFace Hub.

    Safe to call every startup — skips files that already exist.
    Works on localhost (after git clone) and Streamlit Cloud.
    """
    if models_dir is None:
        models_dir = MODELS_DIR

    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        logger.warning("huggingface_hub not installed — cannot auto-download models.")
        return

    models_dir.mkdir(parents=True, exist_ok=True)

    # ── Single-file models ────────────────────────────────────────────────────
    single_files = [
        "bilstm_attention.pt",
        "baseline_pipeline_product.joblib",
        "baseline_pipeline_issue_group.joblib",
    ]
    for fname in single_files:
        if not (models_dir / fname).exists():
            logger.info(f"Downloading {fname} from HuggingFace Hub…")
            hf_hub_download(repo_id=HF_REPO, filename=fname,
                            local_dir=str(models_dir))
            logger.info(f"  Done: {fname}")

    # ── DistilBERT folder (~255 MB) ───────────────────────────────────────────
    distilbert_weights = models_dir / "distilbert_best_product" / "model.safetensors"
    if not distilbert_weights.exists():
        logger.info("Downloading DistilBERT model (~255 MB) from HuggingFace Hub…")
        snapshot_download(
            repo_id=HF_REPO,
            local_dir=str(models_dir),
            allow_patterns="distilbert_best_product/*",
        )
        logger.info("  Done: distilbert_best_product/")

MAX_LEN_BILSTM      = 256
MAX_LEN_DISTILBERT  = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ===========================================================================
# TextRank Summarizer (lightweight, no GPU needed)
# ===========================================================================

def summarize_text(text: str, n_sentences: int = 2) -> str:
    """Generate an extractive summary using TextRank.

    Args:
        text: Input complaint narrative.
        n_sentences: Number of sentences to extract.

    Returns:
        Extractive summary string.
    """
    try:
        import nltk
        from sumy.parsers.plaintext import PlaintextParser
        from sumy.nlp.tokenizers import Tokenizer
        from sumy.summarizers.text_rank import TextRankSummarizer

        try:
            nltk.data.find("tokenizers/punkt")
        except LookupError:
            nltk.download("punkt", quiet=True)
            nltk.download("punkt_tab", quiet=True)

        parser = PlaintextParser.from_string(text, Tokenizer("english"))
        summarizer = TextRankSummarizer()
        summary = summarizer(parser.document, n_sentences)
        result = " ".join(str(s) for s in summary)
        return result if result else text[:300] + "…"
    except Exception:
        # Fallback: return first 300 characters
        return text[:300] + "…"


# ===========================================================================
# Complaint Analyzer
# ===========================================================================

class ComplaintAnalyzer:
    """Unified inference class that wraps all three models.

    Loads models lazily on first use to avoid startup delays. Each model is
    cached after first load.

    Usage:
        analyzer = ComplaintAnalyzer()
        results = analyzer.predict("I cannot access my checking account...")
    """

    def __init__(self):
        # Download any missing models from HuggingFace Hub before first use.
        # This is instant when models already exist locally.
        download_models_if_needed()

        self._baseline = None
        self._bilstm = None
        self._vocab = None
        self._distilbert_model = None
        self._distilbert_tokenizer = None
        self._le_product = None
        self._le_issue = None
        self._label_names = None
        self._issue_label_names = None
        self._comparison_metrics = None

    def _ensure_product_encoder(self):
        if self._le_product is None:
            self._le_product = joblib.load(MODELS_DIR / "label_encoder_product.joblib")
            self._label_names = list(self._le_product.classes_)

    def _ensure_issue_encoder(self):
        if self._le_issue is None:
            self._le_issue = joblib.load(MODELS_DIR / "label_encoder_issue_group.joblib")
            self._issue_label_names = list(self._le_issue.classes_)

    @property
    def label_names(self) -> list[str]:
        self._ensure_product_encoder()
        return self._label_names

    @property
    def issue_label_names(self) -> list[str]:
        self._ensure_issue_encoder()
        return self._issue_label_names

    def _load_baseline(self):
        if self._baseline is None:
            path = MODELS_DIR / "baseline_pipeline_product.joblib"
            if not path.exists():
                raise FileNotFoundError(f"Baseline model not found at {path}. Run notebook 02 first.")
            self._baseline = joblib.load(path)
            logger.info("Baseline model loaded.")

    def _load_bilstm(self):
        if self._bilstm is None:
            self._ensure_product_encoder()
            vocab_path  = MODELS_DIR / "vocab.json"
            model_path  = MODELS_DIR / "bilstm_attention.pt"
            if not vocab_path.exists() or not model_path.exists():
                raise FileNotFoundError("BiLSTM model/vocab not found. Run notebook 03 first.")

            self._vocab = load_vocab(vocab_path)
            model = BiLSTMAttention(
                vocab_size=len(self._vocab),
                num_classes=len(self._label_names),
            )
            model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            self._bilstm = model
            logger.info("BiLSTM model loaded.")

    def _load_distilbert(self):
        if self._distilbert_model is None:
            from src.transformer_model import load_transformer
            model_dir = MODELS_DIR / "distilbert_best_product"
            if not model_dir.exists():
                raise FileNotFoundError(f"DistilBERT model not found at {model_dir}. Run notebook 04 first.")
            self._distilbert_model, self._distilbert_tokenizer = load_transformer(task="product")
            self._distilbert_model.to(DEVICE)
            self._distilbert_model.eval()
            logger.info("DistilBERT model loaded.")

    # -----------------------------------------------------------------------
    # Predict with individual models
    # -----------------------------------------------------------------------

    def predict_baseline(self, text: str) -> dict:
        """Run baseline TF-IDF + LR prediction.

        Returns:
            Dict with 'prediction', 'confidence', 'all_probabilities'.
        """
        self._load_baseline()
        self._ensure_product_encoder()
        clean = clean_text(text)
        import pandas as pd
        probas = self._baseline.predict_proba(pd.Series([clean]))[0]
        pred_idx = probas.argmax()
        return {
            "prediction": self._label_names[pred_idx],
            "confidence": float(probas[pred_idx]),
            "all_probabilities": {
                name: float(p) for name, p in zip(self._label_names, probas)
            },
        }

    def predict_bilstm(self, text: str) -> dict:
        """Run BiLSTM + Attention prediction.

        Returns:
            Dict with 'prediction', 'confidence', 'all_probabilities', 'attention_tokens', 'attention_weights'.
        """
        self._load_bilstm()
        self._ensure_product_encoder()
        from src.attention_model import get_attention_for_text
        tokens, attn_weights, probas = get_attention_for_text(
            self._bilstm, clean_text(text), self._vocab,
            max_len=MAX_LEN_BILSTM, device=DEVICE,
        )
        pred_idx = probas.argmax()
        return {
            "prediction": self._label_names[pred_idx],
            "confidence": float(probas[pred_idx]),
            "all_probabilities": {
                name: float(p) for name, p in zip(self._label_names, probas)
            },
            "attention_tokens": tokens,
            "attention_weights": attn_weights.tolist(),
        }

    def predict_distilbert(self, text: str) -> dict:
        """Run DistilBERT prediction.

        Returns:
            Dict with 'prediction', 'confidence', 'all_probabilities'.
        """
        self._load_distilbert()
        self._ensure_product_encoder()

        inputs = self._distilbert_tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=MAX_LEN_DISTILBERT,
            return_tensors="pt",
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self._distilbert_model(**inputs)
            probas = F.softmax(outputs.logits, dim=-1)[0].cpu().numpy()

        pred_idx = probas.argmax()
        return {
            "prediction": self._label_names[pred_idx],
            "confidence": float(probas[pred_idx]),
            "all_probabilities": {
                name: float(p) for name, p in zip(self._label_names, probas)
            },
        }

    def predict_issue_group(self, text: str) -> dict:
        """Predict issue group using the baseline issue-group classifier.

        Falls back to rule-based mapping if model not available.
        """
        self._ensure_issue_encoder()
        issue_model_path = MODELS_DIR / "baseline_pipeline_issue_group.joblib"
        if issue_model_path.exists():
            import pandas as pd
            issue_pipeline = joblib.load(issue_model_path)
            clean = clean_text(text)
            probas = issue_pipeline.predict_proba(pd.Series([clean]))[0]
            pred_idx = probas.argmax()
            return {
                "prediction": self._issue_label_names[pred_idx],
                "confidence": float(probas[pred_idx]),
                "all_probabilities": {
                    name: float(p) for name, p in zip(self._issue_label_names, probas)
                },
            }
        else:
            # Fallback: keyword-based
            text_lower = text.lower()
            if any(kw in text_lower for kw in ["fraud", "scam", "stolen", "unauthorized", "identity theft"]):
                group = "Fraud/Security"
            elif any(kw in text_lower for kw in ["billing", "payment", "charge", "fee", "refund"]):
                group = "Billing/Payment"
            elif any(kw in text_lower for kw in ["cannot login", "locked", "access", "password"]):
                group = "Account Access"
            elif any(kw in text_lower for kw in ["report", "credit report", "collection", "debt"]):
                group = "Reporting/Collections"
            elif any(kw in text_lower for kw in ["loan", "mortgage", "foreclosure", "servicer"]):
                group = "Loan Servicing"
            else:
                group = "Customer Service"
            return {"prediction": group, "confidence": 0.6, "all_probabilities": {}}

    # -----------------------------------------------------------------------
    # Unified prediction
    # -----------------------------------------------------------------------

    def predict(self, text: str, models: list[str] = None) -> dict:
        """Run all models and return combined results.

        Args:
            text: Complaint narrative text.
            models: List of model names to run. Defaults to all available.

        Returns:
            Dictionary with results from each model plus metadata.
        """
        if models is None:
            models = ["baseline", "bilstm", "distilbert"]

        results = {}

        for m in models:
            try:
                if m == "baseline":
                    results["baseline"] = self.predict_baseline(text)
                elif m == "bilstm":
                    results["bilstm"] = self.predict_bilstm(text)
                elif m == "distilbert":
                    results["distilbert"] = self.predict_distilbert(text)
            except FileNotFoundError as e:
                results[m] = {"error": str(e)}

        # Issue group prediction
        results["issue_group"] = self.predict_issue_group(text)

        # Priority scoring
        results["priority"] = score_priority(text)

        # Extractive summary
        results["summary"] = summarize_text(text, n_sentences=2)

        # Determine most confident model
        confidences = {
            m: results[m]["confidence"]
            for m in ["baseline", "bilstm", "distilbert"]
            if m in results and "confidence" in results[m]
        }
        if confidences:
            results["most_confident_model"] = max(confidences, key=confidences.get)

        return results

    def load_comparison_metrics(self) -> dict | None:
        """Load saved model comparison metrics from results/.

        Returns:
            Dict mapping model_name → metrics, or None if not available.
        """
        if self._comparison_metrics is not None:
            return self._comparison_metrics

        metric_files = {
            "TF-IDF + LR": RESULTS_DIR / "baseline_product_metrics.json",
            "BiLSTM+Attention": RESULTS_DIR / "bilstm_attention_product_metrics.json",
            "DistilBERT": RESULTS_DIR / "distilbert_product_metrics.json",
        }
        metrics = {}
        for name, path in metric_files.items():
            if path.exists():
                with open(path) as f:
                    metrics[name] = json.load(f)

        self._comparison_metrics = metrics if metrics else None
        return self._comparison_metrics
