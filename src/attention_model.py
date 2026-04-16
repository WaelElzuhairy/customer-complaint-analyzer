"""
BiLSTM + Attention Model for Complaint Classification.

Implements a Bidirectional LSTM with a custom additive attention mechanism
for multi-class text classification. The attention layer learns to focus
on the most informative parts of the input sequence.

Architecture:
    Embedding → BiLSTM (2-layer) → Additive Attention → Dense → Softmax

Usage:
    from src.attention_model import BiLSTMAttention, train_attention_model
"""

import json
import logging
import re
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1


# ===========================================================================
# Vocabulary
# ===========================================================================

def tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer.

    Args:
        text: Input text string (should be pre-cleaned).

    Returns:
        List of lowercase tokens.
    """
    return re.findall(r"\b\w+\b", text.lower())


def build_vocab(
    texts: pd.Series,
    max_vocab: int = 50000,
    min_freq: int = 3,
) -> dict[str, int]:
    """Build a word → index vocabulary from training texts.

    Args:
        texts: Series of text strings.
        max_vocab: Maximum vocabulary size.
        min_freq: Minimum word frequency to include.

    Returns:
        Dictionary mapping word to integer index.
    """
    counter = Counter()
    for text in texts:
        counter.update(tokenize(text))

    # Filter by frequency and cap size
    vocab = {PAD_TOKEN: PAD_IDX, UNK_TOKEN: UNK_IDX}
    for word, freq in counter.most_common(max_vocab):
        if freq < min_freq:
            break
        vocab[word] = len(vocab)

    logger.info("Built vocabulary: %d words (from %d unique tokens).", len(vocab), len(counter))
    return vocab


def save_vocab(vocab: dict[str, int], path: Path) -> None:
    """Save vocabulary to JSON file."""
    with open(path, "w") as f:
        json.dump(vocab, f)


def load_vocab(path: Path) -> dict[str, int]:
    """Load vocabulary from JSON file."""
    with open(path) as f:
        return json.load(f)


# ===========================================================================
# Dataset
# ===========================================================================

class ComplaintDataset(Dataset):
    """PyTorch Dataset for tokenized complaint narratives.

    Converts text to integer sequences using the vocabulary, applies padding
    and truncation to a fixed length.
    """

    def __init__(
        self,
        texts: list[str],
        labels: list[int],
        vocab: dict[str, int],
        max_len: int = 256,
    ):
        """
        Args:
            texts: List of text strings.
            labels: List of integer labels.
            vocab: Word → index mapping.
            max_len: Maximum sequence length (truncate/pad to this).
        """
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokenize(self.texts[idx])
        ids = [self.vocab.get(t, UNK_IDX) for t in tokens[: self.max_len]]

        # Pad to max_len
        padding_len = self.max_len - len(ids)
        ids = ids + [PAD_IDX] * padding_len

        return torch.tensor(ids, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


# ===========================================================================
# Model Architecture
# ===========================================================================

class AdditiveAttention(nn.Module):
    """Additive (Bahdanau-style) attention mechanism.

    Computes attention weights over the BiLSTM hidden states using a
    two-layer feedforward network:

        score(h_t) = v^T * tanh(W * h_t + b)
        alpha_t = softmax(score(h_t))  (with padding mask)
        context = sum(alpha_t * h_t)

    This allows the model to learn which parts of the input sequence
    are most relevant for classification, rather than relying solely
    on the final hidden state.
    """

    def __init__(self, hidden_dim: int):
        """
        Args:
            hidden_dim: Size of the input hidden states (2 * lstm_hidden for BiLSTM).
        """
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(
        self,
        lstm_output: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute attention-weighted context vector.

        Args:
            lstm_output: BiLSTM output of shape (batch, seq_len, hidden_dim).
            mask: Boolean mask of shape (batch, seq_len), True for real tokens.

        Returns:
            context: Weighted sum of shape (batch, hidden_dim).
            weights: Attention weights of shape (batch, seq_len).
        """
        # Compute attention scores
        energy = torch.tanh(self.W(lstm_output))  # (B, L, H)
        scores = self.v(energy).squeeze(-1)        # (B, L)

        # Mask padding positions with large negative value
        scores = scores.masked_fill(~mask, -1e9)

        # Normalize to get attention distribution
        weights = F.softmax(scores, dim=1)         # (B, L)

        # Weighted sum of LSTM outputs
        context = torch.bmm(
            weights.unsqueeze(1),                  # (B, 1, L)
            lstm_output,                            # (B, L, H)
        ).squeeze(1)                                # (B, H)

        return context, weights


class BiLSTMAttention(nn.Module):
    """Bidirectional LSTM with Additive Attention for text classification.

    Architecture:
        1. Embedding layer: maps word indices to dense vectors
        2. BiLSTM: captures bidirectional sequential context
        3. Additive Attention: learns to weight important tokens
        4. Classifier: feedforward layers with dropout for prediction

    The attention mechanism addresses the limitation of using only the
    final hidden state, allowing the model to aggregate information
    from all timesteps with learned importance weights.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 128,
        num_classes: int = 11,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        """
        Args:
            vocab_size: Size of the vocabulary.
            embed_dim: Embedding dimension.
            hidden_dim: LSTM hidden state dimension (output will be 2*hidden_dim for BiLSTM).
            num_classes: Number of output classes.
            num_layers: Number of LSTM layers.
            dropout: Dropout probability.
        """
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_IDX)

        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        # Attention operates on BiLSTM output (2 * hidden_dim)
        self.attention = AdditiveAttention(hidden_dim * 2)

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.

        Args:
            x: Input token IDs of shape (batch, seq_len).

        Returns:
            logits: Class logits of shape (batch, num_classes).
            attention_weights: Attention distribution of shape (batch, seq_len).
        """
        # Create padding mask (True = real token, False = padding)
        mask = (x != PAD_IDX)

        # Embed tokens
        embedded = self.embedding(x)                    # (B, L, E)

        # BiLSTM encoding
        lstm_out, _ = self.lstm(embedded)               # (B, L, 2H)

        # Attention-weighted context
        context, attn_weights = self.attention(lstm_out, mask)  # (B, 2H), (B, L)

        # Classification
        logits = self.classifier(context)               # (B, C)

        return logits, attn_weights


# ===========================================================================
# Training
# ===========================================================================

def train_attention_model(
    train_texts: list[str],
    train_labels: list[int],
    val_texts: list[str],
    val_labels: list[int],
    vocab: dict[str, int],
    num_classes: int,
    class_weights: np.ndarray = None,
    max_len: int = 256,
    embed_dim: int = 128,
    hidden_dim: int = 128,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_epochs: int = 15,
    patience: int = 3,
    device: str = None,
) -> tuple[BiLSTMAttention, dict]:
    """Train the BiLSTM+Attention model with early stopping.

    Args:
        train_texts: Training narratives.
        train_labels: Training integer labels.
        val_texts: Validation narratives.
        val_labels: Validation integer labels.
        vocab: Word-to-index vocabulary.
        num_classes: Number of output classes.
        class_weights: Optional class weight array for loss function.
        max_len: Max sequence length.
        embed_dim: Embedding dimension.
        hidden_dim: LSTM hidden dimension.
        batch_size: Batch size.
        learning_rate: Learning rate for AdamW.
        weight_decay: Weight decay for AdamW.
        num_epochs: Maximum number of epochs.
        patience: Early stopping patience (epochs without improvement).
        device: Device string ('cuda' or 'cpu'). Auto-detected if None.

    Returns:
        Tuple of (trained model, training history dict).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Training on device: %s", device)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Create datasets and loaders
    train_dataset = ComplaintDataset(train_texts, train_labels, vocab, max_len)
    val_dataset = ComplaintDataset(val_texts, val_labels, vocab, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Build model
    model = BiLSTMAttention(
        vocab_size=len(vocab),
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        num_classes=num_classes,
    ).to(device)

    logger.info("Model parameters: %d", sum(p.numel() for p in model.parameters()))

    # Loss with optional class weights
    if class_weights is not None:
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weight_tensor)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_f1_macro": [], "val_accuracy": []}
    best_val_f1 = 0.0
    patience_counter = 0

    for epoch in range(num_epochs):
        # --- Train ---
        model.train()
        total_loss = 0.0
        for batch_x, batch_y in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            logits, _ = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            total_loss += loss.item() * batch_x.size(0)

        avg_train_loss = total_loss / len(train_dataset)
        history["train_loss"].append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                logits, _ = model(batch_x)
                loss = criterion(logits, batch_y)
                val_loss += loss.item() * batch_x.size(0)

                preds = logits.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_y.cpu().numpy())

        avg_val_loss = val_loss / len(val_dataset)
        val_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
        val_acc = accuracy_score(all_labels, all_preds)

        history["val_loss"].append(avg_val_loss)
        history["val_f1_macro"].append(val_f1)
        history["val_accuracy"].append(val_acc)

        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f | Val Loss: %.4f | Val F1: %.4f | Val Acc: %.4f | LR: %.6f",
            epoch + 1, num_epochs, avg_train_loss, avg_val_loss, val_f1, val_acc, current_lr,
        )

        # Early stopping
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), MODELS_DIR / "bilstm_attention.pt")
            logger.info("  → New best model saved (F1=%.4f).", best_val_f1)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info("Early stopping at epoch %d (patience=%d).", epoch + 1, patience)
                break

    # Load best model
    model.load_state_dict(torch.load(MODELS_DIR / "bilstm_attention.pt", weights_only=True))

    # Save history
    with open(RESULTS_DIR / "bilstm_training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return model, history


# ===========================================================================
# Inference
# ===========================================================================

def predict_with_attention(
    model: BiLSTMAttention,
    texts: list[str],
    vocab: dict[str, int],
    max_len: int = 256,
    batch_size: int = 64,
    device: str = None,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray]]:
    """Generate predictions, probabilities, and attention weights.

    Args:
        model: Trained BiLSTMAttention model.
        texts: List of text strings.
        vocab: Vocabulary mapping.
        max_len: Max sequence length.
        batch_size: Batch size for inference.
        device: Device string.

    Returns:
        Tuple of (predictions, probability_matrix, list_of_attention_weights).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    dummy_labels = [0] * len(texts)
    dataset = ComplaintDataset(texts, dummy_labels, vocab, max_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    all_preds = []
    all_probas = []
    all_attns = []

    with torch.no_grad():
        for batch_x, _ in loader:
            batch_x = batch_x.to(device)
            logits, attn_weights = model(batch_x)

            probas = F.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_probas.extend(probas.cpu().numpy())
            all_attns.extend(attn_weights.cpu().numpy())

    return np.array(all_preds), np.array(all_probas), all_attns


def get_attention_for_text(
    model: BiLSTMAttention,
    text: str,
    vocab: dict[str, int],
    max_len: int = 256,
    device: str = None,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Get word-level attention weights for a single text.

    Args:
        model: Trained model.
        text: Input text.
        vocab: Vocabulary.
        max_len: Max sequence length.
        device: Device string.

    Returns:
        Tuple of (tokens, attention_weights, class_probabilities).
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()
    model.to(device)

    tokens = tokenize(text)
    ids = [vocab.get(t, UNK_IDX) for t in tokens[:max_len]]
    padding_len = max_len - len(ids)
    ids = ids + [PAD_IDX] * padding_len

    x = torch.tensor([ids], dtype=torch.long).to(device)

    with torch.no_grad():
        logits, attn_weights = model(x)
        probas = F.softmax(logits, dim=1)

    # Only return attention for actual tokens (not padding)
    actual_len = min(len(tokens), max_len)
    attn = attn_weights[0, :actual_len].cpu().numpy()
    display_tokens = tokens[:actual_len]

    return display_tokens, attn, probas[0].cpu().numpy()
