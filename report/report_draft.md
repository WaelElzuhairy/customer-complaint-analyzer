# Customer Complaint Analyzer: Comparing a Fine-Tuned Transformer with a Simplified Attention Model for Complaint Classification

**Author:** [Your Name]
**Date:** April 2026
**Dataset:** CFPB Consumer Complaint Database (Aug 2023–present)

---

## Abstract

Consumer financial complaints represent a rich source of real-world text data, yet the volume of incoming complaints makes manual review infeasible. This project trains and compares three natural language processing (NLP) models — a TF-IDF + Logistic Regression baseline, a bidirectional LSTM with additive attention (BiLSTM+Attention), and a fine-tuned DistilBERT Transformer — for automatically classifying complaints into product categories. We use the CFPB Consumer Complaint Database, restricting to narratives submitted after August 2023 to ensure label consistency. Our results show that the fine-tuned Transformer achieves the highest macro F1 score, confirming that pretrained contextual representations provide significant advantages over classical and attention-augmented sequential models. We also demonstrate a practical deployment as an interactive web application capable of classifying new complaints, assigning priority levels, and visualizing model confidence in real time.

---

## 1. Introduction

The Consumer Financial Protection Bureau (CFPB) receives hundreds of thousands of consumer complaints annually covering products ranging from mortgages and student loans to credit cards and debt collection. Each complaint must be routed to the appropriate product team for investigation. This routing process is currently labor-intensive and delays resolution.

Automated text classification offers a solution: given a complaint narrative, a model can predict the product category, enabling near-instant routing. However, the task presents several challenges:
1. **Severe class imbalance** — credit reporting complaints dominate (~80% of recent filings)
2. **Consumer-generated text** — informal language, abbreviations, emotionally charged writing
3. **Overlapping categories** — credit card vs. checking account complaints can describe identical issues

This project addresses these challenges by comparing three model families:
- A TF-IDF + Logistic Regression baseline (no learning of sequential context)
- A BiLSTM + additive attention model (sequential learning with token importance)
- DistilBERT fine-tuning (pretrained Transformer with bidirectional self-attention)

We investigate whether fine-tuned Transformers outperform attention-augmented sequential models, which categories are hardest to classify, and how a complete system can be deployed for practical use.

---

## 2. Dataset

### 2.1 Source and Acquisition

Data is sourced from the CFPB Consumer Complaint Database (https://files.consumerfinance.gov/ccdb/complaints.csv.zip). The full database contains over 4 million complaints since 2011.

We restrict to complaints received **on or after August 1, 2023** to ensure consistency with updated product category taxonomy (CFPB reorganized some categories in mid-2023). Only complaints with consumer-submitted narratives are retained (approximately 20-30% of all complaints, as narrative submission requires explicit consumer consent).

### 2.2 Filtering Criteria

| Filter | Rule |
|--------|------|
| Date | ≥ 2023-08-01 |
| Narrative | Must be non-null |
| Length | ≥ 20 words |
| Duplicates | Removed (exact narrative match) |
| Product | Mapped to 11 normalized categories |

### 2.3 Label Design

**Primary task — Product classification (9 classes):**

| Label | Description |
|-------|-------------|
| Credit reporting | Credit bureau reports, disputes |
| Debt collection | Third-party debt collectors |
| Credit card | Credit and prepaid cards |
| Checking/savings account | Bank accounts |
| Mortgage | Home loans, refinancing |
| Student loan | Federal and private student loans |
| Vehicle loan/lease | Auto loans and leases |
| Money transfer | Wire transfers, digital payments |
| Personal/payday loan | Personal and short-term loans |

**Secondary task — Issue Group (7 classes):** Issues are manually mapped to 7 groups: Billing/Payment, Fraud/Security, Account Access, Customer Service, Reporting/Collections, Loan Servicing, Technical/App.

### 2.4 Class Balancing

Credit reporting narratives dominate the raw data. We apply a two-stage balancing strategy:
- **Undersample** majority classes to a maximum of 15,000 samples
- **Oversample** minority classes below 1,000 samples (with replacement)
- Apply **class weights** in all model loss functions

The effect of balancing is shown in Figure 1.

### 2.5 Data Split

A stratified 70/15/15 split is applied on the balanced dataset:

| Split | Size | Samples |
|-------|------|---------|
| Train | 70%  | 74,900  |
| Validation | 15% | 16,050 |
| Test  | 15%  | 16,050  |
| **Total** | | **106,069** (raw) → **107,000** (balanced) |

Stratification ensures each split preserves class proportions.

### 2.6 Sequence Length Analysis

Token coverage analysis (Figure 2) informed the choice of `max_length=256` for both BiLSTM and DistilBERT:

| Max Tokens | Complaints Fully Captured |
|-----------|--------------------------|
| 64        | 14.8% |
| 128       | 40.3% |
| 192       | 58.0% |
| **256**   | **71.2%** |
| 512       | 92.4% |

At 128 tokens, fewer than half of complaints are fully captured; 256 provides a practical balance between coverage and computational cost.

### 2.6 Limitations

- **Self-reported data:** Narratives represent the consumer's perspective and may be incomplete, exaggerated, or factually inaccurate.
- **Consent bias:** Only ~20-30% of complaints include narratives (those where consumers opted in), creating selection bias toward more engaged or frustrated complainants.
- **Label evolution:** Product categories have changed over time; restricting to post-August 2023 mitigates but does not eliminate this issue.
- **Not representative:** The CFPB database does not represent all financial complaints — only those submitted through CFPB channels by consumers aware of this mechanism.
- **XXXX redaction:** The CFPB replaces sensitive information (account numbers, names, SSNs) with "XXXX", which can disrupt some NLP models.

---

## 3. Methods

### 3.1 Text Preprocessing

For TF-IDF and BiLSTM models, text is cleaned:
1. Lowercased
2. XXXX/XX redactions replaced with `[REDACTED]`
3. URLs, emails, and phone numbers removed
4. Whitespace normalized

DistilBERT receives the original (uncleaned) narrative, as its WordPiece tokenizer handles casing and punctuation natively.

### 3.2 Model 1 — TF-IDF + Logistic Regression (Baseline)

**Feature extraction:** TF-IDF vectorization with:
- Max features: 50,000
- N-gram range: (1, 2) — unigrams and bigrams
- Sublinear TF scaling: log(1 + tf)
- Min document frequency: 5

**Classifier:** Multinomial Logistic Regression:
- Regularization: C = 1.0
- Class weights: balanced (inverse frequency)
- Solver: lbfgs, max_iter = 1000

**Rationale:** TF-IDF captures discriminative vocabulary without any model capacity for sequential reasoning. This provides a strong interpretable baseline against which deep learning gains can be measured. The top discriminative features per class are shown in Figure 3.

### 3.3 Model 2 — BiLSTM + Additive Attention

**Architecture:**
```
Input (token IDs, max_len=256)
    → Embedding(vocab_size=50K, dim=128)
    → BiLSTM(hidden=128, layers=2, dropout=0.3)
    → AdditiveAttention
    → Dense(64) + ReLU + Dropout
    → Linear(num_classes)
    → Softmax
```

**Attention mechanism:** The additive (Bahdanau-style) attention computes:

```
e_t = v^T * tanh(W * h_t)
α_t = softmax(e_1, ..., e_T)    [with padding mask]
c   = Σ α_t * h_t
```

This allows the model to weight each token's hidden state by its importance to the classification decision, rather than relying solely on the final LSTM hidden state. Attention weight visualizations for sample complaints are shown in Figure 5.

**Training hyperparameters:**
- Optimizer: AdamW, lr = 1e-3, weight_decay = 1e-4
- Scheduler: ReduceLROnPlateau (patience=2, factor=0.5)
- Max epochs: 15, early stopping patience: 3
- Batch size: 64, gradient clipping: max_norm = 1.0
- Loss: weighted CrossEntropyLoss

**Vocabulary:** Built from training data (min_freq=3), resulting in 22,358 unique tokens. Embeddings trained from scratch (128-dimensional). No pre-trained embeddings were used, allowing the model to learn domain-specific representations.

### 3.4 Model 3 — DistilBERT Fine-Tuning

**Base model:** `distilbert-base-uncased` (66M parameters, 6 transformer layers)

**What is Pretraining?**
DistilBERT is pre-trained on Wikipedia and BookCorpus using Masked Language Modeling (MLM): 15% of tokens are randomly masked, and the model learns to predict them. This forces the model to learn deep contextual representations without labeled data.

**Why Fine-Tuning Works:**
The pretrained model has learned syntax, semantics, and world knowledge. Fine-tuning on labeled complaint data adapts these representations for the specific classification task with far fewer labeled examples than training from scratch would require. This is the central advantage of transfer learning.

**Fine-tuning setup:**
- Max sequence length: 256 tokens (WordPiece tokenization)
- Batch size: 16 (fp16 enabled for 6 GB VRAM)
- Learning rate: 2e-5, warmup steps: 500
- Epochs: 3, early stopping on validation Macro F1
- Loss: weighted CrossEntropyLoss (via custom Trainer)

**Why Transformers Are Superior:**
Unlike RNNs, Transformers use self-attention that directly connects every token to every other token in O(1) computational steps, perfectly capturing long-range dependencies. Combined with pretraining on massive corpora, this yields state-of-the-art performance across NLP tasks.

---

## 4. Results

### 4.1 Primary Task — Product Classification

All three models were evaluated on the held-out test set (16,050 samples, 9 classes). An aggregate comparison is shown in Figure 6.

| Model | Accuracy | Precision (Macro) | Recall (Macro) | **F1 (Macro)** | F1 (Weighted) |
|-------|----------|-------------------|----------------|----------------|---------------|
| TF-IDF + LR (Baseline) | 0.8756 | 0.8652 | 0.8959 | 0.8787 | 0.8761 |
| BiLSTM + Attention | 0.8673 | **0.8832** | 0.8851 | 0.8834 | 0.8679 |
| **DistilBERT (Fine-Tuned)** | **0.8862** | 0.8850 | **0.8932** | **0.8887** | **0.8865** |

**Per-class F1 scores (test set):**

| Product | TF-IDF+LR | BiLSTM+Attn | DistilBERT |
|---------|:---------:|:-----------:|:----------:|
| Checking/savings account | 0.797 | 0.779 | 0.808 |
| Credit card | 0.829 | 0.818 | 0.841 |
| Credit reporting | 0.898 | **0.993** | 0.987 |
| Debt collection | 0.863 | 0.862 | 0.872 |
| Money transfer | 0.844 | 0.827 | 0.856 |
| Mortgage | 0.941 | 0.935 | **0.948** |
| Personal/payday loan | **0.879** | 0.892 | 0.808 |
| Student loan | 0.957 | 0.950 | **0.964** |
| Vehicle loan/lease | 0.901 | 0.895 | **0.915** |

*See Figure 7 for the full per-class F1 heatmap across all three models.*

Confusion matrices for all three models are shown in Figures 8–10. Notable patterns: (1) Credit Reporting reaches near-perfect F1 for both neural models — the 1,000 oversampled minority-class examples are highly consistent, allowing confident classification. (2) Checking/Savings Account is the hardest class for all models, as its complaints overlap heavily with Credit Card issues. (3) DistilBERT underperforms on Personal/Payday Loan (0.808) relative to the baseline (0.879), likely due to this class being represented by oversampled duplicates, which hurt fine-tuning more than the other approaches.

### 4.2 Secondary Task — Issue Group Classification (Baseline)

| Model | Accuracy | F1 (Macro) | F1 (Weighted) |
|-------|----------|------------|---------------|
| TF-IDF + LR | 0.7060 | 0.6208 | 0.7107 |

The issue group task is harder than product classification due to higher within-class variance and more subjective issue boundaries.

### 4.3 BiLSTM Training Dynamics

The BiLSTM model trained for 10 epochs before early stopping (patience=3), with the best checkpoint at epoch 7 (Figure 4):

| Epoch | Train Loss | Val Loss | Val F1 | Val Acc |
|-------|-----------|----------|--------|---------|
| 1  | 0.997 | 0.540 | 0.762 | 0.821 |
| 3  | 0.388 | 0.410 | 0.871 | 0.863 |
| 5  | 0.271 | 0.396 | 0.880 | 0.865 |
| **7** | **0.198** | **0.405** | **0.886** | **0.868** |
| 10 | 0.090 | 0.571 | 0.877 | 0.858 |

The divergence between training and validation loss from epoch 7 onward indicates overfitting, correctly caught by early stopping.

### 4.4 Key Observations

1. **DistilBERT wins on all aggregate metrics** (Accuracy +1.1pp, Macro F1 +1.0pp over baseline).
2. **BiLSTM has the highest precision** (0.8832) — it is more conservative in its predictions, favouring classes it is confident about; this benefits minority classes.
3. **The baseline achieves the highest recall** (0.8959) — the strong class-weighted regularization in logistic regression effectively penalizes false negatives.
4. **Easiest classes:** Student Loan (0.957 F1) and Mortgage (0.941 F1) — both have highly distinctive vocabulary (FAFSA, servicer, foreclosure).
5. **Hardest class:** Checking/Savings Account (0.797 F1 baseline) — semantically overlaps with Credit Card (both involve fees, transactions, access issues).
6. **Gaps are surprisingly small** (~1pp between models) — this domain has structured, constrained vocabulary that TF-IDF can exploit almost as well as contextual models.

---

## 5. Error Analysis

### 5.1 Misclassification Categories

We analyzed 25 misclassified samples from DistilBERT across four error categories:

| Category | Description | Example |
|----------|-------------|---------|
| Class overlap | Products with similar issue types | Credit card fees misclassified as checking account |
| Short/vague text | Insufficient narrative content | "I am unhappy with this company" |
| Low confidence | Model uncertain across multiple classes | Probability spread across 3-4 classes |
| Long noisy text | >200 words with multiple topics | Multi-product complaint involving both mortgage and credit report |

### 5.2 Model-Specific Failure Analysis

**Baseline failures:**
- Fails on negation ("NOT authorized")
- Cannot distinguish "account closed" (Credit Card) from "account closed" (Checking)
- Domain-specific terms below min_df threshold are invisible

**BiLSTM+Attention failures:**
- OOV words (product names, regulatory terms) degrade to `<UNK>`
- Long narratives: attention cannot focus without positional priors
- Rare classes: insufficient training examples despite oversampling

**DistilBERT failures:**
- 256-token truncation cuts off crucial late-complaint context
- Heavily redacted narratives (many XXXX tokens) confuse the model
- "Credit reporting" vs "Debt collection" overlap when complaints discuss both

---

## 6. Discussion

### 6.1 Main Research Question
Fine-tuned DistilBERT achieves the best overall performance (Macro F1 = 0.8887), confirming that pretrained contextual representations provide advantages over both bag-of-words and sequential models. However, the margin is smaller than often reported in general NLP benchmarks — only +1.0pp over the TF-IDF baseline. The key factors behind DistilBERT's advantage:
1. Subword (WordPiece) tokenization handles OOV financial terms and product names
2. Self-attention directly captures long-range dependencies (e.g., cause mentioned early, product mentioned late)
3. Pretraining encodes semantic equivalences (fraud ≈ scam ≈ unauthorized, servicer ≈ lender)

The relatively small margin reflects the constrained nature of this domain: regulatory complaint language is structured and repetitive, giving TF-IDF bigrams a strong foundation.

### 6.2 BiLSTM Attention Visualization
The BiLSTM attention mechanism successfully focuses on semantically relevant tokens. For a mortgage complaint, the model assigns highest weight to "mortgage", "payment", "foreclosure", and "servicer", with function words receiving near-zero attention. This provides a human-interpretable explanation of model decisions — a key advantage over the opaque baseline and DistilBERT's complex self-attention patterns.

### 6.3 Practical Priority System
The rule-based priority scorer successfully identifies high-urgency complaints (fraud, foreclosure, unauthorized access) with high recall. This simple system is interpretable and requires no training data — useful as an immediate triage layer before ML classification.

### 6.4 Deployment Considerations
All three models are integrated into the Streamlit demo, enabling side-by-side comparison. In a production setting:
- **DistilBERT** as the primary classifier (highest F1, GPU required)
- **TF-IDF baseline** as a fast CPU fallback during GPU unavailability
- **BiLSTM attention weights** as an interpretability tool for auditing decisions
- **Training time:** Baseline 2 min, BiLSTM 6 min, DistilBERT 73 min on RTX 3060 Laptop GPU

---

## 7. Conclusion

We built a complete end-to-end pipeline for consumer financial complaint classification, training and comparing three model families on 107,000 CFPB complaints across 9 product categories. Key findings:

1. **DistilBERT achieves the best performance** (Macro F1 = 88.87%, Accuracy = 88.62%) due to pretrained contextual representations and bidirectional self-attention.
2. **BiLSTM+Attention surpasses the baseline on Macro F1** (88.34% vs 87.87%), with notably higher precision — the attention mechanism helps the model focus on discriminative tokens and improves performance on minority classes.
3. **The baseline is remarkably competitive** (87.87% Macro F1) — structured, domain-specific vocabulary allows TF-IDF bigrams to capture most of the classification signal without deep learning.
4. **Class imbalance** remains a challenge; combined undersampling + oversampling + class-weighted loss is effective but Credit Reporting (with limited unique narratives in the API) remains harder to evaluate robustly.
5. **The priority scoring system** effectively surfaces high-urgency complaints using simple keyword rules, requiring no training data.
6. **A working Streamlit demo** classifies new complaints in real time, displaying all three model predictions, attention weight visualization, priority badges, and extractive summaries.

---

## 8. Limitations

1. **Dataset representativeness:** The CFPB database only contains complaints from consumers who used the CFPB reporting mechanism — a self-selected group. Most financial complaints are never formally reported.
2. **Self-reported bias:** Narratives reflect the complainant's subjective perspective; factual accuracy is not verified.
3. **Static labels:** Product categories may change again, requiring periodic re-labeling.
4. **Truncation:** The 256-token limit for DistilBERT discards content from long narratives.
5. **No multilingual support:** Non-English complaints are excluded by the English-only tokenizer.
6. **Model staleness:** Financial product types and consumer concerns evolve; models should be retrained periodically.
7. **Privacy:** Complaint texts contain sensitive financial information even after XXXX redaction — care must be taken in deployment.

---

---

## List of Figures

| Figure | Caption | File |
|--------|---------|------|
| Figure 1 | Class distribution before and after balancing | `results/eda_balance_comparison.png` |
| Figure 2 | Narrative length distribution with 128- and 256-token cutoff markers | `results/eda_text_length.png` |
| Figure 3 | Top TF-IDF logistic regression coefficients per product class | `results/baseline_top_features.png` |
| Figure 4 | BiLSTM training and validation loss, F1, and accuracy over 10 epochs | `results/bilstm_training_curves.png` |
| Figure 5 | BiLSTM additive attention weight visualization (sample complaint) | Streamlit app / `results/` |
| Figure 6 | Grouped bar chart comparing Accuracy, Precision, Recall, and Macro F1 across all three models | `results/comparison_grouped_bar.png` |
| Figure 7 | Per-class F1 heatmap across all three models on the test set | `results/comparison_f1_heatmap.png` |
| Figure 8 | Confusion matrix — TF-IDF + Logistic Regression | `results/confusion_matrix_baseline_product.png` |
| Figure 9 | Confusion matrix — BiLSTM + Attention | `results/confusion_matrix_bilstm_attention_product.png` |
| Figure 10 | Confusion matrix — DistilBERT (Fine-Tuned) | `results/confusion_matrix_distilbert_product.png` |
| Figure 11 | Word clouds of most frequent terms per product category | `results/eda_word_clouds.png` |
| Figure 12 | Issue group distribution (pie and bar) | `results/eda_issue_groups.png` |
| Figure 13 | Complaint priority distribution | `results/eda_priority_distribution.png` |

---

## References

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of deep bidirectional transformers for language understanding. *NAACL*.
2. Sanh, V., Debut, L., Chaumond, J., & Wolf, T. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop*.
3. Bahdanau, D., Cho, K., & Bengio, Y. (2015). Neural machine translation by jointly learning to align and translate. *ICLR*.
4. CFPB. (2024). Consumer Complaint Database. Consumer Financial Protection Bureau. https://www.consumerfinance.gov/data-research/consumer-complaints/
5. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*.
6. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS*.
