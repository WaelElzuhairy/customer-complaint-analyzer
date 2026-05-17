# Customer Complaint Analyzer

**Comparing TF-IDF, BiLSTM+Attention, and DistilBERT for CFPB Complaint Classification**

A complete academic ML project that downloads the CFPB Consumer Complaint dataset,
trains three models for multi-class classification, evaluates and compares them,
and serves a live Streamlit demo application.

---
 
## Project Structure

```
customer-complaint-analyzer/
├── data/
│   ├── raw/           # Downloaded CSV (auto-populated)
│   └── processed/     # Filtered train/val/test splits
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_baseline.ipynb         # TF-IDF + Logistic Regression
│   ├── 03_attention_model.ipynb  # BiLSTM + Attention
│   ├── 04_transformer_finetune.ipynb  # DistilBERT fine-tuning
│   └── 05_error_analysis.ipynb   # Comparison & error analysis
├── src/
│   ├── data_prep.py         # Data pipeline
│   ├── baseline.py          # Baseline model
│   ├── attention_model.py   # BiLSTM + Attention (PyTorch)
│   ├── transformer_model.py # DistilBERT fine-tuning
│   └── evaluate.py          # Metrics & visualization
├── app/
│   ├── streamlit_app.py     # Demo web application
│   └── inference.py         # Unified inference module
├── models/                  # Saved model checkpoints
├── results/                 # Metrics JSON, plots, confusion matrices
├── report/
│   └── report_draft.md      # Academic report draft
├── requirements.txt
└── README.md
```

---

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU recommended (runs on CPU but training will be slow)
- ~5 GB free disk space

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Pipeline

### Step 1 — Download & Prepare Data

```bash
python src/data_prep.py
```

This will:
- Fetch the CFPB dataset (~1.8 GB zip → ~3 GB CSV)
- Filter to narratives from Aug 2023 onward with at least 20 words
- Balance classes, encode labels
- Save train/val/test splits to `data/processed/`

**Expected time:** 20–40 minutes (mostly depends on download speed)

### Step 2 — Exploratory Data Analysis

Open and run `notebooks/01_eda.ipynb`. Check cell 8 for the max_length cutoff decision.

### Step 3 — Train All Models (run notebooks in order)

```bash
jupyter notebook
```

Run each notebook sequentially:
1. `notebooks/02_baseline.ipynb` (~5 min)
2. `notebooks/03_attention_model.ipynb` (~30–60 min on GPU)
3. `notebooks/04_transformer_finetune.ipynb` (~45–90 min on GPU)
4. `notebooks/05_error_analysis.ipynb` (after all models are trained)

### Step 4 — Launch the Demo App

```bash
streamlit run app/streamlit_app.py
```

The app will be available at `http://localhost:8501`

---

## Models

Pre-trained models are hosted on HuggingFace Hub:
**[`Wael-Elzuhairy/complaint-analyzer-models`](https://huggingface.co/Wael-Elzuhairy/complaint-analyzer-models)**

| Model | File | Size |
|-------|------|------|
| TF-IDF + LR | `baseline_pipeline_product.joblib` | 5.2 MB |
| Issue Group LR | `baseline_pipeline_issue_group.joblib` | 4.8 MB |
| BiLSTM+Attention | `bilstm_attention.pt` | 13.8 MB |
| DistilBERT | `distilbert_best_product/` | 255 MB |
| Label encoders | `models/label_encoder_*.joblib` | included in repo |
| Vocabulary | `models/vocab.json` | included in repo |

### Download Pre-trained Models

To skip training and use the pre-trained models directly, download them from HuggingFace Hub:

```python
from huggingface_hub import hf_hub_download, snapshot_download
import os

HF_REPO = "Wael-Elzuhairy/complaint-analyzer-models"
os.makedirs("models", exist_ok=True)

# Download BiLSTM + both baselines (fast, ~24 MB total)
for fname in ["bilstm_attention.pt", "baseline_pipeline_product.joblib",
              "baseline_pipeline_issue_group.joblib"]:
    hf_hub_download(repo_id=HF_REPO, filename=fname, local_dir="models")

# Download DistilBERT (~255 MB)
snapshot_download(repo_id=HF_REPO, local_dir="models",
                  allow_patterns="distilbert_best_product/*")

print("Models ready in models/")
```

Or via the Hugging Face CLI:
```bash
pip install huggingface_hub
huggingface-cli download Wael-Elzuhairy/complaint-analyzer-models --local-dir models
```

---

## Results

After training, all results are saved to `results/`:

| File | Content |
|------|---------|
| `comparison_table.csv` | Side-by-side metrics across all models |
| `*_metrics.json` | Detailed per-model metrics |
| `confusion_matrix_*.png` | Normalized confusion matrices |
| `per_class_f1_comparison.png` | Grouped bar chart by class |
| `model_comparison_bar.png` | Accuracy/F1 overview chart |
| `attention_visualization.png` | BiLSTM attention weights on sample inputs |
| `category_summaries.json` | TextRank summaries per product category |

---

## Deployment on Streamlit Community Cloud

### 1. Repository is Already on GitHub

The code lives at: **https://github.com/WaelElzuhairy/customer-complaint-analyzer**

Clone it with:
```bash
git clone https://github.com/WaelElzuhairy/customer-complaint-analyzer.git
cd customer-complaint-analyzer
```

> **Note:** Large model files are excluded from the repo and hosted on HuggingFace Hub.
> Refer to the [Models section](#models) above for download instructions.

### 2. Deploy on Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Sign in with GitHub
3. Click **"New app"**
4. Select your repository, branch `main`, and set the main file to `app/streamlit_app.py`
5. Click **Deploy**

### 3. Streamlit Cloud Requirements

Add a `packages.txt` file for any system-level dependencies if needed:

```
# packages.txt  (leave empty if no system packages are required)
```

Streamlit Cloud will automatically pick up `requirements.txt` from the project root.

### Secrets (if needed)

If models are loaded from HuggingFace Hub or S3, add credentials under:
**Streamlit Cloud → App → Settings → Secrets**

---

## Research Questions

| Question | Answer Source |
|----------|--------------|
| Can DistilBERT outperform BiLSTM+Attention? | `results/comparison_table.csv` |
| Which categories are hardest/easiest to classify? | `results/per_class_f1_comparison.png` |
| Does attention improve over the baseline? | Notebooks 02 vs 03 metrics |
| Where does the Transformer fall short? | `notebooks/05_error_analysis.ipynb` |
| Can this system help prioritize complaints? | `results/eda_priority_distribution.png` |

---

## Citation

If using this project for academic purposes:

```
@misc{complaint-analyzer-2026,
  title  = {Customer Complaint Analyzer: Comparing Fine-Tuned Transformers with Attention Models},
  author = {[Your Name]},
  year   = {2026},
  url    = {https://github.com/WaelElzuhairy/customer-complaint-analyzer}
}
```

---

## License

MIT License. Dataset source: [CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/) (public domain).
