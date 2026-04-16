"""
Data Preparation Module for Customer Complaint Analyzer.

Downloads, filters, preprocesses, and splits the CFPB Consumer Complaint
dataset for multi-class classification tasks.

Usage:
    python src/data_prep.py
"""

import os
import re
import io
import json
import zipfile
import logging
from pathlib import Path
from collections import Counter

import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / "data" / "raw"
DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

DATASET_URL = "https://files.consumerfinance.gov/ccdb/complaints.csv.zip"
RAW_CSV = DATA_RAW / "complaints.csv"   # only created if explicitly saved
RAW_ZIP = DATA_RAW / "complaints.csv.zip"  # temp zip, deleted after streaming

RANDOM_SEED = 42
MIN_WORD_COUNT = 20
DATE_CUTOFF = "2023-08-01"
MAX_PER_CLASS = 15000
MIN_PER_CLASS = 1000

RELEVANT_COLUMNS = [
    "Complaint ID",
    "Date received",
    "Product",
    "Issue",
    "Sub-issue",
    "Consumer complaint narrative",
    "Company response to consumer",
    "Timely response?",
]

# ---------------------------------------------------------------------------
# Product label normalization (post-2023 CFPB labels)
# ---------------------------------------------------------------------------

PRODUCT_LABEL_MAP = {
    "Credit reporting, credit repair services, or personal consumer reports": "Credit reporting",
    "Credit reporting": "Credit reporting",
    "Credit reporting or other personal consumer reports": "Credit reporting",  # post-2023 label
    "Debt or credit management": "Personal/payday loan",  # new post-2023 category
    "Debt collection": "Debt collection",
    "Credit card or prepaid card": "Credit card",
    "Credit card": "Credit card",
    "Prepaid card": "Credit card",
    "Checking or savings account": "Checking/savings account",
    "Mortgage": "Mortgage",
    "Student loan": "Student loan",
    "Vehicle loan or lease": "Vehicle loan/lease",
    "Money transfer, virtual currency, or money service": "Money transfer",
    "Money transfers": "Money transfer",
    "Virtual currency": "Money transfer",
    "Payday loan, title loan, or personal loan": "Personal/payday loan",
    "Payday loan, title loan, personal loan, or advance": "Personal/payday loan",
    "Payday loan": "Personal/payday loan",
    "Personal loan": "Personal/payday loan",
    "Title loan": "Personal/payday loan",
}

# ---------------------------------------------------------------------------
# Issue → Issue-Group mapping (7 categories)
# ---------------------------------------------------------------------------

ISSUE_GROUP_MAP = {
    # Billing / Payment
    "Billing disputes": "Billing/Payment",
    "Billing statement": "Billing/Payment",
    "Problem with a purchase shown on your statement": "Billing/Payment",
    "Fees or interest": "Billing/Payment",
    "Other fee": "Billing/Payment",
    "Charged fees or interest while I was paying on time": "Billing/Payment",
    "Problem with fees": "Billing/Payment",
    "Was charged for something I did not purchase with the card": "Billing/Payment",
    "Late fee": "Billing/Payment",
    "Balance transfer fee": "Billing/Payment",
    "Problem when making payments": "Billing/Payment",
    "Struggling to pay your bill": "Billing/Payment",
    "Unexpected or other fees": "Billing/Payment",
    "Problem with a lender or other company charging your account": "Billing/Payment",

    # Fraud / Security
    "Fraud or scam": "Fraud/Security",
    "Identity theft / Fraud / Embezzlement": "Fraud/Security",
    "Problem with fraud alerts or security freezes": "Fraud/Security",
    "Unauthorized transactions/trans am not responsible for": "Fraud/Security",
    "Took or threatened to take negative or legal action": "Fraud/Security",
    "False statements or representation": "Fraud/Security",

    # Account Access
    "Managing an account": "Account Access",
    "Opening an account": "Account Access",
    "Closing an account": "Account Access",
    "Closing your account": "Account Access",
    "Problem with a company's investigation into an existing problem": "Account Access",
    "Problem with a company's investigation into an existing issue": "Account Access",
    "Deposits and withdrawals": "Account Access",
    "Problem adding money": "Account Access",
    "Getting a credit card": "Account Access",
    "Problem getting a card or closing an account": "Account Access",
    "Closing on a mortgage": "Account Access",

    # Customer Service
    "Customer service / Customer relations": "Customer Service",
    "Getting a line of credit": "Customer Service",
    "Problem with customer service": "Customer Service",
    "Communication tactics": "Customer Service",
    "Written notification about debt": "Customer Service",
    "Advertising": "Customer Service",
    "Advertising and marketing, including promotional offers": "Customer Service",

    # Reporting / Collections
    "Incorrect information on your report": "Reporting/Collections",
    "Credit monitoring or identity theft protection services": "Reporting/Collections",
    "Problem with a credit reporting company's investigation into an existing problem": "Reporting/Collections",
    "Improper use of your report": "Reporting/Collections",
    "Unable to get your credit report or credit score": "Reporting/Collections",
    "Credit report and/or credit score": "Reporting/Collections",
    "Attempts to collect debt not owed": "Reporting/Collections",
    "Cont'd attempts collect debt not owed": "Reporting/Collections",
    "Threatened to contact someone or share information improperly": "Reporting/Collections",

    # Loan Servicing
    "Dealing with your lender or servicer": "Loan Servicing",
    "Struggling to pay mortgage": "Loan Servicing",
    "Trouble during payment process": "Loan Servicing",
    "Struggling to repay your loan": "Loan Servicing",
    "Problem with the payoff process at the end of the loan": "Loan Servicing",
    "Applying for a mortgage or refinancing an existing mortgage": "Loan Servicing",
    "Applying for a mortgage": "Loan Servicing",
    "Loan modification,collection,foreclosure": "Loan Servicing",
    "Loan servicing, payments, escrow account": "Loan Servicing",
    "Getting a loan or lease": "Loan Servicing",
    "Getting a loan": "Loan Servicing",
    "Managing the loan or lease": "Loan Servicing",
    "Problems at the end of the loan or lease": "Loan Servicing",
    "Repossession": "Loan Servicing",
    "Vehicle was repossessed or sold the vehicle": "Loan Servicing",

    # Technical / App Issue
    "Problem with a purchase or transfer": "Technical/App",
    "Money was not available when promised": "Technical/App",
    "Other transaction problem": "Technical/App",
    "Confusing or missing disclosures": "Technical/App",
    "Problem caused by your funds being low": "Technical/App",
    "Other features, terms, or problems": "Technical/App",
    "Overdraft, currentBalance, or other account problems": "Technical/App",
    "Problem with overdraft": "Technical/App",
    "Electronic communications": "Technical/App",
}

# ---------------------------------------------------------------------------
# Priority scoring keywords
# ---------------------------------------------------------------------------

PRIORITY_KEYWORDS = {
    "high": [
        "fraud", "scam", "identity theft", "unauthorized transaction",
        "cannot access money", "foreclosure", "stolen", "identity stolen",
        "unauthorized charge", "garnish", "wage garnishment", "sue",
        "lawsuit", "legal action", "repossession", "harassment",
        "threaten", "fdcpa", "fcra violation",
    ],
    "medium": [
        "billing dispute", "delayed refund", "overcharged", "late fee",
        "incorrect balance", "wrong amount", "error on report",
        "inaccurate information", "dispute", "penalty", "denied",
        "closed account", "cannot open account",
    ],
    "low": [
        "dissatisfaction", "general complaint", "question", "inquiry",
        "information request", "status update", "slow response",
    ],
}

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ===========================================================================
# Download + Filter (streaming — never writes the full 8 GB CSV to disk)
# ===========================================================================

def download_zip(url: str = DATASET_URL, zip_path: Path = RAW_ZIP) -> Path:
    """Download the CFPB zip file to disk (1.8 GB compressed).

    The compressed zip is ~1.8 GB, which is safe on disk. We keep it so
    subsequent runs can skip the download.

    Args:
        url: URL to the zipped CSV.
        zip_path: Where to save the zip file.

    Returns:
        Path to the saved zip file.
    """
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    if zip_path.exists():
        logger.info("Zip already exists at %s — skipping download.", zip_path)
        return zip_path

    logger.info("Downloading dataset zip from %s …", url)
    response = requests.get(url, stream=True, timeout=600)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    with open(zip_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc="Downloading") as pbar:
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Saved zip to %s (%.1f MB)", zip_path, zip_path.stat().st_size / 1e6)
    return zip_path


def stream_and_filter(
    zip_path: Path = RAW_ZIP,
    date_cutoff: str = DATE_CUTOFF,
    min_words: int = MIN_WORD_COUNT,
    chunksize: int = 100_000,
) -> pd.DataFrame:
    """Read the zip directly and filter in chunks — never extracts the full CSV.

    The uncompressed CSV is ~8 GB. This function opens the zip and passes a
    file-like object directly to pandas, reading chunksize rows at a time and
    keeping only surviving rows. Peak extra disk usage is ~0 bytes beyond the
    zip itself.

    Filters applied:
        1. Keep only rows from *date_cutoff* onward.
        2. Drop rows without a consumer complaint narrative.
        3. Keep narratives with >= *min_words* words.
        4. Keep only columns listed in RELEVANT_COLUMNS.
        5. Normalize product labels via PRODUCT_LABEL_MAP.
        6. Drop exact-duplicate narratives.

    Args:
        zip_path: Path to the downloaded zip file.
        date_cutoff: Earliest date to keep (YYYY-MM-DD).
        min_words: Minimum number of words in the narrative.
        chunksize: Rows per chunk.

    Returns:
        Filtered DataFrame.
    """
    logger.info("Streaming and filtering from zip (date >= %s, words >= %d) …", date_cutoff, min_words)

    frames = []
    with zipfile.ZipFile(zip_path) as zf:
        csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not csv_names:
            raise FileNotFoundError("No CSV inside the zip archive.")
        with zf.open(csv_names[0]) as csv_file:
            for chunk in tqdm(
                pd.read_csv(csv_file, chunksize=chunksize, low_memory=False, dtype=str),
                desc="Processing chunks",
            ):
                # Keep relevant columns only
                available_cols = [c for c in RELEVANT_COLUMNS if c in chunk.columns]
                chunk = chunk[available_cols].copy()

                # Date filter
                chunk["Date received"] = pd.to_datetime(chunk["Date received"], errors="coerce")
                chunk = chunk[chunk["Date received"] >= date_cutoff]

                # Narrative must be present
                chunk = chunk[chunk["Consumer complaint narrative"].notna()]
                chunk = chunk[chunk["Consumer complaint narrative"].str.strip().str.len() > 0]

                # Word count filter
                chunk["_word_count"] = chunk["Consumer complaint narrative"].str.split().str.len()
                chunk = chunk[chunk["_word_count"] >= min_words]
                chunk.drop(columns=["_word_count"], inplace=True)

                # Product normalization
                chunk = chunk[chunk["Product"].isin(PRODUCT_LABEL_MAP.keys())]
                chunk["Product"] = chunk["Product"].map(PRODUCT_LABEL_MAP)

                if len(chunk) > 0:
                    frames.append(chunk)

    if not frames:
        raise ValueError("No data survived filtering — check date_cutoff and zip path.")

    df = pd.concat(frames, ignore_index=True)

    # Drop duplicate narratives
    before = len(df)
    df.drop_duplicates(subset=["Consumer complaint narrative"], inplace=True)
    logger.info("Dropped %d duplicate narratives.", before - len(df))

    logger.info("Filtered dataset: %d rows, %d products.", len(df), df["Product"].nunique())
    return df


# ===========================================================================
# Text Preprocessing
# ===========================================================================

def clean_text(text: str) -> str:
    """Clean a complaint narrative for TF-IDF and BiLSTM models.

    Steps:
        1. Lowercase
        2. Replace XXXX / XX redaction patterns with [REDACTED]
        3. Remove URLs
        4. Remove email addresses
        5. Remove phone numbers
        6. Collapse whitespace

    Args:
        text: Raw complaint narrative.

    Returns:
        Cleaned text string.
    """
    text = text.lower()
    text = re.sub(r"\bx{2,}\b", "[redacted]", text)
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"\S+@\S+\.\S+", "", text)
    text = re.sub(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess_texts(df: pd.DataFrame) -> pd.DataFrame:
    """Add cleaned narrative column and word-count column.

    Args:
        df: DataFrame with 'Consumer complaint narrative' column.

    Returns:
        DataFrame with added 'narrative_clean' and 'word_count' columns.
    """
    logger.info("Preprocessing %d narratives …", len(df))
    df["narrative_clean"] = df["Consumer complaint narrative"].apply(clean_text)
    df["word_count"] = df["narrative_clean"].str.split().str.len()
    return df


# ===========================================================================
# Issue Group Mapping
# ===========================================================================

def map_issue_group(issue: str) -> str:
    """Map a raw issue string to one of 7 issue groups.

    Falls back to 'Other' if no mapping is found.

    Args:
        issue: Raw issue string from the dataset.

    Returns:
        One of the 7 issue-group labels or 'Other'.
    """
    if pd.isna(issue):
        return "Other"
    return ISSUE_GROUP_MAP.get(issue, "Other")


def add_issue_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add 'issue_group' column to the DataFrame.

    Args:
        df: DataFrame with 'Issue' column.

    Returns:
        DataFrame with added 'issue_group' column.
    """
    df["issue_group"] = df["Issue"].apply(map_issue_group)
    logger.info("Issue group distribution:\n%s", df["issue_group"].value_counts().to_string())
    return df


# ===========================================================================
# Priority Scoring
# ===========================================================================

def score_priority(text: str) -> str:
    """Assign a priority level to a complaint based on keyword matching.

    Priority levels:
        - Critical: 2+ high-priority keywords found
        - High: 1 high-priority keyword found
        - Medium: 2+ medium-priority keywords found
        - Low: anything else

    Args:
        text: Complaint narrative (raw or cleaned).

    Returns:
        Priority label string.
    """
    text_lower = text.lower()
    high_count = sum(1 for kw in PRIORITY_KEYWORDS["high"] if kw in text_lower)
    med_count = sum(1 for kw in PRIORITY_KEYWORDS["medium"] if kw in text_lower)

    if high_count >= 2:
        return "Critical"
    elif high_count >= 1:
        return "High"
    elif med_count >= 2:
        return "Medium"
    else:
        return "Low"


# ===========================================================================
# Class Balancing
# ===========================================================================

def balance_dataset(
    df: pd.DataFrame,
    label_col: str = "Product",
    max_per_class: int = MAX_PER_CLASS,
    min_per_class: int = MIN_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Balance the dataset by capping and flooring class sizes.

    Majority classes are undersampled to *max_per_class*, minority classes
    with fewer than *min_per_class* samples are oversampled (with replacement).

    Args:
        df: Input DataFrame.
        label_col: Column name for class labels.
        max_per_class: Maximum samples per class.
        min_per_class: Minimum samples per class.
        seed: Random seed.

    Returns:
        Balanced DataFrame, shuffled.
    """
    logger.info("Balancing dataset (max=%d, min=%d per class) …", max_per_class, min_per_class)
    balanced = []
    for label, group in df.groupby(label_col):
        n = len(group)
        if n > max_per_class:
            group = group.sample(max_per_class, random_state=seed)
        elif n < min_per_class:
            group = group.sample(min_per_class, replace=True, random_state=seed)
        balanced.append(group)

    result = pd.concat(balanced).sample(frac=1, random_state=seed).reset_index(drop=True)
    logger.info("Balanced dataset: %d rows.", len(result))
    logger.info("Class distribution:\n%s", result[label_col].value_counts().to_string())
    return result


# ===========================================================================
# Label Encoding
# ===========================================================================

def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Fit and apply label encoders for product and issue_group columns.

    Saves encoders to the models directory.

    Args:
        df: DataFrame with 'Product' and 'issue_group' columns.

    Returns:
        DataFrame with added 'product_encoded' and 'issue_group_encoded' columns.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Product encoder
    le_product = LabelEncoder()
    df["product_encoded"] = le_product.fit_transform(df["Product"])
    joblib.dump(le_product, MODELS_DIR / "label_encoder_product.joblib")
    logger.info("Product classes (%d): %s", len(le_product.classes_), list(le_product.classes_))

    # Issue group encoder
    le_issue = LabelEncoder()
    df["issue_group_encoded"] = le_issue.fit_transform(df["issue_group"])
    joblib.dump(le_issue, MODELS_DIR / "label_encoder_issue_group.joblib")
    logger.info("Issue-group classes (%d): %s", len(le_issue.classes_), list(le_issue.classes_))

    return df


# ===========================================================================
# Train / Val / Test Split
# ===========================================================================

def split_data(
    df: pd.DataFrame,
    stratify_col: str = "product_encoded",
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split data into 70% train / 15% validation / 15% test.

    Uses stratified splitting to maintain class proportions.

    Args:
        df: Input DataFrame.
        stratify_col: Column to stratify on.
        seed: Random seed.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df[stratify_col], random_state=seed,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df[stratify_col], random_state=seed,
    )
    logger.info("Split sizes — train: %d, val: %d, test: %d", len(train_df), len(val_df), len(test_df))
    return train_df, val_df, test_df


# ===========================================================================
# Compute class weights
# ===========================================================================

def compute_class_weights(labels: np.ndarray) -> np.ndarray:
    """Compute inverse-frequency class weights for imbalanced classification.

    Args:
        labels: Array of integer-encoded labels.

    Returns:
        Array of weights indexed by class label.
    """
    counts = np.bincount(labels)
    total = counts.sum()
    weights = total / (len(counts) * counts.astype(float))
    return weights


# ===========================================================================
# Main Pipeline
# ===========================================================================

def run_pipeline() -> None:
    """Execute the full data preparation pipeline.

    Steps:
        1. Download dataset
        2. Load and filter
        3. Preprocess text
        4. Map issue groups
        5. Balance classes
        6. Encode labels
        7. Split and save
    """
    # Ensure output dirs exist
    for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Step 1: Download zip (1.8 GB compressed — safe on disk)
    zip_path = download_zip()

    # Step 2: Stream-filter directly from zip (never writes the 8 GB CSV)
    df = stream_and_filter(zip_path)

    # Step 3: Preprocess
    df = preprocess_texts(df)

    # Step 4: Issue groups
    df = add_issue_groups(df)

    # Step 5: Add priority scores
    df["priority"] = df["Consumer complaint narrative"].apply(score_priority)
    logger.info("Priority distribution:\n%s", df["priority"].value_counts().to_string())

    # Step 6: Balance
    df = balance_dataset(df, label_col="Product")

    # Step 7: Encode labels
    df = encode_labels(df)

    # Step 8: Split and save
    train_df, val_df, test_df = split_data(df)

    train_df.to_csv(DATA_PROCESSED / "train.csv", index=False)
    val_df.to_csv(DATA_PROCESSED / "val.csv", index=False)
    test_df.to_csv(DATA_PROCESSED / "test.csv", index=False)

    # Save full filtered dataset too (for EDA)
    df.to_csv(DATA_PROCESSED / "full_filtered.csv", index=False)

    # Save stats
    stats = {
        "total_filtered": len(df),
        "train_size": len(train_df),
        "val_size": len(val_df),
        "test_size": len(test_df),
        "num_product_classes": int(df["Product"].nunique()),
        "num_issue_groups": int(df["issue_group"].nunique()),
        "product_distribution": df["Product"].value_counts().to_dict(),
        "issue_group_distribution": df["issue_group"].value_counts().to_dict(),
        "priority_distribution": df["priority"].value_counts().to_dict(),
    }
    with open(RESULTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)

    logger.info("Pipeline complete. Files saved to %s", DATA_PROCESSED)
    logger.info("Dataset stats saved to %s", RESULTS_DIR / "dataset_stats.json")


if __name__ == "__main__":
    run_pipeline()
