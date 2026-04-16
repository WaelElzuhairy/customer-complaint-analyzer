"""
Patch script: fetch Credit Reporting complaints via CFPB API and merge into
the existing processed dataset.

The CFPB renamed 'Credit reporting, credit repair services...' to
'Credit reporting or other personal consumer reports' in 2023, causing
the original pipeline to drop all credit reporting data.

This script:
  1. Fetches up to 10,000 credit reporting complaints (API max)
  2. Applies the same filters as data_prep (>=20 words, Aug 2023+)
  3. Merges with existing train/val/test splits
  4. Re-balances and re-encodes labels
  5. Overwrites the processed CSVs

Run from project root:
    python src/patch_credit_reporting.py
"""

import sys
import json
import time
import logging
from pathlib import Path

import requests
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from src.data_prep import (
    clean_text, add_issue_groups, score_priority,
    balance_dataset, compute_class_weights,
    RELEVANT_COLUMNS, DATE_CUTOFF, MIN_WORD_COUNT,
    MAX_PER_CLASS, MIN_PER_CLASS, RANDOM_SEED,
)

DATA_PROCESSED = BASE_DIR / "data" / "processed"
MODELS_DIR     = BASE_DIR / "models"
RESULTS_DIR    = BASE_DIR / "results"

API_URL     = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"
PRODUCT_NEW = "Credit reporting or other personal consumer reports"
PRODUCT_LABEL = "Credit reporting"
BATCH_SIZE  = 100   # max per request
TARGET_N    = 10000  # API hard ceiling is from+size <= 10000

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API fetcher
# ---------------------------------------------------------------------------

def fetch_credit_reporting(target: int = TARGET_N) -> pd.DataFrame:
    """Fetch credit reporting complaints with narratives from the CFPB API.

    Paginates in batches of BATCH_SIZE up to API ceiling of 10 000.

    Returns:
        DataFrame matching the schema of the processed CSVs.
    """
    log.info("Fetching up to %d Credit Reporting complaints from CFPB API...", target)
    rows = []
    fetched = 0

    with tqdm(total=target, desc="Fetching API pages") as pbar:
        while fetched < target:
            params = {
                "size":               BATCH_SIZE,
                "from":               fetched,
                "date_received_min":  DATE_CUTOFF,
                "product":            PRODUCT_NEW,
                "has_narrative":      "true",   # only complaints with actual text
            }
            try:
                r = requests.get(API_URL, params=params, timeout=60)
                r.raise_for_status()
            except requests.RequestException as e:
                log.warning("API request failed: %s — retrying in 5s", e)
                time.sleep(5)
                continue

            hits = r.json().get("hits", {}).get("hits", [])
            if not hits:
                log.info("No more results at offset %d.", fetched)
                break

            for h in hits:
                src = h.get("_source", {})
                narrative = src.get("complaint_what_happened") or ""
                if not narrative.strip():
                    continue
                rows.append({
                    "Complaint ID":                  src.get("complaint_id", ""),
                    "Date received":                 src.get("date_received", "")[:10],
                    "Product":                       PRODUCT_LABEL,   # normalized
                    "Issue":                         src.get("issue", ""),
                    "Sub-issue":                     src.get("sub_issue", ""),
                    "Consumer complaint narrative":  narrative,
                    "Company response to consumer":  src.get("company_response", ""),
                    "Timely response?":              src.get("timely", ""),
                })

            fetched += len(hits)
            pbar.update(len(hits))
            time.sleep(0.1)   # be polite to the API

    df = pd.DataFrame(rows)
    log.info("Fetched %d raw complaints from API.", len(df))
    return df


# ---------------------------------------------------------------------------
# Filter + preprocess
# ---------------------------------------------------------------------------

def filter_and_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same filters as data_prep pipeline."""
    # Word count filter
    df = df[df["Consumer complaint narrative"].str.split().str.len() >= MIN_WORD_COUNT].copy()
    # Date filter
    df["Date received"] = pd.to_datetime(df["Date received"], errors="coerce")
    df = df[df["Date received"] >= DATE_CUTOFF]
    # Remove duplicates
    before = len(df)
    df.drop_duplicates(subset=["Consumer complaint narrative"], inplace=True)
    log.info("After filter: %d rows (dropped %d duplicates).", len(df), before - len(df))

    # Clean text
    df["narrative_clean"] = df["Consumer complaint narrative"].apply(clean_text)
    df["word_count"]      = df["narrative_clean"].str.split().str.len()

    # Issue groups + priority
    df = add_issue_groups(df)
    df["priority"] = df["Consumer complaint narrative"].apply(score_priority)

    return df


# ---------------------------------------------------------------------------
# Merge + re-split
# ---------------------------------------------------------------------------

def merge_and_resplit(new_df: pd.DataFrame) -> None:
    """Merge Credit Reporting into existing processed data and re-split."""

    # Load existing data — drop any incomplete Credit Reporting rows from prior run
    full_df  = pd.read_csv(DATA_PROCESSED / "full_filtered.csv")
    before = len(full_df)
    full_df = full_df[full_df["Product"] != PRODUCT_LABEL]
    log.info("Removed %d old Credit Reporting rows before re-patch.", before - len(full_df))

    # Sample up to MAX_PER_CLASS from new data
    cr_sample = new_df.sample(
        min(len(new_df), MAX_PER_CLASS), random_state=RANDOM_SEED
    ).copy()
    log.info("Credit Reporting sample: %d rows.", len(cr_sample))

    # Combine into full filtered dataset
    full_combined = pd.concat([full_df, cr_sample], ignore_index=True)
    full_combined.to_csv(DATA_PROCESSED / "full_filtered.csv", index=False)
    log.info("Full filtered dataset: %d rows.", len(full_combined))

    # Re-balance using the full combined dataset
    from src.data_prep import balance_dataset
    balanced = balance_dataset(full_combined, label_col="Product",
                               max_per_class=MAX_PER_CLASS, min_per_class=MIN_PER_CLASS)

    # Re-fit label encoders on full label set
    le_product = LabelEncoder()
    balanced["product_encoded"] = le_product.fit_transform(balanced["Product"])
    joblib.dump(le_product, MODELS_DIR / "label_encoder_product.joblib")

    le_issue = LabelEncoder()
    balanced["issue_group_encoded"] = le_issue.fit_transform(balanced["issue_group"])
    joblib.dump(le_issue, MODELS_DIR / "label_encoder_issue_group.joblib")

    log.info("Product classes (%d): %s", len(le_product.classes_), list(le_product.classes_))

    # Re-split 70 / 15 / 15
    train, temp = train_test_split(
        balanced, test_size=0.30, stratify=balanced["product_encoded"], random_state=RANDOM_SEED,
    )
    val, test = train_test_split(
        temp, test_size=0.50, stratify=temp["product_encoded"], random_state=RANDOM_SEED,
    )

    train.to_csv(DATA_PROCESSED / "train.csv", index=False)
    val.to_csv(DATA_PROCESSED / "val.csv",   index=False)
    test.to_csv(DATA_PROCESSED / "test.csv",  index=False)
    log.info("New split: %d train / %d val / %d test", len(train), len(val), len(test))

    # Update stats
    stats = {
        "total_filtered":          len(balanced),
        "train_size":              len(train),
        "val_size":                len(val),
        "test_size":               len(test),
        "num_product_classes":     int(balanced["Product"].nunique()),
        "num_issue_groups":        int(balanced["issue_group"].nunique()),
        "product_distribution":    balanced["Product"].value_counts().to_dict(),
        "issue_group_distribution": balanced["issue_group"].value_counts().to_dict(),
        "priority_distribution":   balanced["priority"].value_counts().to_dict(),
    }
    with open(RESULTS_DIR / "dataset_stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    log.info("Stats saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    for d in [DATA_PROCESSED, MODELS_DIR, RESULTS_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    raw = fetch_credit_reporting(target=TARGET_N)
    if len(raw) == 0:
        log.error("No data fetched — check API connectivity.")
        sys.exit(1)

    cleaned = filter_and_clean(raw)
    log.info("Clean Credit Reporting rows: %d", len(cleaned))

    merge_and_resplit(cleaned)
    log.info("Patch complete. Dataset now includes Credit Reporting.")
