"""
preprocess.py
=============
Handles data loading, data cleaning, exploratory text metrics,
and natural language text preprocessing for the Spam Detection pipeline.
"""

import os
import re
import sys
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Ensure NLTK resources are available
try:
    STOPWORDS = set(stopwords.words("english"))
except LookupError:
    nltk.download("stopwords", quiet=True)
    STOPWORDS = set(stopwords.words("english"))

STEMMER = PorterStemmer()

RAW_DATA_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "SMSSpamCollection"
)
PROCESSED_DATA_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "processed"
)
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "spam_cleaned.csv")


def clean_text(text: str, stem: bool = True) -> str:
    """
    Cleans and normalizes raw text input.

    Steps:
    1. Lowercase text
    2. Remove URLs and web links
    3. Remove email addresses
    4. Remove non-alphabetic characters (numbers, punctuation, emojis)
    5. Tokenize into words
    6. Remove common English stopwords
    7. Apply Porter Stemmer (e.g., 'running' -> 'run', 'calls' -> 'call')
    8. Join back into a single string

    Args:
        text: Raw text string.
        stem: Whether to apply Porter stemming.

    Returns:
        Cleaned and normalized text string.
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercase
    text = text.lower()

    # 2. Remove URLs
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    # 3. Remove email addresses
    text = re.sub(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", " ", text)

    # 4. Remove special characters, numbers, and punctuation
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # 5. Tokenize by whitespace
    tokens = text.split()

    # 6. Remove stopwords and short single-character tokens
    tokens = [w for w in tokens if w not in STOPWORDS and len(w) > 1]

    # 7. Apply Stemming
    if stem:
        tokens = [STEMMER.stem(w) for w in tokens]

    return " ".join(tokens)


def load_raw_dataset(filepath: str = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Loads raw SMSSpamCollection file into a pandas DataFrame.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Raw data file not found at: {filepath}. Run download_data.py first.")

    # The dataset is tab-separated: <label>\t<message>
    df = pd.read_csv(filepath, sep="\t", header=None, names=["label", "message"], quoting=3)
    return df


def preprocess_data(raw_filepath: str = RAW_DATA_PATH, save_processed: bool = True) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Loads raw data
    - Audits missing values and duplicates
    - Deduplicates records to prevent train-test data leakage
    - Computes descriptive text statistics (character length, word count)
    - Applies clean_text function
    - Encodes target variable (ham: 0, spam: 1)
    - Saves clean data to data/processed/spam_cleaned.csv

    Returns:
        Cleaned pandas DataFrame.
    """
    print("[INFO] Loading raw data...")
    df = load_raw_dataset(raw_filepath)

    raw_total = len(df)
    raw_ham = (df["label"] == "ham").sum()
    raw_spam = (df["label"] == "spam").sum()
    missing_count = df.isnull().sum().to_dict()
    duplicate_count = df.duplicated(subset=["message"]).sum()

    print(f"[AUDIT] Raw Dataset Shape: {df.shape}")
    print(f"[AUDIT] Raw Ham Count: {raw_ham} ({raw_ham / raw_total * 100:.2f}%)")
    print(f"[AUDIT] Raw Spam Count: {raw_spam} ({raw_spam / raw_total * 100:.2f}%)")
    print(f"[AUDIT] Missing Values: {missing_count}")
    print(f"[AUDIT] Duplicate Messages: {duplicate_count} ({duplicate_count / raw_total * 100:.2f}%)")

    # Drop duplicate messages to prevent data leakage between train and test sets
    # (e.g. identical spam broadcast appearing in both sets artificially inflates test metrics)
    df_clean = df.drop_duplicates(subset=["message"]).copy()
    clean_total = len(df_clean)
    clean_ham = (df_clean["label"] == "ham").sum()
    clean_spam = (df_clean["label"] == "spam").sum()

    print(f"[INFO] Deduplicated Dataset Shape: {df_clean.shape}")
    print(f"[INFO] Deduplicated Ham Count: {clean_ham} ({clean_ham / clean_total * 100:.2f}%)")
    print(f"[INFO] Deduplicated Spam Count: {clean_spam} ({clean_spam / clean_total * 100:.2f}%)")

    # Add descriptive text metrics
    df_clean["char_count"] = df_clean["message"].apply(len)
    df_clean["word_count"] = df_clean["message"].apply(lambda x: len(str(x).split()))

    # Encode label: ham -> 0, spam -> 1
    df_clean["label_num"] = df_clean["label"].map({"ham": 0, "spam": 1})

    # Clean text
    print("[INFO] Cleaning text and applying Porter Stemmer...")
    df_clean["cleaned_message"] = df_clean["message"].apply(clean_text)

    # Filter out any rare messages that become empty after cleaning
    empty_cleaned = (df_clean["cleaned_message"].str.strip() == "").sum()
    if empty_cleaned > 0:
        print(f"[WARNING] {empty_cleaned} messages became empty after stopword removal. Replacing with original lowercased text.")
        empty_mask = df_clean["cleaned_message"].str.strip() == ""
        df_clean.loc[empty_mask, "cleaned_message"] = df_clean.loc[empty_mask, "message"].str.lower()

    if save_processed:
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
        df_clean.to_csv(PROCESSED_DATA_PATH, index=False)
        print(f"[SUCCESS] Processed dataset saved to: {PROCESSED_DATA_PATH}")

    return df_clean


if __name__ == "__main__":
    df_processed = preprocess_data()
    print("\n--- Processed Data Sample ---")
    print(df_processed[["label", "char_count", "word_count", "cleaned_message"]].head(5))
