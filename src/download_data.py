"""
download_data.py
================
Downloads the UCI SMS Spam Collection dataset directly from the official UCI repository,
extracts the archive, and verifies the downloaded files.

Dataset Details:
- Source: UCI Machine Learning Repository
- Official URL: https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Download URL: https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip
- Target File: SMSSpamCollection (tab-separated: label \t message)
"""

import os
import sys
import zipfile
import urllib.request
import requests

UCI_DATASET_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
ZIP_FILE_PATH = os.path.join(RAW_DATA_DIR, "sms_spam_collection.zip")
DATASET_FILE_PATH = os.path.join(RAW_DATA_DIR, "SMSSpamCollection")


def download_dataset(url: str = UCI_DATASET_URL, target_dir: str = RAW_DATA_DIR) -> str:
    """
    Downloads the zipped dataset from UCI, extracts contents, and verifies integrity.

    Args:
        url: Direct link to zip archive.
        target_dir: Directory where raw files will be saved.

    Returns:
        Path to the extracted SMSSpamCollection file.
    """
    os.makedirs(target_dir, exist_ok=True)
    zip_path = os.path.join(target_dir, "sms_spam_collection.zip")
    extracted_file = os.path.join(target_dir, "SMSSpamCollection")

    # If dataset already extracted, verify and return
    if os.path.exists(extracted_file) and os.path.getsize(extracted_file) > 0:
        print(f"[INFO] Dataset file already exists: {extracted_file} ({os.path.getsize(extracted_file):,} bytes)")
        return extracted_file

    print(f"[INFO] Initiating download from: {url}")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SpamClassifier/1.0"}

    # Attempt download via requests, fallback to urllib
    try:
        response = requests.get(url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print(f"[SUCCESS] Download completed: {zip_path} ({os.path.getsize(zip_path):,} bytes)")
    except Exception as req_err:
        print(f"[WARNING] requests download failed ({req_err}). Retrying with urllib.request...")
        urllib.request.urlretrieve(url, zip_path)
        print(f"[SUCCESS] Download completed via urllib: {zip_path}")

    # Extract contents
    print(f"[INFO] Extracting {zip_path} into {target_dir}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        extracted_members = zip_ref.namelist()
        print(f"[INFO] Archive contains: {extracted_members}")
        zip_ref.extractall(target_dir)

    # Verify extracted file
    if not os.path.exists(extracted_file):
        raise FileNotFoundError(f"Expected file '{extracted_file}' not found after extraction!")

    file_size = os.path.getsize(extracted_file)
    print(f"[SUCCESS] Extracted successfully: {extracted_file}")
    print(f"[INFO] File size: {file_size:,} bytes ({file_size / 1024:.2f} KB)")

    # Read first 3 lines as verification
    with open(extracted_file, "r", encoding="utf-8", errors="replace") as f:
        preview_lines = [f.readline().strip() for _ in range(3)]

    print("\n--- Raw Data Verification Preview (First 3 lines) ---")
    for i, line in enumerate(preview_lines, 1):
        print(f"Line {i}: {line[:100]}...")
    print("----------------------------------------------------\n")

    return extracted_file


if __name__ == "__main__":
    try:
        path = download_dataset()
        print(f"[DONE] Ready for preprocessing at: {path}")
    except Exception as e:
        print(f"[ERROR] Failed to download or verify dataset: {e}", file=sys.stderr)
        sys.exit(1)
