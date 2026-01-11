"""
download.py — FinanceInsight dataset downloader (Windows-ready)

What it does:
- Downloads Kaggle datasets using the kaggle CLI (must have kaggle.json set up)
- Downloads HF datasets via `datasets` library (e.g., FinancialPhraseBank)
- Downloads OHLCV & fundamentals for tickers via yfinance
- Scrapes simple press release pages (configurable list of URLs)
- Creates folder structure under base_dir (default: C:\FinanceInsight\data)

How to use:
1) Install dependencies:
   python -m pip install kaggle datasets yfinance requests beautifulsoup4 tqdm

2) Place your kaggle.json in %USERPROFILE%\.kaggle\kaggle.json (see Kaggle docs)

3) Edit CONFIG below (list datasets/tickers/press_urls)

4) Run:
   python download.py

NOTES:
- This script uses the kaggle CLI via subprocess. Make sure `kaggle` is callable from your shell:
  pip install kaggle
- Hugging Face dataset downloads require internet and disk space.
"""

import os
import sys
import subprocess
import shutil
import json
import time
from pathlib import Path
from typing import List
import logging

# External libs (import inside functions to avoid crash if missing)
# pip install datasets yfinance requests beautifulsoup4 tqdm

# -------------------- CONFIG --------------------
BASE_DIR = Path(r"C:\FinanceInsight\data")  # change if you want
RAW_DIR = BASE_DIR / "raw"
KAGGLE_DIR = RAW_DIR / "kaggle"
HF_DIR = RAW_DIR / "huggingface"
YFINANCE_DIR = RAW_DIR / "yfinance"
PRESS_DIR = RAW_DIR / "press"

# Kaggle dataset slugs (owner/dataset-name)
KAGGLE_DATASETS = [
    "aaron7sun/stocknews",                     # StockNews
    "ankurzing/sentiment-analysis-for-financial-news",  # Financial news sentiment (alt)
    "shivamb/earnings-call-transcripts",       # earnings transcripts
    "samarthagarwal23/annual-reports",         # annual reports collection
    # add more Kaggle dataset slugs as needed
]

# Hugging Face datasets to download (dataset_id, config optional)
HF_DATASETS = [
    ("takala/financial_phrasebank", None),
    # ("some/hf-dataset", "config_name"),
]

# Tick ers for yfinance (add exchange suffix where needed, e.g., "INFY.NS")
YFINANCE_TICKERS = [
    "AAPL", "TSLA", "MSFT", "INFY.NS", "TCS.NS"
]

# Date range for yfinance
YF_START = "2015-01-01"
YF_END = "2025-11-01"

# Press release pages to scrape (basic scraper; site-specific parsing needed for better results)
PRESS_URLS = [
    # Example: Apple's newsroom (we will fetch the list page and save HTML)
    "https://www.apple.com/newsroom/",
    "https://www.sap.com/about/news.html",
    # add company IR pages or press release lists here
]

# Retry & timeout settings
RETRY_COUNT = 3
SLEEP_BETWEEN = 1.0  # seconds

# ------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

def ensure_dirs():
    for d in [BASE_DIR, RAW_DIR, KAGGLE_DIR, HF_DIR, YFINANCE_DIR, PRESS_DIR]:
        d.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created/verified base directories under {BASE_DIR}")

def run_kaggle_download(slug: str, dest: Path):
    """
    Uses `kaggle datasets download -d <slug> -p <dest> --unzip`.
    Requires `kaggle` CLI available and kaggle.json in %USERPROFILE%\.kaggle
    """
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "datasets", "download", "-d", slug, "-p", str(dest), "--unzip"]
    logging.info(f"Running Kaggle download for {slug} -> {dest}")
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            subprocess.check_call(cmd)
            logging.info(f"Downloaded {slug} successfully.")
            return True
        except subprocess.CalledProcessError as e:
            logging.warning(f"Kaggle download failed (attempt {attempt}/{RETRY_COUNT}) for {slug}: {e}")
            time.sleep(SLEEP_BETWEEN * attempt)
    logging.error(f"Failed to download {slug} after {RETRY_COUNT} attempts.")
    return False

def download_hf_dataset(dataset_id: str, config_name: str, dest: Path):
    try:
        from datasets import load_dataset
    except Exception as e:
        logging.error("datasets library not installed. Run: pip install datasets")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    logging.info(f"Loading HF dataset {dataset_id} (config={config_name})")
    try:
        if config_name:
            ds = load_dataset(dataset_id, config_name)
        else:
            ds = load_dataset(dataset_id)
        # save to disk as jsonlines per split
        for split, s in ds.items():
            out_path = dest / f"{dataset_id.replace('/', '_')}_{split}.jsonl"
            with open(out_path, "w", encoding="utf-8") as fo:
                for ex in s:
                    fo.write(json.dumps(ex, ensure_ascii=False) + "\n")
            logging.info(f"Wrote {split} to {out_path} ({len(s)} records)")
        return True
    except Exception as e:
        logging.exception(f"Failed to load HF dataset {dataset_id}: {e}")
        return False

def download_yfinance(tickers: List[str], dest: Path, start: str, end: str):
    try:
        import yfinance as yf
        import pandas as pd
    except Exception as e:
        logging.error("yfinance or pandas not installed. Run: pip install yfinance pandas")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    logging.info(f"Downloading yfinance data for {len(tickers)} tickers")
    for tic in tickers:
        try:
            logging.info(f"Downloading {tic}")
            t = yf.Ticker(tic)
            hist = t.history(start=start, end=end, auto_adjust=False)
            if hist.empty:
                logging.warning(f"No history for {tic}.")
            out_csv = dest / f"{tic.replace('/', '_')}_history.csv"
            hist.to_csv(out_csv)
            # Basic info (short)
            info = t.info if hasattr(t, "info") else {}
            out_info = dest / f"{tic.replace('/', '_')}_info.json"
            with open(out_info, "w", encoding="utf-8") as fo:
                json.dump({k: info.get(k, None) for k in ["symbol", "longName", "sector", "country"] if k in info}, fo, ensure_ascii=False, indent=2)
            logging.info(f"Saved {tic} history to {out_csv}")
            time.sleep(0.5)
        except Exception as e:
            logging.exception(f"Failed to download yfinance for {tic}: {e}")
    return True

def simple_scrape_page(url: str, dest: Path):
    """
    Basic fetch & save of HTML and extracted text. For press release lists, saving HTML is safe.
    For content extraction, site-specific parsing is needed.
    """
    try:
        import requests
        from bs4 import BeautifulSoup
    except Exception:
        logging.error("requests or bs4 not installed. Run: pip install requests beautifulsoup4")
        return False
    dest.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0 (compatible; FinanceInsight/1.0; +https://your-org)"}  # polite UA
    logging.info(f"Fetching {url}")
    for attempt in range(1, RETRY_COUNT + 1):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            html_path = dest / (url.replace("https://", "").replace("http://", "").replace("/", "_") + ".html")
            with open(html_path, "w", encoding="utf-8") as fo:
                fo.write(r.text)
            # try a simple text extraction
            soup = BeautifulSoup(r.text, "html.parser")
            body = soup.get_text(separator=" ", strip=True)
            txt_path = dest / (url.replace("https://", "").replace("http://", "").replace("/", "_") + ".txt")
            with open(txt_path, "w", encoding="utf-8") as fo:
                fo.write(body)
            logging.info(f"Saved HTML -> {html_path} and text -> {txt_path}")
            return True
        except Exception as e:
            logging.warning(f"Scrape attempt {attempt} failed for {url}: {e}")
            time.sleep(SLEEP_BETWEEN * attempt)
    logging.error(f"Failed to fetch {url} after {RETRY_COUNT} attempts.")
    return False

def main():
    ensure_dirs()

    # 1) Kaggle downloads
    logging.info("Starting Kaggle downloads...")
    for slug in KAGGLE_DATASETS:
        safe_slug = slug.replace("/", "_")
        dest = KAGGLE_DIR / safe_slug
        ok = run_kaggle_download(slug, dest)
        if not ok:
            logging.warning(f"Kaggle download failed for {slug}. Check your kaggle.json and CLI install.")

    # 2) Hugging Face datasets
    logging.info("Starting Hugging Face dataset downloads...")
    for dataset_id, config in HF_DATASETS:
        dest = HF_DIR / dataset_id.replace("/", "_")
        ok = download_hf_dataset(dataset_id, config, dest)
        if not ok:
            logging.warning(f"HF dataset download failed for {dataset_id}")

    # 3) yfinance downloads
    logging.info("Starting yfinance downloads...")
    download_yfinance(YFINANCE_TICKERS, YFINANCE_DIR, YF_START, YF_END)

    # 4) Press release page scraping
    logging.info("Starting press release page scraping...")
    for url in PRESS_URLS:
        subdir_name = url.replace("https://", "").replace("http://", "").split("/")[0]
        dest = PRESS_DIR / subdir_name
        simple_scrape_page(url, dest)

    logging.info("All tasks submitted. Check logs above for success/fail messages.")
    logging.info(f"Data folder structure under: {BASE_DIR}")
    logging.info("If any downloads failed, re-run the script after fixing credentials/network.")

if __name__ == "__main__":
    main()
