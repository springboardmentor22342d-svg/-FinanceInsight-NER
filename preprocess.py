# C:\FinanceInsight\preprocess.py
import os
import re
import pandas as pd
from bs4 import BeautifulSoup
import spacy
from pathlib import Path

RAW_DIR = r"C:\FinanceInsight\data\raw"
PROCESSED_DIR = r"C:\FinanceInsight\data\processed"
LOG_PATH = r"C:\FinanceInsight\preprocess_log.txt"

os.makedirs(PROCESSED_DIR, exist_ok=True)

# load spacy model (make sure en_core_web_sm installed on your machine)
try:
    nlp = spacy.load("en_core_web_sm", disable=["ner"])
except Exception as e:
    print("spacy model not found. run: python -m spacy download en_core_web_sm")
    raise

# regex patterns
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
EMAIL_PATTERN = re.compile(r'\S+@\S+')
NON_ALPHANUM = re.compile(r'[^A-Za-z0-9\s]')
MULTISPACE = re.compile(r'\s+')

# candidate names to look for
TEXT_KEYS = [
    "text","content","body","article","description","summary","notes",
    "headline","title","announcement","press"
]

# tried encodings in order
ENCODINGS = ["utf-8", "cp1252", "latin-1", "utf-16"]

def log(msg):
    print(msg)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(msg + "\n")

def try_read_csv(path):
    """Try several encodings and return (df, used_encoding)"""
    last_err = None
    for enc in ENCODINGS:
        try:
            # engine python sometimes helps with messy csvs
            df = pd.read_csv(path, encoding=enc, engine="python")
            return df, enc
        except Exception as e:
            last_err = e
    # final fallback: read with replace to avoid crash
    try:
        df = pd.read_csv(path, encoding="utf-8", engine="python", errors="replace")
        return df, "utf-8(errors=replace)"
    except Exception as e:
        raise last_err or e

def clean_html(text):
    if not isinstance(text, str):
        return ""
    soup = BeautifulSoup(text, "html.parser")
    for tag in soup(["script","style"]):
        tag.decompose()
    return soup.get_text(" ")

def regex_clean(text):
    if not isinstance(text, str):
        return ""
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    text = NON_ALPHANUM.sub(" ", text)
    text = MULTISPACE.sub(" ", text)
    return text.strip()

def spacy_clean(text):
    if not text:
        return ""
    # Limit length a bit to avoid very long docs slowing things down
    doc = nlp(text[:200000])
    toks = []
    for t in doc:
        if t.is_stop or t.is_punct or not t.is_alpha:
            continue
        if len(t.text) <= 2:
            continue
        toks.append(t.lemma_.lower())
    return " ".join(toks)

def detect_text_column(df: pd.DataFrame):
    # first: exact/substring name match
    for col in df.columns:
        low = str(col).lower()
        if any(key in low for key in TEXT_KEYS):
            return col
    # second: prefer object dtype columns with long average length
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return None
    # compute avg length (safe)
    avg_len = {}
    for c in obj_cols:
        try:
            avg_len[c] = df[c].dropna().astype(str).map(len).mean()
        except Exception:
            avg_len[c] = 0
    # pick column with largest avg length if it's reasonable (>20 chars)
    best = max(avg_len, key=lambda k: avg_len[k])
    if avg_len[best] and avg_len[best] > 20:
        return best
    # otherwise return None to signal fallback to combined text
    return None

def combine_object_columns(df):
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    if not obj_cols:
        return None
    # join non-null parts with space
    combined = df[obj_cols].fillna("").astype(str).agg(" ".join, axis=1)
    # if combined has any meaningful length, return it
    if combined.map(len).sum() > 0:
        return combined
    return None

def process_file(path):
    path = Path(path)
    log(f"\n--- Processing: {path.name} ---")
    try:
        df, enc = try_read_csv(str(path))
        log(f"read ok with encoding: {enc} | shape: {df.shape}")
    except Exception as e:
        log(f"FAILED reading {path.name}: {e}")
        return

    text_col = detect_text_column(df)
    used_combined = False
    if text_col is None:
        combined = combine_object_columns(df)
        if combined is not None:
            df["combined_text"] = combined
            text_col = "combined_text"
            used_combined = True
            log("No single text column detected → using combined_text (all object columns concatenated).")
        else:
            log("❌ No text-like columns found and combine failed. Skipping file.")
            return
    else:
        log(f"Detected text column: {text_col}")

    # run cleaning pipeline
    df["_clean_html"] = df[text_col].apply(clean_html)
    df["_clean_regex"] = df["_clean_html"].apply(regex_clean)
    # apply spacy in chunks to avoid memory spikes
    cleaned = []
    batch = 500
    texts = df["_clean_regex"].astype(str).tolist()
    for i in range(0, len(texts), batch):
        batch_texts = texts[i:i+batch]
        for t in batch_texts:
            cleaned.append(spacy_clean(t))
    df["clean_text"] = cleaned

    out_name = path.stem + "_clean.csv"
    out_path = Path(PROCESSED_DIR) / out_name
    df.to_csv(out_path, index=False, encoding="utf-8")
    log(f"✅ saved cleaned file: {out_path} | cleaned column: clean_text")
    if used_combined:
        log("note: cleaned from combined_text (multiple columns merged).")

def main():
    # clear/create log
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write("Preprocess log\n")
    files = [os.path.join(RAW_DIR, fn) for fn in os.listdir(RAW_DIR) if fn.lower().endswith(".csv")]
    if not files:
        log("no csv files found in raw dir.")
        return
    for f in files:
        process_file(f)
    log("\nDone processing all files.")

if __name__ == "__main__":
    main()
