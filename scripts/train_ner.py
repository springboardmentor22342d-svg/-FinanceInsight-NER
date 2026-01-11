i dont want seprate file update this file:
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.matcher import PhraseMatcher
from pathlib import Path
import os
import json

from error_analysis import analyze_ner

# ---------------- CONFIG ----------------

BASE_DIR = Path("C:/Users/aksha/Documents/FinanceInsight")

PROCESSED_DATA = BASE_DIR / "data/processed/data_clean.csv"
TRAIN_DATA_PATH = BASE_DIR / "data/train"
REPORTS_PATH = BASE_DIR / "reports"

os.makedirs(TRAIN_DATA_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

# ---------------- BOOTSTRAP ----------------

def bootstrap_labels():
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

    # Domain vocabulary
    tickers = [
    "AAPL","TSLA","MSFT","GOOGL","AMZN","META","NFLX","NVDA","INTC",
    "ORCL","IBM","ADBE","CRM","UBER","LYFT","SNAP","PYPL","SQ",
    "BABA","TCS","INFY","WIPRO","HDFC","ICICI","SBIN","RELIANCE"
    ]

    metrics = [
    "revenue","profit","earnings","ebitda","net income","gross margin",
    "operating income","free cash flow","dividend","dividend yield",
    "market cap","valuation","guidance","forecast","growth","loss",
    "quarter","annual","yoy","qoq","eps","debt","equity"
    ]

    

    matcher.add("TICKER", [nlp.make_doc(t) for t in tickers])
    matcher.add("FIN_METRIC", [nlp.make_doc(m) for m in metrics])

    df = pd.read_csv(PROCESSED_DATA)

    doc_bin = DocBin()
    error_logs = []

    print("🔎 Generating NER training data...")

    for text in df["clean_text"].astype(str).head(2000):
        doc = nlp(text) 
        # Artificially stress tokenizer to expose boundary issues
        if "$" in text:
            text = text.replace("$", "$ ")
            doc = nlp(text)
  # forces tokenizer & pipeline behavior

        matches = matcher(doc)
        spans = []

        for match_id, start, end in matches:
            spans.append(doc[start:end])

        doc.ents = spacy.util.filter_spans(spans)

        tokens = [t.text for t in doc]
        labels = []

        for token in doc:
            if token.ent_iob_ == "O":
                labels.append("O")
            else:
                labels.append(f"{token.ent_iob_}-{token.ent_type_}")

        report = analyze_ner(tokens, labels)

        if any(report.values()):
            error_logs.append({
                "text": text,
                "tokens": tokens,
                "labels": labels,
                "errors": report
            })
                      # Emergency: inject simple entity if nothing matched
        if len(doc.ents) == 0:
            words = doc.text.split()
            if len(words) >= 2:
                span = doc.char_span(0, len(words[0]), label="ORG")
                if span:
                    doc.ents = [span]

        doc_bin.add(doc)

    doc_bin.to_disk(TRAIN_DATA_PATH / "train.spacy")
    doc_bin.to_disk(TRAIN_DATA_PATH / "dev.spacy")

    with open(REPORTS_PATH / "ner_bootstrap_error_report.json", "w") as f:
        json.dump(error_logs, f, indent=4)

    print("✅ NER data generated")
    print("📊 Error report saved in /reports")

# ---------------- RUN ----------------

if __name__ == "__main__":
    bootstrap_labels()
