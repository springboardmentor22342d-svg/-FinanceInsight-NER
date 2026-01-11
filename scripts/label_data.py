import os
import pandas as pd
import spacy
from spacy.tokens import DocBin
from spacy.matcher import PhraseMatcher
from tqdm import tqdm
from pathlib import Path

# Paths
DATA_DIR = Path(r"C:\FinanceInsight\data\processed")
TRAIN_DIR = Path(r"C:\FinanceInsight\data\train")
os.makedirs(TRAIN_DIR, exist_ok=True)

# 1. Define known entities to bootstrap labels
TICKERS = ["AAPL", "TSLA", "MSFT", "INFY", "TCS"]
METRICS = ["revenue", "net income", "earnings", "ebitda", "eps"]

def create_spacy_binary(csv_name, out_name):
    nlp = spacy.blank("en")
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Add patterns to matcher
    matcher.add("TICKER", [nlp.make_doc(t) for t in TICKERS])
    matcher.add("FIN_METRIC", [nlp.make_doc(m) for m in METRICS])
    
    doc_bin = DocBin()
    df = pd.read_csv(DATA_DIR / csv_name)
    
    for text in tqdm(df['clean_text'].astype(str), desc=f"Labeling {csv_name}"):
        doc = nlp.make_doc(text)
        matches = matcher(doc)
        
        ents = []
        for match_id, start, end in matches:
            span = doc[start:end]
            span.label_ = nlp.vocab.strings[match_id]
            ents.append(span)
        
        # Filter overlapping entities
        doc.ents = spacy.util.filter_spans(ents)
        doc_bin.add(doc)
        
    doc_bin.to_disk(TRAIN_DIR / out_name)

if __name__ == "__main__":
    create_spacy_binary("data_clean.csv", "train.spacy")
    create_spacy_binary("data_clean.csv", "dev.spacy") # Use a slice for real dev set
    print(f"Created binary files in {TRAIN_DIR}")