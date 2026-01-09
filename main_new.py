
"""
Main pipeline for financial text processing and NER extraction.

This script performs:
- Data loading and normalization
- Text preprocessing using spaCy
- Basic EDA
- Named Entity Recognition (NER)
- Rule-based financial metric extraction
- Long-document segmentation
- Table-like numeric parsing
- Structured JSON output generation

Original inspiration:
- Financial NLP pipeline structure adapted from an open-source GitHub project.
- Refactored, documented, and reorganized by <YOUR NAME> for educational use.

No core logic has been altered.
"""

import json
import re
from typing import List, Dict

import pandas as pd
import spacy

# ---------------------------
# CONFIGURATION
# ---------------------------
RAW_DATA_FILES = [
    ("data/raw/financial_news.csv", "intro"),
    ("data/raw/indian_financial_news.csv", "Description"),
]
OUTPUT_JSON_PATH = "output/final_output.json"
SPACY_MODEL = "en_core_web_sm"

SECTION_HEADERS = {
    "MD&A": ["management", "discussion"],
    "Risk Factors": ["risk", "uncertainty"],
    "Financial Statements": ["revenue", "profit", "assets", "liabilities"],
}


# ---------------------------
# INITIALIZATION
# ---------------------------
nlp = spacy.load(SPACY_MODEL)


# ---------------------------
# DATA LOADING
# ---------------------------
def load_and_combine_datasets(files: List[tuple]) -> pd.DataFrame:
    """Load multiple CSV files and standardize text into a single column."""
    frames = []
    for path, text_col in files:
        df = pd.read_csv(path)
        df = df[[text_col]].rename(columns={text_col: "content"})
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined.dropna(inplace=True)
    return combined


# ---------------------------
# TEXT PREPROCESSING
# ---------------------------
def preprocess_text(text: str) -> List[Dict]:
    """Tokenize, lemmatize, and POS-tag text while removing noise."""
    text = re.sub(r'[$€₹]', '', text)
    doc = nlp(text)

    tokens = []
    for token in doc:
        if not token.is_stop and not token.is_punct:
            tokens.append({
                "token": token.text,
                "lemma": token.lemma_,
                "pos": token.pos_
            })
    return tokens


# ---------------------------
# NER EXTRACTION
# ---------------------------
def extract_entities(text: str):
    """Extract named entities using spaCy's pretrained model."""
    doc = nlp(text)
    return [(ent.text, ent.label_) for ent in doc.ents]


# ---------------------------
# RULE-BASED EXTRACTION
# ---------------------------
def extract_by_keyword(text: str, keyword: str) -> List[str]:
    """Extract sentences containing a given financial keyword."""
    return [
        sent.strip()
        for sent in text.split(".")
        if keyword.lower() in sent.lower()
    ]


# ---------------------------
# LONG DOCUMENT HANDLING
# ---------------------------
def segment_document(text: str) -> Dict[str, List[str]]:
    """Segment a long document into financial sections."""
    sections = {"General": []}
    current_section = "General"

    for sentence in text.split("."):
        for section, keywords in SECTION_HEADERS.items():
            if any(k in sentence.lower() for k in keywords):
                current_section = section
                sections.setdefault(section, [])
        sections[current_section].append(sentence.strip())

    return sections


def is_table_line(line: str) -> bool:
    return sum(char.isdigit() for char in line) > 6


def parse_table(lines: List[str]) -> List[Dict]:
    """Parse numeric-heavy lines into key-value pairs."""
    rows = []
    for line in lines:
        parts = line.split()
        if len(parts) > 1:
            rows.append({
                "item": " ".join(parts[:-1]),
                "value": parts[-1]
            })
    return rows


# ---------------------------
# MAIN EXECUTION
# ---------------------------
def main():
    df = load_and_combine_datasets(RAW_DATA_FILES)

    # Sample preprocessing
    df["processed"] = df["content"].head(5).apply(preprocess_text)

    # EDA
    df["text_length"] = df["content"].apply(lambda x: len(str(x).split()))

    # NER
    df["entities"] = df["content"].head(5).apply(extract_entities)

    # Long document simulation
    long_doc = " ".join(df["content"].head(200).tolist())
    sections = segment_document(long_doc)

    table_lines = [l for l in long_doc.split(".") if is_table_line(l)]
    parsed_tables = parse_table(table_lines[:5])

    final_output = {
        "company": "Sample Financial Corpus",
        "sections": sections,
        "sample_entities": df["entities"].head(3).tolist(),
        "custom_extraction_example": extract_by_keyword(df["content"].iloc[0], "profit"),
        "tables": parsed_tables
    }

    with open(OUTPUT_JSON_PATH, "w") as f:
        json.dump(final_output, f, indent=2)


if __name__ == "__main__":
    main()
