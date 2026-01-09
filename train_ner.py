
"""
Lightweight NER fine-tuning script using spaCy.

Purpose:
- Auto-annotate financial text using simple heuristics
- Fine-tune spaCy NER component
- Persist trained model to disk

Refactored and documented for clarity.
"""

import random
import re
from pathlib import Path

import pandas as pd
import spacy
from spacy.training import Example

SPACY_BASE_MODEL = "en_core_web_sm"
TRAIN_DATA_PATH = "data/raw/financial_news.csv"
OUTPUT_MODEL_DIR = "trained_model"
MAX_SAMPLES = 300
EPOCHS = 5


def auto_annotate(text: str):
    entities = []

    for m in re.finditer(r'Rs\.?\s?\d+[\,\d]*', text):
        entities.append((m.start(), m.end(), "MONEY"))

    for org in ["TCS", "Infosys", "ICICI", "HDFC", "Reliance", "REC", "BSE"]:
        for m in re.finditer(org, text):
            entities.append((m.start(), m.end(), "ORG"))

    return text, {"entities": entities}


def main():
    nlp = spacy.load(SPACY_BASE_MODEL)

    df = pd.read_csv(TRAIN_DATA_PATH)
    texts = df.iloc[:, 0].dropna().astype(str).tolist()[:MAX_SAMPLES]

    train_data = [
        auto_annotate(t)
        for t in texts
        if auto_annotate(t)[1]["entities"]
    ]

    ner = nlp.get_pipe("ner")
    optimizer = nlp.resume_training()

    for epoch in range(EPOCHS):
        random.shuffle(train_data)
        losses = {}
        for text, annotations in train_data:
            doc = nlp.make_doc(text)
            example = Example.from_dict(doc, annotations)
            nlp.update([example], sgd=optimizer, losses=losses)
        print(f"Epoch {epoch + 1}, Losses: {losses}")

    Path(OUTPUT_MODEL_DIR).mkdir(exist_ok=True)
    nlp.to_disk(OUTPUT_MODEL_DIR)


if __name__ == "__main__":
    main()
