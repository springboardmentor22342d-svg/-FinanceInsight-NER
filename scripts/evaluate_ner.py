import spacy, json
from pathlib import Path

nlp = spacy.load("models/model-best")

BASE_DIR = Path("C:/FinanceInsight")
DATA_PATH = BASE_DIR / "data/processed/data_clean.csv"
REPORT_PATH = BASE_DIR / "reports/ner_error_report.json"

def analyze():
    import pandas as pd
    df = pd.read_csv(DATA_PATH)

    error_logs = []

    for text in df["clean_text"].astype(str).head(200):
        doc = nlp(text)

        for ent in doc.ents:
            # boundary sanity check for financial values
            if ent.label_ in ["MONEY", "PERCENT"] and " " in ent.text:
                error_logs.append({
                    "text": text,
                    "entity": ent.text,
                    "label": ent.label_,
                    "issue": "POSSIBLE_BOUNDARY_ERROR"
                })

    with open(REPORT_PATH, "w") as f:
        json.dump(error_logs, f, indent=4)

    print("📊 Error report generated.")

analyze()
