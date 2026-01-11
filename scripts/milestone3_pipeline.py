import spacy
import yfinance as yf
import json
import re
import pandas as pd
from pathlib import Path

# Load trained model
nlp = spacy.load("models/model-best")

DATA_PATH = Path("C:/Users/aksha/Documents/FinanceInsight/data/processed/data_clean.csv")
REPORT_DIR = Path("C:/Users/aksha/Documents/FinanceInsight/reports")
REPORT_DIR.mkdir(exist_ok=True)

# ------------------ HELPERS ------------------

def find_ticker_from_company(company):
    try:
        search = yf.search(company)
        if search and "quotes" in search and len(search["quotes"]) > 0:
            return search["quotes"][0]["symbol"]
    except:
        pass
    return None

# ------------------ EXTRACTION ------------------

def extract_entities(text, requested_metrics=None):
    doc = nlp(text)

    results = {
        "companies": [],
        "tickers": [],
        "metrics": [],
        "events": []
    }

    # NER companies
    for ent in doc.ents:
        if ent.label_ == "ORG":
            results["companies"].append(ent.text)

    # Raw-text enrichment
    tokens = re.findall(r"\b[A-Za-z]+\b|\b[A-Z]{1,5}\b", text)

    for tok in tokens:
        lower = tok.lower()

        if tok.isupper() and 1 < len(tok) <= 5:
            results["tickers"].append(tok)

        if lower in ["revenue","earnings","profit","ebitda","growth","loss"]:
            if not requested_metrics or lower in requested_metrics:
                results["metrics"].append(tok)

        if lower in ["merger","acquisition","ipo","split"]:
            results["events"].append(tok)

    # Resolve ticker from company names
    for company in results["companies"]:
        ticker = find_ticker_from_company(company)
        if ticker:
            results["tickers"].append(ticker)

    # Deduplicate
    for k in results:
        results[k] = list(set(results[k]))

    return results

# ------------------ VALIDATION ------------------

def validate_with_yahoo(tickers):
    validated = []
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            validated.append({
                "ticker": ticker,
                "official_name": info.get("longName"),
                "market_price": info.get("regularMarketPrice")
            })
        except:
            validated.append({"ticker": ticker, "error": "Lookup failed"})
    return validated

# ------------------ PIPELINE ------------------

def process_text(text, requested_metrics=None):
    extracted = extract_entities(text, requested_metrics)
    validated = validate_with_yahoo(extracted["tickers"])
    return {
        "text": text,
        "extracted_entities": extracted,
        "validated_tickers": validated
    }

def process_full_dataset():
    df = pd.read_csv(DATA_PATH)

    all_results = []

    for i, row in df.iterrows():
        record = process_text(str(row["clean_text"]), ["revenue", "earnings"])
        record["document_id"] = i
        all_results.append(record)

        if (i + 1) % 50 == 0:
            print(f"Processed {i + 1} documents...")

    with open(REPORT_DIR / "full_dataset_results.json", "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=4)

    print("\n✅ Full dataset processing complete.")

# ------------------ DEMO ------------------

if __name__ == "__main__":

    # Demo: generic text
    demo1 = "accord finnish russian chamber commerce major construction company finland operate russia"
    with open(REPORT_DIR / "demo_generic_text.json", "w") as f:
        json.dump(process_text(demo1, ["revenue", "earnings"]), f, indent=4)

    # Demo: financial news
    demo2 = """
    Apple Inc (AAPL) reported strong revenue growth this quarter.
    The company announced earnings on July 15 and hinted at a possible merger.
    """
    with open(REPORT_DIR / "demo_financial_news.json", "w") as f:
        json.dump(process_text(demo2, ["revenue", "earnings"]), f, indent=4)

    # Full dataset run
    process_full_dataset()
