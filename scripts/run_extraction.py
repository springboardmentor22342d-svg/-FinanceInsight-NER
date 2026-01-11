import spacy
import yfinance as yf
import pandas as pd
import json
from pathlib import Path

# 1. Load your best trained model
MODEL_PATH = Path(r"C:\FinanceInsight\models\model-best")
nlp = spacy.load(MODEL_PATH)

def get_market_validation(ticker):
    """Link extracted data with Yahoo Finance to verify accuracy[cite: 53, 81]."""
    try:
        stock = yf.Ticker(ticker)
        return stock.info.get("regularMarketPrice", "N/A")
    except:
        return "Validation Error"

def run_milestone_3():
    # Load cleaned news events from Milestone 1
    df = pd.read_csv(r"C:\FinanceInsight\data\processed\financial_news_events_clean.csv")
    
    results = []
    # Process the first 50 entries as a test
    for text in df['clean_text'].astype(str).head(50):
        doc = nlp(text)
        entities = []
        for ent in doc.ents:
            info = {"text": ent.text, "label": ent.label_}
            # Milestone 3: Cross-reference Tickers [cite: 118]
            if ent.label_ == "TICKER":
                info["live_price"] = get_market_validation(ent.text)
            entities.append(info)
        
        if entities:
            results.append({"text_source": text, "entities": entities})

    # Save final results [cite: 76]
    with open(r"C:\FinanceInsight\data\final_extracted_data.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Milestone 3 Complete! Data saved to final_extracted_data.json")

if __name__ == "__main__":
    run_milestone_3()