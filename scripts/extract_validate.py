import spacy
import yfinance as yf
import pandas as pd
import json
from pathlib import Path

# Load your best trained model
MODEL_PATH = Path(r"C:\FinanceInsight\models\model-best")
DATA_PATH = Path(r"C:\FinanceInsight\data\processed\financial_news_events_clean.csv")

def get_live_price(ticker):
    """Links extracted ticker to Yahoo Finance for validation [cite: 53, 118]"""
    try:
        stock = yf.Ticker(ticker)
        # Fetching current price for cross-referencing
        return stock.fast_info.get('last_price', "N/A")
    except Exception:
        return "Not Found"

def run_final_extraction():
    if not MODEL_PATH.exists():
        print(f"Error: Model not found at {MODEL_PATH}. Wait for training to finish!")
        return

    nlp = spacy.load(MODEL_PATH)
    df = pd.read_csv(DATA_PATH)
    
    results = []
    # Process a sample of processed news events [cite: 46]
    for text in df['clean_text'].astype(str).head(50):
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            ent_data = {"text": ent.text, "label": ent.label_}
            
            # Milestone 3: Validation integration [cite: 81, 118]
            if ent.label_ == "TICKER":
                ent_data["market_validation_price"] = get_live_price(ent.text)
            
            entities.append(ent_data)
        
        if entities:
            results.append({"original_text": text, "extracted_entities": entities})

    # Save final deliverables [cite: 75, 76]
    output_file = Path(r"C:\FinanceInsight\data\final_extraction_results.json")
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"✅ Milestone 3 Complete! Results saved to {output_file}")

if __name__ == "__main__":
    run_final_extraction()