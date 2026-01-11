import spacy
import yfinance as yf
import json

# Load the best model generated from training [cite: 88, 90]
try:
    nlp = spacy.load(r"C:\FinanceInsight\models\model-best")
except:
    print("Model not found. Please run 'spacy train' first.")
    nlp = spacy.load("en_core_web_sm") # Fallback

def get_market_context(ticker):
    """Link extracted data with financial databases."""
    stock = yf.Ticker(ticker)
    return stock.info.get("regularMarketPrice", "N/A")

def run_extraction(text):
    doc = nlp(text)
    extracted = []
    
    for ent in doc.ents:
        data = {"entity": ent.text, "label": ent.label_}
        # If it's a ticker, get live price for validation [cite: 69]
        if ent.label_ == "TICKER":
            data["live_price"] = get_market_context(ent.text)
        extracted.append(data)
    
    return extracted

if __name__ == "__main__":
    sample_text = "Apple AAPL reported record revenue this quarter."
    results = run_extraction(sample_text)
    print(json.dumps(results, indent=2))