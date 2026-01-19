import os
import json
import csv
import yfinance as yf
from datetime import datetime

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
entities_file = os.path.join(BASE_DIR, "../outputs/entities_output.json")
events_file = os.path.join(BASE_DIR, "../outputs/events_output.json")
json_output_file = os.path.join(BASE_DIR, "../outputs/verified_results.json")
csv_output_file = os.path.join(BASE_DIR, "../outputs/verified_results.csv")

# -----------------------------
# Load extracted entities and events
# -----------------------------
with open(entities_file, "r") as f:
    entities = json.load(f)

with open(events_file, "r") as f:
    events = json.load(f)

# -----------------------------
# Map company names to ticker symbols (expand as needed)
# -----------------------------
company_to_ticker = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Google": "GOOGL",
    "Amazon": "AMZN",
}

# -----------------------------
# Example: detect companies dynamically from entities/events
# -----------------------------
# For simplicity, just hardcode here. In real use, extract from text via NER
companies_in_text = ["Apple", "Microsoft"]  

all_verified_results = []

for company_name in companies_in_text:
    ticker = company_to_ticker.get(company_name)
    if not ticker:
        continue

    stock = yf.Ticker(ticker)
    info = stock.info

    verified_results = {
        "company": company_name,
        "ticker": ticker,
        "financial_entities": [],
        "financial_events": []
    }

    # -----------------------------
    # Verify entities
    # -----------------------------
    for item in entities:
        entity_result = {}
        if "EPS" in item:
            entity_result["EPS"] = {
                "extracted": item["EPS"],
                "actual": info.get("trailingEps")
            }
        if "REVENUE_GROWTH" in item:
            entity_result["REVENUE_GROWTH"] = {
                "extracted": item["REVENUE_GROWTH"],
                "actual": info.get("revenueGrowth")
            }
        if entity_result:
            verified_results["financial_entities"].append(entity_result)

    # -----------------------------
    # Verify events (basic placeholder)
    # -----------------------------
    for event in events:
        event_result = event.copy()
        try:
            event_date = datetime.strptime(event.get("date"), "%m/%d/%Y")
            event_result["verified"] = True  # could be enhanced with actual verification
        except:
            event_result["verified"] = False
        verified_results["financial_events"].append(event_result)

    all_verified_results.append(verified_results)

# -----------------------------
# Save JSON
# -----------------------------
os.makedirs(os.path.dirname(json_output_file), exist_ok=True)
with open(json_output_file, "w") as f:
    json.dump(all_verified_results, f, indent=4)

# -----------------------------
# Save CSV for entities
# -----------------------------
with open(csv_output_file, "w", newline="") as csvfile:
    fieldnames = ["company", "entity", "extracted", "actual"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for res in all_verified_results:
        for entity in res["financial_entities"]:
            for k, v in entity.items():
                writer.writerow({
                    "company": res["company"],
                    "entity": k,
                    "extracted": v["extracted"],
                    "actual": v["actual"]
                })

# -----------------------------
# Print summary
# -----------------------------
print("Verification completed!")
print("JSON saved to:", json_output_file)
print("CSV saved to:", csv_output_file)
