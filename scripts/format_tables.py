import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(r"C:\Users\aksha\Documents\FinanceInsight")
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = BASE_DIR / "reports" / "formatted_financial_report.json"

results = []

for csv_file in PROCESSED_DIR.glob("annual_report_*.csv"):
    df = pd.read_csv(csv_file)

    table_entry = {
        "section": "Financial Statements",
        "table_type": csv_file.stem,
        "units": "USD (millions)",
        "rows": []
    }

    for _, row in df.iterrows():
        item = str(row[0])
        value = str(row[1]) if len(row) > 1 else ""

        table_entry["rows"].append({
            "item": item,
            "value": value
        })

    results.append(table_entry)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=4)

print("✅ Formatted financial report created")
