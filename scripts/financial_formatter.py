import json
import re
from pathlib import Path

INPUT_PATH = Path(r"C:\Users\aksha\Documents\FinanceInsight\reports\milestone4_output.json")
OUTPUT_PATH = Path(r"C:\Users\aksha\Documents\FinanceInsight\reports\final_financial_intelligence.json")

def normalize_number(text):
    match = re.search(r"([\d,]+)", text.replace(",", ""))
    return match.group(1) if match else None

def extract_year(text):
    m = re.search(r"(2023|2024|2025)", text)
    return m.group(1) if m else None

def smart_format():
    raw = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    final = []

    for block in raw:
        section = block["section"]
        table_type = block["table_type"]
        units = block.get("units", "Not specified")

        for row in block["rows"]:
            item = row["item"].lower()
            value = normalize_number(row["item"] + " " + row["value"])
            year = extract_year(row["item"] + " " + row["value"])

            # Detect key financial metrics
            for metric in ["revenue", "net income", "total assets", "total liabilities", "gross margin", "operating income"]:
                if metric in item:
                    final.append({
                        "company": "Apple",
                        "metric": metric,
                        "value": value,
                        "period": year or "Latest",
                        "section": table_type,
                        "units": units
                    })

    OUTPUT_PATH.write_text(json.dumps(final, indent=4))
    print("🔥 Final financial intelligence generated")

if __name__ == "__main__":
    smart_format()
