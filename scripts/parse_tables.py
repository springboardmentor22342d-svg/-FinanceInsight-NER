import pdfplumber
import json
import re
from pathlib import Path

BASE_DIR = Path(r"C:\Users\aksha\Documents\FinanceInsight")
RAW_DIR = BASE_DIR / "data" / "raw"
REPORT_PATH = BASE_DIR / "reports" / "final_output.json"

RAW_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)

PDF_PATH = RAW_DIR / "annual_report.pdf"

# ------------------ VALIDATORS ------------------
BAD_METRICS = {
    "", "level", "subtotal", "total", "thereafter", "beginning balances",
    "ending balances", "other", "current", "deferred"
}

def valid_metric_name(x):
    return x.lower() not in BAD_METRICS and len(x) > 2

def is_year(x):
    return re.fullmatch(r"(19|20)\d{2}", x) is not None

def extract_money(text):
    matches = re.findall(r"\$?\(?\d{1,3}(?:,\d{3})*(?:\.\d+)?\)?", text)
    valid = []
    for m in matches:
        clean = m.replace("$","").replace(",","")
        if not is_year(clean):
            valid.append(m)
    return valid

# ------------------ CLEANING ------------------

def clean_text(x):
    x = re.sub(r"[^\w\s.,%-]", "", x)
    x = re.sub(r"\s+", " ", x)
    return x.strip()

def clean_row(row):
    row = [clean_text(str(c)) for c in row if c and str(c).lower() not in ("nan", "", "none")]
    if not row:
        return None

    full = " ".join(row)
    money = extract_money(full)

    if not money:
        return None   # Not financial

    metric = re.split(r"\$|\d", full, maxsplit=1)[0].strip()

    if not valid_metric_name(metric):
        return None

    # Reject garbage numeric rows
    if all(is_year(m.replace(",", "")) for m in money):
        return {"item": metric, "value": None}

    # If value looks weird → null
    clean_values = []
    for m in money:
        raw = m.replace(",", "")
        if len(raw) < 4:   # too small to be real money
            continue
        clean_values.append(m)

    if not clean_values:
        return {"item": metric, "value": None}

    return {
        "item": metric,
        "value": " | ".join(clean_values)
    }

# ------------------ CLASSIFIER ------------------

def detect_table_type(text):
    text = text.lower()
    if any(x in text for x in ["asset","liabilit","equity"]):
        return "Balance Sheet"
    if any(x in text for x in ["revenue","income","profit","sales","margin"]):
        return "Income Statement"
    if any(x in text for x in ["cash","flow"]):
        return "Cash Flow Statement"
    return None

# ------------------ EXTRACTOR ------------------

def extract_financial_tables():
    if not PDF_PATH.exists():
        print("❌ PDF not found:", PDF_PATH)
        return

    final_json = []

    with pdfplumber.open(PDF_PATH) as pdf:
        for page in pdf.pages:

            page_text = (page.extract_text() or "").lower()
            if any(x in page_text for x in ["table of contents", "index", "contents"]):
                continue  # Skip non-financial pages

            for table in page.extract_tables():
                rows = []
                flat = ""

                for raw in table:
                    cleaned = clean_row(raw)
                    if cleaned:
                        rows.append(cleaned)
                        flat += " " + cleaned["item"].lower()

                if not rows:
                    continue

                table_type = detect_table_type(flat)
                if not table_type:
                    continue   # Not a real financial table

                final_json.append({
                    "section": "Financial Statements",
                    "table_type": table_type,
                    "rows": rows
                })

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(final_json, f, indent=4)

    print("🔥 Final clean financial JSON saved →")
    print(REPORT_PATH)
    print(f"📊 Tables extracted: {len(final_json)}")

# ------------------ RUN ------------------

if __name__ == "__main__":
    extract_financial_tables()
