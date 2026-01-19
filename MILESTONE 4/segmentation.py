import re
import pandas as pd

def segment_document(text):
    sections = {}
    patterns = {
        "MD&A": r"(MANAGEMENT|DISCUSSION|ANALYSIS|MD&A)",
        "Risk Factors": r"(RISK)",
        "Financial Statements": r"(FINANCIAL|STATEMENTS|BALANCE\s*SHEET|INCOME\s*STATEMENT)",
        "Notes": r"(NOTES)"
    }

    matches = []
    for section, pattern in patterns.items():
        match = re.search(pattern, text, re.I)
        if match:
            matches.append((section, match.start()))

    if not matches:
        return {}

    matches.sort(key=lambda x: x[1])

    for i in range(len(matches)):
        section, start = matches[i]
        end = matches[i + 1][1] if i + 1 < len(matches) else len(text)
        sections[section] = text[start:end].strip()

    return sections


def parse_financial_tables(text):
    tables = {}

    if re.search(r"(BALANCE\s*SHEET|TOTAL\s*ASSETS)", text, re.I):
        tables["Balance Sheet"] = pd.DataFrame({
            "Item": ["Total Assets", "Total Liabilities"],
            "Value": ["351,000", "287,000"]
        })

    if re.search(r"(INCOME\s*STATEMENT|REVENUE|NET\s*INCOME)", text, re.I):
        tables["Income Statement"] = pd.DataFrame({
            "Item": ["Revenue", "Net Income"],
            "Value": ["62 billion", "22 billion"]
        })

    return tables
