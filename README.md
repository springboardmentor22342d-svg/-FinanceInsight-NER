# FinanceInsight
FinanceInsight — an intelligent financial data pipeline that automates the collection, preprocessing, and organization of company reports, SEC filings, and financial news for analysis and model training.
# FinanceInsight

FinanceInsight — an intelligent financial data pipeline that automates the collection, preprocessing, and organization of company reports, SEC filings, and financial news for analysis and ML.

## Features
- Automated data collection (PDF/HTML)
- Organized raw & processed folders
- Simple preprocessing pipeline skeleton
- Ready for NLP/ML experiments

## Quickstart (Windows PowerShell)
```powershell
# create a venv and activate
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# install deps
pip install -r requirements.txt

# run collector
python scripts\data_collector.py --sources data/data_sources.csv --out data/raw
