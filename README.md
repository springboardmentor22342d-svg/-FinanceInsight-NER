# Financial NER Extraction Pipeline

Extract structured financial insights from 10-K/10-Q reports using custom-trained FinBERT NER model.

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-31019/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Features

- ✅ **Custom FinBERT NER** trained on scraped finance articles
- ✅ **PDF Table Extraction** with `pdfplumber` (preserves multi-column structure)
- ✅ **Section Detection** (MD&A, Financial Statements, Risk Factors, etc.)
- ✅ **Entity Extraction** (ORG, MONEY, DATE, PERCENT, etc.)
- ✅ **Multi-Year Table Parsing** (2024, 2023, 2022 columns)
- ✅ **Clean JSON Output** (`insights.json`)

---

## 📦 Installation

### Prerequisites
- Python 3.10.19
- CUDA (optional, for GPU training)

### Setup

```bash
# Clone repository
git clone https://github.com/DevilsBreath/financial-ner-extraction.git
cd financial-ner-extraction

# Create virtual environment
python -m venv venv
source venv/bin/activate

# On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

