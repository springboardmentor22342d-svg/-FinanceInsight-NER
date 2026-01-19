# AI Financial Analyst - NER & Data Extraction Tool

## 📌 Project Overview
This project is an automated financial analysis pipeline designed to process unstructured financial documents (like 10-K Annual Reports). It uses a hybrid approach combining **Deep Learning (FinBERT)** and **Rule-Based Logic (Regex)** to extract key financial metrics and Balance Sheet data.

## 📂 Files in this Repository
* **`ner_model.ipynb`**: The complete Python code (Google Colab Notebook) that handles data ingestion, cleaning, AI inference, and report generation.
* **`final_submission.json`**: The structured output file containing extracted entities (Company, Metrics, Values, Periods) and tabular data.
* **`AI_Financial_Report.pdf`**: A generated readable report summarizing the AI's findings for human review.

## 🚀 Key Features
* **PDF Parsing**: Uses `pdfplumber` to ingest raw text from complex PDF documents.
* **Hybrid Extraction**:
    * **Text Analysis**: Uses **FinBERT** (NLP) to identify Organization names.
    * **Logic Layer**: Uses custom Regex patterns to accurately extract numerical values (Revenue, Assets, etc.) and filter out noise.
* **Table Mining**: A robust text-mining algorithm that detects Balance Sheet rows ("Total Assets", "Liabilities") even in PDFs without visible grid lines.
* **Automated Reporting**: Automatically generates a professional PDF summary of the findings.

## 🛠️ Tech Stack
* **Language**: Python
* **AI Model**: FinBERT (Hugging Face Transformers)
* **Libraries**: `pdfplumber`, `reportlab`, `pandas`, `re`, `json`

## 📊 Milestone Summary
* **Milestone 1:** Data Collection & Cleaning (Preprocessing).
* **Milestone 2:** Model Training & Fine-tuning (NER).
* **Milestone 3:** Custom Extraction Logic (Business Rules).
* **Milestone 4:** PDF Integration & Final Reporting (Complete Pipeline).

---
*Submitted by: Nidhi Jat*
# FinanceInsight-NER

## Project Overview
FinanceInsight-NER is a Natural Language Processing (NLP) project focused on
Named Entity Recognition (NER) for financial text data. The goal is to identify
key financial entities such as organizations, monetary values, financial ratios,
and economic indicators from unstructured financial documents.

## Mentorship
This project is developed under the mentorship of **Ms. Dikshita**, as part of
a guided learning and evaluation program.

## Objectives
- Collect and preprocess financial text data
- Apply domain-specific cleaning for financial terminology
- Explore and experiment with NER models for financial entities
- Evaluate model performance using precision, recall, and F1-score

## Technologies Used
- Python
- Jupyter Notebook
- NLP libraries (NLTK / spaCy / Transformers)
- Financial text datasets

## Current Progress
- Financial text preprocessing completed
- Exploratory Data Analysis performed
- Initial NER model experimentation implemented

## Repository Structure
- `infosys_intern (4).ipynb` → Main implementation notebook
- `README.md` → Project documentation

## Future Work
- Improve entity recognition accuracy
- Experiment with transformer-based financial models (FinBERT)
- Perform deeper error analysis and refinement
