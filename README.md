
# Financial NER Pipeline (Refactored)

This repository contains a refactored and well-documented implementation of a
financial text processing and Named Entity Recognition (NER) pipeline.

## Overview
The project demonstrates how unstructured financial news can be transformed into
structured insights using NLP techniques such as tokenization, lemmatization,
NER, rule-based extraction, and document segmentation.

## Features
- Text preprocessing with spaCy
- Named Entity Recognition (baseline + fine-tuned)
- Rule-based extraction for financial metrics
- Long document segmentation
- Table-like numeric parsing
- Structured JSON output

## Technologies
- Python 3.x
- spaCy
- pandas

## How to Run
```bash
pip install pandas spacy
python -m spacy download en_core_web_sm

python main_refactored.py
```

## Attribution
This project is a **refactored derivative work** inspired by an open-source
financial NLP project on GitHub. The code has been reorganized, documented,
and enhanced for educational purposes.

All credit for the original idea and baseline implementation goes to the
original author. Any mistakes introduced during refactoring are mine.
