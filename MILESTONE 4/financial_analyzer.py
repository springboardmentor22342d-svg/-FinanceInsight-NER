# financial_analyzer.py

import streamlit as st
import pandas as pd
import pdfplumber
import json
import re

from segmentation import segment_document, parse_financial_tables

# ============================
# Streamlit App Title
# ============================
st.title("Financial Document Analyzer")

# ============================
# File Upload (CSV, TXT, PDF, DOCX)
# ============================
uploaded_file = st.file_uploader(
    "Upload a financial report (CSV, TXT, PDF, DOCX)",
    type=["csv", "txt", "pdf", "docx"]
)

text = ""

# ============================
# PDF Reader (pdfplumber)
# ============================
def read_pdf(uploaded_file):
    text = ""
    with pdfplumber.open(uploaded_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text

# ============================
# Handle Uploaded Files
# ============================
if uploaded_file:

    # -------- CSV --------
    if uploaded_file.type == "text/csv":
        df = pd.read_csv(uploaded_file)
        text = df.to_string(index=False)

    # -------- PDF --------
    elif uploaded_file.type == "application/pdf":
        text = read_pdf(uploaded_file)

    # -------- TXT --------
    elif uploaded_file.type == "text/plain":
        text = uploaded_file.read().decode("utf-8")

    # -------- DOCX --------
    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        import docx
        doc = docx.Document(uploaded_file)
        text = "\n".join([para.text for para in doc.paragraphs])

# ============================
# Manual Input
# ============================
if not uploaded_file:
    text = st.text_area("Or paste financial report text here")

# ============================
# Helper Functions
# ============================
def clean_table(df):
    return df



def extract_mdna_metrics(mdna_text):
    results = []

    revenue = re.search(
        r"revenue.*?(\$?\d+\.?\d*\s*(billion|million))",
        mdna_text,
        re.I
    )
    year = re.search(r"(20\d{2})", mdna_text)

    if revenue:
        results.append({
            "company": "Apple",
            "metric": "revenue",
            "value": revenue.group(1),
            "period": year.group(1) if year else "2023",
            "section": "MD&A"
        })

    return results


def evaluate_system(sections, tables):
    score = 0
    total = 2  # segmentation + table parsing

    if sections and len(sections) >= 2:
        score += 1

    if tables and len(tables) >= 1:
        score += 1

    return (score / total) * 100

# ============================
# Analyze Button
# ============================
if st.button("Analyze"):

    if not text.strip():
        st.warning("Please upload a document or paste some text to analyze.")

    else:
        final_json = []

        # -------- Section Segmentation --------
        sections = segment_document(text)

        if sections:
            st.subheader("Document Sections")

            for section, content in sections.items():
                with st.expander(section):
                    st.write(content)

                if "MD&A" in section.upper():
                    final_json.extend(extract_mdna_metrics(content))

        else:
            st.info("No sections detected.")

        # -------- Table Parsing --------
        tables = parse_financial_tables(text)

        if tables:
            st.subheader("Financial Tables")

            for table_name, df in tables.items():
                st.write(f"**{table_name}**")

                if isinstance(df, pd.DataFrame):
                    df = clean_table(df)
                    st.dataframe(df)

                    final_json.append({
                        "section": "Financial Statements",
                        "table_type": table_name,
                        "rows": df.head(5).to_dict(orient="records")
                    })
                else:
                    st.write(df)

        else:
            st.info("No financial tables detected.")

        # -------- Final Evaluation --------
        accuracy = evaluate_system(sections, tables)
        st.subheader("System Evaluation")
        st.success(f"Overall System Completion: {accuracy:.0f}%")

        # -------- Final JSON Output --------
        st.subheader("Final JSON Output")
        st.json(final_json)

        st.download_button(
            label="Download JSON",
            data=json.dumps(final_json, indent=4),
            file_name="final_output.json",
            mime="application/json"
        )
