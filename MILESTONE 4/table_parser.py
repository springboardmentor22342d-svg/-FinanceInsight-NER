import streamlit as st
import pandas as pd
from segmentation import segment_document, parse_financial_tables
def clean_table(df):
    df = df.dropna(how="all")
    for col in df.columns[1:]:
        df[col] = df[col].astype(str).str.replace(",", "").astype(float, errors="ignore")
    return df
if st.button("Analyze"):
    if not text.strip():
        st.warning("Please paste some text to analyze.")
    else:
        # Section Segmentation
        sections = segment_document(text)
        if not sections:
            st.info("No sections detected.")
        else:
            st.subheader("Document Sections")
            for section, content in sections.items():
                with st.expander(section, expanded=False):
                    st.write(content)
        # Table Parsing
        tables = parse_financial_tables(text)
        if tables:
            st.subheader("Financial Tables")
            for table_name, df in tables.items():
                st.write(f"**{table_name}**")
                if isinstance(df, pd.DataFrame):
                    df = clean_table(df)  # clean numeric values
                    st.dataframe(df)
                else:
                    st.write(df)
