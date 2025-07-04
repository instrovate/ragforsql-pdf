import streamlit as st
from llama_index.core import VectorStoreIndex, Document
from llama_index.readers.file import PDFReader
import pandas as pd
import sqlite3
import os

st.set_page_config(page_title="Hybrid RAG: SQL + PDF", layout="wide")
st.title("🔍 Hybrid RAG Demo: Query SQL + PDF with LLM")

# Use sample option
use_sample = st.checkbox("Use sample files")

if use_sample:
    db_path = "example.db"
    pdf_path = "example.pdf"
else:
    uploaded_db = st.file_uploader("Upload SQLite database (.db)", type=["db"])
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_db and uploaded_pdf:
        db_path = uploaded_db.name
        pdf_path = uploaded_pdf.name
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
    else:
        st.warning("Please upload both a .db and .pdf file or select sample.")
        st.stop()

# --- Custom SQL loader ---
def load_sql_docs(path):
    conn = sqlite3.connect(path)
    cursor = conn.cursor()
    tables = cursor.execute("SELECT name FROM sqlite_master WHERE type='table';").fetchall()
    docs = []
    for (table_name,) in tables:
        df = pd.read_sql_query(f"SELECT * FROM {table_name} LIMIT 100", conn)
        doc_text = f"Table: {table_name}\n{df.to_markdown(index=False)}"
        docs.append(Document(text=doc_text))
    conn.close()
    return docs

# Load SQL & PDF
sql_docs = load_sql_docs(db_path)
pdf_docs = PDFReader().load_data(file=pdf_path)
all_docs = sql_docs + pdf_docs

index = VectorStoreIndex.from_documents(all_docs)

query = st.text_input("Ask your question")

if query:
    with st.spinner("Thinking..."):
        response = index.as_query_engine().query(query)
        st.markdown("### 📄 Answer")
        st.write(response.response)
