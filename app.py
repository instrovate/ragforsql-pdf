import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage, download_loader
from sqlalchemy import create_engine
import pandas as pd
import os
import fitz  # PyMuPDF
import requests

st.set_page_config(page_title="Hybrid RAG Demo", layout="centered")

st.title("🔍 Hybrid RAG Demo: Query SQL + PDF with LLM")

use_sample = st.checkbox("Use sample files")

# --- Sample file URLs ---
sample_db_url = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.db"
sample_pdf_url = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.pdf"

# --- File loading ---
def download_file(url, local_path):
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

if use_sample:
    db_path = "sample.db"
    pdf_path = "sample.pdf"
    download_file(sample_db_url, db_path)
    download_file(sample_pdf_url, pdf_path)
    st.success("Loaded sample files from GitHub!")
else:
    db_file = st.file_uploader("Upload SQLite database (.db)", type=["db"])
    pdf_file = st.file_uploader("Upload PDF file", type=["pdf"])

    if db_file and pdf_file:
        db_path = os.path.join("temp", db_file.name)
        pdf_path = os.path.join("temp", pdf_file.name)

        os.makedirs("temp", exist_ok=True)
        with open(db_path, "wb") as f:
            f.write(db_file.read())
        with open(pdf_path, "wb") as f:
            f.write(pdf_file.read())
    else:
        st.warning("Please upload both a .db and .pdf file or select sample.")
        st.stop()

# --- Load SQL Data ---
def load_sql_docs(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    tables = engine.table_names()

    all_text = ""
    for table in tables:
        df = pd.read_sql_table(table, engine)
        doc_text = f"Table: {table}\n{df.head(5)}\n\n"
        all_text += doc_text

    return all_text

# --- Load PDF Text ---
def load_pdf_docs(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# --- Create Index ---
@st.cache_resource
def create_index_from_text(text_data):
    with open("combined.txt", "w", encoding="utf-8") as f:
        f.write(text_data)

    reader = SimpleDirectoryReader(input_files=["combined.txt"])
    docs = reader.load_data()
    index = VectorStoreIndex.from_documents(docs)
    return index

# --- Load documents ---
sql_docs = load_sql_docs(db_path)
pdf_docs = load_pdf_docs(pdf_path)

combined_docs = sql_docs + "\n\n" + pdf_docs
index = create_index_from_text(combined_docs)
query_engine = index.as_query_engine()

st.markdown("### 🤖 Ask a question below:")
question = st.text_input("What would you like to ask?")

# --- Sample Questions ---
with st.expander("💡 Sample Questions to Try"):
    st.markdown("""
    - What tables are present in the database?
    - What does the PDF say about the company’s goals?
    - Give me a summary combining both PDF content and SQL data.
    - Show the top 5 rows of the customers table.
    - What are the key products mentioned in the PDF?
    """)

if question:
    with st.spinner("Thinking..."):
        response = query_engine.query(question)
        st.markdown("### 💬 Answer:")
        st.write(response.response)
