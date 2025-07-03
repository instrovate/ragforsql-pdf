
import streamlit as st
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
)
from llama_index.readers.file import PDFReader
from llama_index.readers.database import SQLDatabaseReader
from sqlalchemy import create_engine
import os
import requests

st.set_page_config(page_title="RAG on SQL + PDF", layout="wide")
st.title("🔍 RAG App: Query Your Data from SQL + PDF")
st.markdown("Use your own `.pdf` or `.db` file, or try the sample files below!")

# Sample file URLs from your GitHub
SAMPLE_PDF_URL = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.pdf"
SAMPLE_DB_URL = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.db"

# Utility function to download if not present
def download_file(url, local_path):
    if not os.path.exists(local_path):
        r = requests.get(url)
        with open(local_path, "wb") as f:
            f.write(r.content)

use_sample = st.checkbox("Use preloaded sample PDF and SQLite DB (from GitHub)", value=True)

# File handling logic
if use_sample:
    pdf_path = "example.pdf"
    db_path = "example.db"
    download_file(SAMPLE_PDF_URL, pdf_path)
    download_file(SAMPLE_DB_URL, db_path)
else:
    uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
    uploaded_db = st.file_uploader("Upload your SQLite DB", type=["db"])

    os.makedirs("temp", exist_ok=True)

    if uploaded_pdf:
        pdf_path = os.path.join("temp", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

    if uploaded_db:
        db_path = os.path.join("temp", uploaded_db.name)
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())

# Indexing logic
if ('pdf_path' in locals() and os.path.exists(pdf_path)) and ('db_path' in locals() and os.path.exists(db_path)):
    if st.button("🔎 Index & Preview Data"):
        st.success(f"Using PDF: `{pdf_path}`")
        st.success(f"Using DB: `{db_path}`")

        # Index PDF
        pdf_docs = PDFReader().load_data(file=pdf_path)
        pdf_index = VectorStoreIndex.from_documents(pdf_docs)

        # Index SQLite DB
        db_engine = create_engine(f"sqlite:///{db_path}")
        db_reader = SQLDatabaseReader(db_engine)
        db_docs = db_reader.load_data()
        db_index = VectorStoreIndex.from_documents(db_docs)

        # Save in session
        st.session_state["pdf_index"] = pdf_index
        st.session_state["db_index"] = db_index

# Prompt interface
prompt = st.text_input("💬 Ask a question about your data:")

if prompt:
    if "pdf_index" in st.session_state and "db_index" in st.session_state:
        with st.spinner("Thinking..."):
            pdf_response = st.session_state["pdf_index"].as_query_engine().query(prompt)
            db_response = st.session_state["db_index"].as_query_engine().query(prompt)
            combined_response = f"📄 **From PDF**: {pdf_response}\n\n🗄️ **From SQL DB**: {db_response}"
        st.markdown(combined_response)
    else:
        st.warning("Please click '🔎 Index & Preview Data' to load your sources first.")
