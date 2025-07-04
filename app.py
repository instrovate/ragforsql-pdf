import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, StorageContext, load_index_from_storage
from llama_index.readers.file import PDFReader
from llama_index.readers.database import DatabaseReader
from sqlalchemy import create_engine
import os

# Set page config
st.set_page_config(page_title="Hybrid RAG: SQL + PDF", layout="wide")

st.title("🔍 Hybrid RAG Demo: Query SQL + PDF with LLM")

# Option to load sample files
use_sample = st.checkbox("Use sample database and PDF")

if use_sample:
    db_path = "example.db"
    pdf_path = "example.pdf"
else:
    uploaded_db = st.file_uploader("Upload a SQLite database (.db)", type=["db"])
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    if uploaded_db and uploaded_pdf:
        db_path = uploaded_db.name
        pdf_path = uploaded_pdf.name
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
    else:
        st.warning("Please upload both a PDF and a .db file or select 'Use sample'.")
        st.stop()

# Load SQL
engine = create_engine(f"sqlite:///{db_path}")
sql_reader = DatabaseReader(engine=engine)
sql_docs = sql_reader.load_data()

# Load PDF
pdf_loader = PDFReader()
pdf_docs = pdf_loader.load_data(file=pdf_path)

# Combine
all_docs = sql_docs + pdf_docs
index = VectorStoreIndex.from_documents(all_docs)

query = st.text_input("Ask a question about the data")

if query:
    with st.spinner("Thinking..."):
        response = index.as_query_engine().query(query)
        st.markdown("### 📄 Answer")
        st.write(response.response)
