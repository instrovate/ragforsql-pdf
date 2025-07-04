import streamlit as st
from llama_index.core import (
    VectorStoreIndex, SimpleDirectoryReader, SQLStructStoreIndex, ServiceContext
)
from llama_index.core.objects import SQLDatabase
from llama_index.readers.file import PDFReader
from sqlalchemy import create_engine
import os
import requests

st.set_page_config(page_title="RAG on SQL + PDF", layout="wide")

st.title("🔍 RAG App: Query SQL + PDF with GPT")

st.markdown("Use your own `.pdf` and `.db` file — or try preloaded samples!")

use_sample = st.checkbox("Use GitHub-hosted sample PDF and SQLite DB", value=True)

# Make sure temp folder exists
os.makedirs("temp", exist_ok=True)

if use_sample:
    pdf_url = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.pdf"
    db_url = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.db"

    pdf_path = "temp/example.pdf"
    db_path = "temp/example.db"

    if not os.path.exists(pdf_path):
        with open(pdf_path, "wb") as f:
            f.write(requests.get(pdf_url).content)

    if not os.path.exists(db_path):
        with open(db_path, "wb") as f:
            f.write(requests.get(db_url).content)

else:
    uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
    uploaded_db = st.file_uploader("Upload your SQLite DB", type=["db"])

    if uploaded_pdf:
        pdf_path = os.path.join("temp", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

    if uploaded_db:
        db_path = os.path.join("temp", uploaded_db.name)
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())

if "pdf_path" not in locals() or not os.path.exists(pdf_path):
    st.warning("Upload or load a PDF to continue.")
elif "db_path" not in locals() or not os.path.exists(db_path):
    st.warning("Upload or load a SQLite DB to continue.")
else:
    if st.button("🔎 Index Files"):
        st.success(f"Loaded PDF: {pdf_path}")
        st.success(f"Loaded DB: {db_path}")

        # Index PDF
        pdf_docs = PDFReader().load_data(file=pdf_path)
        pdf_index = VectorStoreIndex.from_documents(pdf_docs)

        # Index SQLite DB
        engine = create_engine(f"sqlite:///{db_path}")
        sql_database = SQLDatabase(engine)
        sql_index = SQLStructStoreIndex.from_sql_database(sql_database)

        st.session_state["pdf_index"] = pdf_index
        st.session_state["sql_index"] = sql_index

prompt = st.text_input("Ask a question about your data:")

if prompt and "pdf_index" in st.session_state and "sql_index" in st.session_state:
    with st.spinner("Thinking..."):
        pdf_ans = st.session_state["pdf_index"].as_query_engine().query(prompt)
        sql_ans = st.session_state["sql_index"].as_query_engine().query(prompt)

        st.markdown("### 📄 Answer from PDF")
        st.write(pdf_ans)

        st.markdown("### 🗃️ Answer from SQL DB")
        st.write(sql_ans)
