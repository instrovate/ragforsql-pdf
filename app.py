import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, SQLDatabase
from llama_index.readers.file import PDFReader
from llama_index.readers.database.base import SQLDatabaseReader
from sqlalchemy import create_engine
import os

st.set_page_config(page_title="RAG on SQL + PDF", layout="wide")
st.title("🔍 RAG App: Query Your Data from SQL + PDF")
st.markdown("Use your own `.pdf` or `.db` file, or try the sample files below!")

use_sample = st.checkbox("Use preloaded sample PDF and SQLite DB", value=True)

if use_sample:
    pdf_path = "example.pdf"
    db_path = "example.db"
else:
    uploaded_pdf = st.file_uploader("Upload your PDF", type=["pdf"])
    uploaded_db = st.file_uploader("Upload your SQLite DB", type=["db"])

    if uploaded_pdf:
        pdf_path = os.path.join("temp", uploaded_pdf.name)
        os.makedirs("temp", exist_ok=True)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())

    if uploaded_db:
        db_path = os.path.join("temp", uploaded_db.name)
        os.makedirs("temp", exist_ok=True)
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())

if ("pdf_path" in locals() and os.path.exists(pdf_path)) and ("db_path" in locals() and os.path.exists(db_path)):
    if st.button("🔎 Index & Preview Data"):
        st.success(f"Using PDF: {pdf_path}")
        st.success(f"Using DB: {db_path}")

        # Index PDF
        pdf_docs = PDFReader().load_data(file=pdf_path)
        pdf_index = VectorStoreIndex.from_documents(pdf_docs)

        # Index SQLite DB
        engine = create_engine(f"sqlite:///{db_path}")
        sql_database = SQLDatabase(engine)
        db_reader = SQLDatabaseReader(sql_database)
        db_docs = db_reader.load_data()
        db_index = VectorStoreIndex.from_documents(db_docs)

        st.session_state["pdf_index"] = pdf_index
        st.session_state["db_index"] = db_index

prompt = st.text_input("Ask a question about your data:")

if prompt and "pdf_index" in st.session_state and "db_index" in st.session_state:
    with st.spinner("Thinking..."):
        pdf_response = st.session_state["pdf_index"].as_query_engine().query(prompt)
        db_response = st.session_state["db_index"].as_query_engine().query(prompt)
        st.markdown(f"📄 **From PDF**: {pdf_response}")
        st.markdown(f"🗄️ **From SQL DB**: {db_response}")
