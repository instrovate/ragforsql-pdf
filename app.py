import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.readers.file import PDFReader
from llama_index.readers.database import SQLDatabaseReader
from sqlalchemy import create_engine
import os

st.set_page_config(page_title="🔍 RAG: SQL + PDF", layout="wide")
st.title("🔍 RAG App: Ask from SQL DB and PDF 📄")

use_sample = st.checkbox("Use Sample PDF and DB", value=True)

os.makedirs("temp", exist_ok=True)

if use_sample:
    pdf_path = "temp/example.pdf"
    db_path = "temp/example.db"

    if not os.path.exists(pdf_path):
        import requests
        r = requests.get("https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.pdf")
        with open(pdf_path, "wb") as f:
            f.write(r.content)

    if not os.path.exists(db_path):
        r = requests.get("https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.db")
        with open(db_path, "wb") as f:
            f.write(r.content)
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

if "pdf_path" in locals() and "db_path" in locals():
    if st.button("🚀 Index and Load"):
        pdf_docs = PDFReader().load_data(file=pdf_path)
        pdf_index = VectorStoreIndex.from_documents(pdf_docs)

        db_engine = create_engine(f"sqlite:///{db_path}")
        db_reader = SQLDatabaseReader(db_engine)
        db_docs = db_reader.load_data()
        db_index = VectorStoreIndex.from_documents(db_docs)

        st.session_state["pdf_index"] = pdf_index
        st.session_state["db_index"] = db_index
        st.success("Indexes created successfully!")

prompt = st.text_input("Ask your question (PDF + SQL):")

if prompt and "pdf_index" in st.session_state and "db_index" in st.session_state:
    with st.spinner("Thinking..."):
        pdf_ans = st.session_state["pdf_index"].as_query_engine().query(prompt)
        sql_ans = st.session_state["db_index"].as_query_engine().query(prompt)

        st.markdown("### 📄 PDF Answer")
        st.write(str(pdf_ans))

        st.markdown("### 🗄️ SQL Answer")
        st.write(str(sql_ans))
