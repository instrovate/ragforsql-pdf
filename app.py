import streamlit as st
import os
import sqlite3
import pandas as pd
from sqlalchemy import create_engine, inspect
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.llms import OpenAI
from llama_index.core.readers.file import PDFReader
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.tools import QueryEngineTool, ToolMetadata

# ----- CONFIG -----
st.set_page_config(page_title="Hybrid RAG Demo", layout="centered")

st.title("🔍 Hybrid RAG Demo: Query SQL + PDF with LLM")

# ----- SAMPLE FILES -----
SAMPLE_DB_URL = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.db"
SAMPLE_PDF_URL = "https://raw.githubusercontent.com/instrovate/ragforsql-pdf/main/example.pdf"

# Download sample files locally if needed
def download_sample_file(url, local_path):
    import requests
    response = requests.get(url)
    with open(local_path, "wb") as f:
        f.write(response.content)

# ----- LOADERS -----
def load_pdf_docs(pdf_path):
    reader = PDFReader()
    return reader.load_data(file=pdf_path)

def load_sql_docs(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    inspector = inspect(engine)
    tables = inspector.get_table_names()

    all_text = ""
    for table in tables:
        df = pd.read_sql_table(table, engine)
        doc_text = f"Table: {table}\n{df.head(5).to_markdown(index=False)}\n\n"
        all_text += doc_text

    return all_text

# ----- INPUT HANDLING -----
use_sample = st.checkbox("Use sample files")

if use_sample:
    db_path = "example.db"
    pdf_path = "example.pdf"

    if not os.path.exists(db_path):
        download_sample_file(SAMPLE_DB_URL, db_path)
    if not os.path.exists(pdf_path):
        download_sample_file(SAMPLE_PDF_URL, pdf_path)

    st.success("Loaded sample files from GitHub!")
else:
    uploaded_db = st.file_uploader("Upload SQLite database (.db)", type=["db"])
    uploaded_pdf = st.file_uploader("Upload PDF file", type=["pdf"])

    if uploaded_db:
        db_path = os.path.join("temp.db")
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())
    else:
        db_path = None

    if uploaded_pdf:
        pdf_path = os.path.join("temp.pdf")
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
    else:
        pdf_path = None

# ----- RAG PIPELINE -----
if (use_sample or (db_path and pdf_path)):

    with st.spinner("🔄 Indexing your files..."):
        # Load docs
        pdf_docs = load_pdf_docs(pdf_path)
        sql_docs = load_sql_docs(db_path)

        # Build tools
        service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-3.5-turbo"))

        pdf_index = VectorStoreIndex.from_documents(pdf_docs, service_context=service_context)
        sql_index = VectorStoreIndex.from_documents(
            [pdf_docs[0].__class__(text=sql_docs)], service_context=service_context
        )

        pdf_tool = QueryEngineTool(
            query_engine=pdf_index.as_query_engine(),
            metadata=ToolMetadata(name="pdf", description="Information from the uploaded PDF file")
        )
        sql_tool = QueryEngineTool(
            query_engine=sql_index.as_query_engine(),
            metadata=ToolMetadata(name="sql", description="Information extracted from the database tables")
        )

        # Hybrid Query Engine
        hybrid_engine = SubQuestionQueryEngine.from_defaults(
            query_engine_tools=[pdf_tool, sql_tool],
            service_context=service_context
        )

    # ----- QUESTION BOX -----
    st.subheader("Ask a question:")
    question = st.text_input("Try questions like:", "What does the PDF say about the 'Order Flow'?")

    if st.button("Get Answer") and question:
        with st.spinner("Thinking..."):
            response = hybrid_engine.query(question)
            st.markdown("### 📌 Answer:")
            st.write(response)

    st.markdown("---")
    st.markdown("#### 🧠 Sample Questions:")
    st.markdown("- What is the most popular category in the database?")
    st.markdown("- Which customer has the highest order total?")
    st.markdown("- What does the PDF say about delivery policies?")
    st.markdown("- Summarize the product return guidelines from the PDF.")
