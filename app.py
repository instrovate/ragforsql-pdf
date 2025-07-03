
import streamlit as st
import os
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    ServiceContext,
    SQLDatabase,
    StorageContext,
)
from llama_index.query_engine import RouterQueryEngine
from llama_index.tools import QueryEngineTool
from llama_index.llms import OpenAI
from sqlalchemy import create_engine

# Set OpenAI Key (replace with your own)
os.environ["OPENAI_API_KEY"] = "your-openai-key"

# Title
st.title("🔄 Hybrid RAG App – SQL + PDF")
st.markdown("Ask a question and get answers from structured (SQLite) and unstructured (PDF) data using GPT.")

# Let user choose to upload or use sample data
use_sample = st.checkbox("Use preloaded sample data from GitHub (example.db + example.pdf)", value=True)

# File upload
if not use_sample:
    uploaded_pdf = st.file_uploader("Upload a PDF file", type=["pdf"])
    uploaded_db = st.file_uploader("Upload a SQLite DB file", type=["db"])
else:
    uploaded_pdf = "data/pdf/example.pdf"
    uploaded_db = "data/example.db"

# Load services only if files are ready
if uploaded_pdf and uploaded_db:

    # Handle uploaded PDF saving
    if not use_sample:
        os.makedirs("data/pdf", exist_ok=True)
        pdf_path = os.path.join("data/pdf", uploaded_pdf.name)
        with open(pdf_path, "wb") as f:
            f.write(uploaded_pdf.read())
    else:
        pdf_path = uploaded_pdf

    # Handle uploaded DB saving
    if not use_sample:
        os.makedirs("data", exist_ok=True)
        db_path = os.path.join("data", uploaded_db.name)
        with open(db_path, "wb") as f:
            f.write(uploaded_db.read())
    else:
        db_path = uploaded_db

    # Setup LLM
    service_context = ServiceContext.from_defaults(llm=OpenAI(model="gpt-4"))

    # PDF RAG tool
    documents = SimpleDirectoryReader("data/pdf").load_data()
    pdf_index = VectorStoreIndex.from_documents(documents, service_context=service_context)
    pdf_tool = QueryEngineTool.from_defaults(
        query_engine=pdf_index.as_query_engine(), name="PDF_RAG", description="PDF-based retrieval"
    )

    # SQL RAG tool
    engine = create_engine(f"sqlite:///{db_path}")
    sql_db = SQLDatabase(engine)
    sql_tool = QueryEngineTool.from_defaults(
        query_engine=sql_db.as_query_engine(), name="SQL_RAG", description="SQL-based retrieval"
    )

    # Router engine
    router = RouterQueryEngine.from_defaults(
        tools=[pdf_tool, sql_tool], service_context=service_context
    )

    # Optional preview
    if st.checkbox("Preview PDF text"):
        st.markdown("Sample content from uploaded PDF:")
        with open(pdf_path, "rb") as file:
            st.text(file.read().decode("latin1")[:1000])

    if st.checkbox("Preview SQL table"):
        st.markdown("Showing first 5 rows of the `financials` table:")
        import sqlite3
        conn = sqlite3.connect(db_path)
        df = conn.execute("SELECT * FROM financials LIMIT 5").fetchall()
        st.table(df)
        conn.close()

    # Ask question
    st.markdown("## 🔍 Ask a Question")
    question = st.text_input("Your question")

    if question:
        with st.spinner("Retrieving..."):
            response = router.query(question)
            st.success("Answer:")
            st.write(response.response)
