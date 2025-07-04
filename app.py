import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.readers.file import PDFReader
from llama_index.readers.database import SQLDatabaseReader
from sqlalchemy import create_engine
import os

# Title
st.title("Hybrid RAG with SQL + PDF")

# File Upload
pdf_file = st.file_uploader("Upload a PDF", type=["pdf"])
db_file = st.file_uploader("Upload a SQLite .db File", type=["db"])
query = st.text_input("Ask your question:")

if pdf_file and db_file and query:
    with open("temp.pdf", "wb") as f:
        f.write(pdf_file.read())

    pdf_docs = PDFReader().load_data(file=Path("temp.pdf"))
    pdf_index = VectorStoreIndex.from_documents(pdf_docs)

    db_path = f"sqlite:///{db_file.name}"
    with open(db_file.name, "wb") as f:
        f.write(db_file.read())

    db_engine = create_engine(db_path)
    db_reader = SQLDatabaseReader(db_engine)
    db_docs = db_reader.load_data()
    db_index = VectorStoreIndex.from_documents(db_docs)

    # Combine indexes (basic version, you can do reranking later)
    combined_nodes = pdf_index.docstore.docs.values() + db_index.docstore.docs.values()
    hybrid_index = VectorStoreIndex.from_documents(list(combined_nodes))
    query_engine = hybrid_index.as_query_engine()
    response = query_engine.query(query)
    st.write(response)
