import streamlit as st
import os
from sqlalchemy import create_engine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.indices.vector_store import GPTVectorStoreIndex
from llama_index.retrievers import SQLRetriever, VectorIndexRetriever, RouterRetriever
from llama_index.query_engine import RetrieverQueryEngine, RouterQueryEngine
from llama_index.tools.query_engine import QueryEngineTool
from llama_index.tools.types import ToolMetadata
from llama_index import Settings

st.set_page_config(page_title="Hybrid RAG Demo", layout="wide")
st.title("🔍 Hybrid RAG Demo: Query SQL + PDF using LLM")

# === Set API Key ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# === Use sample toggle ===
use_sample = st.checkbox("Use sample files", value=False)

if use_sample:
    db_path = "sample/sample.db"
    pdf_dir = "sample"
else:
    db_file = st.file_uploader("Upload SQLite DB File", type=["db"])
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if db_file and pdf_files:
        os.makedirs("temp", exist_ok=True)
        db_path = os.path.join("temp", db_file.name)
        with open(db_path, "wb") as f:
            f.write(db_file.read())

        pdf_dir = "temp/pdf_uploads"
        os.makedirs(pdf_dir, exist_ok=True)
        for pdf in pdf_files:
            with open(os.path.join(pdf_dir, pdf.name), "wb") as f:
                f.write(pdf.read())
    else:
        st.stop()

# === SQL Retriever ===
def load_sql_docs(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    tables = engine.table_names()
    return SQLRetriever(sql_engine=engine, tables=tables)

sql_retriever = load_sql_docs(db_path)
sql_tool = QueryEngineTool(
    query_engine=RetrieverQueryEngine.from_args(sql_retriever),
    metadata=ToolMetadata(name="sql_tool", description="Query the database tables")
)

# === PDF Retriever ===
pdf_docs = SimpleDirectoryReader(pdf_dir).load_data()
pdf_index = GPTVectorStoreIndex.from_documents(pdf_docs)
pdf_retriever = VectorIndexRetriever(index=pdf_index)
pdf_tool = QueryEngineTool(
    query_engine=RetrieverQueryEngine.from_args(pdf_retriever),
    metadata=ToolMetadata(name="pdf_tool", description="Query the unstructured PDF documents")
)

# === Router ===
router_engine = RouterQueryEngine(
    selector=RouterRetriever.from_tools([sql_tool, pdf_tool]),
    query_engine_tools=[sql_tool, pdf_tool]
)

# === Input ===
query = st.text_input("Ask a question about the SQL or PDF content:")
if query:
    response = router_engine.query(query)
    st.markdown("### 🤖 Response:")
    st.markdown(response.response)
