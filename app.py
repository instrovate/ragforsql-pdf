import streamlit as st
from sqlalchemy import create_engine
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext, StorageContext, load_index_from_storage
from llama_index.core.llms import OpenAI
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.embeddings import OpenAIEmbedding
from llama_index.core.retrievers import VectorIndexRetriever, RouterRetriever, SQLRetriever
from llama_index.core.query_engine import RetrieverQueryEngine, SQLAutoVectorQueryEngine, RouterQueryEngine
from llama_index.core.indices.vector_store import GPTVectorStoreIndex
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core import Settings
import os

st.set_page_config(page_title="Hybrid RAG Demo", layout="wide")

st.title("🔍 Hybrid RAG Demo: Query SQL + PDF with LLM")

# === CONFIGURATION ===
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
Settings.embed_model = OpenAIEmbedding()
Settings.llm = OpenAI(model="gpt-3.5-turbo")

# === LOAD FILES ===
use_sample = st.checkbox("Use sample files", value=False)

if use_sample:
    st.success("Loaded sample files from GitHub!")
    db_path = "sample/sample.db"
    pdf_dir = "sample"
else:
    db_file = st.file_uploader("Upload SQLite DB File", type=["db"])
    pdf_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

    if db_file and pdf_files:
        db_path = f"temp/{db_file.name}"
        os.makedirs("temp", exist_ok=True)
        with open(db_path, "wb") as f:
            f.write(db_file.read())

        pdf_dir = "temp/pdf_uploads"
        os.makedirs(pdf_dir, exist_ok=True)
        for pdf in pdf_files:
            with open(os.path.join(pdf_dir, pdf.name), "wb") as f:
                f.write(pdf.read())
    else:
        st.stop()

# === LOAD SQL ===
def load_sql_docs(db_path):
    engine = create_engine(f"sqlite:///{db_path}")
    tables = engine.table_names() if hasattr(engine, "table_names") else engine.inspect().get_table_names()
    return SQLRetriever(engine=engine, tables=tables)

sql_retriever = load_sql_docs(db_path)
sql_tool = QueryEngineTool(
    query_engine=RetrieverQueryEngine.from_args(sql_retriever),
    metadata=ToolMetadata(name="sql_tool", description="Query structured database tables")
)

# === LOAD PDF ===
pdf_docs = SimpleDirectoryReader(pdf_dir).load_data()
pdf_index = VectorStoreIndex.from_documents(pdf_docs)
pdf_retriever = VectorIndexRetriever(index=pdf_index)
pdf_tool = QueryEngineTool(
    query_engine=RetrieverQueryEngine.from_args(pdf_retriever),
    metadata=ToolMetadata(name="pdf_tool", description="Query unstructured PDF content")
)

# === ROUTER ===
router_engine = RouterQueryEngine(
    selector=RouterRetriever.from_tools([sql_tool, pdf_tool]),
    query_engine_tools=[sql_tool, pdf_tool]
)

# === CHAT UI ===
query = st.text_input("Ask something about the SQL or PDF content:")
if query:
    response = router_engine.query(query)
    st.write("### 🤖 Response:")
    st.markdown(response.response)
