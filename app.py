import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index import Settings

# ---- Config ----
st.set_page_config(page_title="PDF Q&A RAG", layout="wide")
st.title("📄 Ask Questions from Your PDF")

# ---- Set API Keys ----
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = OpenAIEmbedding()

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    filepath = os.path.join("temp", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing PDF..."):
        documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

    query = st.text_input("Ask a question from your PDF:")
    if query:
        response = query_engine.query(query)
        st.markdown("### 🤖 Response:")
        st.write(response.response)
