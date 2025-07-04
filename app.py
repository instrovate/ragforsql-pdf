import streamlit as st
import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms import OpenAI as OpenAI_LLM
from llama_index import set_global_service_context

# ---- UI Setup ----
st.set_page_config(page_title="PDF Q&A RAG", layout="wide")
st.title("📄 Ask Questions from Your PDF")

# ---- OpenAI Key Setup ----
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ---- Set Embedding & LLM for LlamaIndex ----
embed_model = OpenAIEmbedding()
llm = OpenAI_LLM(model="gpt-3.5-turbo")
service_context = ServiceContext.from_defaults(llm=llm, embed_model=embed_model)
set_global_service_context(service_context)

# ---- File Upload ----
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    os.makedirs("temp", exist_ok=True)
    filepath = os.path.join("temp", uploaded_file.name)
    with open(filepath, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Indexing your PDF..."):
        documents = SimpleDirectoryReader(input_files=[filepath]).load_data()
        index = VectorStoreIndex.from_documents(documents)
        query_engine = index.as_query_engine()

    query = st.text_input("Ask a question from your PDF:")
    if query:
        response = query_engine.query(query)
        st.markdown("### 🤖 Response:")
        st.write(response.response)
