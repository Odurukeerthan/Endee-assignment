import streamlit as st
import requests
import msgpack
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8090/api/v1"
HEADERS = {
    "Authorization": "admin-secret-123",
    "Content-Type": "application/json"
}
INDEX_NAME = "tech_docs"

# --- MOCK DATABASE (To show text in UI) ---
# In a real production app, we would fetch this from Endee using /vector/get
# For this assignment, a lookup dict is faster and 100% reliable.
DOC_LOOKUP = {
    "doc_1": "Endee is a high-performance vector database optimized for SIMD.",
    "doc_2": "Run Endee on Windows using Docker Desktop and WSL2.",
    "doc_3": "Data is persisted in the /data directory inside the container.",
    "doc_4": "RAG allows LLMs to answer questions using private data."
}

# --- SETUP PAGE ---
st.set_page_config(page_title="Endee AI Search", page_icon="⚡")

st.title("⚡ Endee.io Semantic Search")
st.markdown("This project uses **Endee Vector DB** + **SentenceTransformers** to perform RAG-style retrieval.")

# --- CACHED MODEL LOADER ---
# We cache this so we don't reload the AI model on every click
@st.cache_resource
def load_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

try:
    model = load_model()
    st.success("✅ AI Model Loaded")
except Exception as e:
    st.error(f"Failed to load model: {e}")

# --- SEARCH FUNCTION ---
def search_endee(query):
    # 1. Vectorize
    vector = model.encode(query).tolist()
    
    # 2. Payload
    payload = {
        "vector": vector,
        "k": 3,
        "include_vectors": False
    }
    
    try:
        url = f"{BASE_URL}/index/{INDEX_NAME}/search"
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            return msgpack.unpackb(response.content, raw=False)
        else:
            st.error(f"Endee API Error: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return []

# --- UI INPUT ---
query = st.text_input("Ask a question about Endee:", placeholder="e.g., How do I install on Windows?")

if query:
    with st.spinner("Searching Vector Space..."):
        results = search_endee(query)
    
    st.divider()
    
    if results:
        st.subheader("🔍 Top Matches")
        for res in results:
            # Parse Endee Response [score, id, ...]
            score = res[0]
            doc_id = res[1]
            
            # Lookup the actual text
            content = DOC_LOOKUP.get(doc_id, "Unknown Document Content")
            
            # Display Card
            with st.container():
                st.markdown(f"**📄 Document ID:** `{doc_id}` | **Confidence:** `{score:.4f}`")
                st.info(content)
    else:
        st.warning("No results found.")

# --- SIDEBAR INFO ---
with st.sidebar:
    st.header("System Status")
    try:
        health = requests.get(f"{BASE_URL}/health").json()
        st.success(f"🟢 Endee is Online")
    except:
        st.error("🔴 Endee is Offline")
    
    st.markdown("---")
    st.caption("Built for Endee Assignment")