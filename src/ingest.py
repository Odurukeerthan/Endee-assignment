import requests
import sys
import time
from sentence_transformers import SentenceTransformer

# --- CONFIGURATION ---
BASE_URL = "http://localhost:8090/api/v1"
HEADERS = {
    "Authorization": "admin-secret-123",
    "Content-Type": "application/json"
}
INDEX_NAME = "tech_docs"

# --- DATASET ---
raw_data = [
    {"id": "doc_1", "text": "Endee is a high-performance vector database optimized for SIMD.", "category": "overview"},
    {"id": "doc_2", "text": "Run Endee on Windows using Docker Desktop and WSL2.", "category": "installation"},
    {"id": "doc_3", "text": "Data is persisted in the /data directory inside the container.", "category": "storage"},
    {"id": "doc_4", "text": "RAG allows LLMs to answer questions using private data.", "category": "ai_concepts"}
]

def create_index():
    print(f"🔨 Creating Index '{INDEX_NAME}'...")
    
    # ✅ CORRECT PAYLOAD BASED ON C++ SOURCE CODE
    payload = {
        "index_name": INDEX_NAME,  # Changed from "name"
        "dim": 384,                # Changed from "dimension"
        "space_type": "cosine",    # Changed from "metric"
        
        # Optional settings found in source code:
        "M": 16,                   # Connections per node
        "ef_con": 200,             # Construction search depth
        "precision": "float32"     # Options: "int8", "float32"
    }
    
    try:
        url = f"{BASE_URL}/index/create"
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            print(f"   ✅ Index created successfully!")
        elif "already exists" in response.text:
            print(f"   ⚠️ Index already exists (Proceeding...)")
        else:
            print(f"   ❌ Create Failed: {response.status_code} - {response.text}")
            sys.exit(1) # Stop if we can't create the index
            
    except Exception as e:
        print(f"   ⚠️ Connection Error: {e}")
        sys.exit(1)

def main():
    # 1. Create Index
    create_index()

    # 2. Load Model
    print(f"🧠 Loading AI Model...")
    model = SentenceTransformer('all-MiniLM-L6-v2')

    success = 0
    print(f"🚀 Processing {len(raw_data)} documents...")
    
    # 3. Insert Data
    for doc in raw_data:
        vector = model.encode(doc["text"]).tolist()
        
        # Structure found in source code:
        # It handles a LIST of objects or a single object.
        # We will send one by one for simplicity.
        payload = {
            "id": doc["id"],
            "vector": vector,
            # Metadata isn't explicitly shown in the insert handler, 
            # but usually handled or we might need to store it separately.
            # For now, we send vector + id.
        }
        
        # URL Logic from source: /api/v1/index/<index_name>/vector/insert
        url = f"{BASE_URL}/index/{INDEX_NAME}/vector/insert"
        
        try:
            # The C++ code checks if body is a List or Object. We send Object.
            resp = requests.post(url, json=payload, headers=HEADERS)
            
            if resp.status_code == 200:
                print(f"   ✅ Indexed: {doc['id']}")
                success += 1
            else:
                print(f"   ❌ Failed {doc['id']}: {resp.status_code} - {resp.text}")
        except Exception as e:
            print(f"   ⚠️ Error: {e}")

    print(f"\n🎉 Finished. {success}/{len(raw_data)} documents active.")

if __name__ == "__main__":
    main()