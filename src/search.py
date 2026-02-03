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

print("🧠 Loading AI Model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

def search(query_text, k=2):
    print(f"\n🔎 Query: '{query_text}'")
    
    query_vector = model.encode(query_text).tolist()
    
    payload = {
        "vector": query_vector,
        "k": k,
        "include_vectors": False
    }
    
    url = f"{BASE_URL}/index/{INDEX_NAME}/search"
    
    try:
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            results = msgpack.unpackb(response.content, raw=False)
            
            print(f"   ✅ Found {len(results)} matches:")
            
            for i, res in enumerate(results):
                # FIXED MAPPING BASED ON DEBUG OUTPUT:
                # [0] = Score (Float)
                # [1] = ID (String)
                score = res[0]
                doc_id = res[1]
                
                print(f"      [{i+1}] ID: {doc_id} | Score: {score:.4f}")
                
        else:
            print(f"   ❌ Search Failed: {response.status_code} - {response.text}")

    except Exception as e:
        print(f"   ⚠️ Error: {e}")

if __name__ == "__main__":
    search("How do I run Endee on Windows?")
    search("What is RAG?")