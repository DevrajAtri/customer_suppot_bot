import json
import os
import time
from pinecone import Pinecone
from servicess.vector_engine import VectorEngine
from dotenv import load_dotenv
from tqdm import tqdm # Progress bar

load_dotenv()

# --- CONFIGURATION ---
JSONL_PATH = "chunks_out/aq_chunks.jsonl" # Path to your file
INDEX_NAME = "ecom-bot" # As seen in your screenshot
BATCH_SIZE = 50

# Map 'source_citation' from JSON to Pinecone Namespaces
NAMESPACE_MAP = {
    "privacy": "privacy",  
    "terms": "terms",      # Just look for "terms", it's safer than the full string
    "faq": "faq",          # Lowercase
    "grievance": "grievance"
}

def upsert_chunks():
    # 1. Initialize Pinecone
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in environment variables")
        
    pc = Pinecone(api_key=api_key)
    index = pc.Index(INDEX_NAME)
    
    # 2. Initialize Engine
    engine = VectorEngine()
    
    # 3. Read Data
    print(f"Reading chunks from {JSONL_PATH}...")
    with open(JSONL_PATH, 'r', encoding='utf-8') as f:
        # Read all lines first to group them (optional, but cleaner)
        all_lines = [json.loads(line) for line in f]

    # 4. Process in Batches
    # We group data by namespace first to minimize API switching
    data_by_ns = {}
    
    for record in all_lines:
        raw_source = record.get("source_citation", "unknown")
        source_lower = raw_source.lower()
        
        # Determine Namespace (Default to 'general' if no match)
        # We try strict matching first, then partial
        ns = "general"
        for key, val in NAMESPACE_MAP.items():
            if key in source_lower:
                ns = val
                break
        
        if ns not in data_by_ns:
            data_by_ns[ns] = []
        data_by_ns[ns].append(record)

    # 5. Upsert Loop
    for ns, records in data_by_ns.items():
        print(f"\nProcessing Namespace: '{ns}' ({len(records)} chunks)")
        
        for i in range(0, len(records), BATCH_SIZE):
            batch = records[i : i + BATCH_SIZE]
            
            # Prepare texts for vectorization
            texts = [b["text"] for b in batch]
            ids = [b["chunk_id"] for b in batch]
            
            # Vectorize Batch (Dense + Sparse)
            dense_vecs, sparse_vecs = engine.encode_batch(texts, is_query=False)
            
            # Create Pinecone Payload
            to_upsert = []
            for j, item in enumerate(batch):
                to_upsert.append({
                    "id": ids[j],
                    "values": dense_vecs[j],
                    "sparse_values": sparse_vecs[j],
                    "metadata": {
                        "text": item["text"],
                        "source": item.get("source_citation"),
                        "doc_id": item.get("doc_id")
                    }
                })
            
            # Upload
            try:
                index.upsert(vectors=to_upsert, namespace=ns)
                print(f"  Upserted batch {i // BATCH_SIZE + 1}...")
            except Exception as e:
                print(f"  !!! Error upserting batch: {e}")
                time.sleep(2) # Brief backoff

    print("\n--- Ingestion Complete ---")

if __name__ == "__main__":
    upsert_chunks()