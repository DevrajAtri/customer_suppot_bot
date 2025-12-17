import os
from pinecone import Pinecone
from dotenv import load_dotenv

# 1. Load your keys
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ecom-bot" # Ensure this matches your index name

def clear_all_namespaces():
    # 2. Connect
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(INDEX_NAME)
    
    print(f"--- Connecting to Index: {INDEX_NAME} ---")
    
    # 3. Get list of current namespaces
    stats = index.describe_index_stats()
    namespaces = stats.get('namespaces', {}).keys()
    
    if not namespaces:
        print("Index is already empty!")
        return

    # 4. Iterate and Delete
    for ns in namespaces:
        print(f"Deleting all vectors in namespace: '{ns}'...")
        try:
            # delete_all=True is the fastest way to wipe a namespace
            index.delete(delete_all=True, namespace=ns)
            print(f" -> Successfully cleared '{ns}'")
        except Exception as e:
            print(f" -> Error clearing '{ns}': {e}")

    print("\n--- All data cleared. Ready for fresh upsert. ---")

if __name__ == "__main__":
    clear_all_namespaces()