import os
import torch
from sentence_transformers import SentenceTransformer
from pinecone_text.sparse import SpladeEncoder

# Fix for memory fragmentation if using GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

class VectorEngine:
    def __init__(self):
        # 1. Setup Device (GPU/CPU)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"--- VectorEngine: Loading models on {self.device} ---")

        # 2. Dense Model (BGE-M3)
        # We use the ID. The library automatically looks in your local 'models--BAAI--bge-m3' folder.
        self.dense_id = "BAAI/bge-m3"
        self.dense_model = SentenceTransformer(self.dense_id, device=self.device)
        self.dense_model.eval() # Optimization: Set to eval mode

        # 3. Sparse Model (SPLADE)
        # ERROR FIXED: We do NOT pass 'model_name_or_path'. 
        # It defaults to 'naver/splade-cocondenser-ensemble-distilbert', 
        # which matches the folder in your screenshot.
        self.splade_model = SpladeEncoder(
            device=self.device
        )
        print("--- Models Loaded Successfully ---")

    def encode(self, text: str):
        """
        Generates both dense and sparse vectors for a single text.
        Used by the Retriever (Live Bot).
        """
        # Dense
        dense_vec = self.dense_model.encode(text, normalize_embeddings=True)
        
        # Sparse (encode_queries is used for search queries)
        # We MUST use encode_queries for the user's question.
        sparse_vec = self.splade_model.encode_queries(text)
        
        return {
            "dense": dense_vec.tolist(),
            "sparse": sparse_vec
        }

    def encode_batch(self, texts: list, is_query: bool = False):
        """
        Generates vectors for a batch of texts.
        Used by the Upsert Script.
        """
        # Dense
        dense_vecs = self.dense_model.encode(
            texts, 
            normalize_embeddings=True, 
            batch_size=32,
            show_progress_bar=False
        )
        
        # Sparse
        if is_query:
            sparse_vecs = self.splade_model.encode_queries(texts)
        else:
            sparse_vecs = self.splade_model.encode_documents(texts)
            
        # Convert numpy arrays to lists for JSON serialization
        return [v.tolist() for v in dense_vecs], sparse_vecs