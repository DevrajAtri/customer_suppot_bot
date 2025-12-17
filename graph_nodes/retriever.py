import os
import logging
import concurrent.futures
from typing import Dict, Any, List, Set
from pinecone import Pinecone
# We import CrossEncoder for the Re-ranking step
from sentence_transformers import CrossEncoder 

# --- IMPORT SETUP ---
# Adjust these imports based on your actual folder structure
try:
    from servicess.vector_engine import VectorEngine
    from graph_nodes.state import AgentState
except ImportError:
    # Fallback for local testing
    from servicess.vector_engine import VectorEngine
    from state import AgentState

# Setup Logger
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "ecom-bot"
ALL_NAMESPACES = ["faq", "privacy", "terms", "grievance"]
DEFAULT_ALPHA = 0.5 

# --- SINGLETON RESOURCES ---
_pc = None
_index = None
_engine = None
_cross_encoder = None

def get_resources():
    """
    Lazy loader for Pinecone, VectorEngine, and CrossEncoder.
    """
    global _pc, _index, _engine, _cross_encoder
    
    # 1. Connect to Pinecone
    if _index is None:
        if not PINECONE_API_KEY:
            raise ValueError("PINECONE_API_KEY not found in environment variables.")
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(INDEX_NAME)
        logger.info(f"Connected to Pinecone Index: {INDEX_NAME}")

    # 2. Load Vector Engine (Bi-Encoder + Sparse)
    if _engine is None:
        logger.info("Loading VectorEngine...")
        _engine = VectorEngine()

    # 3. Load Cross Encoder (The Re-ranker)
    # This model is specifically trained to score (Query, Document) pairs
    if _cross_encoder is None:
        logger.info("Loading CrossEncoder (ms-marco-MiniLM-L-6-v2)...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

    return _index, _engine, _cross_encoder

# =========================================================================
# HELPER: PARALLEL SEARCH
# =========================================================================

def search_namespace(index, namespace: str, dense_vec: list, sparse_vec: dict, alpha: float):
    """
    Executes a single hybrid query against a specific namespace.
    """
    try:
        # Prepare Hybrid Vectors
        weighted_dense = [v * alpha for v in dense_vec]
        weighted_sparse = {
            "indices": sparse_vec["indices"],
            "values": [v * (1.0 - alpha) for v in sparse_vec["values"]]
        }

        results = index.query(
            namespace=namespace,
            vector=weighted_dense,
            sparse_vector=weighted_sparse,
            top_k=5, # Fetch top 5 from EACH namespace (we will filter later)
            include_metadata=True
        )
        return results["matches"]
    
    except Exception as e:
        logger.warning(f"Search failed for namespace '{namespace}': {e}")
        return []

# =========================================================================
# MAIN NODE
# =========================================================================

def retriever_node(state: AgentState) -> Dict[str, Any]:
    """
    The Main Logic:
    1. Embed 'current_task_query'.
    2. Hybrid Search in Pinecone.
    3. Re-rank results using CrossEncoder.
    4. Deduplicate against existing state chunks.
    5. Append ONLY new, relevant chunks.
    """
    logger.info("--- KNOWLEDGE RETRIEVER: Running ---")
    
    # Use the REFINED query (Standalone), not the chat history
    query = state.get("current_task_query")
    if not query:
        logger.warning("No current_task_query found. Skipping retrieval.")
        return {"retrieval_count": state.get("retrieval_count", 0)}

    try:
        index, engine, cross_encoder = get_resources()

        # 1. Encode the Query
        vectors = engine.encode(query)
        dense_vec = vectors["dense"]
        sparse_vec = vectors["sparse"]

        # 2. Parallel Search
        # We search ALL namespaces to ensure we don't miss anything.
        all_matches = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(search_namespace, index, ns, dense_vec, sparse_vec, DEFAULT_ALPHA)
                for ns in ALL_NAMESPACES
            ]
            for future in concurrent.futures.as_completed(futures):
                all_matches.extend(future.result())

        # 3. Deduplicate by ID (Before Re-ranking)
        # We don't want to re-rank the same doc twice if it appeared in multiple searches
        unique_candidates = {}
        for match in all_matches:
            if match['id'] not in unique_candidates:
                unique_candidates[match['id']] = match
        
        candidates = list(unique_candidates.values())

        if not candidates:
            return {"retrieval_count": state["retrieval_count"] + 1}

        # 4. Cross-Encoder Re-ranking
        # Prepare pairs: [(Query, Doc1), (Query, Doc2)...]
        pairs = []
        for match in candidates:
            text = match['metadata'].get('text', '')
            pairs.append([query, text])
        
        # Get scores (higher is better)
        scores = cross_encoder.predict(pairs)

        # Attach scores to matches
        for i, match in enumerate(candidates):
            match['re_rank_score'] = scores[i]

        # Sort by Cross-Encoder Score (Descending)
        candidates.sort(key=lambda x: x['re_rank_score'], reverse=True)

        # Take Top 8 (as per your architecture)
        top_candidates = candidates[:8]

        # 5. Global Deduplication (Against State)
        # Check if we already have this text in the state to avoid repeating ourselves
        existing_chunks_set = set(state.get("retrieved_chunks", []))
        new_chunks_to_add = []

        for match in top_candidates:
            text_content = match['metadata'].get('text', '')
            # Simple deduplication: Check if the exact string is already in history
            if text_content and text_content not in existing_chunks_set:
                new_chunks_to_add.append(text_content)
        
        logger.info(f"Retriever found {len(top_candidates)} candidates. {len(new_chunks_to_add)} are new.")

        # Return updates
        # LangGraph will automatically APPEND 'retrieved_chunks' because of operator.add in state.py
        return {
            "retrieved_chunks": new_chunks_to_add,
            "retrieval_count": state["retrieval_count"] + 1
        }

    except Exception as e:
        logger.error(f"!!! RETRIEVER ERROR: {e}", exc_info=True)
        return {"retrieval_count": state.get("retrieval_count", 0) + 1}