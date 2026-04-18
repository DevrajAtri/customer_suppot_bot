# import os
# import logging
# import concurrent.futures
# from typing import Dict, Any, List, Set
# from pinecone import Pinecone
# # We import CrossEncoder for the Re-ranking step
# from sentence_transformers import CrossEncoder 

# # --- IMPORT SETUP ---
# # Adjust these imports based on your actual folder structure
# try:
#     from servicess.vector_engine import VectorEngine
#     from graph_nodes.state import AgentState
# except ImportError:
#     # Fallback for local testing
#     from servicess.vector_engine import VectorEngine
#     from state import AgentState

# # Setup Logger
# logger = logging.getLogger(__name__)

# # --- CONFIGURATION ---
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
# INDEX_NAME = "ecom-bot"
# ALL_NAMESPACES = ["faq", "privacy", "terms", "grievance"]
# DEFAULT_ALPHA = 0.5 

# # --- SINGLETON RESOURCES ---
# _pc = None
# _index = None
# _engine = None
# _cross_encoder = None

# def get_resources():
#     """
#     Lazy loader for Pinecone, VectorEngine, and CrossEncoder.
#     """
#     global _pc, _index, _engine, _cross_encoder
    
#     # 1. Connect to Pinecone
#     if _index is None:
#         if not PINECONE_API_KEY:
#             raise ValueError("PINECONE_API_KEY not found in environment variables.")
#         _pc = Pinecone(api_key=PINECONE_API_KEY)
#         _index = _pc.Index(INDEX_NAME)
#         logger.info(f"Connected to Pinecone Index: {INDEX_NAME}")

#     # 2. Load Vector Engine (Bi-Encoder + Sparse)
#     if _engine is None:
#         logger.info("Loading VectorEngine...")
#         _engine = VectorEngine()

#     # 3. Load Cross Encoder (The Re-ranker)
#     # This model is specifically trained to score (Query, Document) pairs
#     if _cross_encoder is None:
#         logger.info("Loading CrossEncoder (ms-marco-MiniLM-L-6-v2)...")
#         _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

#     return _index, _engine, _cross_encoder

# # =========================================================================
# # HELPER: PARALLEL SEARCH
# # =========================================================================

# def search_namespace(index, namespace: str, dense_vec: list, sparse_vec: dict, alpha: float):
#     """
#     Executes a single hybrid query against a specific namespace.
#     """
#     try:
#         # Prepare Hybrid Vectors
#         weighted_dense = [v * alpha for v in dense_vec]
#         weighted_sparse = {
#             "indices": sparse_vec["indices"],
#             "values": [v * (1.0 - alpha) for v in sparse_vec["values"]]
#         }

#         results = index.query(
#             namespace=namespace,
#             vector=weighted_dense,
#             sparse_vector=weighted_sparse,
#             top_k=5, # Fetch top 5 from EACH namespace (we will filter later)
#             include_metadata=True
#         )
#         return results["matches"]
    
#     except Exception as e:
#         logger.warning(f"Search failed for namespace '{namespace}': {e}")
#         return []

# # =========================================================================
# # MAIN NODE
# # =========================================================================

# def retriever_node(state: AgentState) -> Dict[str, Any]:
#     """
#     The Main Logic:
#     1. Embed 'current_task_query'.
#     2. Hybrid Search in Pinecone.
#     3. Re-rank results using CrossEncoder.
#     4. Deduplicate against existing state chunks.
#     5. Append ONLY new, relevant chunks.
#     """
#     logger.info("--- KNOWLEDGE RETRIEVER: Running ---")
    
#     # Use the REFINED query (Standalone), not the chat history
#     query = state.get("current_task_query")
#     if not query:
#         logger.warning("No current_task_query found. Skipping retrieval.")
#         return {"retrieval_count": state.get("retrieval_count", 0)}

#     try:
#         index, engine, cross_encoder = get_resources()

#         # 1. Encode the Query
#         vectors = engine.encode(query)
#         dense_vec = vectors["dense"]
#         sparse_vec = vectors["sparse"]

#         # 2. Parallel Search
#         # We search ALL namespaces to ensure we don't miss anything.
#         all_matches = []
#         with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
#             futures = [
#                 executor.submit(search_namespace, index, ns, dense_vec, sparse_vec, DEFAULT_ALPHA)
#                 for ns in ALL_NAMESPACES
#             ]
#             for future in concurrent.futures.as_completed(futures):
#                 all_matches.extend(future.result())

#         # 3. Deduplicate by ID (Before Re-ranking)
#         # We don't want to re-rank the same doc twice if it appeared in multiple searches
#         unique_candidates = {}
#         for match in all_matches:
#             if match['id'] not in unique_candidates:
#                 unique_candidates[match['id']] = match
        
#         candidates = list(unique_candidates.values())

#         if not candidates:
#             return {"retrieval_count": state["retrieval_count"] + 1}

#         # 4. Cross-Encoder Re-ranking
#         # Prepare pairs: [(Query, Doc1), (Query, Doc2)...]
#         pairs = []
#         for match in candidates:
#             text = match['metadata'].get('text', '')
#             pairs.append([query, text])
        
#         # Get scores (higher is better)
#         scores = cross_encoder.predict(pairs)

#         # Attach scores to matches
#         for i, match in enumerate(candidates):
#             match['re_rank_score'] = scores[i]

#         # Sort by Cross-Encoder Score (Descending)
#         candidates.sort(key=lambda x: x['re_rank_score'], reverse=True)

#         # Take Top 8 (as per your architecture)
#         top_candidates = candidates[:8]

#         # 5. Global Deduplication (Against State)
#         # Check if we already have this text in the state to avoid repeating ourselves
#         existing_chunks_set = set(state.get("retrieved_chunks", []))
#         new_chunks_to_add = []

#         for match in top_candidates:
#             text_content = match['metadata'].get('text', '')
#             # Simple deduplication: Check if the exact string is already in history
#             if text_content and text_content not in existing_chunks_set:
#                 new_chunks_to_add.append(text_content)
        
#         logger.info(f"Retriever found {len(top_candidates)} candidates. {len(new_chunks_to_add)} are new.")

#         # Return updates
#         # LangGraph will automatically APPEND 'retrieved_chunks' because of operator.add in state.py
#         return {
#             "retrieved_chunks": new_chunks_to_add,
#             "retrieval_count": state["retrieval_count"] + 1
#         }

#     except Exception as e:
#         logger.error(f"!!! RETRIEVER ERROR: {e}", exc_info=True)
#         return {"retrieval_count": state.get("retrieval_count", 0) + 1}

import os
import logging
import math
import re
import concurrent.futures
from collections import Counter
from typing import Dict, Any, List

import requests
from pinecone import Pinecone
import cohere

# --- IMPORT SETUP ---
try:
    from graph_nodes.state import AgentState
except ImportError:
    from state import AgentState

logger = logging.getLogger(__name__)

# --- CONFIG ---
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY   = os.getenv("GEMINI_API_KEY")
COHERE_API_KEY   = os.getenv("COHERE_API_KEY")

INDEX_NAME = "ecom-bot"
ALL_NAMESPACES = ["faq", "privacy", "terms", "grievance"]

# 👉 Set to 1.0 unless you reindexed sparse properly
DEFAULT_ALPHA = 1.0  

GEMINI_EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent"

# --- SINGLETONS ---
_pc = None
_index = None
_co = None


# =========================================================================
# INIT
# =========================================================================

def get_resources():
    global _pc, _index, _co

    if not PINECONE_API_KEY:
        raise ValueError("Missing PINECONE_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("Missing GEMINI_API_KEY")
    if not COHERE_API_KEY:
        raise ValueError("Missing COHERE_API_KEY")

    if _index is None:
        _pc = Pinecone(api_key=PINECONE_API_KEY)
        _index = _pc.Index(INDEX_NAME)
        logger.info("Pinecone connected")

    if _co is None:
        _co = cohere.Client(COHERE_API_KEY)
        logger.info("Cohere initialized")

    return _index, _co


# =========================================================================
# EMBEDDING (Gemini)
# =========================================================================

def gemini_embed(text: str) -> List[float]:
    payload = {
        "model": "models/text-embedding-004",
        "content": {"parts": [{"text": text}]},
        "taskType": "RETRIEVAL_QUERY"
    }

    resp = requests.post(
        GEMINI_EMBED_URL,
        params={"key": GEMINI_API_KEY},
        json=payload,
        timeout=15
    )
    resp.raise_for_status()

    return resp.json()["embedding"]["values"]


# =========================================================================
# SPARSE VECTOR (optional fallback)
# =========================================================================

def _tokenize(text: str):
    return re.findall(r'\b\w+\b', text.lower())


def build_sparse_vector(text: str):
    tokens = _tokenize(text)
    if not tokens:
        return {"indices": [0], "values": [0.0]}

    counts = Counter(tokens)
    total = len(tokens)

    indices, values = [], []

    for token, count in counts.items():
        tf = count / total
        weight = tf * math.log(1 + 1 / tf)
        idx = hash(token) % (2 ** 16)

        indices.append(idx)
        values.append(weight)

    norm = math.sqrt(sum(v * v for v in values)) or 1.0
    values = [v / norm for v in values]

    return {"indices": indices, "values": values}


# =========================================================================
# SEARCH
# =========================================================================

def search_namespace(index, namespace, dense_vec, sparse_vec, alpha):
    try:
        weighted_dense = [v * alpha for v in dense_vec]

        weighted_sparse = {
            "indices": sparse_vec["indices"],
            "values": [v * (1 - alpha) for v in sparse_vec["values"]]
        }

        results = index.query(
            namespace=namespace,
            vector=weighted_dense,
            sparse_vector=weighted_sparse,
            top_k=5,
            include_metadata=True
        )

        return results["matches"]

    except Exception as e:
        logger.warning(f"Search failed for {namespace}: {e}")
        return []


# =========================================================================
# COHERE RERANK (REAL CROSS-ENCODER)
# =========================================================================

def cohere_rerank(co, query: str, candidates: List[dict]) -> List[dict]:
    docs = [match["metadata"].get("text", "") for match in candidates]

    if not docs:
        return candidates

    try:
        response = co.rerank(
            model="rerank-english-v3.0",
            query=query,
            documents=docs,
            top_n=len(docs)
        )

        # Map scores back
        for r in response.results:
            idx = r.index
            candidates[idx]["re_rank_score"] = r.relevance_score

        candidates.sort(key=lambda x: x["re_rank_score"], reverse=True)
        return candidates

    except Exception as e:
        logger.warning(f"Cohere rerank failed: {e}")
        return candidates  # fallback


# =========================================================================
# MAIN NODE
# =========================================================================

def retriever_node(state: AgentState) -> Dict[str, Any]:
    logger.info("--- RETRIEVER RUNNING ---")

    query = state.get("current_task_query")
    if not query:
        return {"retrieval_count": state.get("retrieval_count", 0)}

    try:
        index, co = get_resources()

        # 1. EMBEDDING
        dense_vec = gemini_embed(query)

        # ⚠️ Sparse only if your index supports it
        sparse_vec = {"indices": [], "values": []}

        # 2. SEARCH
        all_matches = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [
                executor.submit(search_namespace, index, ns, dense_vec, sparse_vec, DEFAULT_ALPHA)
                for ns in ALL_NAMESPACES
            ]
            for future in concurrent.futures.as_completed(futures):
                all_matches.extend(future.result())

        # 3. DEDUP
        unique = {}
        for m in all_matches:
            if m["id"] not in unique:
                unique[m["id"]] = m

        candidates = list(unique.values())

        if not candidates:
            return {"retrieval_count": state.get("retrieval_count", 0) + 1}

        # 🔥 Limit before rerank (important for cost)
        candidates = candidates[:15]

        # 4. RERANK
        candidates = cohere_rerank(co, query, candidates)

        top_candidates = candidates[:8]

        # 5. FINAL DEDUP (state)
        existing = set(state.get("retrieved_chunks", []))

        new_chunks = []
        for match in top_candidates:
            text = match["metadata"].get("text", "")
            if text and text not in existing:
                new_chunks.append(text)

        logger.info(f"Added {len(new_chunks)} new chunks")

        return {
            "retrieved_chunks": new_chunks,
            "retrieval_count": state.get("retrieval_count", 0) + 1
        }

    except Exception as e:
        logger.error(f"Retriever error: {e}", exc_info=True)
        return {"retrieval_count": state.get("retrieval_count", 0) + 1}