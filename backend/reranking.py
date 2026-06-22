"""
backend/reranking.py
Reranking algorithms: MMR, BM25, and true Hybrid search via Reciprocal
Rank Fusion (RRF) — combining dense (FAISS) and sparse (BM25) retrieval.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Tuple

#MMR — Maximal Marginal Relevance

def mmr_rerank(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    candidate_indices: List[int],
    lambda_score: float = 0.5,
    top_n: int = 5
) -> List[int]:
    """
    Rerank candidate documents using Maximal Marginal Relevance.
    Balances relevance to query vs diversity among selected docs.
    lambda_score=1.0 -> pure relevance, lambda_score=0.0 -> pure diversity.
    """
    if not candidate_indices:
        return []

    query_emb = np.array(query_embedding).reshape(1, -1)
    candidate_embs = np.array([doc_embeddings[i] for i in candidate_indices])

    sim_to_query = cosine_similarity(candidate_embs, query_emb).flatten()
    sim_between = cosine_similarity(candidate_embs)

    selected_local = []
    remaining = list(range(len(candidate_indices)))

    while len(selected_local) < min(top_n, len(candidate_indices)):
        if not selected_local:
            idx = int(np.argmax(sim_to_query))
        else:
            best_score = float("-inf")
            idx = remaining[0]
            for i in remaining:
                max_sim_to_selected = max(sim_between[i][j] for j in selected_local)
                mmr_score = (
                    lambda_score * sim_to_query[i]
                    - (1 - lambda_score) * max_sim_to_selected
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    idx = i
        selected_local.append(idx)
        remaining.remove(idx)

    return [candidate_indices[i] for i in selected_local]


#BM25 — Keyword-based Sparse Retrieval

def bm25_rerank(
    query: str,
    paper_texts: List[str],
    top_n: int = 5
) -> List[Tuple[int, float]]:
    """
    Rank documents using BM25 keyword scoring (the same family of
    algorithm used by Elasticsearch). Captures exact keyword matches
    that semantic search sometimes misses.
    Returns list of (index, score) tuples sorted by relevance.
    """
    tokenized_corpus = [text.lower().split() for text in paper_texts]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(int(idx), float(scores[idx])) for idx in top_indices]


#Hybrid Search — Reciprocal Rank Fusion (Dense + Sparse)

def reciprocal_rank_fusion(
    rankings: List[List[int]],
    k: int = 60,
    top_n: int = 5
) -> List[int]:
    """
    Fuse multiple ranked lists of document indices into one ranking
    using Reciprocal Rank Fusion (RRF).

    Combines rankings by RANK POSITION, not raw score — this avoids
    the scale-mismatch problem between FAISS cosine similarity (0-1)
    and BM25 scores (unbounded). A document ranking decently in
    multiple lists can outscore one that's #1 in only one list.

    k=60 is the standard constant from the original RRF paper
    (Cormack et al., 2009).
    """
    fused_scores = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)

    sorted_docs = sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in sorted_docs[:top_n]]


def hybrid_search(
    query: str,
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    paper_texts: List[str],
    dense_top_k: int = 30,
    top_n: int = 5
) -> List[int]:
    """
    True hybrid dense + sparse retrieval.

    Runs FAISS-equivalent dense cosine similarity AND BM25 keyword
    scoring independently over the same candidate pool, then fuses
    both rankings with RRF. This is the standard "hybrid search"
    pattern used in production RAG systems (Elasticsearch, Weaviate,
    LlamaIndex) — it catches semantic matches BM25 would miss
    (synonyms, paraphrasing) AND exact keyword matches embeddings
    sometimes miss (acronyms, rare technical terms, model names).

    Note: this computes cosine similarity directly via sklearn rather
    than calling the FAISS index object, since reranking.py is kept
    independent of embeddings.py (no circular dependency). The result
    is mathematically identical to FAISS's IndexFlatIP search.
    """
    sim_scores = cosine_similarity(
        np.array(doc_embeddings), np.array(query_embedding).reshape(1, -1)
    ).flatten()
    dense_ranking = list(np.argsort(sim_scores)[::-1][:dense_top_k])

    bm25_result = bm25_rerank(query, paper_texts, top_n=dense_top_k)
    sparse_ranking = [idx for idx, _ in bm25_result]

    return reciprocal_rank_fusion([dense_ranking, sparse_ranking], top_n=top_n)


#Sort by Date

def sort_papers_by_date(papers: List[dict], order: str = "Latest") -> List[dict]:
    """Sort papers list by published date. order: 'Latest', 'Oldest', or 'Relevance'."""
    if order == "Latest":
        return sorted(papers, key=lambda x: x["published"], reverse=True)
    elif order == "Oldest":
        return sorted(papers, key=lambda x: x["published"])
    return papers