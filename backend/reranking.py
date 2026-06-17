"""
Reranking algorithms: MMR, Hybrid(FAISS+MMR), and BM25.
Implements dense-sparsed hybrid retrival for production-grade RAG.
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rank_bm25 import BM25Okapi
from typing import List, Tuple

#MMR - Maximal Marginal Relevance

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
    lambda_score=1.0 → pure relevance
    lambda_score=0.0 → pure diversity

    Args:
        query_embedding: 1D vector of the query
        doc_embeddings: 2D array of all document embeddings
        candidate_indices: indices of FAISS-retrieved candidates to rerank
        lambda_score: balance between relevance and diversity
        top_n: number of documents to return

    Returns:
        List of selected document indices in ranked order
    """
    if not candidate_indices:
        return []

    query_emb = np.array(query_embedding).reshape(1, -1)
    candidate_embs = np.array([doc_embeddings[i] for i in candidate_indices])

    # Similarity of each candidate to the query
    sim_to_query = cosine_similarity(candidate_embs, query_emb).flatten()

    # Pairwise similarity between candidates
    sim_between = cosine_similarity(candidate_embs)

    selected_local = []   # indices within candidate_indices
    remaining = list(range(len(candidate_indices)))

    while len(selected_local) < min(top_n, len(candidate_indices)):
        if not selected_local:
            # First pick: most relevant to query
            idx = int(np.argmax(sim_to_query))
        else:
            best_score = float("-inf")
            idx = remaining[0]
            for i in remaining:
                max_sim_to_selected = max(
                    sim_between[i][j] for j in selected_local
                )
                mmr_score = (
                    lambda_score * sim_to_query[i]
                    - (1 - lambda_score) * max_sim_to_selected
                )
                if mmr_score > best_score:
                    best_score = mmr_score
                    idx = i

        selected_local.append(idx)
        remaining.remove(idx)

    # Map local indices back to original document indices
    return [candidate_indices[i] for i in selected_local]

# Hybrid Re-Ranking (FAISS candidates + MMR diversification) 

def hybrid_rerank(
    query_embedding: np.ndarray,
    doc_embeddings: np.ndarray,
    faiss_indices: List[int],
    lambda_score: float = 0.5,
    top_n: int = 5
) -> List[int]:
    """
    Hybrid dense-sparse style reranking.
    Uses FAISS top-k as candidate pool, then applies MMR for diversity.

    This mirrors production RAG pipelines where a fast retriever
    (FAISS) feeds a slower but smarter reranker (MMR).
    """
    return mmr_rerank(
        query_embedding=query_embedding,
        doc_embeddings=doc_embeddings,
        candidate_indices=faiss_indices,
        lambda_score=lambda_score,
        top_n=top_n
    )

# BM25 - Keyword-based sparse retrieval

def bm25_rerank(
    query: str,
    paper_texts: List[str],
    top_n: int = 5
) -> List[Tuple[int, float]]:
    """
    Rank documents using BM25 (Best Match 25) keyword scoring.

    BM25 is a classical TF-IDF-style algorithm used by Elasticsearch
    and major search engines. Captures exact keyword matches that
    semantic search sometimes misses.

    Returns list of (index, score) tuples sorted by relevance.
    """
    # Tokenize: lowercase and split on whitespace
    tokenized_corpus = [text.lower().split() for text in paper_texts]
    tokenized_query = query.lower().split()

    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_query)

    # Get top_n indices sorted by score descending
    top_indices = np.argsort(scores)[::-1][:top_n]
    return [(int(idx), float(scores[idx])) for idx in top_indices]

# Sort by date

def sort_papers_by_date(
    papers: List[dict],
    order: str = "Latest"
) -> List[dict]:
    """
    Sort papers list by published date.
    order: 'Latest', 'Oldest', or 'Relevance' (no sort)
    """
    if order == "Latest":
        return sorted(papers, key=lambda x: x["published"], reverse=True)
    elif order == "Oldest":
        return sorted(papers, key=lambda x: x["published"])
    return papers
