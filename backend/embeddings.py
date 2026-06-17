"""
backend/embeddings.py
Converts paper text into semantic vectors and builds FAISS index.
Uses cosine similarity throughout (IndexFlatIP + normalized vectors).
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import faiss
faiss.omp_set_num_threads(1)

from sentence_transformers import SentenceTransformer
from typing import List, Tuple


#Model name as a constant - easy to upgrade later
EMBEDDING_MODEL = "all-mpnet-base-v2"

def load_embedding_model() -> SentenceTransformer:
    """Load and return the sentence embedding model"""
    return SentenceTransformer(EMBEDDING_MODEL)

def build_paper_texts(papers: List[dict]) -> List[str]:
    """
    Combine paper field into single rich text string for embedding.
    Including title,authors,abstract gives better semantic coverage.
    """
    texts = []
    for paper in papers:
        text = (
            f"Title: {paper['title']}\n"
            f"Authors: {', '.join(paper['authors'])}\n"
            f"Published: {paper['published']}\n"
            f"Abstract: {paper['abstract']}"
        )
        texts.append(text)
    return texts

def embed_texts(texts: List[str], model: SentenceTransformer) -> np.ndarray:
    """
    Encode a list of texts into normalized embeddings.
    Normalization ensures cosine similarity = inner product,
    which is required for IndexFlatIP to work correctly.
    """

    embeddings = model.encode(texts, convert_to_tensor=False, show_progress_bar=False)
    embeddings = np.array(embeddings).astype("float32")

    #Normalize each vector to unit length
    faiss.normalize_L2(embeddings)
    return embeddings

def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Build a faiss index using Inner Product (= cosine sim on normalized vectors).
    Higher Score = More Similar (opposite of L2 where lower = more similar).
    """
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    return index

def search_index(
        query: str, 
        model: SentenceTransformer, 
        index: faiss.IndexFlatIP, 
        top_k: int = 20
        ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Embed the query and search the FAISS index.
    Returns (scores, indices) - scores are cosine similarities.
    """
    query_embedding = model.encode(query, convert_to_tensor=False)
    query_embedding = np.array(query_embedding).astype("float32").reshape(1, -1)
    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, k=top_k)
    return scores[0], indices[0]

