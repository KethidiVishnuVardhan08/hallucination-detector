"""
Embedding backend used to compute semantic similarity between the N
self-consistency samples.

Two backends are supported:

  - "sentence-transformers": loads a small local embedding model
    (all-MiniLM-L6-v2 by default). Much better semantic quality —
    this is the backend used in the SelfCheckGPT paper's BERTScore-style
    variant. Requires a one-time model download (~90MB) the first time
    it runs, so it needs internet access on first use only; after that
    it's fully local and free per-call.

  - "tfidf": a pure scikit-learn TF-IDF + cosine similarity fallback.
    No downloads, works fully offline, noticeably lower quality
    (misses paraphrases that share no words), but useful for quick
    testing or restricted environments.

The scorer imports `embed_and_similarity()` and doesn't care which
backend produced the numbers.
"""
import numpy as np
from typing import List, Tuple

from app.config import settings

_model = None  # lazy-loaded singleton


def _get_sentence_transformer():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(settings.embedding_model_name)
    return _model


def _cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vectors, axis=1, keepdims=True)
    norm[norm == 0] = 1e-8
    unit = vectors / norm
    return unit @ unit.T


def embed_and_similarity(texts: List[str]) -> np.ndarray:
    """
    Returns an (N, N) pairwise cosine similarity matrix for the given texts.
    Values are in [-1, 1], typically [0, 1] for natural language.
    """
    if settings.embedding_backend == "tfidf":
        return _tfidf_similarity(texts)
    try:
        model = _get_sentence_transformer()
        vectors = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        return _cosine_similarity_matrix(vectors)
    except Exception as e:
        # Graceful degradation: if the embedding model can't be downloaded
        # (e.g. no internet in this environment), fall back to TF-IDF rather
        # than crashing the whole request.
        print(f"[embeddings] sentence-transformers unavailable ({e}); falling back to TF-IDF.")
        return _tfidf_similarity(texts)


def _tfidf_similarity(texts: List[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    if len(set(texts)) == 1:
        # All identical strings -> perfect similarity, avoid vectorizer edge case.
        n = len(texts)
        return np.ones((n, n))

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(texts)
    return cosine_similarity(matrix)


def mean_off_diagonal(sim_matrix: np.ndarray) -> float:
    """Average pairwise similarity, excluding the diagonal (self-similarity=1)."""
    n = sim_matrix.shape[0]
    if n < 2:
        return 1.0
    off_diag_sum = sim_matrix.sum() - np.trace(sim_matrix)
    count = n * (n - 1)
    return float(off_diag_sum / count)
