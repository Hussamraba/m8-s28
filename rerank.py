"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from sentence_transformers import CrossEncoder
from hybrid import hybrid_search  # helper موجود بالريبو

# 1. Cross-encoder rerank function
def cross_encoder_rerank(query, candidates, k_out=5):
    ce = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    # candidates: list of dicts { "id": ..., "text": ... }
    pairs = [(query, c["text"]) for c in candidates]
    scores = ce.predict(pairs)
    # ربط السكور مع الـ id
    scored = [(c["id"], s) for c, s in zip(candidates, scores)]
    # ترتيب تنازلي
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in ranked[:k_out]]

# 2. Two-stage retriever
def rerank_search(client, query, embedder, k_in=50, k_out=5):
    # Hybrid retrieve
    candidates = hybrid_search(client, query, embedder, k=k_in)
    # Cross-encoder rerank
    return cross_encoder_rerank(query, candidates, k_out=k_out)
