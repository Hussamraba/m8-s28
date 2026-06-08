"""Module 8 — Thursday Stretch (Honors Track): Cross-Encoder Re-Ranking.

Add a cross-encoder re-ranking stage to the lab's hybrid retriever and
evaluate the cost/benefit. Cross-encoders score (query, passage) pairs
jointly rather than independently — they produce a more discriminative
ranking, but at a real latency cost.

Use cross-encoder/ms-marco-MiniLM-L-6-v2 from sentence-transformers.
"""

from __future__ import annotations

import weaviate
from sentence_transformers import CrossEncoder


from retrieval_helpers import hybrid_search

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
_ce = CrossEncoder(CROSS_ENCODER_MODEL)  


def cross_encoder_rerank(query: str, candidates: list[dict], k_out: int = 5) -> list[str]:
    """Re-rank a candidate list using a cross-encoder.

    candidates: list of {"doc_id": str, "text": str}
    """
    pairs = [(query, c["text"]) for c in candidates]
    scores = _ce.predict(pairs)

    scored = [(c["doc_id"], s) for c, s in zip(candidates, scores)]
    ranked = sorted(scored, key=lambda x: x[1], reverse=True)

    return [doc_id for doc_id, _ in ranked[:k_out]]


def rerank_search(
    client: weaviate.Client,
    query: str,
    embedder,
    k_in: int = 50,
    k_out: int = 5,
) -> list[str]:
    """Two-stage retriever: hybrid retrieve k_in, cross-encoder re-rank to k_out."""
   
    doc_ids = hybrid_search(client, query, k_in, embedder)

    
    candidates = []
    for doc_id in doc_ids:
        result = (
            client.query.get("Post", ["doc_id", "text"])
            .with_where({
                "path": "doc_id",
                "operator": "Equal",
                "valueString": doc_id,
            })
            .do()
        )

        if "data" in result and "Get" in result["data"]:
            docs = result["data"]["Get"].get("Post", [])
            if docs:
                candidates.append(docs[0])

    
    return cross_encoder_rerank(query, candidates, k_out=k_out)
