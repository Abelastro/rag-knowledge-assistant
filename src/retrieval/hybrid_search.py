import numpy as np
from typing import List


def _compute_tf(words: List[str]) -> dict:
    tf = {}
    for w in words:
        tf[w] = tf.get(w, 0) + 1
    return tf


def _bm25_score(query_words: List[str], doc_words: List[str], avg_dl: float, k1: float = 1.5, b: float = 0.75) -> float:
    doc_tf = _compute_tf(doc_words)
    dl = len(doc_words)
    score = 0.0

    for q in query_words:
        if q in doc_tf:
            tf_val = doc_tf[q]
            numerator = tf_val * (k1 + 1)
            denominator = tf_val + k1 * (1 - b + b * dl / avg_dl)
            score += numerator / denominator

    return score


async def hybrid_search(
    query: str,
    documents: List[dict],
    vectorstore,
    alpha: float = 0.7,
    top_k: int = 5
) -> List[dict]:
    if not documents:
        return []

    vector_results = await vectorstore.search(query, top_k=min(top_k * 2, len(documents)))

    vector_scores = {}
    for r in vector_results:
        source = r["document"]["metadata"]["source"]
        vector_scores[source] = r["score"]

    query_words = query.lower().split()
    all_doc_words = [doc["content"].lower().split() for doc in documents]
    avg_dl = sum(len(dw) for dw in all_doc_words) / len(all_doc_words) if all_doc_words else 1

    bm25_scores = {}
    for doc in documents:
        doc_words = doc["content"].lower().split()
        bm25_scores[doc["metadata"]["source"]] = _bm25_score(query_words, doc_words, avg_dl)

    max_bm25 = max(bm25_scores.values()) if bm25_scores else 1.0
    max_vector = max(vector_scores.values()) if vector_scores else 1.0

    combined = []
    for doc in documents:
        source = doc["metadata"]["source"]
        v_score = vector_scores.get(source, 0.0) / max_vector if max_vector else 0.0
        k_score = bm25_scores.get(source, 0.0) / max_bm25 if max_bm25 else 0.0

        combined_score = alpha * v_score + (1 - alpha) * k_score
        combined.append({
            "document": doc,
            "score": combined_score,
            "vector_score": v_score,
            "keyword_score": k_score
        })

    combined.sort(key=lambda x: x["score"], reverse=True)
    return combined[:top_k]
