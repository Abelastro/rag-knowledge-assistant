from typing import List


def rerank_results(
    query: str,
    results: List[dict],
    keyword_weight: float = 0.6,
    position_weight: float = 0.4
) -> List[dict]:
    if not results:
        return []

    query_words = set(query.lower().split())
    scored = []

    for idx, result in enumerate(results):
        doc_words = set(result["document"]["content"].lower().split())
        overlap = len(query_words.intersection(doc_words))
        keyword_score = overlap / len(query_words) if query_words else 0.0

        position_penalty = idx / len(results)
        combined = (keyword_weight * keyword_score) - (position_weight * position_penalty)

        scored.append({
            **result,
            "rerank_score": combined,
            "keyword_score": keyword_score,
            "original_rank": idx
        })

    scored.sort(key=lambda x: x["rerank_score"], reverse=True)
    return scored
