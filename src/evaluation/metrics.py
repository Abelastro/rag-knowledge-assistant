from typing import List


def calculate_retrieval_precision(retrieved: List[dict], relevant_threshold: float = 0.7) -> float:
    if not retrieved:
        return 0.0

    relevant = sum(1 for r in retrieved if r.get("score", 0) >= relevant_threshold)
    return relevant / len(retrieved)


def calculate_grounding_score(answer: str, context: List[dict]) -> float:
    context_words = set()
    for r in context:
        context_words.update(r["document"]["content"].lower().split())

    answer_words = set(answer.lower().split())

    if not answer_words:
        return 0.0

    grounded = len(answer_words.intersection(context_words))
    return grounded / len(answer_words)


def calculate_answer_relevancy(answer: str, query: str) -> float:
    query_words = set(query.lower().split())
    answer_words = set(answer.lower().split())

    if not query_words or not answer_words:
        return 0.0

    overlap = len(query_words.intersection(answer_words))
    return overlap / len(query_words)


def calculate_context_relevance(query: str, context_chunks: List[dict]) -> float:
    query_words = set(query.lower().split())

    if not query_words or not context_chunks:
        return 0.0

    total_overlap = 0
    for chunk in context_chunks:
        content = chunk["content"] if "content" in chunk else chunk.get("document", {}).get("content", "")
        doc_words = set(content.lower().split())
        total_overlap += len(query_words.intersection(doc_words))

    avg_overlap = total_overlap / len(context_chunks)
    return min(avg_overlap / len(query_words), 1.0) if query_words else 0.0


def detect_hallucination(answer: str, context: List[dict]) -> dict:
    context_words = set()
    for r in context:
        context_words.update(r["document"]["content"].lower().split())

    answer_words = set(answer.lower().split())

    if not answer_words:
        return {"hallucinated_words": [], "hallucination_ratio": 0.0, "is_hallucinated": False}

    hallucinated = answer_words - context_words
    ratio = len(hallucinated) / len(answer_words)

    return {
        "hallucinated_words": list(hallucinated),
        "hallucination_ratio": ratio,
        "is_hallucinated": ratio > 0.3
    }
