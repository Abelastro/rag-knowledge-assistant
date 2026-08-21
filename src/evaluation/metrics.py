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
