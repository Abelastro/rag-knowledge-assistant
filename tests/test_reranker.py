import pytest
from src.retrieval.reranker import rerank_results


def test_reranker_reorders_by_keyword():
    results = [
        {"document": {"content": "unrelated text here", "metadata": {"source": "a.md"}}, "score": 0.9},
        {"document": {"content": "python programming guide", "metadata": {"source": "b.md"}}, "score": 0.5},
        {"document": {"content": "java basics tutorial", "metadata": {"source": "c.md"}}, "score": 0.7},
    ]
    reranked = rerank_results("python programming", results, keyword_weight=1.0, position_weight=0.0)
    assert reranked[0]["document"]["metadata"]["source"] == "b.md"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]


def test_reranker_position_penalty():
    results = [
        {"document": {"content": "python programming guide", "metadata": {"source": "a.md"}}, "score": 0.9},
        {"document": {"content": "python programming guide", "metadata": {"source": "b.md"}}, "score": 0.8},
    ]
    reranked = rerank_results("python programming", results, keyword_weight=1.0, position_weight=1.0)
    assert reranked[0]["document"]["metadata"]["source"] == "a.md"
    assert reranked[0]["rerank_score"] > reranked[1]["rerank_score"]
