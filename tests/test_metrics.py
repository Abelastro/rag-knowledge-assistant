import pytest
from src.evaluation.metrics import (
    calculate_retrieval_precision,
    calculate_grounding_score,
    calculate_answer_relevancy,
    calculate_context_relevance,
    detect_hallucination,
)


def test_retrieval_precision_perfect():
    retrieved = [{"score": 0.9}, {"score": 0.95}, {"score": 1.0}]
    assert calculate_retrieval_precision(retrieved, relevant_threshold=0.7) == 1.0


def test_retrieval_precision_empty():
    assert calculate_retrieval_precision([]) == 0.0


def test_grounding_score_full():
    context = [{"document": {"content": "the cat sat on the mat"}}]
    score = calculate_grounding_score("the cat sat on the mat", context)
    assert score == 1.0


def test_grounding_score_empty():
    context = [{"document": {"content": ""}}]
    score = calculate_grounding_score("the cat sat", context)
    assert score == 0.0


def test_answer_relevancy():
    score = calculate_answer_relevancy("the quick brown fox", "the quick brown fox jumps")
    assert score >= 0.8


def test_context_relevance():
    chunks = [
        {"content": "python programming language features"},
        {"content": "python is used for data science"},
    ]
    score = calculate_context_relevance("python programming", chunks)
    assert score > 0.0


def test_detect_hallucination_clean():
    context = [{"document": {"content": "the cat sat on the mat"}}]
    result = detect_hallucination("the cat sat on the mat", context)
    assert not result["is_hallucinated"]
    assert result["hallucination_ratio"] < 0.3


def test_detect_hallucination_hallucinated():
    context = [{"document": {"content": "the cat sat"}}]
    result = detect_hallucination("the quantum physics explains teleportation", context)
    assert result["is_hallucinated"]
    assert result["hallucination_ratio"] > 0.3
