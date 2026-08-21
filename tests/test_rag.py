import pytest
from src.ingestion.chunker import chunk_documents
from src.evaluation.metrics import calculate_retrieval_precision, calculate_grounding_score


def test_chunk_documents():
    docs = [{"content": "Test content " * 100, "metadata": {"source": "test.md"}}]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1


def test_retrieval_precision():
    retrieved = [{"score": 0.9}, {"score": 0.8}, {"score": 0.5}]
    precision = calculate_retrieval_precision(retrieved, relevant_threshold=0.7)
    assert precision == 2 / 3


def test_grounding_score():
    context = [{"document": {"content": "the cat sat on the mat"}}]
    score = calculate_grounding_score("the cat sat", context)
    assert score > 0
