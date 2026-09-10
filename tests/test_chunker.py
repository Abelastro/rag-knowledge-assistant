import pytest
from src.ingestion.chunker import chunk_documents, chunk_recursive, chunk_semantic


def test_chunk_documents_fixed_size():
    content = "word " * 200
    docs = [{"content": content, "metadata": {"source": "test.md"}}]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["content"]) <= 100
        assert "source" in chunk["metadata"]
        assert "chunk_index" in chunk["metadata"]


def test_chunk_documents_recursive():
    content = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph with enough words to exceed limit."
    docs = [{"content": content, "metadata": {"source": "test.md"}}]
    chunks = chunk_recursive(docs, chunk_size=50, chunk_overlap=0)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "source" in chunk["metadata"]


def test_chunk_documents_semantic():
    content = "Introduction paragraph about something important.\n\nBody paragraph with more details.\n\nConclusion paragraph."
    docs = [{"content": content, "metadata": {"source": "test.md"}}]
    chunks = chunk_semantic(docs, min_paragraph_size=10, merge_threshold=200)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert "source" in chunk["metadata"]
        assert "content" in chunk


def test_chunk_overlap():
    content = "a" * 200
    docs = [{"content": content, "metadata": {"source": "test.md"}}]
    chunks_no_overlap = chunk_documents(docs, chunk_size=100, chunk_overlap=0)
    chunks_with_overlap = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
    assert len(chunks_with_overlap) >= len(chunks_no_overlap)


def test_empty_documents():
    docs = []
    chunks = chunk_documents(docs)
    assert chunks == []


def test_single_character_document():
    docs = [{"content": "x", "metadata": {"source": "test.md"}}]
    chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=0)
    assert len(chunks) == 1
    assert chunks[0]["content"] == "x"
