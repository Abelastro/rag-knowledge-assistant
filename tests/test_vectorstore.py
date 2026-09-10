import pytest
import numpy as np
from unittest.mock import AsyncMock, patch
from src.retrieval.vectorstore import VectorStore


@pytest.fixture
def mock_embeddings():
    async def fake_embeddings(texts, model="text-embedding-ada-002"):
        np.random.seed(42)
        return [np.random.rand(8).tolist() for _ in texts]
    return fake_embeddings


@pytest.mark.asyncio
async def test_add_documents(mock_embeddings):
    with patch("src.retrieval.vectorstore.generate_embeddings", mock_embeddings):
        store = VectorStore()
        docs = [
            {"content": "hello world", "metadata": {"source": "a.md"}},
            {"content": "foo bar", "metadata": {"source": "b.md"}},
        ]
        await store.add_documents(docs)
        assert len(store.documents) == 2
        assert store.embeddings.shape == (2, 8)


@pytest.mark.asyncio
async def test_search_returns_top_k(mock_embeddings):
    with patch("src.retrieval.vectorstore.generate_embeddings", mock_embeddings):
        store = VectorStore()
        docs = [
            {"content": f"document number {i}", "metadata": {"source": f"{i}.md"}}
            for i in range(10)
        ]
        await store.add_documents(docs)
        results = await store.search("document", top_k=3)
        assert len(results) == 3
        assert all("score" in r for r in results)
        assert all("document" in r for r in results)


@pytest.mark.asyncio
async def test_search_empty_store():
    store = VectorStore()
    results = await store.search("query", top_k=3)
    assert results == []


@pytest.mark.asyncio
async def test_cosine_similarity_calculation():
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([1.0, 0.0, 0.0])
    cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    assert abs(cos_sim - 1.0) < 1e-6

    c = np.array([0.0, 1.0, 0.0])
    cos_sim_orth = np.dot(a, c) / (np.linalg.norm(a) * np.linalg.norm(c))
    assert abs(cos_sim_orth) < 1e-6
