import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient


@pytest.fixture
def mock_deps():
    with patch("src.api.app.vectorstore") as mock_vs, \
         patch("src.api.app.retriever") as mock_ret, \
         patch("src.api.app.llm") as mock_llm:
        mock_ret.retrieve = AsyncMock(return_value=[
            {"document": {"content": "test content", "metadata": {"source": "test.md"}}, "score": 0.9}
        ])
        mock_llm.generate = AsyncMock(return_value="Test answer based on context.")
        mock_vs.search = AsyncMock(return_value=[])
        mock_vs.add_documents = AsyncMock()
        mock_vs.documents = []
        mock_vs.embeddings = None
        yield mock_vs, mock_ret, mock_llm


@pytest.fixture
def client(mock_deps):
    from src.api.app import app
    return TestClient(app)


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_query_endpoint(client, mock_deps):
    mock_vs, mock_ret, mock_llm = mock_deps
    mock_ret.retrieve = AsyncMock(return_value=[
        {"document": {"content": "context text", "metadata": {"source": "doc.md"}}, "score": 0.85}
    ])
    mock_llm.generate = AsyncMock(return_value="The answer is context text.")

    response = client.post("/query", json={"query": "what is context", "top_k": 1})
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
    assert "retrieval_precision" in data
    assert "grounding_score" in data


def test_ingest_endpoint(client, mock_deps):
    mock_vs, mock_ret, mock_llm = mock_deps
    response = client.post(
        "/ingest",
        files={"file": ("test.txt", b"Hello world content", "text/plain")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["documents_added"] == 1
    assert data["chunks_created"] >= 1


def test_documents_endpoint(client, mock_deps):
    response = client.get("/documents")
    assert response.status_code == 200
    data = response.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)
