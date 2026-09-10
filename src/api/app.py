from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional
from ..retrieval.vectorstore import VectorStore
from ..retrieval.retriever import Retriever
from ..generation.llm import LLM
from ..evaluation.metrics import calculate_retrieval_precision, calculate_grounding_score
from ..ingestion.loader import load_documents
from ..ingestion.chunker import chunk_documents


app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")

vectorstore = VectorStore()
retriever = Retriever(vectorstore)
llm = LLM()

ingested_documents: List[dict] = []


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieval_precision: float
    grounding_score: float


class IngestResponse(BaseModel):
    message: str
    documents_added: int
    chunks_created: int


class MetricsResponse(BaseModel):
    total_documents: int
    total_chunks: int
    embedding_dimension: Optional[int] = None


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    results = await retriever.retrieve(request.query, top_k=request.top_k)
    answer = await llm.generate(request.query, results)

    return QueryResponse(
        answer=answer,
        sources=[{"source": r["document"]["metadata"]["source"], "score": r["score"]} for r in results],
        retrieval_precision=calculate_retrieval_precision(results),
        grounding_score=calculate_grounding_score(answer, results),
    )


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    content = await file.read()
    text = content.decode("utf-8")

    doc = {
        "content": text,
        "metadata": {"source": file.filename, "filename": file.filename}
    }

    ingested_documents.append(doc)
    chunks = chunk_documents([doc])
    await vectorstore.add_documents(chunks)

    return IngestResponse(
        message=f"Successfully ingested {file.filename}",
        documents_added=1,
        chunks_created=len(chunks)
    )


@app.get("/documents")
async def list_documents():
    return {
        "documents": [
            {"filename": doc["metadata"]["filename"], "source": doc["metadata"]["source"]}
            for doc in ingested_documents
        ]
    }


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics():
    total_chunks = len(vectorstore.documents)
    embedding_dim = vectorstore.embeddings.shape[1] if vectorstore.embeddings is not None else None

    return MetricsResponse(
        total_documents=len(ingested_documents),
        total_chunks=total_chunks,
        embedding_dimension=embedding_dim
    )
