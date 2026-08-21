from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
from ..retrieval.vectorstore import VectorStore
from ..retrieval.retriever import Retriever
from ..generation.llm import LLM
from ..evaluation.metrics import calculate_retrieval_precision, calculate_grounding_score


app = FastAPI(title="RAG Knowledge Assistant", version="0.1.0")

vectorstore = VectorStore()
retriever = Retriever(vectorstore)
llm = LLM()


class QueryRequest(BaseModel):
    query: str
    top_k: int = 3


class QueryResponse(BaseModel):
    answer: str
    sources: List[dict]
    retrieval_precision: float
    grounding_score: float


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
