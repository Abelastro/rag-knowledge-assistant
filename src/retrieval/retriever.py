from typing import List
from .vectorstore import VectorStore


class Retriever:
    def __init__(self, vectorstore: VectorStore):
        self.vectorstore = vectorstore

    async def retrieve(self, query: str, top_k: int = 3) -> List[dict]:
        results = await self.vectorstore.search(query, top_k=top_k)
        return results
