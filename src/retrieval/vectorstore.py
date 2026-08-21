import numpy as np
from typing import List, Optional
from ..ingestion.embeddings import generate_embeddings


class VectorStore:
    def __init__(self):
        self.documents: List[dict] = []
        self.embeddings: Optional[np.ndarray] = None

    async def add_documents(self, documents: List[dict]):
        texts = [doc["content"] for doc in documents]
        new_embeddings = await generate_embeddings(texts)

        if self.embeddings is None:
            self.embeddings = np.array(new_embeddings)
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        self.documents.extend(documents)

    async def search(self, query: str, top_k: int = 3) -> List[dict]:
        if self.embeddings is None or len(self.documents) == 0:
            return []

        query_embedding = await generate_embeddings([query])
        query_vec = np.array(query_embedding[0])

        similarities = np.dot(self.embeddings, query_vec) / (
            np.linalg.norm(self.embeddings, axis=1) * np.linalg.norm(query_vec)
        )

        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                "document": self.documents[idx],
                "score": float(similarities[idx])
            })

        return results
