from typing import List


def chunk_documents(
    documents: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[dict]:
    chunks = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]

        for i in range(0, len(content), chunk_size - chunk_overlap):
            chunk_text = content[i:i + chunk_size]
            chunks.append({
                "content": chunk_text,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "start_char": i,
                    "end_char": min(i + chunk_size, len(content))
                }
            })

    return chunks
