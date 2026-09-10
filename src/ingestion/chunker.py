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


def chunk_recursive(
    documents: List[dict],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    separators: List[str] = None
) -> List[dict]:
    if separators is None:
        separators = ["\n\n", "\n", ". ", " "]

    chunks = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        pieces = _recursive_split(content, separators, chunk_size)

        for piece in pieces:
            if not piece.strip():
                continue
            chunks.append({
                "content": piece,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "start_char": content.find(piece),
                    "end_char": content.find(piece) + len(piece)
                }
            })

    return chunks


def _recursive_split(text: str, separators: List[str], chunk_size: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else ""
    remaining_separators = separators[1:] if len(separators) > 1 else []

    if sep:
        parts = text.split(sep)
    else:
        parts = [text]

    result = []
    current = ""

    for part in parts:
        candidate = (current + sep + part) if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                result.append(current)
            if len(part) > chunk_size and remaining_separators:
                result.extend(_recursive_split(part, remaining_separators, chunk_size))
            else:
                current = part

    if current:
        result.append(current)

    return result


def chunk_semantic(
    documents: List[dict],
    min_paragraph_size: int = 100,
    merge_threshold: int = 200
) -> List[dict]:
    chunks = []

    for doc in documents:
        content = doc["content"]
        metadata = doc["metadata"]
        paragraphs = content.split("\n\n")

        merged = []
        buffer = ""

        for para in paragraphs:
            if buffer and len(buffer) + len(para) + 2 > merge_threshold:
                merged.append(buffer)
                buffer = para
            elif buffer:
                buffer = buffer + "\n\n" + para
            else:
                buffer = para

        if buffer:
            merged.append(buffer)

        for para in merged:
            if len(para.strip()) < min_paragraph_size and merged:
                if merged.index(para) > 0:
                    prev = merged[merged.index(para) - 1]
                    merged[merged.index(para) - 1] = prev + "\n\n" + para
                    continue
            chunks.append({
                "content": para,
                "metadata": {
                    **metadata,
                    "chunk_index": len(chunks),
                    "start_char": content.find(para),
                    "end_char": content.find(para) + len(para)
                }
            })

    return chunks
