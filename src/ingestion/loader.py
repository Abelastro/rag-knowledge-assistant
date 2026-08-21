from pathlib import Path
from typing import List


def load_documents(path: str) -> List[dict]:
    documents = []
    path = Path(path)

    if path.is_file():
        documents.append(_load_file(path))
    elif path.is_dir():
        for file in path.glob("**/*.md"):
            documents.append(_load_file(file))
        for file in path.glob("**/*.txt"):
            documents.append(_load_file(file))

    return documents


def _load_file(file_path: Path) -> dict:
    content = file_path.read_text(encoding="utf-8")
    return {
        "content": content,
        "metadata": {"source": str(file_path), "filename": file_path.name}
    }
