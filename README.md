# RAG Knowledge Assistant

A Retrieval-Augmented Generation system with **document ingestion**, **chunking strategies**, **vector retrieval**, and **grounded responses** with source citations.

## Why I Built This

This project demonstrates practical RAG engineering — going beyond a simple "PDF chatbot" to build a system with proper chunking, evaluation metrics, hallucination detection, and source attribution.

## Architecture

```
Documents
    ↓
Chunking
    ↓
Embeddings
    ↓
Vector Store
    ↓
Retrieval
    ↓
LLM Generation
    ↓
Grounded Response
```

## Features

- **Multiple chunking strategies** (fixed-size, recursive, semantic)
- **Vector store integration** (FAISS/Chroma)
- **Retrieval evaluation** with relevance scoring
- **Hallucination-aware responses** with confidence tracking
- **Source citations** for every answer
- **FastAPI REST API**
- **Docker containerization**

## Tech Stack

- Python 3.10+
- LangChain
- FAISS / ChromaDB
- FastAPI
- OpenAI / Hugging Face embeddings

## Project Structure

```
rag-knowledge-assistant/
├── src/
│   ├── ingestion/
│   │   ├── loader.py        # Document loaders
│   │   ├── chunker.py       # Text chunking
│   │   └── embeddings.py    # Embedding generation
│   ├── retrieval/
│   │   ├── vectorstore.py   # Vector DB interface
│   │   └── retriever.py     # Retrieval logic
│   ├── generation/
│   │   ├── llm.py           # LLM interface
│   │   └── prompts.py       # Prompt templates
│   ├── evaluation/
│   │   └── metrics.py       # Retrieval metrics
│   ├── api/
│   │   └── app.py           # FastAPI
│   └── utils/
│       └── config.py
├── data/sample_docs/
├── tests/
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Installation

```bash
git clone https://github.com/Abelastro/rag-knowledge-assistant.git
cd rag-knowledge-assistant
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Ingest Documents

```python
from src.ingestion.loader import load_documents
from src.ingestion.chunker import chunk_documents
from src.retrieval.vectorstore import VectorStore

docs = load_documents("data/sample_docs/")
chunks = chunk_documents(docs)
vectorstore = VectorStore()
vectorstore.add_documents(chunks)
```

### Query

```python
response = await vectorstore.query("What is the main topic?")
print(response.answer)
print(response.sources)
```

## Evaluation

The system tracks:
- **Retrieval precision**: How many retrieved chunks are relevant
- **Answer groundedness**: How well the answer is supported by retrieved chunks
- **Source attribution**: Percentage of answer backed by sources

## Limitations

- Prototype system, not production-ready
- Requires API key for LLM access
- Vector store is local (not distributed)

## Future Improvements

- Add reranking step
- Implement hybrid search
- Add evaluation dashboard
- Support multiple document formats
