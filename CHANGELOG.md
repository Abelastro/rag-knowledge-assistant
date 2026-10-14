# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Multiple chunking strategies: fixed-size, recursive, and semantic.
- Reranking layer using keyword-overlap scoring with a position penalty.
- Hybrid search combining vector similarity with BM25-style keyword matching.
- Evaluation metrics: retrieval precision, answer relevancy, context relevance, and hallucination detection.
- REST endpoints for ingestion (`POST /ingest`), listing documents (`GET /documents`), and metrics (`GET /metrics`).
- Test suite covering chunking, vector store, metrics, reranking, and API behavior.

### Changed
- Expanded the FastAPI application with new document and metrics endpoints.
