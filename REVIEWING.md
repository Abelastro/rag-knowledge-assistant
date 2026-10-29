# Reviewing This Project

A short guide for anyone reviewing code or tests in this repository.

## What this project is

A retrieval-augmented generation system: documents are loaded, chunked, embedded,
searched, reranked, and passed to an LLM that must answer only from the retrieved
context. The interesting engineering is in chunking quality, retrieval evaluation,
and hallucination detection.

## Where the risk lives

1. **Chunking** (`src/ingestion/chunker.py`): chunks that split mid-sentence destroy
   context. A good chunk preserves paragraph boundaries when possible.
2. **Retrieval** (`src/retrieval/`): a high similarity score is not relevance. Check
   that hybrid search and reranking are tested against cases where scores mislead.
3. **Generation** (`src/generation/llm.py`): the model must not be able to answer from
   prior knowledge. Look for prompt hardening and grounding enforcement.
4. **Evaluation** (`src/evaluation/metrics.py`): metrics that are vacuously true
   (always 0.0 or 1.0) are worse than no metrics. Ask what each metric measures.

## Review checklist

- [ ] Every behavior change has a test that fails without the change
- [ ] Chunk overlap respects document structure, not just character offsets
- [ ] Retrieval tests assert order and relevance, not only that something was found
- [ ] The API response ties every answer claim back to a source
- [ ] Metric edge cases are covered: empty context, empty answer, zero-length docs
- [ ] No secrets or `.env` content in tests or fixtures

## What a reviewer should notice

- Tests that pass trivially (e.g., asserting an unchanging constant) count as red
  flags for the "test validity" review dimension.
- A test that mocks the LLM entirely can validate plumbing but proves nothing about
  generation quality. Prefer a deterministic fake that behaves like the prompt.
- The strongest signal of a sound RAG system is that the evaluation suite would fail
  if the prompt were weakened. Consider asserting on the metrics, not the prose.
