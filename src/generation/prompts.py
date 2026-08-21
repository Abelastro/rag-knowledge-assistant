RAG_SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
1. Only answer based on the provided context
2. Cite your sources when possible
3. If the context doesn't contain enough information, say so
4. Be concise and accurate
5. Do not make up information"""

HALLUCINATION_CHECK_PROMPT = """Check if the following answer is supported by the given context.
Return "SUPPORTED" if the answer is grounded in the context, or "UNSUPPORTED" if it contains information not found in the context.

Context: {context}
Answer: {answer}"""
