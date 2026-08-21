from typing import List, Optional


class LLM:
    def __init__(self, model: str = "gpt-4"):
        self.model = model

    async def generate(
        self,
        query: str,
        context: List[dict],
        system_prompt: Optional[str] = None
    ) -> str:
        from openai import OpenAI
        client = OpenAI()

        context_text = "\n\n".join([
            f"Source: {r['document']['metadata']['source']}\n{r['document']['content']}"
            for r in context
        ])

        messages = [
            {"role": "system", "content": system_prompt or "Answer based on the provided context. Cite sources."},
            {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {query}"}
        ]

        response = client.chat.completions.create(
            model=self.model,
            messages=messages
        )

        return response.choices[0].message.content
