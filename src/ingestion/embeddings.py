from typing import List


async def generate_embeddings(texts: List[str], model: str = "text-embedding-ada-002") -> List[List[float]]:
    from openai import OpenAI
    client = OpenAI()

    response = client.embeddings.create(
        model=model,
        input=texts
    )

    return [item.embedding for item in response.data]
