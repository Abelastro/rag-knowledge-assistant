from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    openai_api_key: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    llm_model: str = "gpt-4"
    chunk_size: int = 500
    chunk_overlap: int = 50
    vector_store_path: str = "./vectorstore"

    class Config:
        env_file = ".env"


settings = Settings()
