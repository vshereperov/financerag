from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Required
    openrouter_api_key: str
    financebench_dir: str

    # Embeddings
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dim: int = 3072
    sparse_model: str = "Qdrant/bm25"

    # Vector store
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "financerag"

    # Generation & evaluation
    llm_model: str = "openai/gpt-4o-mini"
    judge_model: str = "openai/gpt-4o"

    # Retrieval
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    rerank: bool = True
    rerank_model: str = "cohere/rerank-4-fast"
    top_k: int = 10


settings = Settings()  # type: ignore[call-arg]  # values come from the environment
