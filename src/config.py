from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Connection
    openrouter_api_key: str
    financebench_dir: str
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "financerag"

    # Embedding
    summary_model: str = "openai/gpt-4o-mini"
    embedding_model: str = "openai/text-embedding-3-large"
    embedding_dim: int = 3072
    sparse_model: str = "Qdrant/bm25"

    # Retrieval
    retrieval_mode: Literal["dense", "hybrid"] = "hybrid"
    rerank: bool = True
    rerank_model: str = "cohere/rerank-4-fast"
    candidates: int = 30
    top_k: int = 10

    # Generation
    generator_model: str = "openai/gpt-4o-mini"

    # Evaluation
    judge_model: str = "openai/gpt-5.1"


settings = Settings()  # type: ignore[call-arg]  # values come from the environment
