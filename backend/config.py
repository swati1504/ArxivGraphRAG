from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


_PROJECT_ROOT = Path(__file__).resolve().parents[1]
_ENV_FILE = str(_PROJECT_ROOT / ".env")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str | None = Field(default=None, alias="OPENAI_API_KEY")
    anthropic_api_key: str | None = Field(default=None, alias="ANTHROPIC_API_KEY")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")

    pinecone_api_key: str | None = Field(default=None, alias="PINECONE_API_KEY")
    pinecone_index_name: str = Field(default="arxiv-graphrag", alias="PINECONE_INDEX_NAME")
    pinecone_namespace: str = Field(default="default", alias="PINECONE_NAMESPACE")
    pinecone_cloud: str = Field(default="aws", alias="PINECONE_CLOUD")
    pinecone_region: str = Field(default="us-east-1", alias="PINECONE_REGION")

    neo4j_uri: str | None = Field(default=None, alias="NEO4J_URI")
    neo4j_username: str | None = Field(default=None, alias="NEO4J_USERNAME")
    neo4j_password: str | None = Field(default=None, alias="NEO4J_PASSWORD")
    neo4j_database: str = Field(default="", alias="NEO4J_DATABASE")

    embeddings_provider: str = Field(default="ollama", alias="EMBEDDINGS_PROVIDER")
    embeddings_model: str = Field(default="nomic-embed-text", alias="EMBEDDINGS_MODEL")
    embeddings_dim: int = Field(default=0, alias="EMBEDDINGS_DIM")
    embeddings_max_chars: int = Field(default=6000, alias="EMBEDDINGS_MAX_CHARS")

    rag_reasoning_model: str = Field(default="claude-3-5-sonnet-latest", alias="RAG_REASONING_MODEL")
    rag_provider: str = Field(default="ollama", alias="RAG_PROVIDER")
    rag_ollama_model: str = Field(default="", alias="RAG_OLLAMA_MODEL")
    rag_gemini_model: str = Field(default="gemini-1.5-flash-latest", alias="RAG_GEMINI_MODEL")
    rag_input_cost_per_1k: float | None = Field(default=None, alias="RAG_INPUT_COST_PER_1K")
    rag_output_cost_per_1k: float | None = Field(default=None, alias="RAG_OUTPUT_COST_PER_1K")

    graph_provider: str = Field(default="gemini", alias="GRAPH_PROVIDER")
    graph_gemini_model: str = Field(default="gemini-1.5-flash-latest", alias="GRAPH_GEMINI_MODEL")
    graph_ollama_model: str = Field(default="", alias="GRAPH_OLLAMA_MODEL")

    eval_provider: str = Field(default="gemini", alias="EVAL_PROVIDER")
    eval_gemini_model: str = Field(default="gemini-1.5-flash-latest", alias="EVAL_GEMINI_MODEL")

    ollama_host: str = Field(default="http://localhost:11434", alias="OLLAMA_HOST")

    chunk_size: int = Field(default=4000, alias="CHUNK_SIZE")
    chunk_overlap: int = Field(default=400, alias="CHUNK_OVERLAP")


def get_settings() -> Settings:
    return Settings()
