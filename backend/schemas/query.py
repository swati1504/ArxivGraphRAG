from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class Citation(BaseModel):
    paper_id: str
    title: str | None = None
    chunk_index: int | None = None


class RetrievedContext(BaseModel):
    text: str
    citation: Citation


class UsageMetrics(BaseModel):
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None


class QueryResponse(BaseModel):
    answer: str
    contexts: list[RetrievedContext]
    metrics: UsageMetrics
