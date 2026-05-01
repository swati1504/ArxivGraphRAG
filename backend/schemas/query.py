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


class GraphEdge(BaseModel):
    source_paper_id: str
    rel_type: str
    target_type: str
    target_id_or_name: str
    chunk_index: int | None = None
    evidence: str | None = None
    confidence: float | None = None


class GraphDebug(BaseModel):
    seed_paper_ids: list[str] = Field(default_factory=list)
    related_paper_ids: list[str] = Field(default_factory=list)
    edges: list[GraphEdge] = Field(default_factory=list)


class QueryResponse(BaseModel):
    answer: str
    contexts: list[RetrievedContext]
    metrics: UsageMetrics
    graph: GraphDebug | None = None
    trace_id: str | None = None
