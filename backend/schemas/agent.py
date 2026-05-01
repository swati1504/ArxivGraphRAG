from pydantic import BaseModel, Field

from backend.schemas.query import Citation, UsageMetrics


class AgentQueryRequest(BaseModel):
    question: str = Field(min_length=1)
    top_k: int = Field(default=8, ge=1, le=50)


class AgentRetrievedContext(BaseModel):
    text: str
    citation: Citation
    score: float | None = None
    source: str = Field(default="vector")


class AgentPlan(BaseModel):
    sub_questions: list[str] = Field(default_factory=list)
    entities: dict[str, list[str]] = Field(default_factory=dict)
    run_vector: bool = True
    run_graph: bool = False
    prompt_style: str = Field(default="concise_factual")


class AgentTraceItem(BaseModel):
    agent: str
    decision: str | None = None
    latency_ms: int
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    total_cost_usd: float | None = None


class AgentQueryResponse(BaseModel):
    query_type: str
    route: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    routing_reason: str = Field(default="")
    plan: AgentPlan | None = None
    answer: str
    contexts: list[AgentRetrievedContext]
    contested_claims: list[str] = Field(default_factory=list)
    graph_edges: list[str] = Field(default_factory=list)
    metrics: UsageMetrics
    agent_trace: list[AgentTraceItem]
    trace_id: str | None = None
