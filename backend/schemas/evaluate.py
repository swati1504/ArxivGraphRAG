from pydantic import BaseModel, Field

from backend.schemas.agent import AgentQueryResponse
from backend.schemas.query import QueryResponse


class EvaluateRequest(BaseModel):
    questions_path: str = Field(default="data/benchmark/questions.json")
    top_k: int = Field(default=8, ge=1, le=50)
    limit: int = Field(default=0, ge=0, le=200)
    include_llm_judge: bool = Field(default=True)
    include_agent: bool = Field(default=True)


class JudgeMetrics(BaseModel):
    faithfulness: float | None = None
    answer_relevancy: float | None = None
    context_precision: float | None = None
    context_recall: float | None = None
    answer_correctness: float | None = None


class EvaluatePerQuestion(BaseModel):
    id: str
    category: str
    question: str
    rag: QueryResponse
    graphrag: QueryResponse
    agent: AgentQueryResponse | None = None
    rag_judge: JudgeMetrics | None = None
    graphrag_judge: JudgeMetrics | None = None
    agent_judge: JudgeMetrics | None = None


class EvaluateSummary(BaseModel):
    rag_avg_latency_ms: float | None = None
    graphrag_avg_latency_ms: float | None = None
    agent_avg_latency_ms: float | None = None
    rag_avg_cost_usd: float | None = None
    graphrag_avg_cost_usd: float | None = None
    agent_avg_cost_usd: float | None = None
    rag_avg_faithfulness: float | None = None
    graphrag_avg_faithfulness: float | None = None
    agent_avg_faithfulness: float | None = None
    rag_avg_answer_relevancy: float | None = None
    graphrag_avg_answer_relevancy: float | None = None
    agent_avg_answer_relevancy: float | None = None
    rag_avg_context_precision: float | None = None
    graphrag_avg_context_precision: float | None = None
    agent_avg_context_precision: float | None = None
    rag_avg_context_recall: float | None = None
    graphrag_avg_context_recall: float | None = None
    agent_avg_context_recall: float | None = None
    rag_avg_answer_correctness: float | None = None
    graphrag_avg_answer_correctness: float | None = None
    agent_avg_answer_correctness: float | None = None


class EvaluateResponse(BaseModel):
    results: list[EvaluatePerQuestion]
    summary: EvaluateSummary
    warnings: list[str] = Field(default_factory=list)
    trace_id: str | None = None
