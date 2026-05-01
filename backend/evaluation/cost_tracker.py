from __future__ import annotations

from dataclasses import dataclass

from backend.schemas.agent import AgentQueryResponse
from backend.schemas.query import QueryResponse


@dataclass(frozen=True)
class AggregateMetrics:
    avg_latency_ms: float | None
    avg_cost_usd: float | None


def aggregate_query_metrics(responses: list[QueryResponse | AgentQueryResponse]) -> AggregateMetrics:
    latencies = [r.metrics.latency_ms for r in responses if r.metrics.latency_ms is not None]
    costs = [r.metrics.total_cost_usd for r in responses if r.metrics.total_cost_usd is not None]
    avg_latency = (sum(latencies) / len(latencies)) if latencies else None
    avg_cost = (sum(costs) / len(costs)) if costs else None
    return AggregateMetrics(avg_latency_ms=avg_latency, avg_cost_usd=avg_cost)
