export type QueryRequest = {
  question: string
  top_k: number
}

export type Citation = {
  paper_id: string
  title?: string | null
  chunk_index?: number | null
}

export type RetrievedContext = {
  text: string
  citation: Citation
}

export type UsageMetrics = {
  latency_ms: number
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
  total_cost_usd?: number | null
}

export type GraphEdge = {
  source_paper_id: string
  rel_type: string
  target_type: string
  target_id_or_name: string
  chunk_index?: number | null
  evidence?: string | null
  confidence?: number | null
}

export type GraphDebug = {
  seed_paper_ids: string[]
  related_paper_ids: string[]
  edges: GraphEdge[]
}

export type QueryResponse = {
  answer: string
  contexts: RetrievedContext[]
  metrics: UsageMetrics
  graph?: GraphDebug | null
}

export type AgentPlan = {
  sub_questions: string[]
  entities: Record<string, string[]>
  run_vector: boolean
  run_graph: boolean
  prompt_style: string
}

export type AgentTraceItem = {
  agent: string
  decision?: string | null
  latency_ms: number
  prompt_tokens?: number | null
  completion_tokens?: number | null
  total_tokens?: number | null
  total_cost_usd?: number | null
}

export type AgentRetrievedContext = {
  text: string
  citation: Citation
  score?: number | null
  source: string
}

export type AgentQueryResponse = {
  query_type: string
  route: string
  confidence: number
  routing_reason: string
  plan?: AgentPlan | null
  answer: string
  contexts: AgentRetrievedContext[]
  contested_claims: string[]
  graph_edges: string[]
  metrics: UsageMetrics
  agent_trace: AgentTraceItem[]
}

export type JudgeMetrics = {
  faithfulness?: number | null
  answer_relevancy?: number | null
  context_precision?: number | null
  context_recall?: number | null
  answer_correctness?: number | null
}

export type EvaluateRequest = {
  questions_path: string
  top_k: number
  limit: number
  include_llm_judge: boolean
  include_agent: boolean
}

export type EvaluatePerQuestion = {
  id: string
  category: string
  question: string
  rag: QueryResponse
  graphrag: QueryResponse
  agent?: AgentQueryResponse | null
  rag_judge?: JudgeMetrics | null
  graphrag_judge?: JudgeMetrics | null
  agent_judge?: JudgeMetrics | null
}

export type EvaluateSummary = {
  rag_avg_latency_ms?: number | null
  graphrag_avg_latency_ms?: number | null
  agent_avg_latency_ms?: number | null
  rag_avg_cost_usd?: number | null
  graphrag_avg_cost_usd?: number | null
  agent_avg_cost_usd?: number | null
  rag_avg_faithfulness?: number | null
  graphrag_avg_faithfulness?: number | null
  agent_avg_faithfulness?: number | null
  rag_avg_answer_relevancy?: number | null
  graphrag_avg_answer_relevancy?: number | null
  agent_avg_answer_relevancy?: number | null
  rag_avg_context_precision?: number | null
  graphrag_avg_context_precision?: number | null
  agent_avg_context_precision?: number | null
  rag_avg_context_recall?: number | null
  graphrag_avg_context_recall?: number | null
  agent_avg_context_recall?: number | null
  rag_avg_answer_correctness?: number | null
  graphrag_avg_answer_correctness?: number | null
  agent_avg_answer_correctness?: number | null
}

export type EvaluateResponse = {
  results: EvaluatePerQuestion[]
  summary: EvaluateSummary
  warnings: string[]
}

async function postJson<T>(path: string, payload: unknown, signal?: AbortSignal): Promise<T> {
  const res = await fetch(path, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
    signal,
  })

  if (res.ok) return (await res.json()) as T

  let detail = `${res.status} ${res.statusText}`
  try {
    const body = (await res.json()) as { detail?: unknown }
    if (typeof body?.detail === 'string' && body.detail.trim()) detail = body.detail
  } catch {}
  throw new Error(detail)
}

export function queryRag(payload: QueryRequest, signal?: AbortSignal) {
  return postJson<QueryResponse>('/api/query/rag', payload, signal)
}

export function queryGraphRag(payload: QueryRequest, signal?: AbortSignal) {
  return postJson<QueryResponse>('/api/query/graphrag', payload, signal)
}

export function queryAgent(payload: QueryRequest, signal?: AbortSignal) {
  return postJson<AgentQueryResponse>('/api/query/agent', payload, signal)
}

export function evaluate(payload: EvaluateRequest, signal?: AbortSignal) {
  return postJson<EvaluateResponse>('/api/evaluate', payload, signal)
}

