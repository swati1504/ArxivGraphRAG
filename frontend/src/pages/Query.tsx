import type { ChangeEvent } from 'react'
import { useMemo, useRef, useState } from 'react'
import type { AgentQueryResponse, QueryResponse } from '../api'
import { queryAgent, queryGraphRag, queryRag } from '../api'

type QueryRun = {
  rag?: QueryResponse
  graphrag?: QueryResponse
  agent?: AgentQueryResponse
  errors: Partial<Record<'rag' | 'graphrag' | 'agent', string>>
}

function formatCost(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  if (v === 0) return '$0.00'
  if (v < 0.01) return `$${v.toFixed(4)}`
  return `$${v.toFixed(3)}`
}

function formatInt(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  return String(Math.round(v))
}

async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {}
}

function TraceLine(props: { traceId?: string | null }) {
  const id = (props.traceId || '').trim()
  if (!id) return null
  return (
    <div className="metaRow subtle">
      <div className="metaItem">trace {id}</div>
      <button
        className="btn"
        style={{ padding: '6px 10px', borderRadius: 8, fontSize: 12 }}
        onClick={() => copyToClipboard(id)}
        type="button"
      >
        Copy
      </button>
      <a href="https://smith.langchain.com/" target="_blank" rel="noreferrer">
        Open LangSmith
      </a>
    </div>
  )
}

function AnswerCard(props: { title: string; resp?: QueryResponse | AgentQueryResponse; error?: string }) {
  const { title, resp, error } = props
  const metrics = resp?.metrics
  const traceId = resp ? ('trace_id' in resp ? resp.trace_id : null) : null
  return (
    <div className="card">
      <div className="cardHeader">
        <div className="cardTitle">{title}</div>
        <div className="metaRow">
          <div className="metaItem">latency {metrics ? `${formatInt(metrics.latency_ms)} ms` : '—'}</div>
          <div className="metaItem">tokens {metrics ? formatInt(metrics.total_tokens) : '—'}</div>
          <div className="metaItem">cost {metrics ? formatCost(metrics.total_cost_usd) : '—'}</div>
        </div>
      </div>
      {error && <div className="errorBox">{error}</div>}
      {resp && (
        <>
          <TraceLine traceId={traceId} />
          {'graph' in resp && resp.graph?.edges ? (
            <div className="metaRow subtle">
              <div className="metaItem">graph edges {resp.graph.edges.length}</div>
              <div className="metaItem">seed papers {resp.graph.seed_paper_ids.length}</div>
              <div className="metaItem">related papers {resp.graph.related_paper_ids.length}</div>
            </div>
          ) : null}
          {'route' in resp ? (
            <div className="metaRow subtle">
              <div className="metaItem">route {resp.route}</div>
              <div className="metaItem">type {resp.query_type}</div>
              <div className="metaItem">confidence {resp.confidence.toFixed(2)}</div>
            </div>
          ) : null}
          <div className="answerBox">{resp.answer}</div>
          {'contexts' in resp && resp.contexts?.length ? (
            <details className="details">
              <summary>Contexts ({resp.contexts.length})</summary>
              <div className="contexts">
                {resp.contexts.map((c, idx) => (
                  <div key={`${c.citation.paper_id}:${c.citation.chunk_index}:${idx}`} className="contextItem">
                    <div className="contextMeta">
                      {c.citation.paper_id}:{c.citation.chunk_index ?? '—'} {c.citation.title ? `— ${c.citation.title}` : ''}
                      {'source' in c ? ` (${c.source})` : ''}
                    </div>
                    <div className="contextText">{c.text}</div>
                  </div>
                ))}
              </div>
            </details>
          ) : null}
          {'graph_edges' in resp && resp.graph_edges?.length ? (
            <details className="details">
              <summary>Graph edges (agent) ({resp.graph_edges.length})</summary>
              <div className="monoList">
                {resp.graph_edges.map((l, i) => (
                  <div key={i} className="monoLine">
                    {l}
                  </div>
                ))}
              </div>
            </details>
          ) : null}
        </>
      )}
    </div>
  )
}

export function QueryPage() {
  const [question, setQuestion] = useState(
    'What are the main approaches for retrieval-augmented generation, and how does GraphRAG change the failure modes?'
  )
  const [topK, setTopK] = useState(8)
  const [run, setRun] = useState<QueryRun>({ errors: {} })
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const payload = useMemo(() => ({ question: question.trim(), top_k: topK }), [question, topK])

  const onRun = async () => {
    const q = payload.question
    if (!q) return
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac

    setLoading(true)
    setRun({ errors: {} })
    const [rag, graphrag, agent] = await Promise.allSettled([
      queryRag(payload, ac.signal),
      queryGraphRag(payload, ac.signal),
      queryAgent(payload, ac.signal),
    ])

    const next: QueryRun = { errors: {} }
    if (rag.status === 'fulfilled') next.rag = rag.value
    else next.errors.rag = rag.reason instanceof Error ? rag.reason.message : String(rag.reason)
    if (graphrag.status === 'fulfilled') next.graphrag = graphrag.value
    else next.errors.graphrag = graphrag.reason instanceof Error ? graphrag.reason.message : String(graphrag.reason)
    if (agent.status === 'fulfilled') next.agent = agent.value
    else next.errors.agent = agent.reason instanceof Error ? agent.reason.message : String(agent.reason)
    setRun(next)
    setLoading(false)
  }

  const onCancel = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setLoading(false)
  }

  return (
    <div className="page">
      <div className="controls">
        <div className="field">
          <label className="label">Question</label>
          <textarea
            className="textarea"
            value={question}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setQuestion(e.target.value)}
            rows={4}
            placeholder="Ask a question…"
          />
        </div>
        <div className="row">
          <div className="field inline">
            <label className="label">top_k</label>
            <input
              className="input"
              type="number"
              min={1}
              max={50}
              value={topK}
              onChange={(e: ChangeEvent<HTMLInputElement>) =>
                setTopK(Math.max(1, Math.min(50, Number(e.target.value) || 8)))
              }
            />
          </div>
          <button className="btn primary" onClick={onRun} disabled={loading || !payload.question}>
            {loading ? 'Running…' : 'Run (RAG vs GraphRAG vs Agent)'}
          </button>
          <button className="btn" onClick={onCancel} disabled={!loading}>
            Cancel
          </button>
        </div>
      </div>

      <div className="grid3">
        <AnswerCard title="Vector RAG" resp={run.rag} error={run.errors.rag} />
        <AnswerCard title="GraphRAG" resp={run.graphrag} error={run.errors.graphrag} />
        <AnswerCard title="Agent RAG" resp={run.agent} error={run.errors.agent} />
      </div>
    </div>
  )
}
