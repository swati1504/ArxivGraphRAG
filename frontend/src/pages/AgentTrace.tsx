import type { ChangeEvent } from 'react'
import { useMemo, useRef, useState } from 'react'
import type { AgentQueryResponse, AgentTraceItem } from '../api'
import { queryAgent } from '../api'

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

function totalCost(items: AgentTraceItem[]): number | null {
  const vals = items.map((i) => i.total_cost_usd).filter((v): v is number => typeof v === 'number' && !Number.isNaN(v))
  if (!vals.length) return null
  return vals.reduce((a, b) => a + b, 0)
}

export function AgentTracePage() {
  const [question, setQuestion] = useState(
    'Compare two papers and explain any contradictions. Focus on citations and show the routing decision.'
  )
  const [topK, setTopK] = useState(8)
  const [resp, setResp] = useState<AgentQueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const payload = useMemo(() => ({ question: question.trim(), top_k: topK }), [question, topK])

  const run = async () => {
    const q = payload.question
    if (!q) return
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    setLoading(true)
    setError(null)
    setResp(null)
    try {
      const out = await queryAgent(payload, ac.signal)
      setResp(out)
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setLoading(false)
    }
  }

  const cancel = () => {
    abortRef.current?.abort()
    abortRef.current = null
    setLoading(false)
  }

  const trace = resp?.agent_trace ?? []
  const totalLatency = trace.reduce((a: number, t: AgentTraceItem) => a + (t.latency_ms || 0), 0) || 1
  const maxLatency = Math.max(1, ...trace.map((t: AgentTraceItem) => t.latency_ms || 0))
  const totalTraceCost = totalCost(trace)

  return (
    <div className="page">
      <div className="controls">
        <div className="field">
          <label className="label">Question</label>
          <textarea
            className="textarea"
            value={question}
            onChange={(e: ChangeEvent<HTMLTextAreaElement>) => setQuestion(e.target.value)}
            rows={3}
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
          <button className="btn primary" onClick={run} disabled={loading || !payload.question}>
            {loading ? 'Running…' : 'Run Agent'}
          </button>
          <button className="btn" onClick={cancel} disabled={!loading}>
            Cancel
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}

      {resp && (
        <div className="stack">
          <div className="card">
            <div className="cardHeader">
              <div className="cardTitle">Routing</div>
              <div className="metaRow">
                <div className="metaItem">route {resp.route}</div>
                <div className="metaItem">type {resp.query_type}</div>
                <div className="metaItem">confidence {resp.confidence.toFixed(2)}</div>
              </div>
            </div>
            <div className="subtle">{resp.routing_reason || '—'}</div>
            {resp.plan ? (
              <details className="details">
                <summary>Plan</summary>
                <div className="kv">
                  <div>
                    <div className="kvKey">prompt_style</div>
                    <div className="kvVal">{resp.plan.prompt_style}</div>
                  </div>
                  <div>
                    <div className="kvKey">run_vector</div>
                    <div className="kvVal">{String(resp.plan.run_vector)}</div>
                  </div>
                  <div>
                    <div className="kvKey">run_graph</div>
                    <div className="kvVal">{String(resp.plan.run_graph)}</div>
                  </div>
                </div>
                {resp.plan.sub_questions?.length ? (
                  <div className="list">
                    {resp.plan.sub_questions.map((s, i) => (
                      <div key={i} className="listItem">
                        {s}
                      </div>
                    ))}
                  </div>
                ) : null}
              </details>
            ) : null}
          </div>

          <div className="card">
            <div className="cardHeader">
              <div className="cardTitle">Timeline</div>
              <div className="metaRow">
                <div className="metaItem">total {formatInt(resp.metrics.latency_ms)} ms</div>
                <div className="metaItem">trace cost {formatCost(totalTraceCost)}</div>
              </div>
            </div>
            <div className="timeline">
              {trace.map((t, idx) => {
                const w = Math.max(2, Math.round((100 * t.latency_ms) / totalLatency))
                return (
                  <div key={idx} className="timelineRow">
                    <div className="timelineLabel">{t.agent}</div>
                    <div className="timelineBarWrap">
                      <div className="timelineBar" style={{ width: `${w}%` }} />
                    </div>
                    <div className="timelineMeta">{formatInt(t.latency_ms)} ms</div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="card">
            <div className="cardHeader">
              <div className="cardTitle">Cost per step</div>
              <div className="metaRow">
                <div className="metaItem">total {formatCost(resp.metrics.total_cost_usd)}</div>
              </div>
            </div>
            <div className="bars">
              {trace.map((t, idx) => {
                const cost = t.total_cost_usd ?? 0
                const latency = t.latency_ms ?? 0
                const lw = Math.max(1, Math.round((100 * latency) / maxLatency))
                const cw = totalTraceCost ? Math.max(1, Math.round((100 * cost) / totalTraceCost)) : 0
                return (
                  <div key={idx} className="barRow">
                    <div className="barLabel">{t.agent}</div>
                    <div className="barGroup">
                      <div className="barItem">
                        <div className="barFill" style={{ width: `${lw}%` }} />
                        <div className="barText">{formatInt(latency)} ms</div>
                      </div>
                      <div className="barItem">
                        <div className="barFill alt" style={{ width: `${cw}%` }} />
                        <div className="barText">{formatCost(cost)}</div>
                      </div>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          <div className="card">
            <div className="cardHeader">
              <div className="cardTitle">Answer</div>
              <div className="metaRow">
                <div className="metaItem">tokens {formatInt(resp.metrics.total_tokens)}</div>
                <div className="metaItem">cost {formatCost(resp.metrics.total_cost_usd)}</div>
              </div>
            </div>
            <div className="answerBox">{resp.answer}</div>
          </div>
        </div>
      )}
    </div>
  )
}
