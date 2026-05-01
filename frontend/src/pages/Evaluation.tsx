import type { ChangeEvent } from 'react'
import { useMemo, useRef, useState } from 'react'
import type { EvaluateResponse, EvaluateSummary } from '../api'
import { evaluate } from '../api'

function formatCost(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  if (v === 0) return '$0.00'
  if (v < 0.01) return `$${v.toFixed(4)}`
  return `$${v.toFixed(3)}`
}

function formatMs(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  return `${Math.round(v)} ms`
}

function formatScore(v: number | null | undefined): string {
  if (v == null || Number.isNaN(v)) return '—'
  return v.toFixed(2)
}

type BarMetric = { key: string; label: string; value: number | null | undefined; fmt: (v: number | null | undefined) => string }

function MetricBars(props: { title: string; metrics: BarMetric[] }) {
  const vals = props.metrics
    .map((m) => m.value)
    .filter((v): v is number => typeof v === 'number' && !Number.isNaN(v) && v >= 0)
  const max = Math.max(1, ...vals)
  return (
    <div className="card">
      <div className="cardHeader">
        <div className="cardTitle">{props.title}</div>
      </div>
      <div className="bars">
        {props.metrics.map((m) => {
          const v = typeof m.value === 'number' && !Number.isNaN(m.value) ? m.value : null
          const w = v == null ? 0 : Math.max(2, Math.round((100 * v) / max))
          return (
            <div key={m.key} className="barRow">
              <div className="barLabel">{m.label}</div>
              <div className="barGroup">
                <div className="barItem">
                  <div className="barFill" style={{ width: `${w}%` }} />
                  <div className="barText">{m.fmt(m.value)}</div>
                </div>
              </div>
            </div>
          )
        })}
      </div>
    </div>
  )
}

function SummaryGrid(props: { summary: EvaluateSummary }) {
  const s = props.summary
  const showAgent =
    s.agent_avg_latency_ms != null ||
    s.agent_avg_cost_usd != null ||
    s.agent_avg_faithfulness != null ||
    s.agent_avg_answer_relevancy != null

  return (
    <div className="grid2">
      <MetricBars
        title="Latency (avg)"
        metrics={[
          { key: 'rag', label: 'RAG', value: s.rag_avg_latency_ms, fmt: formatMs },
          { key: 'gr', label: 'GraphRAG', value: s.graphrag_avg_latency_ms, fmt: formatMs },
          ...(showAgent ? [{ key: 'ag', label: 'Agent', value: s.agent_avg_latency_ms, fmt: formatMs }] : []),
        ]}
      />
      <MetricBars
        title="Cost (avg)"
        metrics={[
          { key: 'rag', label: 'RAG', value: s.rag_avg_cost_usd, fmt: formatCost },
          { key: 'gr', label: 'GraphRAG', value: s.graphrag_avg_cost_usd, fmt: formatCost },
          ...(showAgent ? [{ key: 'ag', label: 'Agent', value: s.agent_avg_cost_usd, fmt: formatCost }] : []),
        ]}
      />
      <MetricBars
        title="Faithfulness (avg)"
        metrics={[
          { key: 'rag', label: 'RAG', value: s.rag_avg_faithfulness, fmt: formatScore },
          { key: 'gr', label: 'GraphRAG', value: s.graphrag_avg_faithfulness, fmt: formatScore },
          ...(showAgent ? [{ key: 'ag', label: 'Agent', value: s.agent_avg_faithfulness, fmt: formatScore }] : []),
        ]}
      />
      <MetricBars
        title="Answer relevancy (avg)"
        metrics={[
          { key: 'rag', label: 'RAG', value: s.rag_avg_answer_relevancy, fmt: formatScore },
          { key: 'gr', label: 'GraphRAG', value: s.graphrag_avg_answer_relevancy, fmt: formatScore },
          ...(showAgent ? [{ key: 'ag', label: 'Agent', value: s.agent_avg_answer_relevancy, fmt: formatScore }] : []),
        ]}
      />
    </div>
  )
}

export function EvaluationPage() {
  const [questionsPath, setQuestionsPath] = useState('data/benchmark/questions_reference.json')
  const [topK, setTopK] = useState(8)
  const [limit, setLimit] = useState(20)
  const [includeJudge, setIncludeJudge] = useState(true)
  const [includeAgent, setIncludeAgent] = useState(true)
  const [resp, setResp] = useState<EvaluateResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const abortRef = useRef<AbortController | null>(null)

  const payload = useMemo(
    () => ({
      questions_path: questionsPath.trim(),
      top_k: topK,
      limit: Math.max(0, Math.min(200, limit)),
      include_llm_judge: includeJudge,
      include_agent: includeAgent,
    }),
    [questionsPath, topK, limit, includeJudge, includeAgent]
  )

  const run = async () => {
    if (!payload.questions_path) return
    abortRef.current?.abort()
    const ac = new AbortController()
    abortRef.current = ac
    setLoading(true)
    setError(null)
    setResp(null)
    try {
      const out = await evaluate(payload, ac.signal)
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

  return (
    <div className="page">
      <div className="controls">
        <div className="row">
          <div className="field inline grow">
            <label className="label">questions_path</label>
            <input
              className="input"
              value={questionsPath}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setQuestionsPath(e.target.value)}
              placeholder="data/benchmark/questions.json"
            />
          </div>
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
          <div className="field inline">
            <label className="label">limit</label>
            <input
              className="input"
              type="number"
              min={0}
              max={200}
              value={limit}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setLimit(Math.max(0, Math.min(200, Number(e.target.value) || 0)))}
            />
          </div>
        </div>

        <div className="row">
          <label className="check">
            <input
              type="checkbox"
              checked={includeJudge}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setIncludeJudge(e.target.checked)}
            />
            <span>LLM As Judge</span>
          </label>
          <label className="check">
            <input
              type="checkbox"
              checked={includeAgent}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setIncludeAgent(e.target.checked)}
            />
            <span>Include Agent</span>
          </label>
          <button className="btn primary" onClick={run} disabled={loading || !payload.questions_path}>
            {loading ? 'Running…' : 'Run evaluation'}
          </button>
          <button className="btn" onClick={cancel} disabled={!loading}>
            Cancel
          </button>
        </div>
      </div>

      {error && <div className="errorBox">{error}</div>}
      {resp?.warnings?.length ? (
        <div className="warningBox">
          {resp.warnings.map((w, i) => (
            <div key={i}>{w}</div>
          ))}
        </div>
      ) : null}

      {resp && (
        <div className="stack">
          <SummaryGrid summary={resp.summary} />

          <div className="card">
            <div className="cardHeader">
              <div className="cardTitle">Per-question comparison</div>
              <div className="metaRow">
                <div className="metaItem">questions {resp.results.length}</div>
              </div>
            </div>
            <div className="tableWrap">
              <table className="table">
                <thead>
                  <tr>
                    <th>id</th>
                    <th>category</th>
                    <th>question</th>
                    <th>RAG latency</th>
                    <th>GraphRAG latency</th>
                    <th>Agent latency</th>
                    <th>RAG cost</th>
                    <th>GraphRAG cost</th>
                    <th>Agent cost</th>
                    <th>faithfulness</th>
                    <th>answer relevancy</th>
                  </tr>
                </thead>
                <tbody>
                  {resp.results.map((r) => (
                    <tr key={r.id}>
                      <td className="mono">{r.id}</td>
                      <td>{r.category}</td>
                      <td className="cellWrap">{r.question}</td>
                      <td className="mono">{formatMs(r.rag.metrics.latency_ms)}</td>
                      <td className="mono">{formatMs(r.graphrag.metrics.latency_ms)}</td>
                      <td className="mono">{r.agent ? formatMs(r.agent.metrics.latency_ms) : '—'}</td>
                      <td className="mono">{formatCost(r.rag.metrics.total_cost_usd)}</td>
                      <td className="mono">{formatCost(r.graphrag.metrics.total_cost_usd)}</td>
                      <td className="mono">{r.agent ? formatCost(r.agent.metrics.total_cost_usd) : '—'}</td>
                      <td className="mono">
                        {formatScore(r.agent_judge?.faithfulness ?? r.graphrag_judge?.faithfulness ?? r.rag_judge?.faithfulness)}
                      </td>
                      <td className="mono">
                        {formatScore(
                          r.agent_judge?.answer_relevancy ?? r.graphrag_judge?.answer_relevancy ?? r.rag_judge?.answer_relevancy
                        )}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <details className="details">
            <summary>Answers</summary>
            <div className="stack">
              {resp.results.map((r) => (
                <div key={r.id} className="card">
                  <div className="cardHeader">
                    <div className="cardTitle">
                      {r.id} — {r.category}
                    </div>
                    <div className="metaRow">
                      <div className="metaItem">judge {includeJudge ? 'on' : 'off'}</div>
                    </div>
                  </div>
                  <div className="subtle">{r.question}</div>
                  <div className="grid3">
                    <div className="miniCard">
                      <div className="miniTitle">RAG</div>
                      <div className="miniAnswer">{r.rag.answer}</div>
                    </div>
                    <div className="miniCard">
                      <div className="miniTitle">GraphRAG</div>
                      <div className="miniAnswer">{r.graphrag.answer}</div>
                    </div>
                    <div className="miniCard">
                      <div className="miniTitle">Agent</div>
                      <div className="miniAnswer">{r.agent?.answer ?? '—'}</div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  )
}
