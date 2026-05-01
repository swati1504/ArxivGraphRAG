import type { ChangeEvent } from 'react'
import { useMemo, useRef, useState } from 'react'
import type { GraphEdge, QueryResponse } from '../api'
import { queryGraphRag } from '../api'

type Node = { key: string; label: string; kind: 'Paper' | 'Other' }
type LayoutNode = Node & { x: number; y: number; seed: boolean }

function edgeColor(relType: string): string {
  const t = relType.toUpperCase()
  if (t === 'CONTRADICTS') return '#ef4444'
  if (t === 'CITES') return '#60a5fa'
  if (t === 'PROPOSES') return '#34d399'
  if (t === 'STUDIES') return '#fbbf24'
  return '#94a3b8'
}

function toNodesAndEdges(graphEdges: GraphEdge[]) {
  const nodesByKey = new Map<string, Node>()
  const add = (key: string, label: string, kind: Node['kind']) => {
    if (!nodesByKey.has(key)) nodesByKey.set(key, { key, label, kind })
  }

  for (const e of graphEdges) {
    add(`Paper:${e.source_paper_id}`, e.source_paper_id, 'Paper')
    const tkey = e.target_type === 'Paper' ? `Paper:${e.target_id_or_name}` : `${e.target_type}:${e.target_id_or_name}`
    add(tkey, e.target_id_or_name, e.target_type === 'Paper' ? 'Paper' : 'Other')
  }
  return { nodes: [...nodesByKey.values()], edges: graphEdges }
}

function circleLayout(nodes: Node[], seedPaperIds: string[], w: number, h: number): LayoutNode[] {
  const seedSet = new Set(seedPaperIds.map((s) => `Paper:${s}`))
  const n = nodes.length || 1
  const cx = w / 2
  const cy = h / 2
  const r = Math.min(w, h) * 0.38
  return nodes.map((node, idx) => {
    const a = (2 * Math.PI * idx) / n
    const x = cx + r * Math.cos(a)
    const y = cy + r * Math.sin(a)
    return { ...node, x, y, seed: seedSet.has(node.key) }
  })
}

async function copyToClipboard(text: string) {
  try {
    await navigator.clipboard.writeText(text)
  } catch {}
}

export function GraphPage() {
  const [question, setQuestion] = useState('Show the key concepts and citation relationships relevant to GraphRAG.')
  const [topK, setTopK] = useState(8)
  const [resp, setResp] = useState<QueryResponse | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [selectedKey, setSelectedKey] = useState<string | null>(null)
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
    setSelectedKey(null)
    try {
      const out = await queryGraphRag(payload, ac.signal)
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

  const graph = resp?.graph ?? null
  const { nodes, edges } = useMemo(() => toNodesAndEdges(graph?.edges ?? []), [graph?.edges])

  const W = 900
  const H = 520
  const laidOut = useMemo(() => circleLayout(nodes, graph?.seed_paper_ids ?? [], W, H), [nodes, graph?.seed_paper_ids])
  const pos = useMemo(() => new Map(laidOut.map((n: LayoutNode) => [n.key, n])), [laidOut])

  const selected = selectedKey ? pos.get(selectedKey) : null
  const selectedEdges = useMemo(() => {
    if (!selectedKey) return []
    return (graph?.edges ?? []).filter((e: GraphEdge) => {
      const sk = `Paper:${e.source_paper_id}`
      const tk = e.target_type === 'Paper' ? `Paper:${e.target_id_or_name}` : `${e.target_type}:${e.target_id_or_name}`
      return sk === selectedKey || tk === selectedKey
    })
  }, [graph?.edges, selectedKey])

  return (
    <div className="page">
      <div className="controls">
        <div className="field">
          <label className="label">Question (GraphRAG)</label>
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
            {loading ? 'Running…' : 'Run GraphRAG'}
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
              <div className="cardTitle">Answer</div>
              <div className="metaRow">
                <div className="metaItem">edges {graph?.edges?.length ?? 0}</div>
                <div className="metaItem">seed papers {graph?.seed_paper_ids?.length ?? 0}</div>
                <div className="metaItem">related papers {graph?.related_paper_ids?.length ?? 0}</div>
              </div>
            </div>
            {resp.trace_id ? (
              <div className="metaRow subtle">
                <div className="metaItem">trace {resp.trace_id}</div>
                <button
                  className="btn"
                  style={{ padding: '6px 10px', borderRadius: 8, fontSize: 12 }}
                  onClick={() => copyToClipboard(resp.trace_id || '')}
                  type="button"
                >
                  Copy
                </button>
                <a href="https://smith.langchain.com/" target="_blank" rel="noreferrer">
                  Open LangSmith
                </a>
              </div>
            ) : null}
            <div className="answerBox">{resp.answer}</div>
          </div>

          {!graph?.edges?.length ? (
            <div className="card">
              <div className="cardHeader">
                <div className="cardTitle">Graph</div>
              </div>
              <div className="subtle">No edges were returned. This usually means the graph has not been built yet.</div>
            </div>
          ) : (
            <div className="grid2">
              <div className="card">
                <div className="cardHeader">
                  <div className="cardTitle">Subgraph</div>
                  <div className="metaRow">
                    <div className="metaItem">nodes {nodes.length}</div>
                    <div className="metaItem">edges {edges.length}</div>
                  </div>
                </div>
                <div className="svgWrap">
                  <svg width={W} height={H} className="graphSvg" role="img">
                    {edges.map((e, i) => {
                      const sk = `Paper:${e.source_paper_id}`
                      const tk = e.target_type === 'Paper' ? `Paper:${e.target_id_or_name}` : `${e.target_type}:${e.target_id_or_name}`
                      const s = pos.get(sk)
                      const t = pos.get(tk)
                      if (!s || !t) return null
                      const isSelected = selectedKey && (sk === selectedKey || tk === selectedKey)
                      return (
                        <line
                          key={i}
                          x1={s.x}
                          y1={s.y}
                          x2={t.x}
                          y2={t.y}
                          stroke={edgeColor(e.rel_type)}
                          strokeWidth={isSelected ? 2.5 : 1.3}
                          opacity={isSelected || !selectedKey ? 0.9 : 0.25}
                        />
                      )
                    })}
                    {laidOut.map((n) => {
                      const isSelected = selectedKey === n.key
                      const stroke = n.seed ? '#a78bfa' : '#64748b'
                      const fill = n.kind === 'Paper' ? '#0b1220' : '#111827'
                      return (
                        <g
                          key={n.key}
                          onClick={() => setSelectedKey(n.key)}
                          style={{ cursor: 'pointer' }}
                          opacity={isSelected || !selectedKey ? 1 : 0.6}
                        >
                          <circle cx={n.x} cy={n.y} r={isSelected ? 9 : 7} fill={fill} stroke={stroke} strokeWidth={2} />
                          <text x={n.x + 10} y={n.y + 4} fontSize={12} fill="#cbd5e1">
                            {n.label}
                          </text>
                        </g>
                      )
                    })}
                  </svg>
                </div>
                <div className="subtle">Click a node to highlight incident edges.</div>
              </div>

              <div className="card">
                <div className="cardHeader">
                  <div className="cardTitle">Selection</div>
                  <div className="metaRow">
                    <div className="metaItem">{selected ? selected.label : '—'}</div>
                    <div className="metaItem">{selected ? selected.kind : '—'}</div>
                    <div className="metaItem">edges {selectedEdges.length}</div>
                  </div>
                </div>
                {selected ? (
                  <>
                    <div className="monoList">
                      {selectedEdges.slice(0, 30).map((e, i) => (
                        <div key={i} className="monoLine">
                          {e.source_paper_id} -{e.rel_type}-&gt; {e.target_id_or_name}
                          {e.chunk_index != null ? ` [${e.source_paper_id}:${e.chunk_index}]` : ''}
                          {e.evidence ? ` | ${e.evidence}` : ''}
                        </div>
                      ))}
                    </div>
                  </>
                ) : (
                  <div className="subtle">Select a node to inspect its edges and evidence.</div>
                )}
              </div>
            </div>
          )}

          {graph?.edges?.length ? (
            <details className="details">
              <summary>Edges table ({graph.edges.length})</summary>
              <div className="tableWrap">
                <table className="table">
                  <thead>
                    <tr>
                      <th>source</th>
                      <th>rel</th>
                      <th>target</th>
                      <th>chunk</th>
                      <th>confidence</th>
                      <th>evidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {graph.edges.map((e, i) => (
                      <tr key={i}>
                        <td className="mono">{e.source_paper_id}</td>
                        <td>{e.rel_type}</td>
                        <td className="mono">{e.target_id_or_name}</td>
                        <td className="mono">{e.chunk_index ?? '—'}</td>
                        <td className="mono">{e.confidence == null ? '—' : e.confidence.toFixed(2)}</td>
                        <td className="cellWrap">{e.evidence ?? '—'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </details>
          ) : null}
        </div>
      )}
    </div>
  )
}
