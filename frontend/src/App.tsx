import './App.css'
import { useEffect, useMemo, useState } from 'react'
import { AgentTracePage } from './pages/AgentTrace'
import { EvaluationPage } from './pages/Evaluation'
import { GraphPage } from './pages/Graph'
import { QueryPage } from './pages/Query'

type PageKey = 'query' | 'agent-trace' | 'graph' | 'evaluation'

function parseHashPage(hash: string): PageKey {
  const h = (hash || '').replace(/^#/, '')
  const path = h.startsWith('/') ? h : `/${h}`
  if (path.startsWith('/agent-trace')) return 'agent-trace'
  if (path.startsWith('/graph')) return 'graph'
  if (path.startsWith('/evaluation')) return 'evaluation'
  return 'query'
}

function useHashPage(): PageKey {
  const [page, setPage] = useState<PageKey>(() => parseHashPage(window.location.hash))
  useEffect(() => {
    const onChange = () => setPage(parseHashPage(window.location.hash))
    window.addEventListener('hashchange', onChange)
    return () => window.removeEventListener('hashchange', onChange)
  }, [])
  return page
}

function App() {
  const page = useHashPage()
  const [backendStatus, setBackendStatus] = useState<'unknown' | 'up' | 'down'>('unknown')

  useEffect(() => {
    let cancelled = false
    const run = async () => {
      try {
        const res = await fetch('/api/health')
        if (!cancelled) setBackendStatus(res.ok ? 'up' : 'down')
      } catch {
        if (!cancelled) setBackendStatus('down')
      }
    }
    run()
    const id = window.setInterval(run, 5000)
    return () => {
      cancelled = true
      window.clearInterval(id)
    }
  }, [])

  const pageTitle = useMemo(() => {
    switch (page) {
      case 'agent-trace':
        return 'Agent Trace'
      case 'graph':
        return 'Graph'
      case 'evaluation':
        return 'Evaluation'
      default:
        return 'Query'
    }
  }, [page])

  return (
    <div className="appShell">
      <aside className="sidebar">
        <div className="sidebarHeader">
          <div className="brand">arXiv GraphRAG</div>
          <div className="backendStatus">
            <span className={`statusDot status-${backendStatus}`} />
            <span className="statusLabel">Backend {backendStatus}</span>
          </div>
        </div>
        <nav className="nav">
          <a className={page === 'query' ? 'navLink active' : 'navLink'} href="#/query">
            Query
          </a>
          <a className={page === 'agent-trace' ? 'navLink active' : 'navLink'} href="#/agent-trace">
            AgentTrace
          </a>
          <a className={page === 'graph' ? 'navLink active' : 'navLink'} href="#/graph">
            Graph
          </a>
          <a className={page === 'evaluation' ? 'navLink active' : 'navLink'} href="#/evaluation">
            Evaluation
          </a>
        </nav>
      </aside>
      <main className="main">
        <div className="pageHeader">
          <h1 className="pageTitle">{pageTitle}</h1>
        </div>
        {page === 'query' && <QueryPage />}
        {page === 'agent-trace' && <AgentTracePage />}
        {page === 'graph' && <GraphPage />}
        {page === 'evaluation' && <EvaluationPage />}
      </main>
    </div>
  )
}

export default App
