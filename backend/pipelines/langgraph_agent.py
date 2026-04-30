import time
from dataclasses import dataclass
import re

from backend.graph_store.graph_retriever import GraphRetriever
from backend.pipelines.rag_pipeline import RAGPipeline
from backend.schemas.agent import AgentPlan, AgentQueryResponse, AgentRetrievedContext, AgentTraceItem
from backend.schemas.query import Citation, RetrievedContext, UsageMetrics
from backend.storage.chunk_store import ChunkStore
from backend.vector_store.pinecone_client import PineconeIndex
from backend.vector_store.retriever import VectorRetriever

try:
    from langgraph.graph import END, StateGraph
except Exception:
    END = None
    StateGraph = None


@dataclass(frozen=True)
class AgentDecision:
    query_type: str
    route: str
    confidence: float
    reason: str


class LangGraphAgentPipeline:
    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        pinecone_index: PineconeIndex,
        namespace: str,
        rag: RAGPipeline,
        graph_retriever: GraphRetriever | None,
        chunk_store: ChunkStore,
    ) -> None:
        if StateGraph is None:
            raise RuntimeError("LangGraph is not installed. Install with: pip install langgraph")
        self._vector_retriever = vector_retriever
        self._index = pinecone_index
        self._namespace = namespace
        self._rag = rag
        self._graph = graph_retriever
        self._chunk_store = chunk_store
        self._app = self._build_graph()

    def run(self, *, question: str, top_k: int) -> AgentQueryResponse:
        start_ts = time.perf_counter()
        state = {
            "question": question,
            "top_k": int(top_k),
            "start_ts": start_ts,
            "trace": [],
            "decision": None,
            "plan": None,
            "prompt_style": "concise_factual",
            "vector_scored": [],
            "vector_contexts": [],
            "seed_papers": [],
            "graph_edges": [],
            "contested_claims": [],
            "graph_contexts": [],
            "extra_contexts": [],
            "merged_contexts": [],
            "agent_contexts": [],
            "answer": "",
            "usage": {},
        }
        out = self._app.invoke(state)
        decision: AgentDecision = out["decision"]
        plan: AgentPlan | None = out.get("plan")
        usage = out.get("usage") or {}
        metrics = UsageMetrics(
            latency_ms=int(out.get("latency_ms") or 0),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            total_cost_usd=usage.get("total_cost_usd"),
        )
        return AgentQueryResponse(
            query_type=decision.query_type,
            route=decision.route,
            confidence=float(decision.confidence),
            routing_reason=str(decision.reason or ""),
            plan=plan,
            answer=str(out.get("answer") or ""),
            contexts=out.get("agent_contexts") or [],
            contested_claims=out.get("contested_claims") or [],
            graph_edges=out.get("graph_edges") or [],
            metrics=metrics,
            agent_trace=out.get("trace") or [],
        )

    def _build_graph(self):
        graph = StateGraph(dict)
        graph.add_node("classify", self._node_classify)
        graph.add_node("plan", self._node_plan)
        graph.add_node("vector_retrieve", self._node_vector_retrieve)
        graph.add_node("graph_retrieve", self._node_graph_retrieve)
        graph.add_node("merge", self._node_merge)
        graph.add_node("synthesize", self._node_synthesize)

        graph.set_entry_point("classify")
        def _route_from_classify(state: dict) -> str:
            decision = state.get("decision")
            if not isinstance(decision, AgentDecision):
                return "vector_retrieve"
            if decision.route == "hybrid_parallel":
                return "plan"
            return "vector_retrieve"

        graph.add_conditional_edges("classify", _route_from_classify, {"plan": "plan", "vector_retrieve": "vector_retrieve"})

        graph.add_edge("plan", "vector_retrieve")

        def _route_from_vector(state: dict) -> str:
            decision = state.get("decision")
            plan = state.get("plan")
            if not isinstance(decision, AgentDecision):
                return "merge"
            if decision.route == "hybrid_parallel" and isinstance(plan, AgentPlan) and plan.run_graph and not state.get("graph_unavailable"):
                return "graph_retrieve"
            return "merge"

        graph.add_conditional_edges("vector_retrieve", _route_from_vector, {"merge": "merge", "graph_retrieve": "graph_retrieve"})
        graph.add_edge("graph_retrieve", "merge")
        graph.add_edge("merge", "synthesize")
        graph.add_edge("synthesize", END)
        return graph.compile()

    def _node_classify(self, state: dict) -> dict:
        out = dict(state)
        t0 = time.perf_counter()
        q = str(state.get("question") or "")
        decision = self._classify(question=q, graph_available=self._graph is not None)
        trace = list(state.get("trace") or [])
        trace.append(
            AgentTraceItem(
                agent="QueryClassifier",
                decision=f"{decision.route} ({decision.query_type}) | {decision.reason}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
        out["decision"] = decision
        out["trace"] = trace
        return out

    def _node_plan(self, state: dict) -> dict:
        out = dict(state)
        t0 = time.perf_counter()
        decision = out.get("decision")
        q = str(out.get("question") or "")
        plan = self._plan(question=q, decision=decision, graph_available=self._graph is not None)
        trace = list(out.get("trace") or [])
        trace.append(
            AgentTraceItem(
                agent="Planner",
                decision=f"sub_questions={len(plan.sub_questions)} entities={sorted(plan.entities.keys())} run_vector={plan.run_vector} run_graph={plan.run_graph} style={plan.prompt_style}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
        out["plan"] = plan
        out["prompt_style"] = plan.prompt_style
        out["trace"] = trace
        return out

    def _node_vector_retrieve(self, state: dict) -> dict:
        out = dict(state)
        t0 = time.perf_counter()
        question = str(state.get("question") or "")
        top_k = int(state.get("top_k") or 8)
        scored = self._vector_retriever.retrieve_scored(question=question, top_k=top_k)
        vector_contexts = [c for c, _ in scored]
        seed_papers = sorted({c.citation.paper_id for c in vector_contexts if c.citation.paper_id})
        trace = list(state.get("trace") or [])
        trace.append(
            AgentTraceItem(
                agent="VectorRetriever",
                decision=f"top_k={top_k} seed_papers={len(seed_papers)} contexts={len(vector_contexts)}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
        graph_unavailable = self._graph is None
        out["vector_scored"] = scored
        out["vector_contexts"] = vector_contexts
        out["seed_papers"] = seed_papers
        out["trace"] = trace
        out["graph_unavailable"] = graph_unavailable
        return out

    def _node_graph_retrieve(self, state: dict) -> dict:
        out = dict(state)
        t0 = time.perf_counter()
        trace = list(state.get("trace") or [])
        if self._graph is None:
            trace.append(
                AgentTraceItem(
                    agent="GraphRetriever",
                    decision="skipped (unavailable)",
                    latency_ms=int((time.perf_counter() - t0) * 1000),
                )
            )
            out["trace"] = trace
            return out

        seed_papers = list(state.get("seed_papers") or [])
        g = self._graph.expand_from_seed_papers(seed_paper_ids=seed_papers, limit_edges=80)
        graph_edges = self._graph.edges_to_lines(g.edges, limit=40)
        contested_claims = g.contested_claims
        graph_contexts = g.contexts

        extra_contexts: list[RetrievedContext] = []
        if g.related_paper_ids:
            question = str(state.get("question") or "")
            top_k = int(state.get("top_k") or 8)
            qvec = self._vector_retriever.embed_query(question=question)
            matches = self._index.query(
                vector=qvec,
                top_k=min(12, max(4, top_k)),
                namespace=self._namespace,
                filter={"paper_id": {"$in": g.related_paper_ids[:12]}},
            )
            for m in matches:
                md = m.metadata or {}
                paper_id = str(md.get("paper_id", "")).strip()
                try:
                    chunk_index = int(md.get("chunk_index"))
                except Exception:
                    continue
                if not paper_id:
                    continue
                try:
                    chunk = self._chunk_store.read_chunk(paper_id=paper_id, chunk_index=chunk_index)
                except Exception:
                    continue
                extra_contexts.append(
                    RetrievedContext(
                        text=chunk.text,
                        citation=Citation(paper_id=paper_id, title=md.get("title"), chunk_index=chunk_index),
                    )
                )

        trace.append(
            AgentTraceItem(
                agent="GraphRetriever",
                decision=f"edges={len(g.edges)} graph_contexts={len(graph_contexts)} extra_contexts={len(extra_contexts)}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
        out["graph_edges"] = graph_edges
        out["contested_claims"] = contested_claims
        out["graph_contexts"] = graph_contexts
        out["extra_contexts"] = extra_contexts
        out["trace"] = trace
        return out

    def _node_merge(self, state: dict) -> dict:
        out = dict(state)
        t0 = time.perf_counter()
        merged = self._dedupe_contexts(
            (state.get("vector_contexts") or []) + (state.get("graph_contexts") or []) + (state.get("extra_contexts") or [])
        )
        agent_contexts = self._to_agent_contexts(
            vector_scored=state.get("vector_scored") or [],
            graph_contexts=state.get("graph_contexts") or [],
            extra_contexts=state.get("extra_contexts") or [],
        )
        trace = list(state.get("trace") or [])
        trace.append(
            AgentTraceItem(
                agent="EvidenceMerger",
                decision=f"merged_contexts={len(agent_contexts)} contested={len(state.get('contested_claims') or [])} edges={len(state.get('graph_edges') or [])}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        )
        out["merged_contexts"] = merged
        out["agent_contexts"] = agent_contexts
        out["trace"] = trace
        return out

    def _node_synthesize(self, state: dict) -> dict:
        out = dict(state)
        t0 = time.perf_counter()
        question = str(state.get("question") or "")
        merged: list[RetrievedContext] = state.get("merged_contexts") or []
        decision: AgentDecision = state["decision"]
        system, user = self._build_prompt(
            question=question,
            contexts=merged,
            graph_edges=state.get("graph_edges") or [],
            contested_claims=state.get("contested_claims") or [],
            route=decision.route,
            prompt_style=str(state.get("prompt_style") or "concise_factual"),
        )
        answer, usage = self._rag.synthesize_with_prompt(system=system, user=user)
        trace = list(state.get("trace") or [])
        trace.append(
            AgentTraceItem(
                agent="Synthesizer",
                decision=decision.route,
                latency_ms=int((time.perf_counter() - t0) * 1000),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                total_cost_usd=usage.get("total_cost_usd"),
            )
        )
        latency_ms = int((time.perf_counter() - float(state.get("start_ts") or time.perf_counter())) * 1000)
        out["answer"] = answer
        out["usage"] = usage
        out["trace"] = trace
        out["latency_ms"] = latency_ms
        return out

    def _classify(self, *, question: str, graph_available: bool) -> AgentDecision:
        q = (question or "").lower()
        summarization = any(k in q for k in ["summarize", "summary", "overview", "key takeaways", "high level"])
        entity_centric = any(k in q for k in ["who is", "what is", "define", "definition of", "arxiv", "paper "])
        relational = any(k in q for k in ["relationship", "connection", "influence", "cite", "cites", "contradict", "challeng", "co-author"])
        multi_hop = any(k in q for k in ["how does", "how are", "compare", "contrast", "chain", "influence chain"])

        score = 0.2
        if summarization:
            score += 0.35
        if entity_centric:
            score += 0.25
        if relational:
            score += 0.35
        if multi_hop:
            score += 0.25
        confidence = max(0.0, min(1.0, score))

        if relational or multi_hop:
            if graph_available:
                return AgentDecision(query_type=("multi_hop" if multi_hop else "relational"), route="hybrid_parallel", confidence=confidence, reason="relational/multi-hop signal")
            return AgentDecision(query_type=("multi_hop" if multi_hop else "relational"), route="vector_only", confidence=confidence, reason="graph unavailable")
        if summarization:
            return AgentDecision(query_type="summarization", route="vector_only", confidence=confidence, reason="summarization signal")
        if entity_centric:
            return AgentDecision(query_type="entity_centric", route="vector_only", confidence=confidence, reason="entity-centric signal")
        return AgentDecision(query_type="simple_factual", route="vector_only", confidence=confidence, reason="default")

    def _dedupe_contexts(self, contexts: list[RetrievedContext]) -> list[RetrievedContext]:
        seen: set[tuple[str, int | None]] = set()
        out: list[RetrievedContext] = []
        for c in contexts:
            key = (c.citation.paper_id, c.citation.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(c)
        return out

    def _build_prompt(
        self,
        *,
        question: str,
        contexts: list[RetrievedContext],
        graph_edges: list[str],
        contested_claims: list[str],
        route: str,
        prompt_style: str,
    ) -> tuple[str, str]:
        context_block = "\n\n".join(
            [
                f"Source: {c.citation.paper_id}:{c.citation.chunk_index}\nTitle: {c.citation.title or ''}\nText:\n{c.text}"
                for c in contexts
            ]
        )
        base_system = (
            "You answer questions using only the provided sources. "
            "If the sources are insufficient, say what is missing. "
            "Cite claims with brackets like [paper_id:chunk_index]. "
            "Do not ask follow-up questions. Do not add extra sections like 'Do you want me to...'. "
            "Every paragraph must include at least one citation."
        )

        style = (prompt_style or "concise_factual").strip().lower()
        if style == "comparison_table":
            style_instr = "Return a compact comparison table (Markdown) with rows as methods/papers and columns as key differences."
        elif style == "literature_synthesis":
            style_instr = "Organize the answer as a literature synthesis with themes and cite supporting papers for each theme."
        elif style == "contested_findings":
            style_instr = "Explicitly surface contested findings and summarize the disagreement, citing both sides."
        else:
            style_instr = "Be concise and direct."

        if route == "vector_only" or not graph_edges:
            system = f"{base_system} {style_instr}"
            user = f"Question:\n{question}\n\nSources:\n{context_block}"
            return system, user

        edges_block = "\n".join(graph_edges[:30])
        contested_block = "\n".join(contested_claims[:10])
        system = (
            "You answer questions using only the provided sources. "
            "You are given BOTH text passages and graph relationships extracted from those passages. "
            "Use graph relationships to organize the answer into clusters or influence chains. "
            "If CONTRADICTS appears, surface contested findings explicitly. "
            "Cite claims with brackets like [paper_id:chunk_index]. "
            "Do not ask follow-up questions. Do not add extra sections like 'Do you want me to...'. "
            "Every paragraph must include at least one citation. "
            + style_instr
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Graph relationships:\n{edges_block}\n\n"
            f"Contested claims:\n{contested_block}\n\n"
            f"Text sources:\n{context_block}"
        )
        return system, user

    def _plan(self, *, question: str, decision: object, graph_available: bool) -> AgentPlan:
        q = (question or "").strip()
        ql = q.lower()
        if any(k in ql for k in ["compare", "contrast", "vs", "versus"]):
            style = "comparison_table"
        elif any(k in ql for k in ["contradict", "challeng", "contested"]):
            style = "contested_findings"
        elif isinstance(decision, AgentDecision) and decision.query_type == "summarization":
            style = "literature_synthesis"
        else:
            style = "concise_factual"

        paper_ids = re.findall(r"\b\d{4}\.\d{5}(?:v\d+)?\b", q)
        entities: dict[str, list[str]] = {}
        if paper_ids:
            entities["paper_ids"] = sorted(set(paper_ids))
        kws = [w for w in re.split(r"\W+", q) if len(w) >= 8]
        if kws:
            entities["keywords"] = kws[:10]

        sub: list[str] = []
        if paper_ids and any(k in ql for k in ["cite", "contradict", "influence chain", "influence"]):
            for pid in paper_ids[:2]:
                sub.append(f"What papers cite {pid}?")
                sub.append(f"What papers contradict {pid}?")
        if not sub and q:
            sub.append(q)

        run_graph = isinstance(decision, AgentDecision) and decision.route == "hybrid_parallel" and graph_available
        return AgentPlan(sub_questions=sub, entities=entities, run_vector=True, run_graph=run_graph, prompt_style=style)

    def _to_agent_contexts(
        self,
        *,
        vector_scored: list[tuple[RetrievedContext, float]],
        graph_contexts: list[RetrievedContext],
        extra_contexts: list[RetrievedContext],
    ) -> list[AgentRetrievedContext]:
        out: list[AgentRetrievedContext] = []
        seen: set[tuple[str, int | None]] = set()
        for ctx, score in vector_scored:
            key = (ctx.citation.paper_id, ctx.citation.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(AgentRetrievedContext(text=ctx.text, citation=ctx.citation, score=float(score), source="vector"))
        for ctx in graph_contexts:
            key = (ctx.citation.paper_id, ctx.citation.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(AgentRetrievedContext(text=ctx.text, citation=ctx.citation, score=None, source="graph"))
        for ctx in extra_contexts:
            key = (ctx.citation.paper_id, ctx.citation.chunk_index)
            if key in seen:
                continue
            seen.add(key)
            out.append(AgentRetrievedContext(text=ctx.text, citation=ctx.citation, score=None, source="vector_filtered"))
        return out
