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


@dataclass(frozen=True)
class AgentDecision:
    query_type: str
    route: str
    confidence: float
    reason: str


class AgentPipeline:
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
        self._vector_retriever = vector_retriever
        self._index = pinecone_index
        self._namespace = namespace
        self._rag = rag
        self._graph = graph_retriever
        self._chunk_store = chunk_store

    def run(self, *, question: str, top_k: int) -> AgentQueryResponse:
        start = time.perf_counter()
        trace: list[AgentTraceItem] = []

        t0 = time.perf_counter()
        decision = self._classify(question=question, graph_available=self._graph is not None)
        trace.append(
            AgentTraceItem(
                agent="QueryClassifier",
                decision=f"{decision.route} ({decision.query_type}) confidence={decision.confidence:.2f} | {decision.reason}",
                latency_ms=int((time.perf_counter() - t0) * 1000),
            )
        )

        t1 = time.perf_counter()
        scored = self._vector_retriever.retrieve_scored(question=question, top_k=top_k)
        vector_contexts = [c for c, _ in scored]
        seed_papers = sorted({c.citation.paper_id for c in vector_contexts if c.citation.paper_id})
        trace.append(
            AgentTraceItem(
                agent="VectorRetriever",
                decision=f"top_k={top_k} seed_papers={len(seed_papers)} contexts={len(vector_contexts)}",
                latency_ms=int((time.perf_counter() - t1) * 1000),
            )
        )

        graph_edges: list[str] = []
        contested_claims: list[str] = []
        graph_contexts: list[RetrievedContext] = []
        extra_contexts: list[RetrievedContext] = []

        if decision.route in {"graph_only", "hybrid_parallel"} and self._graph is not None:
            t2 = time.perf_counter()
            g = self._graph.expand_from_seed_papers(seed_paper_ids=seed_papers, limit_edges=80)
            graph_contexts = g.contexts
            graph_edges = self._graph.edges_to_lines(g.edges, limit=40)
            contested_claims = g.contested_claims
            trace.append(
                AgentTraceItem(
                    agent="GraphRetriever",
                    decision=f"edges={len(g.edges)} graph_contexts={len(graph_contexts)} related_papers={len(g.related_paper_ids)}",
                    latency_ms=int((time.perf_counter() - t2) * 1000),
                )
            )

            if g.related_paper_ids:
                t3 = time.perf_counter()
                qvec = self._vector_retriever.embed_query(question=question)
                matches = self._index.query(
                    vector=qvec,
                    top_k=min(12, max(4, top_k)),
                    namespace=self._namespace,
                    filter={"paper_id": {"$in": g.related_paper_ids[:12]}},
                )
                for m in matches:
                    md = m.metadata or {}
                    pid = str(md.get("paper_id", "")).strip()
                    try:
                        ci = int(md.get("chunk_index"))
                    except Exception:
                        continue
                    if not pid:
                        continue
                    try:
                        chunk = self._chunk_store.read_chunk(paper_id=pid, chunk_index=ci)
                    except Exception:
                        continue
                    extra_contexts.append(
                        RetrievedContext(
                            text=chunk.text,
                            citation=Citation(paper_id=pid, title=md.get("title"), chunk_index=ci),
                        )
                    )
                trace.append(
                    AgentTraceItem(
                        agent="VectorRetrieverFiltered",
                        decision=f"papers={min(12, len(g.related_paper_ids))} contexts={len(extra_contexts)}",
                        latency_ms=int((time.perf_counter() - t3) * 1000),
                    )
                )

        t4 = time.perf_counter()
        merged_contexts = self._dedupe_contexts(vector_contexts + graph_contexts + extra_contexts)
        agent_contexts = self._to_agent_contexts(
            vector_scored=scored,
            graph_contexts=graph_contexts,
            extra_contexts=extra_contexts,
        )
        trace.append(
            AgentTraceItem(
                agent="EvidenceMerger",
                decision=f"merged_contexts={len(agent_contexts)} contested={len(contested_claims)} edges={len(graph_edges)}",
                latency_ms=int((time.perf_counter() - t4) * 1000),
            )
        )

        t5 = time.perf_counter()
        plan = self._plan(question=question, decision=decision, graph_available=self._graph is not None)
        system, user = self._build_prompt(
            question=question,
            contexts=merged_contexts,
            graph_edges=graph_edges,
            contested_claims=contested_claims,
            route=decision.route,
            prompt_style=plan.prompt_style,
        )
        answer, usage = self._rag.synthesize_with_prompt(system=system, user=user)
        trace.append(
            AgentTraceItem(
                agent="Synthesizer",
                decision=f"{decision.route} style={plan.prompt_style}",
                latency_ms=int((time.perf_counter() - t5) * 1000),
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                total_cost_usd=usage.get("total_cost_usd"),
            )
        )

        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics = UsageMetrics(
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            total_cost_usd=usage.get("total_cost_usd"),
        )
        return AgentQueryResponse(
            query_type=decision.query_type,
            route=decision.route,
            confidence=decision.confidence,
            routing_reason=decision.reason,
            plan=plan,
            answer=answer,
            contexts=agent_contexts,
            contested_claims=contested_claims,
            graph_edges=graph_edges,
            metrics=metrics,
            agent_trace=trace,
        )

    def _classify(self, *, question: str, graph_available: bool) -> AgentDecision:
        q = (question or "").lower()
        summarization = any(k in q for k in ["summarize", "summary", "overview", "key takeaways", "high level"])
        entity_centric = any(k in q for k in ["who is", "what is", "define", "definition of", "paper ", "arxiv"])
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

    def _plan(self, *, question: str, decision: AgentDecision, graph_available: bool) -> AgentPlan:
        q = (question or "").strip()
        ql = q.lower()
        if any(k in ql for k in ["compare", "contrast", "vs", "versus"]):
            style = "comparison_table"
        elif any(k in ql for k in ["contradict", "challeng", "contested"]):
            style = "contested_findings"
        elif decision.query_type == "summarization":
            style = "literature_synthesis"
        else:
            style = "concise_factual"

        run_graph = decision.route in {"graph_only", "hybrid_parallel"} and graph_available
        entities: dict[str, list[str]] = {}
        entities["paper_ids"] = []
        entities["keywords"] = [w for w in re.split(r"\W+", q) if len(w) >= 8][:8]
        sub = [q] if q else []
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
            out.append(AgentRetrievedContext(text=ctx.text, citation=ctx.citation, score=score, source="vector"))
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
