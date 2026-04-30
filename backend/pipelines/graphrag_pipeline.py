import time
from dataclasses import dataclass

from backend.graph_store.neo4j_client import Neo4jClient
from backend.schemas.query import Citation, QueryResponse, RetrievedContext, UsageMetrics
from backend.storage.chunk_store import ChunkStore
from backend.vector_store.pinecone_client import PineconeIndex
from backend.vector_store.retriever import VectorRetriever
from backend.pipelines.rag_pipeline import RAGPipeline


@dataclass(frozen=True)
class GraphEdgeEvidence:
    source_paper_id: str
    rel_type: str
    target_type: str
    target_id_or_name: str
    chunk_index: int | None
    evidence: str | None
    confidence: float | None


class GraphRAGPipeline:
    def __init__(
        self,
        *,
        vector_retriever: VectorRetriever,
        neo4j: Neo4jClient,
        pinecone_index: PineconeIndex,
        namespace: str,
        chunk_store: ChunkStore,
    ) -> None:
        self._vector_retriever = vector_retriever
        self._neo4j = neo4j
        self._index = pinecone_index
        self._namespace = namespace
        self._chunk_store = chunk_store
        self._rag = RAGPipeline(retriever=vector_retriever)

    def run(self, *, question: str, top_k: int) -> QueryResponse:
        start = time.perf_counter()

        seed_contexts = self._vector_retriever.retrieve(question=question, top_k=top_k)
        seed_papers = sorted({c.citation.paper_id for c in seed_contexts if c.citation.paper_id})

        edges = self._fetch_graph_edges(seed_papers=seed_papers, limit=60)
        related_papers = sorted(self._related_papers(seed_papers=seed_papers, edges=edges))[:12]

        graph_contexts = self._contexts_from_edges(edges)

        extra_contexts: list[RetrievedContext] = []
        if related_papers:
            qvec = self._vector_retriever.embed_query(question=question)
            matches = self._index.query(
                vector=qvec,
                top_k=min(12, max(4, top_k)),
                namespace=self._namespace,
                filter={"paper_id": {"$in": related_papers}},
            )
            for m in matches:
                md = m.metadata or {}
                paper_id = str(md.get("paper_id", "")).strip()
                chunk_index = md.get("chunk_index")
                try:
                    chunk_i = int(chunk_index)
                except Exception:
                    continue
                if not paper_id:
                    continue
                try:
                    chunk = self._chunk_store.read_chunk(paper_id=paper_id, chunk_index=chunk_i)
                except Exception:
                    continue
                extra_contexts.append(
                    RetrievedContext(
                        text=chunk.text,
                        citation=Citation(paper_id=paper_id, title=md.get("title"), chunk_index=chunk_i),
                    )
                )

        merged_contexts = self._dedupe_contexts(seed_contexts + graph_contexts + extra_contexts)

        if not merged_contexts:
            return self._rag.run_with_contexts(question=question, contexts=[], start=start)

        system, user = self._build_graphrag_prompt(question=question, contexts=merged_contexts, edges=edges)
        answer, usage = self._rag.synthesize_with_prompt(system=system, user=user)
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics = UsageMetrics(
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            total_cost_usd=usage.get("total_cost_usd"),
        )
        return QueryResponse(answer=answer, contexts=merged_contexts, metrics=metrics)

    def _fetch_graph_edges(self, *, seed_papers: list[str], limit: int) -> list[GraphEdgeEvidence]:
        if not seed_papers:
            return []
        rows = self._neo4j.run_read(
            """
            UNWIND $paper_ids AS pid
            MATCH (p:Paper {id: pid})-[r]->(t)
            WHERE type(r) IN ['PROPOSES','STUDIES','CITES','CONTRADICTS']
            RETURN p.id AS source_paper_id,
                   type(r) AS rel_type,
                   labels(t)[0] AS target_type,
                   coalesce(t.id, t.name, t.key) AS target_id_or_name,
                   r.chunk_index AS chunk_index,
                   r.evidence AS evidence,
                   r.confidence AS confidence
            UNION
            UNWIND $paper_ids AS pid
            MATCH (s:Paper)-[r]->(p:Paper {id: pid})
            WHERE type(r) IN ['CITES','CONTRADICTS']
            RETURN s.id AS source_paper_id,
                   type(r) AS rel_type,
                   'Paper' AS target_type,
                   p.id AS target_id_or_name,
                   r.chunk_index AS chunk_index,
                   r.evidence AS evidence,
                   r.confidence AS confidence
            LIMIT $limit
            """,
            {"paper_ids": seed_papers, "limit": int(limit)},
        )
        out: list[GraphEdgeEvidence] = []
        for row in rows:
            try:
                out.append(
                    GraphEdgeEvidence(
                        source_paper_id=str(row.get("source_paper_id") or ""),
                        rel_type=str(row.get("rel_type") or ""),
                        target_type=str(row.get("target_type") or ""),
                        target_id_or_name=str(row.get("target_id_or_name") or ""),
                        chunk_index=int(row["chunk_index"]) if row.get("chunk_index") is not None else None,
                        evidence=str(row.get("evidence") or "").strip() or None,
                        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
                    )
                )
            except Exception:
                continue
        return out

    def _related_papers(self, *, seed_papers: list[str], edges: list[GraphEdgeEvidence]) -> set[str]:
        seeds = set(seed_papers)
        out: set[str] = set()
        for e in edges:
            if e.rel_type in {"CITES", "CONTRADICTS"} and e.target_type == "Paper":
                if e.source_paper_id and e.source_paper_id not in seeds:
                    out.add(e.source_paper_id)
                if e.target_id_or_name and e.target_id_or_name not in seeds:
                    out.add(e.target_id_or_name)
        return out

    def _build_graphrag_prompt(
        self,
        *,
        question: str,
        contexts: list[RetrievedContext],
        edges: list[GraphEdgeEvidence],
    ) -> tuple[str, str]:
        context_block = "\n\n".join(
            [
                f"Source: {c.citation.paper_id}:{c.citation.chunk_index}\nTitle: {c.citation.title or ''}\nText:\n{c.text}"
                for c in contexts
            ]
        )
        edge_lines = []
        for e in edges[:30]:
            cite = f"[{e.source_paper_id}:{e.chunk_index}]" if (e.source_paper_id and e.chunk_index is not None) else ""
            edge_lines.append(
                f"{e.source_paper_id} -{e.rel_type}-> {e.target_id_or_name} {cite} "
                + (f"evidence: {e.evidence}" if e.evidence else "")
            )
        edges_block = "\n".join(edge_lines).strip()

        system = (
            "You answer questions using only the provided sources. "
            "You are given BOTH text passages and graph relationships extracted from those passages. "
            "Use graph relationships to organize the answer into clusters (methods, concepts, influence chains). "
            "If any CONTRADICTS relationships appear, surface them explicitly as contested findings. "
            "Cite claims with brackets like [paper_id:chunk_index]. "
            "Do not ask follow-up questions. Do not add extra sections like 'Do you want me to...'. "
            "Every paragraph must include at least one citation."
        )
        user = (
            f"Question:\n{question}\n\n"
            f"Graph relationships (structured signals):\n{edges_block}\n\n"
            f"Text sources:\n{context_block}"
        )
        return system, user

    def _contexts_from_edges(self, edges: list[GraphEdgeEvidence]) -> list[RetrievedContext]:
        out: list[RetrievedContext] = []
        for e in edges:
            if not e.source_paper_id or e.chunk_index is None:
                continue
            try:
                chunk = self._chunk_store.read_chunk(paper_id=e.source_paper_id, chunk_index=int(e.chunk_index))
            except Exception:
                continue
            out.append(
                RetrievedContext(
                    text=chunk.text,
                    citation=Citation(
                        paper_id=e.source_paper_id,
                        title=None,
                        chunk_index=int(e.chunk_index),
                    ),
                )
            )
        return out

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
