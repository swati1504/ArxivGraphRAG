import time
from dataclasses import dataclass

from backend.graph_store.neo4j_client import Neo4jClient
from backend.schemas.query import Citation, RetrievedContext
from backend.storage.chunk_store import ChunkStore


@dataclass(frozen=True)
class GraphEdge:
    source_paper_id: str
    rel_type: str
    target_type: str
    target_id_or_name: str
    chunk_index: int | None
    evidence: str | None
    confidence: float | None


@dataclass(frozen=True)
class GraphRetrieveResult:
    edges: list[GraphEdge]
    contexts: list[RetrievedContext]
    contested_claims: list[str]
    related_paper_ids: list[str]
    latency_ms: int


class GraphRetriever:
    def __init__(self, *, neo4j: Neo4jClient, chunk_store: ChunkStore) -> None:
        self._neo4j = neo4j
        self._chunk_store = chunk_store

    def expand_from_seed_papers(self, *, seed_paper_ids: list[str], limit_edges: int = 80) -> GraphRetrieveResult:
        start = time.perf_counter()
        seed = [p for p in seed_paper_ids if str(p).strip()]
        if not seed:
            return GraphRetrieveResult(
                edges=[],
                contexts=[],
                contested_claims=[],
                related_paper_ids=[],
                latency_ms=int((time.perf_counter() - start) * 1000),
            )

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
            {"paper_ids": seed, "limit": int(limit_edges)},
        )

        edges: list[GraphEdge] = []
        for row in rows:
            try:
                edges.append(
                    GraphEdge(
                        source_paper_id=str(row.get("source_paper_id") or "").strip(),
                        rel_type=str(row.get("rel_type") or "").strip().upper(),
                        target_type=str(row.get("target_type") or "").strip(),
                        target_id_or_name=str(row.get("target_id_or_name") or "").strip(),
                        chunk_index=int(row["chunk_index"]) if row.get("chunk_index") is not None else None,
                        evidence=str(row.get("evidence") or "").strip() or None,
                        confidence=float(row["confidence"]) if row.get("confidence") is not None else None,
                    )
                )
            except Exception:
                continue

        contexts = self._contexts_from_edges(edges)
        contested = self._contested_claims(edges)
        related = self._related_papers(seed_paper_ids=seed, edges=edges)
        latency_ms = int((time.perf_counter() - start) * 1000)
        return GraphRetrieveResult(
            edges=edges,
            contexts=contexts,
            contested_claims=contested,
            related_paper_ids=related,
            latency_ms=latency_ms,
        )

    def _contexts_from_edges(self, edges: list[GraphEdge]) -> list[RetrievedContext]:
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
                    citation=Citation(paper_id=e.source_paper_id, title=None, chunk_index=int(e.chunk_index)),
                )
            )
        return out

    def _related_papers(self, *, seed_paper_ids: list[str], edges: list[GraphEdge]) -> list[str]:
        seeds = set(seed_paper_ids)
        out: set[str] = set()
        for e in edges:
            if e.target_type == "Paper" and e.target_id_or_name and e.target_id_or_name not in seeds:
                out.add(e.target_id_or_name)
            if e.source_paper_id and e.source_paper_id not in seeds:
                if e.rel_type in {"CITES", "CONTRADICTS"}:
                    out.add(e.source_paper_id)
        return sorted(out)

    def _contested_claims(self, edges: list[GraphEdge]) -> list[str]:
        out: list[str] = []
        for e in edges:
            if e.rel_type != "CONTRADICTS":
                continue
            cite = f"[{e.source_paper_id}:{e.chunk_index}]" if e.chunk_index is not None else ""
            if e.evidence:
                out.append(f"{e.source_paper_id} CONTRADICTS {e.target_id_or_name} {cite} evidence: {e.evidence}")
            else:
                out.append(f"{e.source_paper_id} CONTRADICTS {e.target_id_or_name} {cite}".strip())
        return out[:10]

    def edges_to_lines(self, edges: list[GraphEdge], *, limit: int = 40) -> list[str]:
        lines: list[str] = []
        for e in edges[:limit]:
            cite = f"[{e.source_paper_id}:{e.chunk_index}]" if e.chunk_index is not None else ""
            base = f"{e.source_paper_id} -{e.rel_type}-> {e.target_id_or_name} {cite}".strip()
            if e.evidence:
                base += f" | {e.evidence}"
            lines.append(base)
        return lines

