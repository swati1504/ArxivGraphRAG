from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

from backend.graph_store.neo4j_client import Neo4jClient


NodeType = Literal["Paper", "Author", "Institution", "Concept", "Method"]
RelType = Literal["AUTHORED", "AFFILIATED_WITH", "STUDIES", "PROPOSES", "CITES", "CONTRADICTS"]


def _normalize_key(value: str) -> str:
    v = (value or "").strip().lower()
    v = re.sub(r"\s+", " ", v)
    v = re.sub(r"[^\w\s\-:.]", "", v)
    return v.strip()


def _normalize_author_key(name: str) -> str:
    v = (name or "").strip()
    v = v.replace(".", " ")
    if "," in v:
        parts = [p.strip() for p in v.split(",", 1)]
        if len(parts) == 2 and parts[0] and parts[1]:
            v = f"{parts[1]} {parts[0]}"
    return _normalize_key(v)


@dataclass(frozen=True)
class NodeRef:
    type: NodeType
    id: str | None = None
    name: str | None = None

    def key(self) -> str:
        if self.type == "Paper":
            if not self.id:
                raise ValueError("Paper nodes require id.")
            return str(self.id)
        if not self.name:
            raise ValueError(f"{self.type} nodes require name.")
        if self.type == "Author":
            return _normalize_author_key(self.name)
        return _normalize_key(self.name)


@dataclass(frozen=True)
class ExtractedRelation:
    source: NodeRef
    type: RelType
    target: NodeRef
    chunk_index: int
    evidence: str | None = None
    confidence: float | None = None


class GraphWriter:
    def __init__(self, *, client: Neo4jClient) -> None:
        self._client = client

    def ensure_schema(self) -> None:
        stmts: list[tuple[str, dict[str, Any]]] = [
            ("CREATE CONSTRAINT paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.id IS UNIQUE", {}),
            ("CREATE CONSTRAINT author_key IF NOT EXISTS FOR (a:Author) REQUIRE a.key IS UNIQUE", {}),
            ("CREATE CONSTRAINT institution_key IF NOT EXISTS FOR (i:Institution) REQUIRE i.key IS UNIQUE", {}),
            ("CREATE CONSTRAINT concept_key IF NOT EXISTS FOR (c:Concept) REQUIRE c.key IS UNIQUE", {}),
            ("CREATE CONSTRAINT method_key IF NOT EXISTS FOR (m:Method) REQUIRE m.key IS UNIQUE", {}),
        ]
        self._client.run_write_many(stmts)

    def upsert_paper(
        self,
        *,
        paper_id: str,
        title: str | None,
        summary: str | None,
        categories: list[str] | None,
        published: str | None,
        updated: str | None,
        pdf_url: str | None,
        entry_id: str | None,
    ) -> None:
        self._client.run_write(
            """
            MERGE (p:Paper {id: $paper_id})
            SET p.title = $title,
                p.summary = $summary,
                p.categories = $categories,
                p.published = $published,
                p.updated = $updated,
                p.pdf_url = $pdf_url,
                p.entry_id = $entry_id
            """,
            {
                "paper_id": paper_id,
                "title": title,
                "summary": summary,
                "categories": categories,
                "published": published,
                "updated": updated,
                "pdf_url": pdf_url,
                "entry_id": entry_id,
            },
        )

    def upsert_author_authorship(self, *, paper_id: str, author_name: str) -> None:
        key = _normalize_author_key(author_name)
        self._client.run_write(
            """
            MERGE (a:Author {key: $key})
            SET a.name = coalesce(a.name, $name)
            MERGE (p:Paper {id: $paper_id})
            MERGE (a)-[:AUTHORED]->(p)
            """,
            {"key": key, "name": author_name, "paper_id": paper_id},
        )

    def upsert_coauthored_for_paper(self, *, paper_id: str) -> None:
        self._client.run_write(
            """
            MATCH (a1:Author)-[:AUTHORED]->(p:Paper {id: $paper_id})<-[:AUTHORED]-(a2:Author)
            WHERE a1.key < a2.key
            MERGE (a1)-[:CO_AUTHORED]->(a2)
            """,
            {"paper_id": paper_id},
        )

    def write_relations(self, *, relations: list[ExtractedRelation]) -> None:
        stmts: list[tuple[str, dict[str, Any]]] = []
        for r in relations:
            s_cypher, s_props = self._merge_node(r.source, var="s", prefix="s_")
            t_cypher, t_props = self._merge_node(r.target, var="t", prefix="t_")
            rel_type = r.type
            stmt = (
                f"{s_cypher}\n{t_cypher}\n"
                f"MERGE (s)-[rel:{rel_type} {{chunk_index: $chunk_index}}]->(t)\n"
                "SET rel.evidence = $evidence, rel.confidence = $confidence"
            )
            params: dict[str, Any] = {
                **s_props,
                **t_props,
                "chunk_index": int(r.chunk_index),
                "evidence": r.evidence,
                "confidence": float(r.confidence) if r.confidence is not None else None,
            }
            stmts.append((stmt, params))
        if stmts:
            self._client.run_write_many(stmts)

    def _merge_node(self, node: NodeRef, *, var: str, prefix: str) -> tuple[str, dict[str, Any]]:
        if node.type == "Paper":
            pid = str(node.id or "").strip()
            if not pid:
                raise ValueError("Paper nodes require id.")
            return (f"MERGE ({var}:Paper {{id: ${prefix}id}})", {f"{prefix}id": pid})

        key = node.key()
        name = str(node.name or "").strip()
        if not name:
            raise ValueError(f"{node.type} nodes require name.")
        cypher = f"MERGE ({var}:{node.type} {{key: ${prefix}key}})\nSET {var}.name = coalesce({var}.name, ${prefix}name)"
        return (cypher, {f"{prefix}key": key, f"{prefix}name": name})
