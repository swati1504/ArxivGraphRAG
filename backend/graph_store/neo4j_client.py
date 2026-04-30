from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from neo4j import GraphDatabase
from neo4j.exceptions import Neo4jError


class Neo4jClient:
    def __init__(self, *, uri: str, username: str, password: str, database: str | None = None) -> None:
        if not uri:
            raise RuntimeError("NEO4J_URI is required.")
        if not username:
            raise RuntimeError("NEO4J_USERNAME is required.")
        if not password:
            raise RuntimeError("NEO4J_PASSWORD is required.")
        self._driver = GraphDatabase.driver(uri, auth=(username, password))
        self._database = (database or "").strip() or None

    def close(self) -> None:
        self._driver.close()

    def verify_connectivity(self) -> None:
        try:
            self._driver.verify_connectivity()
        except Neo4jError as e:
            raise RuntimeError("Neo4j connectivity check failed. Verify NEO4J_* settings and DB status.") from e

    def run_write(self, cypher: str, parameters: dict[str, Any] | None = None) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                session.execute_write(lambda tx: tx.run(cypher, parameters or {}).consume())
        except Neo4jError as e:
            raise RuntimeError("Neo4j write query failed.") from e

    def run_write_many(self, statements: Iterable[tuple[str, dict[str, Any]]]) -> None:
        try:
            with self._driver.session(database=self._database) as session:
                def _fn(tx):  # type: ignore[no-untyped-def]
                    for cypher, params in statements:
                        tx.run(cypher, params).consume()

                session.execute_write(_fn)
        except Neo4jError as e:
            raise RuntimeError("Neo4j write transaction failed.") from e

    def run_read(self, cypher: str, parameters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        try:
            with self._driver.session(database=self._database) as session:
                def _fn(tx):  # type: ignore[no-untyped-def]
                    return [dict(r) for r in tx.run(cypher, parameters or {})]

                return session.execute_read(_fn)
        except Neo4jError as e:
            raise RuntimeError("Neo4j read query failed.") from e
