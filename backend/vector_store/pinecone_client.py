from __future__ import annotations

from dataclasses import dataclass

from pinecone import Pinecone, ServerlessSpec
from pinecone.exceptions import PineconeException


@dataclass(frozen=True)
class PineconeMatch:
    id: str
    score: float
    metadata: dict | None


class PineconeIndex:
    def __init__(
        self,
        *,
        api_key: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-east-1",
    ) -> None:
        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._dimension = dimension
        self._metric = metric
        self._cloud = cloud
        self._region = region
        self._ensure_index()
        self._index = self._pc.Index(index_name)

    def _existing_index_names(self) -> set[str]:
        try:
            resp = self._pc.list_indexes()
        except PineconeException as e:
            raise RuntimeError("Pinecone authentication/connection failed. Check PINECONE_API_KEY.") from e
        candidates = None
        if isinstance(resp, dict):
            candidates = resp.get("indexes", None)
        else:
            candidates = getattr(resp, "indexes", None)
        if candidates is None:
            candidates = resp

        names: set[str] = set()
        for item in candidates or []:
            if isinstance(item, str):
                names.add(item)
            elif isinstance(item, dict):
                name = item.get("name")
                if name:
                    names.add(str(name))
            else:
                name = getattr(item, "name", None)
                if name:
                    names.add(str(name))
        return names

    def _describe_index_dimension(self) -> int | None:
        try:
            desc = self._pc.describe_index(self._index_name)
        except Exception:
            return None

        if isinstance(desc, dict):
            dim = desc.get("dimension", None)
        else:
            dim = getattr(desc, "dimension", None)
        try:
            return int(dim) if dim is not None else None
        except Exception:
            return None

    def _ensure_index(self) -> None:
        existing = self._existing_index_names()
        if self._index_name in existing:
            dim = self._describe_index_dimension()
            if dim is not None and dim != self._dimension:
                raise RuntimeError(
                    f"Pinecone index dimension mismatch for '{self._index_name}': "
                    f"index dimension is {dim} but EMBEDDINGS_DIM is {self._dimension}. "
                    "Use a new PINECONE_INDEX_NAME or recreate the index with the correct dimension."
                )
            return
        self._pc.create_index(
            name=self._index_name,
            dimension=self._dimension,
            metric=self._metric,
            spec=ServerlessSpec(cloud=self._cloud, region=self._region),
        )

    def upsert(self, *, vectors: list[tuple[str, list[float], dict]], namespace: str) -> None:
        try:
            self._index.upsert(vectors=vectors, namespace=namespace)
        except PineconeException as e:
            raise RuntimeError("Pinecone upsert failed. Check API key, index, and namespace.") from e

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str,
        filter: dict | None = None,
    ) -> list[PineconeMatch]:
        try:
            resp = self._index.query(
                vector=vector,
                top_k=top_k,
                namespace=namespace,
                include_metadata=True,
                filter=filter,
            )
        except PineconeException as e:
            raise RuntimeError("Pinecone query failed. Check API key, index, and namespace.") from e
        if isinstance(resp, dict):
            matches = resp.get("matches", [])
        else:
            matches = getattr(resp, "matches", None) or []
        out: list[PineconeMatch] = []
        for m in matches:
            if isinstance(m, dict):
                mid = m.get("id")
                score = m.get("score")
                metadata = m.get("metadata")
            else:
                mid = getattr(m, "id", None)
                score = getattr(m, "score", None)
                metadata = getattr(m, "metadata", None)
            if mid is None or score is None:
                continue
            out.append(PineconeMatch(id=str(mid), score=float(score), metadata=metadata))
        return out
