import time

from backend.ingestion.embedder import Embedder
from backend.observability import ls_span
from backend.schemas.query import Citation, RetrievedContext
from backend.storage.chunk_store import ChunkStore
from backend.vector_store.pinecone_client import PineconeIndex


class VectorRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        index: PineconeIndex,
        namespace: str,
        chunk_store: ChunkStore,
        expected_dimension: int,
    ) -> None:
        self._embedder = embedder
        self._index = index
        self._namespace = namespace
        self._chunk_store = chunk_store
        self._expected_dimension = expected_dimension

    def retrieve(self, *, question: str, top_k: int) -> list[RetrievedContext]:
        with ls_span(
            name="embed_query",
            run_type="embedding",
            inputs={"question_chars": len(question or "")},
            metadata={"expected_dimension": int(self._expected_dimension)},
        ):
            qvec = self.embed_query(question=question)
        with ls_span(
            name="vector_retrieve",
            run_type="retriever",
            inputs={"top_k": int(top_k), "namespace": self._namespace},
        ):
            matches = self._index.query(vector=qvec, top_k=top_k, namespace=self._namespace)
        out: list[RetrievedContext] = []
        for m in matches:
            md = m.metadata or {}
            paper_id = str(md.get("paper_id", ""))
            chunk_index = int(md.get("chunk_index", -1))
            title = md.get("title")
            if not paper_id or chunk_index < 0:
                continue
            try:
                chunk = self._chunk_store.read_chunk(paper_id=paper_id, chunk_index=chunk_index)
            except Exception:
                continue
            out.append(
                RetrievedContext(
                    text=chunk.text,
                    citation=Citation(paper_id=paper_id, title=title, chunk_index=chunk_index),
                )
            )
        return out

    def retrieve_scored(self, *, question: str, top_k: int) -> list[tuple[RetrievedContext, float]]:
        with ls_span(
            name="embed_query",
            run_type="embedding",
            inputs={"question_chars": len(question or "")},
            metadata={"expected_dimension": int(self._expected_dimension)},
        ):
            qvec = self.embed_query(question=question)
        with ls_span(
            name="vector_retrieve",
            run_type="retriever",
            inputs={"top_k": int(top_k), "namespace": self._namespace},
        ):
            matches = self._index.query(vector=qvec, top_k=top_k, namespace=self._namespace)
        out: list[tuple[RetrievedContext, float]] = []
        for m in matches:
            md = m.metadata or {}
            paper_id = str(md.get("paper_id", ""))
            chunk_index = int(md.get("chunk_index", -1))
            title = md.get("title")
            if not paper_id or chunk_index < 0:
                continue
            try:
                chunk = self._chunk_store.read_chunk(paper_id=paper_id, chunk_index=chunk_index)
            except Exception:
                continue
            out.append(
                (
                    RetrievedContext(
                        text=chunk.text,
                        citation=Citation(paper_id=paper_id, title=title, chunk_index=chunk_index),
                    ),
                    float(m.score),
                )
            )
        return out

    def embed_query(self, *, question: str) -> list[float]:
        start = time.perf_counter()
        qvec = self._embedder.embed_texts([question])[0]
        try:
            elapsed = int((time.perf_counter() - start) * 1000)
        except Exception:
            elapsed = 0
        try:
            from backend.observability import ls_update_current

            ls_update_current(metadata={"embedding_latency_ms": elapsed})
        except Exception:
            pass
        if len(qvec) != self._expected_dimension:
            raise RuntimeError(
                f"Embedding dimension mismatch: got {len(qvec)} but expected {self._expected_dimension}. "
                "Set EMBEDDINGS_DIM to match your embedding model and ensure the Pinecone index dimension matches."
            )
        return qvec
