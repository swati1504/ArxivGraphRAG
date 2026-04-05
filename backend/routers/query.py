from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.ingestion.embedder import build_embedder
from backend.pipelines.rag_pipeline import RAGPipeline
from backend.schemas.query import QueryRequest, QueryResponse
from backend.storage.chunk_store import ChunkStore
from backend.vector_store.pinecone_client import PineconeIndex
from backend.vector_store.retriever import VectorRetriever


router = APIRouter(prefix="/query", tags=["query"])

_rag_pipeline: RAGPipeline | None = None


def _get_rag_pipeline() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is not None:
        return _rag_pipeline

    settings = get_settings()
    rag_provider = (settings.rag_provider or "").strip().lower()
    if rag_provider == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is required")
    if rag_provider == "gemini" and not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is required")
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is required")

    try:
        embedder = build_embedder(
            provider=settings.embeddings_provider,
            model=settings.embeddings_model,
            dimension=settings.embeddings_dim,
            ollama_host=settings.ollama_host,
            embeddings_max_chars=settings.embeddings_max_chars,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e

    emb_dim = int(settings.embeddings_dim)
    if emb_dim <= 0:
        try:
            emb_dim = len(embedder.embed_texts(["dimension probe"])[0])
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) from e

    index = PineconeIndex(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=emb_dim,
        metric="cosine",
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    retriever = VectorRetriever(
        embedder=embedder,
        index=index,
        namespace=settings.pinecone_namespace,
        chunk_store=ChunkStore(),
        expected_dimension=emb_dim,
    )
    _rag_pipeline = RAGPipeline(retriever=retriever)
    return _rag_pipeline


@router.post("/rag", response_model=QueryResponse)
def query_rag(payload: QueryRequest) -> QueryResponse:
    pipeline = _get_rag_pipeline()
    return pipeline.run(question=payload.question, top_k=payload.top_k)


@router.post("/graphrag", response_model=QueryResponse)
def query_graphrag(_: QueryRequest) -> QueryResponse:
    raise HTTPException(status_code=501, detail="Not implemented yet")
