from fastapi import APIRouter, HTTPException
import logging
import json

from backend.config import get_settings
from backend.graph_store.graph_retriever import GraphRetriever
from backend.graph_store.neo4j_client import Neo4jClient
from backend.ingestion.embedder import build_embedder
from backend.pipelines.agent_pipeline import AgentPipeline
from backend.pipelines.graphrag_pipeline import GraphRAGPipeline
from backend.pipelines.rag_pipeline import RAGPipeline
from backend.schemas.agent import AgentQueryRequest, AgentQueryResponse
from backend.schemas.query import QueryRequest, QueryResponse
from backend.storage.chunk_store import ChunkStore
from backend.vector_store.pinecone_client import PineconeIndex
from backend.vector_store.retriever import VectorRetriever


router = APIRouter(prefix="/query", tags=["query"])
_log = logging.getLogger("arxiv_graphrag.metrics")

_rag_pipeline: RAGPipeline | None = None
_graphrag_pipeline: GraphRAGPipeline | None = None
_agent_pipeline: AgentPipeline | None = None


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


def _get_graphrag_pipeline() -> GraphRAGPipeline:
    global _graphrag_pipeline
    if _graphrag_pipeline is not None:
        return _graphrag_pipeline

    settings = get_settings()
    rag_provider = (settings.rag_provider or "").strip().lower()
    if rag_provider == "anthropic" and not settings.anthropic_api_key:
        raise HTTPException(status_code=500, detail="ANTHROPIC_API_KEY is required")
    if rag_provider == "gemini" and not settings.gemini_api_key:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY is required")
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is required")
    if not settings.neo4j_uri or not settings.neo4j_username or not settings.neo4j_password:
        raise HTTPException(status_code=500, detail="NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD are required")

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
    chunk_store = ChunkStore()
    retriever = VectorRetriever(
        embedder=embedder,
        index=index,
        namespace=settings.pinecone_namespace,
        chunk_store=chunk_store,
        expected_dimension=emb_dim,
    )

    neo4j = Neo4jClient(
        uri=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        database=(settings.neo4j_database or "").strip() or None,
    )
    neo4j.verify_connectivity()

    _graphrag_pipeline = GraphRAGPipeline(
        vector_retriever=retriever,
        neo4j=neo4j,
        pinecone_index=index,
        namespace=settings.pinecone_namespace,
        chunk_store=chunk_store,
    )
    return _graphrag_pipeline


def _get_agent_pipeline() -> AgentPipeline:
    global _agent_pipeline
    if _agent_pipeline is not None:
        return _agent_pipeline

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
    chunk_store = ChunkStore()
    retriever = VectorRetriever(
        embedder=embedder,
        index=index,
        namespace=settings.pinecone_namespace,
        chunk_store=chunk_store,
        expected_dimension=emb_dim,
    )
    rag = RAGPipeline(retriever=retriever)

    graph_retriever: GraphRetriever | None = None
    if settings.neo4j_uri and settings.neo4j_username and settings.neo4j_password:
        try:
            neo4j = Neo4jClient(
                uri=settings.neo4j_uri,
                username=settings.neo4j_username,
                password=settings.neo4j_password,
                database=(settings.neo4j_database or "").strip() or None,
            )
            neo4j.verify_connectivity()
            graph_retriever = GraphRetriever(neo4j=neo4j, chunk_store=chunk_store)
        except Exception:
            graph_retriever = None

    _agent_pipeline = AgentPipeline(
        vector_retriever=retriever,
        pinecone_index=index,
        namespace=settings.pinecone_namespace,
        rag=rag,
        graph_retriever=graph_retriever,
        chunk_store=chunk_store,
    )
    return _agent_pipeline


@router.post("/rag", response_model=QueryResponse)
def query_rag(payload: QueryRequest) -> QueryResponse:
    pipeline = _get_rag_pipeline()
    resp = pipeline.run(question=payload.question, top_k=payload.top_k)
    _log.info(
        json.dumps(
            {
                "query": payload.question,
                "pipeline": "rag",
                "latency_ms": resp.metrics.latency_ms,
                "input_tokens": resp.metrics.prompt_tokens,
                "output_tokens": resp.metrics.completion_tokens,
                "cost_usd": resp.metrics.total_cost_usd,
            },
            ensure_ascii=False,
        )
    )
    return resp


@router.post("/graphrag", response_model=QueryResponse)
def query_graphrag(payload: QueryRequest) -> QueryResponse:
    pipeline = _get_graphrag_pipeline()
    resp = pipeline.run(question=payload.question, top_k=payload.top_k)
    _log.info(
        json.dumps(
            {
                "query": payload.question,
                "pipeline": "graphrag",
                "latency_ms": resp.metrics.latency_ms,
                "input_tokens": resp.metrics.prompt_tokens,
                "output_tokens": resp.metrics.completion_tokens,
                "cost_usd": resp.metrics.total_cost_usd,
            },
            ensure_ascii=False,
        )
    )
    return resp


@router.post("/agent", response_model=AgentQueryResponse)
def query_agent(payload: AgentQueryRequest) -> AgentQueryResponse:
    pipeline = _get_agent_pipeline()
    resp = pipeline.run(question=payload.question, top_k=payload.top_k)
    _log.info(
        json.dumps(
            {
                "query": payload.question,
                "pipeline": "agent",
                "route": resp.route,
                "query_type": resp.query_type,
                "latency_ms": resp.metrics.latency_ms,
                "input_tokens": resp.metrics.prompt_tokens,
                "output_tokens": resp.metrics.completion_tokens,
                "cost_usd": resp.metrics.total_cost_usd,
            },
            ensure_ascii=False,
        )
    )
    return resp
