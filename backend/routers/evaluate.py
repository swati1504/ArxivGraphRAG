from fastapi import APIRouter, HTTPException

from backend.config import get_settings
from backend.evaluation.benchmark_questions import load_benchmark_questions
from backend.evaluation.cost_tracker import aggregate_query_metrics
from backend.evaluation.ragas_evaluator import LLMJudge
from backend.graph_store.graph_retriever import GraphRetriever
from backend.graph_store.neo4j_client import Neo4jClient
from backend.ingestion.embedder import build_embedder
from backend.pipelines.agent_pipeline import AgentPipeline
from backend.pipelines.graphrag_pipeline import GraphRAGPipeline
from backend.pipelines.rag_pipeline import RAGPipeline
from backend.schemas.evaluate import EvaluatePerQuestion, EvaluateRequest, EvaluateResponse, EvaluateSummary
from backend.schemas.query import QueryResponse
from backend.storage.chunk_store import ChunkStore
from backend.vector_store.pinecone_client import PineconeIndex
from backend.vector_store.retriever import VectorRetriever


router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("", response_model=EvaluateResponse)
@router.post("/", response_model=EvaluateResponse)
def evaluate(payload: EvaluateRequest) -> EvaluateResponse:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise HTTPException(status_code=500, detail="PINECONE_API_KEY is required")
    if not settings.neo4j_uri or not settings.neo4j_username or not settings.neo4j_password:
        raise HTTPException(status_code=500, detail="NEO4J_URI/NEO4J_USERNAME/NEO4J_PASSWORD are required")

    neo4j: Neo4jClient | None = None
    try:
        embedder = build_embedder(
            provider=settings.embeddings_provider,
            model=settings.embeddings_model,
            dimension=settings.embeddings_dim,
            ollama_host=settings.ollama_host,
            embeddings_max_chars=settings.embeddings_max_chars,
        )
        emb_dim = int(settings.embeddings_dim)
        if emb_dim <= 0:
            emb_dim = len(embedder.embed_texts(["dimension probe"])[0])

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

        neo4j = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=(settings.neo4j_database or "").strip() or None,
        )
        neo4j.verify_connectivity()
        graphrag = GraphRAGPipeline(
            vector_retriever=retriever,
            neo4j=neo4j,
            pinecone_index=index,
            namespace=settings.pinecone_namespace,
            chunk_store=chunk_store,
        )

        try:
            qs = load_benchmark_questions(payload.questions_path)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to load questions_path: {e}") from e
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    if payload.limit:
        qs = qs[: payload.limit]

    judge = LLMJudge() if payload.include_llm_judge else None
    warnings: list[str] = []
    if payload.include_llm_judge and (settings.eval_provider or "").strip().lower() == "gemini" and not settings.gemini_api_key:
        warnings.append("LLM judge enabled but GEMINI_API_KEY is missing; judge metrics will be null.")

    results: list[EvaluatePerQuestion] = []
    rag_resps: list[QueryResponse] = []
    graphrag_resps: list[QueryResponse] = []
    agent_resps: list = []

    agent_pipeline = None
    if payload.include_agent:
        try:
            graph_retriever = GraphRetriever(neo4j=neo4j, chunk_store=chunk_store) if neo4j is not None else None
        except Exception:
            graph_retriever = None
        try:
            from backend.pipelines.langgraph_agent import LangGraphAgentPipeline

            agent_pipeline = LangGraphAgentPipeline(
                vector_retriever=retriever,
                pinecone_index=index,
                namespace=settings.pinecone_namespace,
                rag=rag,
                graph_retriever=graph_retriever,
                chunk_store=chunk_store,
            )
        except Exception:
            agent_pipeline = AgentPipeline(
                vector_retriever=retriever,
                pinecone_index=index,
                namespace=settings.pinecone_namespace,
                rag=rag,
                graph_retriever=graph_retriever,
                chunk_store=chunk_store,
            )

    for q in qs:
        rag_resp = rag.run(question=q.question, top_k=payload.top_k)
        graphrag_resp = graphrag.run(question=q.question, top_k=payload.top_k)
        rag_resps.append(rag_resp)
        graphrag_resps.append(graphrag_resp)

        rag_j = None
        graphrag_j = None
        agent_resp = None
        agent_j = None

        if agent_pipeline is not None:
            agent_resp = agent_pipeline.run(question=q.question, top_k=payload.top_k)
            agent_resps.append(agent_resp)
        if judge is not None:
            rag_papers = [c.citation.paper_id for c in rag_resp.contexts if c.citation.paper_id]
            graphrag_papers = [c.citation.paper_id for c in graphrag_resp.contexts if c.citation.paper_id]
            rag_j = judge.score(
                question=q.question,
                answer=rag_resp.answer,
                contexts=[c.text for c in rag_resp.contexts],
                reference_answer=q.reference_answer,
                reference_paper_ids=q.reference_paper_ids,
                retrieved_paper_ids=rag_papers,
            )
            graphrag_j = judge.score(
                question=q.question,
                answer=graphrag_resp.answer,
                contexts=[c.text for c in graphrag_resp.contexts],
                reference_answer=q.reference_answer,
                reference_paper_ids=q.reference_paper_ids,
                retrieved_paper_ids=graphrag_papers,
            )

            if agent_resp is not None:
                agent_papers = [c.citation.paper_id for c in agent_resp.contexts if c.citation.paper_id]
                agent_j = judge.score(
                    question=q.question,
                    answer=agent_resp.answer,
                    contexts=[c.text for c in agent_resp.contexts],
                    reference_answer=q.reference_answer,
                    reference_paper_ids=q.reference_paper_ids,
                    retrieved_paper_ids=agent_papers,
                )
        results.append(
            EvaluatePerQuestion(
                id=q.id,
                category=q.category,
                question=q.question,
                rag=rag_resp,
                graphrag=graphrag_resp,
                agent=agent_resp,
                rag_judge=rag_j,
                graphrag_judge=graphrag_j,
                agent_judge=agent_j,
            )
        )


    if neo4j is not None:
        neo4j.close()

    rag_agg = aggregate_query_metrics(rag_resps)
    graphrag_agg = aggregate_query_metrics(graphrag_resps)
    agent_agg = aggregate_query_metrics(agent_resps) if agent_resps else None

    summary = EvaluateSummary(
        rag_avg_latency_ms=rag_agg.avg_latency_ms,
        graphrag_avg_latency_ms=graphrag_agg.avg_latency_ms,
        rag_avg_cost_usd=rag_agg.avg_cost_usd,
        graphrag_avg_cost_usd=graphrag_agg.avg_cost_usd,
    )
    if agent_agg is not None:
        summary.agent_avg_latency_ms = agent_agg.avg_latency_ms
        summary.agent_avg_cost_usd = agent_agg.avg_cost_usd

    if judge is not None:
        rag_f = [r.rag_judge.faithfulness for r in results if r.rag_judge and r.rag_judge.faithfulness is not None]
        gr_f = [r.graphrag_judge.faithfulness for r in results if r.graphrag_judge and r.graphrag_judge.faithfulness is not None]
        ag_f = [r.agent_judge.faithfulness for r in results if r.agent_judge and r.agent_judge.faithfulness is not None]
        rag_ar = [r.rag_judge.answer_relevancy for r in results if r.rag_judge and r.rag_judge.answer_relevancy is not None]
        gr_ar = [r.graphrag_judge.answer_relevancy for r in results if r.graphrag_judge and r.graphrag_judge.answer_relevancy is not None]
        ag_ar = [r.agent_judge.answer_relevancy for r in results if r.agent_judge and r.agent_judge.answer_relevancy is not None]
        rag_cp = [r.rag_judge.context_precision for r in results if r.rag_judge and r.rag_judge.context_precision is not None]
        gr_cp = [
            r.graphrag_judge.context_precision for r in results if r.graphrag_judge and r.graphrag_judge.context_precision is not None
        ]
        ag_cp = [r.agent_judge.context_precision for r in results if r.agent_judge and r.agent_judge.context_precision is not None]
        rag_cr = [r.rag_judge.context_recall for r in results if r.rag_judge and r.rag_judge.context_recall is not None]
        gr_cr = [r.graphrag_judge.context_recall for r in results if r.graphrag_judge and r.graphrag_judge.context_recall is not None]
        ag_cr = [r.agent_judge.context_recall for r in results if r.agent_judge and r.agent_judge.context_recall is not None]
        rag_ac = [r.rag_judge.answer_correctness for r in results if r.rag_judge and r.rag_judge.answer_correctness is not None]
        gr_ac = [
            r.graphrag_judge.answer_correctness
            for r in results
            if r.graphrag_judge and r.graphrag_judge.answer_correctness is not None
        ]
        ag_ac = [r.agent_judge.answer_correctness for r in results if r.agent_judge and r.agent_judge.answer_correctness is not None]
        summary.rag_avg_faithfulness = (sum(rag_f) / len(rag_f)) if rag_f else None
        summary.graphrag_avg_faithfulness = (sum(gr_f) / len(gr_f)) if gr_f else None
        summary.agent_avg_faithfulness = (sum(ag_f) / len(ag_f)) if ag_f else None
        summary.rag_avg_answer_relevancy = (sum(rag_ar) / len(rag_ar)) if rag_ar else None
        summary.graphrag_avg_answer_relevancy = (sum(gr_ar) / len(gr_ar)) if gr_ar else None
        summary.agent_avg_answer_relevancy = (sum(ag_ar) / len(ag_ar)) if ag_ar else None
        summary.rag_avg_context_precision = (sum(rag_cp) / len(rag_cp)) if rag_cp else None
        summary.graphrag_avg_context_precision = (sum(gr_cp) / len(gr_cp)) if gr_cp else None
        summary.agent_avg_context_precision = (sum(ag_cp) / len(ag_cp)) if ag_cp else None
        summary.rag_avg_context_recall = (sum(rag_cr) / len(rag_cr)) if rag_cr else None
        summary.graphrag_avg_context_recall = (sum(gr_cr) / len(gr_cr)) if gr_cr else None
        summary.agent_avg_context_recall = (sum(ag_cr) / len(ag_cr)) if ag_cr else None
        summary.rag_avg_answer_correctness = (sum(rag_ac) / len(rag_ac)) if rag_ac else None
        summary.graphrag_avg_answer_correctness = (sum(gr_ac) / len(gr_ac)) if gr_ac else None
        summary.agent_avg_answer_correctness = (sum(ag_ac) / len(ag_ac)) if ag_ac else None

    return EvaluateResponse(results=results, summary=summary, warnings=warnings)
