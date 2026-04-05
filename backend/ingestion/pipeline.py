import json
from pathlib import Path

from backend.config import get_settings
from backend.ingestion.chunker import chunk_text
from backend.ingestion.embedder import build_embedder
from backend.ingestion.pdf_parser import parse_pdf_cached
from backend.storage.chunk_store import ChunkStore, StoredChunk
from backend.vector_store.pinecone_client import PineconeIndex


def ingest_papers(
    *,
    papers_metadata_path: Path,
    parsed_cache_dir: Path = Path("data/processed/parsed"),
    chunk_store: ChunkStore | None = None,
    batch_size: int = 64,
    skip_existing: bool = True,
    force: bool = False,
) -> dict:
    settings = get_settings()
    if not settings.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required for Pinecone.")

    embedder = build_embedder(
        provider=settings.embeddings_provider,
        model=settings.embeddings_model,
        dimension=settings.embeddings_dim,
        ollama_host=settings.ollama_host,
        embeddings_max_chars=settings.embeddings_max_chars,
    )

    emb_dim = int(settings.embeddings_dim)
    if emb_dim <= 0:
        probe = embedder.embed_texts(["dimension probe"])[0]
        emb_dim = len(probe)

    index = PineconeIndex(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index_name,
        dimension=emb_dim,
        metric="cosine",
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    chunks_store = chunk_store or ChunkStore()

    papers = json.loads(papers_metadata_path.read_text(encoding="utf-8"))

    total_papers = len(papers)
    processed_papers = 0
    skipped_papers = 0
    total_chunks = 0
    total_vectors = 0

    for paper in papers:
        paper_id = str(paper["arxiv_id"])
        title = str(paper.get("title", "")).strip() or None
        pdf_path = Path(paper["pdf_path"])
        if skip_existing and not force and chunks_store.has_paper(paper_id=paper_id):
            skipped_papers += 1
            continue

        parsed_cache_path = parsed_cache_dir / f"{paper_id}.json"
        text = parse_pdf_cached(pdf_path=pdf_path, cache_path=parsed_cache_path)

        chunks = chunk_text(text, chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        total_chunks += len(chunks)
        stored = [StoredChunk(paper_id=paper_id, chunk_index=c.chunk_index, text=c.text) for c in chunks]
        chunks_store.write_paper_chunks(paper_id=paper_id, chunks=stored)

        ids: list[str] = []
        texts: list[str] = []
        metas: list[dict] = []

        for c in chunks:
            ids.append(f"{paper_id}:{c.chunk_index}")
            texts.append(c.text)
            metas.append({"paper_id": paper_id, "chunk_index": c.chunk_index, "title": title})

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            batch_ids = ids[i : i + batch_size]
            batch_metas = metas[i : i + batch_size]
            vectors = embedder.embed_texts(batch_texts)
            if vectors and len(vectors[0]) != emb_dim:
                raise RuntimeError(
                    f"Embedding dimension mismatch: got {len(vectors[0])} but EMBEDDINGS_DIM is {emb_dim}. "
                    "Ensure your Pinecone index dimension matches the embedding model dimension, then retry."
                )
            upserts = [(vid, vec, md) for vid, vec, md in zip(batch_ids, vectors, batch_metas)]
            index.upsert(vectors=upserts, namespace=settings.pinecone_namespace)
            total_vectors += len(upserts)

        processed_papers += 1

    return {
        "total_papers": total_papers,
        "processed_papers": processed_papers,
        "skipped_papers": skipped_papers,
        "total_chunks": total_chunks,
        "total_vectors": total_vectors,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--papers", type=str, default="data/raw/papers.json")
    args = parser.parse_args()

    ingest_papers(papers_metadata_path=Path(args.papers))
    print("Ingestion complete")
