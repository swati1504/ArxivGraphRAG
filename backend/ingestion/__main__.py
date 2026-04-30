import argparse
import json
from pathlib import Path

from backend.config import get_settings
from backend.graph_store.entity_extractor import EntityExtractor
from backend.graph_store.graph_writer import GraphWriter
from backend.graph_store.neo4j_client import Neo4jClient


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m backend.ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    fetch = sub.add_parser("fetch", help="Fetch paper metadata and download PDFs from arXiv")
    fetch.add_argument("--max-results", type=int, default=20)
    fetch.add_argument("--pdf-dir", type=str, default="data/raw/pdfs")
    fetch.add_argument("--metadata-path", type=str, default="data/raw/papers.json")
    fetch.add_argument("--query", type=str, default=None)
    fetch.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        default=["hallucination", "reasoning", "chain-of-thought", "RLHF", "retrieval-augmented generation"],
    )
    fetch.add_argument("--categories", type=str, nargs="*", default=["cs.AI", "cs.CL", "cs.LG"])

    ingest = sub.add_parser("ingest", help="Parse, chunk, embed, and upsert papers into Pinecone")
    ingest.add_argument("--papers", type=str, default="data/raw/papers.json")
    ingest.add_argument("--parsed-cache-dir", type=str, default="data/processed/parsed")
    ingest.add_argument("--batch-size", type=int, default=64)
    ingest.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", default=True)
    ingest.add_argument("--force", action="store_true", default=False)

    both = sub.add_parser("fetch-ingest", help="Fetch PDFs then ingest into Pinecone")
    both.add_argument("--max-results", type=int, default=20)
    both.add_argument("--pdf-dir", type=str, default="data/raw/pdfs")
    both.add_argument("--metadata-path", type=str, default="data/raw/papers.json")
    both.add_argument("--query", type=str, default=None)
    both.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        default=["hallucination", "reasoning", "chain-of-thought", "RLHF", "retrieval-augmented generation"],
    )
    both.add_argument("--categories", type=str, nargs="*", default=["cs.AI", "cs.CL", "cs.LG"])
    both.add_argument("--parsed-cache-dir", type=str, default="data/processed/parsed")
    both.add_argument("--batch-size", type=int, default=64)
    both.add_argument("--no-skip-existing", action="store_false", dest="skip_existing", default=True)
    both.add_argument("--force", action="store_true", default=False)

    graph = sub.add_parser("graph", help="Extract entities/relations from chunks and write to Neo4j")
    graph.add_argument("--papers", type=str, default="data/raw/papers.json")
    graph.add_argument("--chunks-dir", type=str, default="data/processed/chunks")
    graph.add_argument("--limit-papers", type=int, default=0)
    graph.add_argument("--max-chunks-per-paper", type=int, default=18)
    graph.add_argument("--skip-extraction", action="store_true", default=False)

    health = sub.add_parser("graph-health", help="Run basic graph health checks in Neo4j")
    health.add_argument("--neo4j-db", type=str, default=None)

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in {"fetch", "fetch-ingest"}:
        from backend.ingestion.arxiv_fetcher import default_arxiv_query, fetch_and_download

        q = args.query or default_arxiv_query(keywords=args.keywords, categories=args.categories)
        papers = fetch_and_download(
            query=q,
            max_results=args.max_results,
            output_dir=Path(args.pdf_dir),
            metadata_path=Path(args.metadata_path),
        )
        print(f"Fetched {len(papers)} papers")
        print(f"Metadata: {args.metadata_path}")
        print(f"PDFs: {args.pdf_dir}")

    if args.command in {"ingest", "fetch-ingest"}:
        from backend.ingestion.pipeline import ingest_papers

        stats = ingest_papers(
            papers_metadata_path=Path(args.papers if args.command == "ingest" else args.metadata_path),
            parsed_cache_dir=Path(args.parsed_cache_dir),
            batch_size=int(args.batch_size),
            skip_existing=bool(args.skip_existing),
            force=bool(args.force),
        )
        print(
            "Ingestion stats: "
            f"total_papers={stats['total_papers']} "
            f"processed_papers={stats['processed_papers']} "
            f"skipped_papers={stats['skipped_papers']} "
            f"total_chunks={stats['total_chunks']} "
            f"total_vectors={stats['total_vectors']}"
        )

    if args.command == "graph":
        settings = get_settings()
        if not settings.neo4j_uri or not settings.neo4j_username or not settings.neo4j_password:
            raise RuntimeError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are required for graph ingestion.")

        papers = json.loads(Path(args.papers).read_text(encoding="utf-8"))
        if not isinstance(papers, list):
            raise RuntimeError("Papers metadata must be a JSON list.")

        limit_papers = int(getattr(args, "limit_papers", 0) or 0)
        if limit_papers > 0:
            papers = papers[:limit_papers]

        chunks_dir = Path(args.chunks_dir)
        max_chunks = int(getattr(args, "max_chunks_per_paper", 12) or 12)

        db = (getattr(args, "neo4j_db", None) or settings.neo4j_database or "").strip() or None
        client = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=db,
        )
        client.verify_connectivity()
        writer = GraphWriter(client=client)
        writer.ensure_schema()
        extractor = EntityExtractor()

        total_relations = 0
        total_chunks = 0
        parse_failures = 0
        request_failures = 0
        failure_examples: list[str] = []
        for paper in papers:
            paper_id = str(paper["arxiv_id"])
            title = str(paper.get("title", "")).strip() or None
            summary = str(paper.get("summary", "")).strip() or None
            categories = paper.get("categories")
            if not isinstance(categories, list):
                categories = None
            published = str(paper.get("published", "")).strip() or None
            updated = str(paper.get("updated", "")).strip() or None
            pdf_url = str(paper.get("pdf_url", "")).strip() or None
            entry_id = str(paper.get("entry_id", "")).strip() or None

            writer.upsert_paper(
                paper_id=paper_id,
                title=title,
                summary=summary,
                categories=categories,
                published=published,
                updated=updated,
                pdf_url=pdf_url,
                entry_id=entry_id,
            )
            authors = paper.get("authors") or []
            if isinstance(authors, list):
                for a in authors:
                    name = str(a or "").strip()
                    if name:
                        writer.upsert_author_authorship(paper_id=paper_id, author_name=name)
                writer.upsert_coauthored_for_paper(paper_id=paper_id)

            chunks_path = chunks_dir / f"{paper_id}.json"
            if not chunks_path.exists():
                continue
            chunk_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            if not isinstance(chunk_payload, list):
                continue
            chunk_payload = _sample_items(chunk_payload, k=max_chunks)

            if not bool(getattr(args, "skip_extraction", False)):
                relations = []
                for item in chunk_payload:
                    if not isinstance(item, dict):
                        continue
                    chunk_index = int(item.get("chunk_index", -1))
                    text = str(item.get("text", "") or "").strip()
                    if chunk_index < 0 or not text:
                        continue
                    total_chunks += 1
                    try:
                        rels = extractor.extract_from_chunk(
                            paper_id=paper_id,
                            paper_title=title,
                            paper_authors=[str(a or "").strip() for a in authors if str(a or "").strip()]
                            if isinstance(authors, list)
                            else None,
                            chunk_index=chunk_index,
                            chunk_text=text,
                        )
                        relations.extend(rels)
                    except RuntimeError as e:
                        msg = str(e)
                        if "parse json" in msg.lower():
                            parse_failures += 1
                        else:
                            request_failures += 1
                        if len(failure_examples) < 3:
                            failure_examples.append(msg[:220])
                        continue

                writer.write_relations(relations=relations)
                total_relations += len(relations)
                print(f"Wrote paper={paper_id} relations={len(relations)} total_relations={total_relations}")

        if not bool(getattr(args, "skip_extraction", False)):
            print(
                json.dumps(
                    {
                        "graph_extraction_total_chunks": total_chunks,
                        "graph_extraction_parse_failures": parse_failures,
                        "graph_extraction_request_failures": request_failures,
                        "graph_extraction_total_relations": total_relations,
                        "graph_extraction_failure_examples": failure_examples,
                    },
                    ensure_ascii=False,
                )
            )
        client.close()

    if args.command == "graph-health":
        settings = get_settings()
        if not settings.neo4j_uri or not settings.neo4j_username or not settings.neo4j_password:
            raise RuntimeError("NEO4J_URI, NEO4J_USERNAME, and NEO4J_PASSWORD are required for graph health checks.")
        db = (getattr(args, "neo4j_db", None) or settings.neo4j_database or "").strip() or None
        client = Neo4jClient(
            uri=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            database=db,
        )
        client.verify_connectivity()

        checks = [
            ("node_counts", "MATCH (n) RETURN labels(n)[0] as label, count(n) as count ORDER BY count DESC"),
            ("rel_counts", "MATCH ()-[r]->() RETURN type(r) as rel, count(r) as count ORDER BY count DESC"),
            ("isolated_nodes", "MATCH (n) WHERE NOT (n)--() RETURN count(n) as isolated_nodes"),
        ]
        for name, cypher in checks:
            rows = client.run_read(cypher)
            print(name + ":")
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        client.close()


def _sample_items(items: list, *, k: int) -> list:
    if k <= 0 or len(items) <= k:
        return items
    n = len(items)
    if k == 1:
        return [items[0]]
    idxs = [round(i * (n - 1) / (k - 1)) for i in range(k)]
    seen = set()
    out = []
    for i in idxs:
        if i in seen:
            continue
        seen.add(i)
        out.append(items[int(i)])
    return out


if __name__ == "__main__":
    main()
