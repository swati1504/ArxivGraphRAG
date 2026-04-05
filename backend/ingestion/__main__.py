import argparse
from pathlib import Path

from backend.ingestion.arxiv_fetcher import default_arxiv_query, fetch_and_download
from backend.ingestion.pipeline import ingest_papers


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

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command in {"fetch", "fetch-ingest"}:
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


if __name__ == "__main__":
    main()
