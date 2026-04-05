import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import arxiv
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class ArxivPaper:
    arxiv_id: str
    title: str
    summary: str
    authors: list[str]
    categories: list[str]
    published: str
    updated: str
    pdf_url: str
    entry_id: str
    pdf_path: str


def _to_iso(dt: datetime) -> str:
    return dt.replace(tzinfo=None).isoformat()


def _paper_id(result: arxiv.Result) -> str:
    return result.get_short_id()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def _download_pdf(result: arxiv.Result, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    if tmp_path.exists():
        tmp_path.unlink()
    result.download_pdf(filename=str(tmp_path))
    tmp_path.replace(out_path)


def fetch_and_download(
    *,
    query: str,
    max_results: int,
    output_dir: Path,
    metadata_path: Path,
) -> list[ArxivPaper]:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )

    client = arxiv.Client(page_size=min(100, max_results), delay_seconds=3.0, num_retries=3)
    papers: list[ArxivPaper] = []

    for result in client.results(search):
        pid = _paper_id(result)
        pdf_path = output_dir / f"{pid}.pdf"
        if not pdf_path.exists():
            _download_pdf(result, pdf_path)

        paper = ArxivPaper(
            arxiv_id=pid,
            title=result.title.strip(),
            summary=result.summary.strip(),
            authors=[a.name for a in result.authors],
            categories=list(result.categories),
            published=_to_iso(result.published),
            updated=_to_iso(result.updated),
            pdf_url=str(result.pdf_url),
            entry_id=str(result.entry_id),
            pdf_path=str(pdf_path),
        )
        papers.append(paper)

    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(p) for p in papers], f, ensure_ascii=False, indent=2)

    return papers


def default_arxiv_query(*, keywords: list[str], categories: list[str]) -> str:
    parts: list[str] = []
    if categories:
        parts.append("(" + " OR ".join([f"cat:{c}" for c in categories]) + ")")
    if keywords:
        parts.append("(" + " OR ".join([f'all:"{k}"' for k in keywords]) + ")")
    if not parts:
        raise ValueError("Provide at least one keyword or category.")
    return " AND ".join(parts)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--data-dir", type=str, default="data/raw")
    parser.add_argument(
        "--keywords",
        type=str,
        nargs="*",
        default=["hallucination", "reasoning", "chain-of-thought", "RLHF", "retrieval-augmented generation"],
    )
    parser.add_argument("--categories", type=str, nargs="*", default=["cs.AI", "cs.CL", "cs.LG"])
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    pdf_dir = data_dir / "pdfs"
    metadata_path = data_dir / "papers.json"
    query = default_arxiv_query(keywords=args.keywords, categories=args.categories)

    fetched = fetch_and_download(
        query=query,
        max_results=args.max_results,
        output_dir=pdf_dir,
        metadata_path=metadata_path,
    )

    print(f"Downloaded {len(fetched)} papers")
