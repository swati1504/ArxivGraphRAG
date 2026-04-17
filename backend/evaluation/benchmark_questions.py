import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    category: str
    question: str
    reference_answer: str | None = None
    reference_paper_ids: list[str] | None = None


def load_benchmark_questions(path: str | Path = "data/benchmark/questions.json") -> list[BenchmarkQuestion]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: list[BenchmarkQuestion] = []
    for q in payload.get("questions", []):
        ref_answer = q.get("reference_answer")
        if ref_answer is not None:
            ref_answer = str(ref_answer)
        ref_papers = q.get("reference_paper_ids")
        if not isinstance(ref_papers, list):
            ref_papers = None
        else:
            ref_papers = [str(x) for x in ref_papers if str(x).strip()]
        out.append(
            BenchmarkQuestion(
                id=q["id"],
                category=q["category"],
                question=q["question"],
                reference_answer=ref_answer,
                reference_paper_ids=ref_papers,
            )
        )
    return out
