import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BenchmarkQuestion:
    id: str
    category: str
    question: str


def load_benchmark_questions(path: str | Path = "data/benchmark/questions.json") -> list[BenchmarkQuestion]:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    out: list[BenchmarkQuestion] = []
    for q in payload.get("questions", []):
        out.append(BenchmarkQuestion(id=q["id"], category=q["category"], question=q["question"]))
    return out
