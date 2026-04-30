import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class StoredChunk:
    paper_id: str
    chunk_index: int
    text: str


class ChunkStore:
    def __init__(self, *, base_dir: str | Path = "data/processed/chunks") -> None:
        self._base_dir = Path(base_dir)

    def _paper_path(self, paper_id: str) -> Path:
        return self._base_dir / f"{paper_id}.json"

    def _sanitize_text(self, text: str) -> str:
        t = text or ""
        t = t.replace("\x00", "")
        t = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F]", "", t)
        return t

    def write_paper_chunks(self, *, paper_id: str, chunks: list[StoredChunk]) -> None:
        self._base_dir.mkdir(parents=True, exist_ok=True)
        payload = [{"chunk_index": c.chunk_index, "text": self._sanitize_text(c.text)} for c in chunks]
        self._paper_path(paper_id).write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def has_paper(self, *, paper_id: str) -> bool:
        return self._paper_path(paper_id).exists()

    def read_chunk(self, *, paper_id: str, chunk_index: int) -> StoredChunk:
        payload = json.loads(self._paper_path(paper_id).read_text(encoding="utf-8"))
        for item in payload:
            if int(item["chunk_index"]) == int(chunk_index):
                return StoredChunk(
                    paper_id=paper_id,
                    chunk_index=chunk_index,
                    text=self._sanitize_text(str(item.get("text") or "")),
                )
        raise KeyError(f"Chunk not found: paper_id={paper_id} chunk_index={chunk_index}")
