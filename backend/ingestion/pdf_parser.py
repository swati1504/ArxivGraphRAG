import json
import re
from pathlib import Path

from tenacity import retry, stop_after_attempt, wait_exponential


_MULTI_SPACE = re.compile(r"[ \t]{2,}")
_MULTI_NEWLINE = re.compile(r"\n{3,}")


def clean_extracted_text(text: str) -> str:
    t = text.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"(\w)-\n(\w)", r"\1\2", t)
    t = _MULTI_SPACE.sub(" ", t)
    t = _MULTI_NEWLINE.sub("\n\n", t)
    return t.strip()


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=6))
def parse_pdf_to_text(pdf_path: Path) -> str:
    try:
        from unstructured.partition.pdf import partition_pdf
    except Exception as e:
        raise RuntimeError(
            "unstructured is required for PDF parsing. Install requirements.txt dependencies."
        ) from e

    elements = partition_pdf(filename=str(pdf_path), strategy="fast")
    raw = "\n".join([getattr(el, "text", "") or "" for el in elements])
    return clean_extracted_text(raw)


def parse_pdf_cached(*, pdf_path: Path, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        return payload["text"]

    text = parse_pdf_to_text(pdf_path)
    cache_path.write_text(json.dumps({"text": text}, ensure_ascii=False), encoding="utf-8")
    return text
