from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass(frozen=True)
class TextChunk:
    text: str
    chunk_index: int


def chunk_text(
    text: str,
    *,
    chunk_size: int = 4000,
    chunk_overlap: int = 400,
) -> list[TextChunk]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_text(text)
    return [TextChunk(text=c, chunk_index=i) for i, c in enumerate(chunks)]
