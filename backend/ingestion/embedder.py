import hashlib
import math
from collections.abc import Sequence
from typing import Protocol

import httpx


class Embedder(Protocol):
    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]: ...


class HashingEmbedder:
    def __init__(self, *, dimension: int) -> None:
        self._dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self._dimension
            tokens = (t or "").lower().split()
            for tok in tokens:
                h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(h, "big") % self._dimension
                vec[idx] += 1.0
            norm = math.sqrt(sum(v * v for v in vec))
            if norm > 0:
                vec = [v / norm for v in vec]
            out.append(vec)
        return out


class OllamaEmbedder:
    def __init__(self, *, host: str, model: str, max_chars: int = 6000) -> None:
        self._host = host.rstrip("/")
        self._model = model
        self._max_chars = max(200, int(max_chars))
        self._client = httpx.Client(timeout=120.0)

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []

        url = f"{self._host}/api/embed"
        payload = {"model": self._model, "input": [self._truncate(t) for t in texts]}
        try:
            resp = self._client.post(url, json=payload)
        except httpx.RequestError as e:
            raise RuntimeError(
                "Ollama embeddings request failed. Ensure Ollama is installed and running "
                "(ollama serve) and OLLAMA_HOST is correct."
            ) from e

        if resp.status_code == 404:
            return self._embed_texts_fallback(texts)

        if resp.status_code == 400 and "context length" in (resp.text or "").lower():
            return self._embed_texts_fallback(texts)

        if resp.status_code != 200:
            raise RuntimeError(f"Ollama embeddings failed with status {resp.status_code}: {resp.text[:300]}")

        data = resp.json()
        embeddings = data.get("embeddings")
        if not isinstance(embeddings, list) or not embeddings:
            raise RuntimeError("Ollama embeddings response missing embeddings.")
        return [[float(v) for v in e] for e in embeddings]

    def _embed_texts_fallback(self, texts: Sequence[str]) -> list[list[float]]:
        url = f"{self._host}/api/embeddings"
        out: list[list[float]] = []
        for t in texts:
            prompt = self._truncate(t)
            payload = {"model": self._model, "prompt": prompt}
            try:
                resp = self._client.post(url, json=payload)
            except httpx.RequestError as e:
                raise RuntimeError(
                    "Ollama embeddings request failed. Ensure Ollama is installed and running "
                    "(ollama serve) and OLLAMA_HOST is correct."
                ) from e

            if resp.status_code != 200:
                if resp.status_code == 400 and "context length" in (resp.text or "").lower():
                    prompt2 = prompt
                    while len(prompt2) > 200:
                        prompt2 = self._truncate(prompt2[: max(200, len(prompt2) // 2)])
                        resp = self._client.post(url, json={"model": self._model, "prompt": prompt2})
                        if resp.status_code == 200:
                            break
                        if resp.status_code != 400 or "context length" not in (resp.text or "").lower():
                            raise RuntimeError(
                                f"Ollama embeddings failed with status {resp.status_code}: {resp.text[:300]}"
                            )
                    if resp.status_code != 200:
                        raise RuntimeError("Ollama embeddings input too long even after truncation.")
                else:
                    raise RuntimeError(
                        f"Ollama embeddings failed with status {resp.status_code}: {resp.text[:300]}"
                    )
            data = resp.json()
            emb = data.get("embedding")
            if not isinstance(emb, list):
                raise RuntimeError("Ollama embeddings response missing embedding.")
            out.append([float(v) for v in emb])
        return out

    def _truncate(self, text: str) -> str:
        t = (text or "").strip()
        if len(t) <= self._max_chars:
            return t
        return t[: self._max_chars]


def build_embedder(
    *,
    provider: str,
    model: str,
    dimension: int,
    ollama_host: str | None = None,
    embeddings_max_chars: int | None = None,
) -> Embedder:
    p = (provider or "").strip().lower()
    if p in {"", "ollama"}:
        if not model:
            raise RuntimeError("EMBEDDINGS_MODEL is required for Ollama embeddings (e.g. nomic-embed-text).")
        host = ollama_host or "http://localhost:11434"
        max_chars = 6000 if embeddings_max_chars is None else int(embeddings_max_chars)
        return OllamaEmbedder(host=host, model=model, max_chars=max_chars)
    if p in {"hash", "hashing"}:
        return HashingEmbedder(dimension=dimension)
    raise ValueError(f"Unsupported EMBEDDINGS_PROVIDER: {provider}")
