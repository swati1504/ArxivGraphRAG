import time

import httpx
from anthropic import Anthropic

from backend.config import get_settings
from backend.schemas.query import QueryResponse, UsageMetrics
from backend.vector_store.retriever import VectorRetriever


class RAGPipeline:
    def __init__(self, *, retriever: VectorRetriever) -> None:
        self._retriever = retriever

    def run(self, *, question: str, top_k: int) -> QueryResponse:
        settings = get_settings()
        start = time.perf_counter()
        contexts = self._retriever.retrieve(question=question, top_k=top_k)
        if not contexts:
            latency_ms = int((time.perf_counter() - start) * 1000)
            metrics = UsageMetrics(
                latency_ms=latency_ms,
                prompt_tokens=None,
                completion_tokens=None,
                total_tokens=None,
                total_cost_usd=None,
            )
            answer = (
                "Insufficient evidence in the retrieved sources. "
                "Try ingesting more papers, increasing top_k, or rephrasing the question."
            )
            return QueryResponse(answer=answer, contexts=[], metrics=metrics)

        context_block = "\n\n".join(
            [
                f"Source: {c.citation.paper_id}:{c.citation.chunk_index}\nTitle: {c.citation.title or ''}\nText:\n{c.text}"
                for c in contexts
            ]
        )

        system = (
            "You answer questions using only the provided sources. "
            "If the sources are insufficient, say what is missing. "
            "Cite claims with brackets like [paper_id:chunk_index]."
        )
        user = f"Question:\n{question}\n\nSources:\n{context_block}"

        provider = (settings.rag_provider or "").strip().lower()
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for answer synthesis.")
            client = Anthropic(api_key=settings.anthropic_api_key)
            msg = client.messages.create(
                model=settings.rag_reasoning_model,
                max_tokens=900,
                temperature=0.2,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            answer = "".join([b.text for b in msg.content if getattr(b, "type", None) == "text"]).strip()
        elif provider == "gemini":
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for answer synthesis.")

            model = (settings.rag_gemini_model or "gemini-1.5-flash-latest").strip()
            url = self._gemini_generate_url(model=model, api_key=settings.gemini_api_key)
            try:
                resp = httpx.post(
                    url,
                    json={
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": user}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
                    },
                    timeout=180.0,
                )
            except httpx.RequestError as e:
                raise RuntimeError("Gemini synthesis request failed. Ensure GEMINI_API_KEY is valid.") from e
            if resp.status_code == 404 and "not found" in (resp.text or "").lower():
                model2 = self._pick_gemini_model(api_key=settings.gemini_api_key)
                resp = httpx.post(
                    self._gemini_generate_url(model=model2, api_key=settings.gemini_api_key),
                    json={
                        "systemInstruction": {"parts": [{"text": system}]},
                        "contents": [{"role": "user", "parts": [{"text": user}]}],
                        "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
                    },
                    timeout=180.0,
                )
            if resp.status_code != 200:
                raise RuntimeError(f"Gemini synthesis failed with status {resp.status_code}: {resp.text[:300]}")
            payload = resp.json()
            candidates = payload.get("candidates") if isinstance(payload, dict) else None
            if not isinstance(candidates, list) or not candidates:
                raise RuntimeError("Gemini synthesis returned no candidates.")
            content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
            parts = content.get("parts") if isinstance(content, dict) else None
            if not isinstance(parts, list) or not parts:
                raise RuntimeError("Gemini synthesis returned an empty response.")
            answer = "".join([str(p.get("text") or "") for p in parts if isinstance(p, dict)]).strip()
        else:
            host = (settings.ollama_host or "http://localhost:11434").rstrip("/")
            model = (settings.rag_ollama_model or "").strip()
            if not model:
                model = self._pick_ollama_model(host=host)
            prompt = f"System:\n{system}\n\nUser:\n{user}\n"
            try:
                resp = httpx.post(
                    f"{host}/api/generate",
                    json={
                        "model": model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2},
                    },
                    timeout=180.0,
                )
            except httpx.RequestError as e:
                raise RuntimeError(
                    "Ollama generation request failed. Ensure Ollama is running and RAG_OLLAMA_MODEL is pulled."
                ) from e
            if resp.status_code == 404 and "not found" in (resp.text or "").lower():
                model2 = self._pick_ollama_model(host=host)
                resp = httpx.post(
                    f"{host}/api/generate",
                    json={
                        "model": model2,
                        "prompt": prompt,
                        "stream": False,
                        "options": {"temperature": 0.2},
                    },
                    timeout=180.0,
                )
            if resp.status_code != 200:
                raise RuntimeError(f"Ollama generation failed with status {resp.status_code}: {resp.text[:300]}")
            payload = resp.json()
            answer = str(payload.get("response") or "").strip()
        latency_ms = int((time.perf_counter() - start) * 1000)
        metrics = UsageMetrics(
            latency_ms=latency_ms,
            prompt_tokens=None,
            completion_tokens=None,
            total_tokens=None,
            total_cost_usd=None,
        )

        return QueryResponse(answer=answer, contexts=contexts, metrics=metrics)

    def _gemini_generate_url(self, *, model: str, api_key: str) -> str:
        m = model.strip()
        if m.startswith("models/"):
            m = m[len("models/") :]
        return f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"

    def _pick_gemini_model(self, *, api_key: str) -> str:
        try:
            resp = httpx.get(
                "https://generativelanguage.googleapis.com/v1beta/models",
                params={"key": api_key},
                timeout=20.0,
            )
        except httpx.RequestError as e:
            raise RuntimeError("Gemini is not reachable.") from e
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list Gemini models: {resp.status_code}")
        payload = resp.json()
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list) or not models:
            raise RuntimeError("Failed to list Gemini models.")

        candidates: list[str] = []
        for m in models:
            if not isinstance(m, dict):
                continue
            name = m.get("name")
            if not isinstance(name, str) or not name:
                continue
            methods = m.get("supportedGenerationMethods")
            if not isinstance(methods, list) or "generateContent" not in methods:
                continue
            candidates.append(name)
        if not candidates:
            raise RuntimeError("No Gemini model found that supports generateContent.")

        preferred = [n for n in candidates if "gemini-1.5-flash" in n]
        if preferred:
            return preferred[0]
        return candidates[0]

    def _pick_ollama_model(self, *, host: str) -> str:
        try:
            resp = httpx.get(f"{host}/api/tags", timeout=20.0)
        except httpx.RequestError as e:
            raise RuntimeError("Ollama is not reachable at OLLAMA_HOST.") from e
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to list Ollama models: {resp.status_code}")
        payload = resp.json()
        models = payload.get("models") if isinstance(payload, dict) else None
        if not isinstance(models, list):
            raise RuntimeError("Failed to list Ollama models.")
        for m in models:
            name = m.get("name") if isinstance(m, dict) else None
            if not name:
                continue
            if "embed" in str(name).lower():
                continue
            return str(name)
        raise RuntimeError("No non-embedding Ollama model found. Pull one (e.g. ollama pull gemma3:1b).")
