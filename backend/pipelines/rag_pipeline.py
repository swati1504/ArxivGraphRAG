import time

import httpx
from anthropic import Anthropic

from backend.config import get_settings
from backend.observability import ls_span, ls_update_current
from backend.schemas.query import QueryResponse, RetrievedContext, UsageMetrics
from backend.vector_store.retriever import VectorRetriever


class RAGPipeline:
    def __init__(self, *, retriever: VectorRetriever) -> None:
        self._retriever = retriever

    def run(self, *, question: str, top_k: int) -> QueryResponse:
        start = time.perf_counter()
        with ls_span(
            name="rag_run",
            run_type="chain",
            inputs={"question_chars": len(question or ""), "top_k": int(top_k)},
            metadata={"pipeline": "rag"},
        ) as meta:
            contexts = self._retriever.retrieve(question=question, top_k=top_k)
            meta["retrieval.chunks_retrieved"] = len(contexts)
        return self.run_with_contexts(question=question, contexts=contexts, start=start)

    def run_with_contexts(
        self,
        *,
        question: str,
        contexts: list[RetrievedContext],
        start: float | None = None,
    ) -> QueryResponse:
        t0 = time.perf_counter() if start is None else start
        if not contexts:
            latency_ms = int((time.perf_counter() - t0) * 1000)
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

        answer, usage = self.synthesize(question=question, contexts=contexts)
        latency_ms = int((time.perf_counter() - t0) * 1000)
        metrics = UsageMetrics(
            latency_ms=latency_ms,
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            total_cost_usd=usage.get("total_cost_usd"),
        )

        return QueryResponse(answer=answer, contexts=contexts, metrics=metrics)

    def synthesize(self, *, question: str, contexts: list[RetrievedContext]) -> tuple[str, dict]:
        system, user = self._build_prompt(question=question, contexts=contexts)
        return self.synthesize_with_prompt(system=system, user=user)

    def synthesize_with_prompt(self, *, system: str, user: str) -> tuple[str, dict]:
        settings = get_settings()
        provider = (settings.rag_provider or "").strip().lower()
        model_name = None
        if provider == "anthropic":
            if not settings.anthropic_api_key:
                raise RuntimeError("ANTHROPIC_API_KEY is required for answer synthesis.")
            model_name = settings.rag_reasoning_model
            client = Anthropic(api_key=settings.anthropic_api_key)
            with ls_span(
                name="synthesize",
                run_type="llm",
                inputs={"system_chars": len(system or ""), "user_chars": len(user or "")},
                metadata={"llm.provider": "anthropic", "llm.model": str(model_name or "")},
            ):
                msg = client.messages.create(
                    model=settings.rag_reasoning_model,
                    max_tokens=900,
                    temperature=0.2,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                )
            answer = "".join([b.text for b in msg.content if getattr(b, "type", None) == "text"]).strip()
            prompt_tokens = getattr(getattr(msg, "usage", None), "input_tokens", None)
            completion_tokens = getattr(getattr(msg, "usage", None), "output_tokens", None)
            usage = self._usage_dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            ls_update_current(
                metadata={
                    "llm.provider": "anthropic",
                    "llm.model": str(model_name or ""),
                    "llm.prompt_tokens": usage.get("prompt_tokens"),
                    "llm.completion_tokens": usage.get("completion_tokens"),
                    "llm.total_cost_usd": usage.get("total_cost_usd"),
                }
            )
            return answer, usage

        if provider == "gemini":
            if not settings.gemini_api_key:
                raise RuntimeError("GEMINI_API_KEY is required for answer synthesis.")
            model = (settings.rag_gemini_model or "gemini-1.5-flash-latest").strip()
            model_name = model
            payload = {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": user}]}],
                "generationConfig": {"temperature": 0.2, "maxOutputTokens": 900},
            }
            with ls_span(
                name="synthesize",
                run_type="llm",
                inputs={"system_chars": len(system or ""), "user_chars": len(user or "")},
                metadata={"llm.provider": "gemini", "llm.model": model},
            ):
                resp = self._gemini_generate_with_retries(
                    model=model,
                    api_key=settings.gemini_api_key,
                    payload=payload,
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
            usage_meta = payload.get("usageMetadata") if isinstance(payload, dict) else None
            prompt_tokens = usage_meta.get("promptTokenCount") if isinstance(usage_meta, dict) else None
            completion_tokens = usage_meta.get("candidatesTokenCount") if isinstance(usage_meta, dict) else None
            usage = self._usage_dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
            ls_update_current(
                metadata={
                    "llm.provider": "gemini",
                    "llm.model": model,
                    "llm.prompt_tokens": usage.get("prompt_tokens"),
                    "llm.completion_tokens": usage.get("completion_tokens"),
                    "llm.total_cost_usd": usage.get("total_cost_usd"),
                }
            )
            return answer, usage

        host = (settings.ollama_host or "http://localhost:11434").rstrip("/")
        model = (settings.rag_ollama_model or "").strip()
        if not model:
            model = self._pick_ollama_model(host=host)
        model_name = model
        prompt = f"System:\n{system}\n\nUser:\n{user}\n"
        try:
            with ls_span(
                name="synthesize",
                run_type="llm",
                inputs={"prompt_chars": len(prompt or "")},
                metadata={"llm.provider": "ollama", "llm.model": model},
            ):
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
            with ls_span(
                name="synthesize_retry",
                run_type="llm",
                inputs={"prompt_chars": len(prompt or "")},
                metadata={"llm.provider": "ollama", "llm.model": model2},
            ):
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
            model_name = model2
        if resp.status_code != 200:
            raise RuntimeError(f"Ollama generation failed with status {resp.status_code}: {resp.text[:300]}")
        payload = resp.json()
        answer = str(payload.get("response") or "").strip()
        prompt_tokens = payload.get("prompt_eval_count")
        completion_tokens = payload.get("eval_count")
        usage = self._usage_dict(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
        ls_update_current(
            metadata={
                "llm.provider": "ollama",
                "llm.model": str(model_name or ""),
                "llm.prompt_tokens": usage.get("prompt_tokens"),
                "llm.completion_tokens": usage.get("completion_tokens"),
                "llm.total_cost_usd": usage.get("total_cost_usd"),
            }
        )
        return answer, usage

    def _usage_dict(self, *, prompt_tokens, completion_tokens) -> dict:
        pt = int(prompt_tokens) if prompt_tokens is not None else None
        ct = int(completion_tokens) if completion_tokens is not None else None
        total = (pt + ct) if (pt is not None and ct is not None) else None
        settings = get_settings()
        cost = None
        if settings.rag_input_cost_per_1k is not None and settings.rag_output_cost_per_1k is not None:
            if pt is not None and ct is not None:
                cost = (pt / 1000.0) * float(settings.rag_input_cost_per_1k) + (ct / 1000.0) * float(
                    settings.rag_output_cost_per_1k
                )
        return {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": total, "total_cost_usd": cost}

    def _build_prompt(self, *, question: str, contexts: list[RetrievedContext]) -> tuple[str, str]:
        context_block = "\n\n".join(
            [
                f"Source: {c.citation.paper_id}:{c.citation.chunk_index}\nTitle: {c.citation.title or ''}\nText:\n{c.text}"
                for c in contexts
            ]
        )
        system = (
            "You answer questions using only the provided sources. "
            "If the sources are insufficient, say what is missing. "
            "Cite claims with brackets like [paper_id:chunk_index]. "
            "Do not ask follow-up questions. Do not add extra sections like 'Do you want me to...'. "
            "Every paragraph must include at least one citation."
        )
        user = f"Question:\n{question}\n\nSources:\n{context_block}"
        return system, user

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

    def _gemini_generate_with_retries(self, *, model: str, api_key: str, payload: dict) -> httpx.Response:
        url = self._gemini_generate_url(model=model, api_key=api_key)
        resp = self._gemini_post_with_retries(url=url, payload=payload)
        if resp.status_code == 404 and "not found" in (resp.text or "").lower():
            model2 = self._pick_gemini_model(api_key=api_key)
            url2 = self._gemini_generate_url(model=model2, api_key=api_key)
            resp = self._gemini_post_with_retries(url=url2, payload=payload)
        return resp

    def _gemini_post_with_retries(self, *, url: str, payload: dict, max_attempts: int = 4) -> httpx.Response:
        last_resp: httpx.Response | None = None
        for attempt in range(1, max_attempts + 1):
            try:
                resp = httpx.post(url, json=payload, timeout=180.0)
            except httpx.RequestError:
                if attempt == max_attempts:
                    raise RuntimeError("Gemini synthesis request failed after retries.")
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            last_resp = resp
            if resp.status_code in {429, 503} and attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            return resp
        if last_resp is not None:
            return last_resp
        raise RuntimeError("Gemini synthesis request failed without a response.")

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
