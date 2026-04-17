from __future__ import annotations

import json
import re
import time
from typing import Any

import httpx

from backend.config import get_settings
from backend.schemas.evaluate import JudgeMetrics


_JSON_OBJ = re.compile(r"\{[\s\S]*\}")


class LLMJudge:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.Client(timeout=180.0)

    def score(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str | None = None,
        reference_paper_ids: list[str] | None = None,
        retrieved_paper_ids: list[str] | None = None,
    ) -> JudgeMetrics:
        provider = (self._settings.eval_provider or "").strip().lower()
        if provider != "gemini":
            return JudgeMetrics()
        if not self._settings.gemini_api_key:
            return JudgeMetrics()

        system = (
            "You are an evaluation judge for RAG outputs. "
            "Score each metric between 0.0 and 1.0. Return JSON only."
        )
        user = self._build_prompt(
            question=question,
            answer=answer,
            contexts=contexts,
            reference_answer=reference_answer,
        )
        model = (self._settings.eval_gemini_model or "gemini-1.5-flash-latest").strip()
        url = self._gemini_generate_url(model=model, api_key=self._settings.gemini_api_key)

        try:
            resp = self._post_with_retries(
                url=url,
                payload={
                    "systemInstruction": {"parts": [{"text": system}]},
                    "contents": [{"role": "user", "parts": [{"text": user}]}],
                    "generationConfig": {"temperature": 0.0, "maxOutputTokens": 350},
                },
            )
        except httpx.RequestError:
            return JudgeMetrics()
        if resp.status_code != 200:
            return JudgeMetrics()
        text = self._extract_text(resp.json())
        data = self._parse_json(text)
        return JudgeMetrics(
            faithfulness=_to_float_01(data.get("faithfulness")),
            answer_relevancy=_to_float_01(data.get("answer_relevancy")),
            context_precision=_to_float_01(data.get("context_precision")),
            context_recall=_paper_recall(reference_paper_ids, retrieved_paper_ids),
            answer_correctness=_to_float_01(data.get("answer_correctness")),
        )

    def _build_prompt(
        self,
        *,
        question: str,
        answer: str,
        contexts: list[str],
        reference_answer: str | None = None,
    ) -> str:
        ctx = "\n\n".join([f"CONTEXT {i+1}:\n{c}" for i, c in enumerate(contexts[:8])])
        schema = {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_precision": 0.0,
            "answer_correctness": None,
        }
        ref_block = f"\nREFERENCE ANSWER:\n{reference_answer}\n" if reference_answer else ""
        return (
            "Return a JSON object with these keys:\n"
            + json.dumps(schema, ensure_ascii=False)
            + "\n\n"
            "Definitions:\n"
            "- faithfulness: answer claims are supported by the provided contexts\n"
            "- answer_relevancy: answer addresses the question directly\n"
            "- context_precision: how much of provided context is relevant/useful\n"
            "- answer_correctness: if reference answer is provided, semantic correctness vs reference (0..1), else null\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nCONTEXTS:\n{ctx}\n"
            + ref_block
        )

    def _extract_text(self, payload: Any) -> str:
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or not candidates:
            return ""
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts:
            return ""
        return "".join([str(p.get("text") or "") for p in parts if isinstance(p, dict)]).strip()

    def _parse_json(self, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        try:
            obj = json.loads(raw)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
        m = _JSON_OBJ.search(raw)
        if not m:
            return {}
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    def _gemini_generate_url(self, *, model: str, api_key: str) -> str:
        m = model.strip()
        if m.startswith("models/"):
            m = m[len("models/") :]
        return f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"

    def _post_with_retries(self, *, url: str, payload: dict, max_attempts: int = 4) -> httpx.Response:
        last_resp: httpx.Response | None = None
        for attempt in range(1, max_attempts + 1):
            resp = self._client.post(url, json=payload)
            last_resp = resp
            if resp.status_code in {429, 503} and attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
                continue
            return resp
        return last_resp if last_resp is not None else self._client.post(url, json=payload)


def _to_float_01(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except Exception:
        return None
    if v < 0.0:
        return 0.0
    if v > 1.0:
        return 1.0
    return v


def _paper_recall(reference_paper_ids: list[str] | None, retrieved_paper_ids: list[str] | None) -> float | None:
    if not reference_paper_ids:
        return None
    ref = {str(x).strip() for x in reference_paper_ids if str(x).strip()}
    if not ref:
        return None
    got = {str(x).strip() for x in (retrieved_paper_ids or []) if str(x).strip()}
    if not got:
        return 0.0
    hit = len(ref.intersection(got))
    return hit / len(ref)
