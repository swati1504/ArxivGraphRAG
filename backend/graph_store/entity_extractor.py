from __future__ import annotations

import json
import re
from typing import Any

import httpx

from backend.config import get_settings
from backend.graph_store.graph_writer import ExtractedRelation, NodeRef


_JSON_BLOCK = re.compile(r"\{[\s\S]*\}")


class EntityExtractor:
    def __init__(self) -> None:
        self._settings = get_settings()
        self._client = httpx.Client(timeout=180.0)

    def extract_from_chunk(
        self,
        *,
        paper_id: str,
        paper_title: str | None,
        paper_authors: list[str] | None,
        chunk_index: int,
        chunk_text: str,
    ) -> list[ExtractedRelation]:
        provider = (self._settings.graph_provider or "").strip().lower()
        if provider != "gemini":
            raise RuntimeError(f"Unsupported GRAPH_PROVIDER: {self._settings.graph_provider}")
        if not self._settings.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is required for graph extraction (Gemini).")

        model = (self._settings.graph_gemini_model or "gemini-1.5-flash-latest").strip()
        url = self._gemini_generate_url(model=model, api_key=self._settings.gemini_api_key)

        system = (
            "Extract a small, high-precision knowledge graph from the given research paper text chunk. "
            "Return JSON only."
        )
        user = self._build_user_prompt(paper_id=paper_id, paper_title=paper_title, chunk_text=chunk_text)
        payload = {
            "systemInstruction": {"parts": [{"text": system}]},
            "contents": [{"role": "user", "parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.1, "maxOutputTokens": 700},
        }

        try:
            resp = self._client.post(url, json=payload)
        except httpx.RequestError as e:
            raise RuntimeError("Gemini graph extraction request failed.") from e
        if resp.status_code == 404 and "not found" in (resp.text or "").lower():
            model2 = self._pick_gemini_model(api_key=self._settings.gemini_api_key)
            resp = self._client.post(self._gemini_generate_url(model=model2, api_key=self._settings.gemini_api_key), json=payload)
        if resp.status_code != 200:
            raise RuntimeError(f"Gemini graph extraction failed with status {resp.status_code}: {resp.text[:300]}")

        text = self._extract_text(resp.json())
        data = self._parse_json(text)
        author_map = self._author_key_map(paper_authors or [])
        return self._to_relations(paper_id=paper_id, chunk_index=chunk_index, payload=data, author_map=author_map)

    def _build_user_prompt(self, *, paper_id: str, paper_title: str | None, chunk_text: str) -> str:
        title = (paper_title or "").strip()
        chunk = (chunk_text or "").strip()
        if len(chunk) > 8000:
            chunk = chunk[:8000]
        schema = {
            "entities": [{"type": "Concept|Method|Institution|Author", "name": "string"}],
            "relations": [
                {
                    "type": "PROPOSES|STUDIES|CITES|AFFILIATED_WITH",
                    "source": {"type": "Paper|Author|Institution|Concept|Method", "id": "paper_id_if_Paper", "name": "string_if_not_Paper"},
                    "target": {"type": "Paper|Author|Institution|Concept|Method", "id": "paper_id_if_Paper", "name": "string_if_not_Paper"},
                    "evidence": "short quote or paraphrase from the chunk",
                    "confidence": 0.0,
                }
            ],
        }
        return (
            f"Paper id: {paper_id}\n"
            f"Paper title: {title}\n\n"
            "Allowed node types: Paper, Author, Institution, Concept, Method.\n"
            "Allowed relation types:\n"
            "- PROPOSES: Paper -> Method\n"
            "- STUDIES: Paper -> Concept\n"
            "- CITES: Paper -> Paper\n"
            "- CONTRADICTS: Paper -> Paper (semantic opposition; only if explicitly stated)\n"
            "- AFFILIATED_WITH: Author -> Institution\n\n"
            "Rules:\n"
            "- Output JSON only (no markdown, no code fences).\n"
            "- Prefer precision over recall; extract at most 6 relations.\n"
            "- If unsure, omit the relation.\n"
            "- Use the given paper id for the Paper node.\n\n"
            f"JSON schema example (types only):\n{json.dumps(schema, ensure_ascii=False)}\n\n"
            f"Chunk:\n{chunk}\n"
        )

    def _extract_text(self, payload: Any) -> str:
        candidates = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(candidates, list) or not candidates:
            raise RuntimeError("Gemini returned no candidates for graph extraction.")
        content = candidates[0].get("content") if isinstance(candidates[0], dict) else None
        parts = content.get("parts") if isinstance(content, dict) else None
        if not isinstance(parts, list) or not parts:
            raise RuntimeError("Gemini returned empty content for graph extraction.")
        return "".join([str(p.get("text") or "") for p in parts if isinstance(p, dict)]).strip()

    def _parse_json(self, text: str) -> dict[str, Any]:
        raw = (text or "").strip()
        try:
            data = json.loads(raw)
            if isinstance(data, dict):
                return data
        except Exception:
            pass
        m = _JSON_BLOCK.search(raw)
        if not m:
            raise RuntimeError("Failed to parse JSON from Gemini graph extraction response.")
        try:
            data = json.loads(m.group(0))
        except Exception as e:
            raise RuntimeError("Failed to parse JSON from Gemini graph extraction response.") from e
        if not isinstance(data, dict):
            raise RuntimeError("Gemini graph extraction JSON must be an object.")
        return data

    def _to_relations(
        self,
        *,
        paper_id: str,
        chunk_index: int,
        payload: dict[str, Any],
        author_map: dict[str, str],
    ) -> list[ExtractedRelation]:
        rels = payload.get("relations")
        if not isinstance(rels, list):
            return []
        out: list[ExtractedRelation] = []
        for r in rels:
            if not isinstance(r, dict):
                continue
            rtype_raw = str(r.get("type") or "").strip().upper()
            rtype = "CONTRADICTS" if rtype_raw in {"CHALLENGES", "CONTRADICTS"} else rtype_raw
            if rtype not in {"PROPOSES", "STUDIES", "CITES", "AFFILIATED_WITH", "CONTRADICTS"}:
                continue
            src = self._parse_node_ref(r.get("source"), default_paper_id=paper_id, author_map=author_map)
            tgt = self._parse_node_ref(r.get("target"), default_paper_id=None, author_map=author_map)
            if src is None or tgt is None:
                continue
            if rtype == "PROPOSES" and not (src.type == "Paper" and tgt.type == "Method"):
                continue
            if rtype == "STUDIES" and not (src.type == "Paper" and tgt.type == "Concept"):
                continue
            if rtype == "CITES" and not (src.type == "Paper" and tgt.type == "Paper"):
                continue
            if rtype == "AFFILIATED_WITH" and not (src.type == "Author" and tgt.type == "Institution"):
                continue
            if rtype == "CONTRADICTS" and not (src.type == "Paper" and tgt.type == "Paper"):
                continue
            evidence = str(r.get("evidence") or "").strip() or None
            conf = r.get("confidence")
            try:
                conf_f = float(conf) if conf is not None else None
            except Exception:
                conf_f = None
            out.append(
                ExtractedRelation(
                    source=src,
                    type=rtype,  # type: ignore[arg-type]
                    target=tgt,
                    chunk_index=int(chunk_index),
                    evidence=evidence,
                    confidence=conf_f,
                )
            )
        return out

    def _parse_node_ref(
        self,
        obj: Any,
        *,
        default_paper_id: str | None,
        author_map: dict[str, str],
    ) -> NodeRef | None:
        if not isinstance(obj, dict):
            if default_paper_id:
                return NodeRef(type="Paper", id=default_paper_id)
            return None
        ntype = str(obj.get("type") or "").strip()
        if ntype not in {"Paper", "Author", "Institution", "Concept", "Method"}:
            return None
        if ntype == "Paper":
            pid = str(obj.get("id") or "").strip() or (default_paper_id or "")
            if not pid:
                return None
            return NodeRef(type="Paper", id=pid)
        name = str(obj.get("name") or "").strip()
        if not name:
            return None
        if ntype == "Author":
            k = self._author_key(name)
            canonical = author_map.get(k)
            if not canonical:
                return None
            return NodeRef(type="Author", name=canonical)
        return NodeRef(type=ntype, name=name)  # type: ignore[arg-type]

    def _author_key(self, name: str) -> str:
        v = (name or "").strip()
        v = v.replace(".", " ")
        if "," in v:
            parts = [p.strip() for p in v.split(",", 1)]
            if len(parts) == 2 and parts[0] and parts[1]:
                v = f"{parts[1]} {parts[0]}"
        v = re.sub(r"\s+", " ", v)
        v = re.sub(r"[^\w\s\-:.]", "", v)
        return v.strip().lower()

    def _author_key_map(self, authors: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for a in authors:
            name = str(a or "").strip()
            if not name:
                continue
            out[self._author_key(name)] = name
        return out

    def _gemini_generate_url(self, *, model: str, api_key: str) -> str:
        m = model.strip()
        if m.startswith("models/"):
            m = m[len("models/") :]
        return f"https://generativelanguage.googleapis.com/v1beta/models/{m}:generateContent?key={api_key}"

    def _pick_gemini_model(self, *, api_key: str) -> str:
        try:
            resp = self._client.get(
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
        preferred = [n for n in candidates if "gemini-1.5-flash" in n]
        if preferred:
            return preferred[0]
        if candidates:
            return candidates[0]
        raise RuntimeError("No Gemini model found that supports generateContent.")
