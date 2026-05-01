from __future__ import annotations

import os
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

_TRACE_ID: ContextVar[str | None] = ContextVar("arxiv_graphrag_trace_id", default=None)


def new_trace_id() -> str:
    trace_id = uuid.uuid4().hex
    _TRACE_ID.set(trace_id)
    return trace_id


def get_trace_id() -> str | None:
    return _TRACE_ID.get()


def set_trace_id(trace_id: str | None) -> Token[str | None]:
    return _TRACE_ID.set(trace_id)


def reset_trace_id(token: Token[str | None]) -> None:
    _TRACE_ID.reset(token)


def _truthy_env(name: str) -> bool:
    v = (os.getenv(name) or "").strip().lower()
    return v in {"1", "true", "yes", "y", "on"}


def langsmith_enabled() -> bool:
    if _truthy_env("LANGSMITH_TRACING") or _truthy_env("LANGSMITH_TRACING_V2"):
        return True
    return False


@contextmanager
def langsmith_request_context(*, trace_id: str, method: str, path: str) -> Iterator[None]:
    if not langsmith_enabled():
        yield
        return
    try:
        import langsmith as ls
    except Exception:
        yield
        return

    with ls.tracing_context(
        tags=["arxiv-graph-rag", f"http:{method.upper()}", f"path:{path}"],
        metadata={"trace_id": trace_id, "http.method": method.upper(), "http.path": path},
    ):
        yield


def ls_update_current(metadata: dict[str, Any] | None = None, tags: list[str] | None = None) -> None:
    if not langsmith_enabled():
        return
    try:
        import langsmith as ls
    except Exception:
        return
    try:
        rt = ls.get_current_run_tree()
    except Exception:
        return
    if rt is None:
        return
    if tags:
        try:
            rt.tags.extend(tags)
        except Exception:
            pass
    if metadata:
        try:
            rt.metadata.update(metadata)
        except Exception:
            pass


@contextmanager
def ls_span(
    *,
    name: str,
    run_type: str,
    inputs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    tags: list[str] | None = None,
) -> Iterator[dict[str, Any]]:
    start = time.perf_counter()
    out_meta: dict[str, Any] = {}
    if metadata:
        out_meta.update(metadata)

    if not langsmith_enabled():
        try:
            yield out_meta
        finally:
            out_meta["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return

    try:
        import langsmith as ls
    except Exception:
        try:
            yield out_meta
        finally:
            out_meta["latency_ms"] = int((time.perf_counter() - start) * 1000)
        return

    with ls.trace(name=name, run_type=run_type, inputs=inputs or {}, metadata=out_meta, tags=tags or []) as rt:
        try:
            yield out_meta
        except Exception as e:
            out_meta["latency_ms"] = int((time.perf_counter() - start) * 1000)
            try:
                rt.end(error=str(e))
            except Exception:
                pass
            raise
        else:
            out_meta["latency_ms"] = int((time.perf_counter() - start) * 1000)
            try:
                rt.end(outputs={"ok": True, "metadata": out_meta})
            except Exception:
                pass
