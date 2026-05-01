from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

try:
    from dotenv import load_dotenv

    _ENV_FILE = Path(__file__).resolve().parents[1] / ".env"
    load_dotenv(_ENV_FILE, override=False)
except Exception:
    pass

from backend.observability import langsmith_request_context, new_trace_id, reset_trace_id, set_trace_id
from backend.routers.query import router as query_router
from backend.routers.evaluate import router as evaluate_router


def create_app() -> FastAPI:
    app = FastAPI(title="arXiv GraphRAG", version="0.1.0")
    static_dir = Path(__file__).resolve().parent / "static"
    invite_page = static_dir / "invite" / "engagement.html"

    try:
        from langsmith.middleware import TracingMiddleware

        app.add_middleware(TracingMiddleware)
    except Exception:
        pass

    @app.middleware("http")
    async def trace_id_middleware(request: Request, call_next):
        incoming = (request.headers.get("x-trace-id") or request.headers.get("x-request-id") or "").strip()
        trace_id = incoming or new_trace_id()
        token = set_trace_id(trace_id)
        try:
            with langsmith_request_context(trace_id=trace_id, method=request.method, path=request.url.path):
                response = await call_next(request)
        finally:
            reset_trace_id(token)
        response.headers["X-Trace-Id"] = trace_id
        return response

    app.include_router(query_router)
    app.include_router(evaluate_router)
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        detail = str(exc) or "Internal Server Error"
        if len(detail) > 500:
            detail = detail[:500]
        return JSONResponse(status_code=500, content={"detail": detail, "error_type": exc.__class__.__name__})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/invite/engagement", include_in_schema=False)
    def engagement_invite() -> FileResponse:
        return FileResponse(invite_page)

    return app


app = create_app()
