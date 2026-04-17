from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from backend.routers.query import router as query_router
from backend.routers.evaluate import router as evaluate_router


def create_app() -> FastAPI:
    app = FastAPI(title="arXiv GraphRAG", version="0.1.0")
    app.include_router(query_router)
    app.include_router(evaluate_router)

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
        detail = str(exc) or "Internal Server Error"
        if len(detail) > 500:
            detail = detail[:500]
        return JSONResponse(status_code=500, content={"detail": detail, "error_type": exc.__class__.__name__})

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
