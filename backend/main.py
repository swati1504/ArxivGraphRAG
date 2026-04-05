from fastapi import FastAPI

from backend.routers.query import router as query_router


def create_app() -> FastAPI:
    app = FastAPI(title="arXiv GraphRAG", version="0.1.0")
    app.include_router(query_router)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    return app


app = create_app()
