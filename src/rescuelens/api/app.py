"""FastAPI application factory."""

from fastapi import FastAPI
from pydantic import BaseModel

from rescuelens import __version__
from rescuelens.config import get_settings
from rescuelens.logging import configure_logging


class Health(BaseModel):
    status: str
    version: str


def create_app() -> FastAPI:
    settings = get_settings()
    configure_logging(settings.log_level, json=settings.log_json)

    app = FastAPI(title="RescueLens", version=__version__)

    @app.get("/api/health")
    def health() -> Health:
        return Health(status="ok", version=__version__)

    return app


app = create_app()
