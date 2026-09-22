"""Application factory.

Engines are loaded during startup rather than on first request: loading the
spaCy pipeline takes several seconds, and doing it lazily means the first
caller after every deploy pays for it and the service looks intermittently
slow for reasons that never reproduce.

There is no module-level ``app`` instance, so importing this module does not
require a working configuration. Run it with:

    uvicorn app.main:create_app --factory
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from .config import Settings, get_settings
from .database import init_db
from .logging_config import configure_logging
from .routes import router
from .service import AnonymizationService

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings

    init_db(settings)

    # A service injected through create_app() is kept. Without this the only
    # way to start the app is to load the real models, which makes the test
    # suite slow and memory-hungry for no benefit.
    if not hasattr(app.state, "service"):
        app.state.service = AnonymizationService(settings)

    logger.info(
        "startup complete",
        extra={
            "engines": sorted(app.state.service.engines),
            "default_engine": settings.default_engine,
            "store_input_text": settings.store_input_text,
        },
    )
    yield
    logger.info("shutdown")


def create_app(
    settings: Settings | None = None, service: AnonymizationService | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title=settings.app_name,
        description=(
            "Detects and redacts sensitive entities in text. One document per "
            "call, or a batch. Every call is recorded."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs" if settings.environment == "dev" else None,
        redoc_url=None,
    )

    app.state.settings = settings
    if service is not None:
        app.state.service = service

    app.include_router(router)

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(
            url="/docs" if settings.environment == "dev" else "/health"
        )

    return app
