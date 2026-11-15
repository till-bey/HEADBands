"""FastAPI entry point and service container lifespan."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.logging import configure_logging, get_logger

logger = get_logger(__name__)


def _resolve_frontend_dir() -> Path | None:
    candidates = [
        Path(__file__).resolve().parents[2] / "frontend",
        Path("/srv/frontend"),
    ]
    for candidate in candidates:
        if (candidate / "index.html").is_file():
            return candidate
    return None


def create_app(settings_override: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings_override or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        logger.info(
            "Starting Swiss Multilingual Voice & Robotics Assistant",
            extra={
                "app_env": settings.app_env,
                "gcp_region": settings.gcp_region,
                "whisper_device": settings.whisper_device,
                "whisper_compute_type": settings.whisper_compute_type,
                "robotics_enabled": settings.robotics_enabled,
            },
        )
        # Initialize runtime container if present
        from app.container import build_service_container

        app.state.container = build_service_container(settings)
        app.state.ready = True
        try:
            yield
        finally:
            app.state.ready = False
            await app.state.container.shutdown()
            logger.info("Backend shut down cleanly")

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        description=(
            "Multilingual voice & embodied robotics platform for Swiss German, "
            "Standard German, and French on Google Cloud (europe-west6)."
        ),
        lifespan=lifespan,
    )

    if settings_override is not None:
        app.dependency_overrides[get_settings] = lambda: settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register API routers
    app.include_router(health_router)

    from app.api.conversation import router as conversation_router
    from app.api.robotics import router as robotics_router
    from app.api.voice import router as voice_router

    app.include_router(voice_router)
    app.include_router(conversation_router)
    app.include_router(robotics_router)

    # Optionally serve zero-credential Web UI directly from Cloud Run / local uvicorn
    if settings.serve_frontend:
        frontend_dir = _resolve_frontend_dir()
        if frontend_dir is not None:
            static_dir = frontend_dir / "assets"
            if static_dir.is_dir():
                app.mount("/assets", StaticFiles(directory=str(static_dir)), name="assets")

            @app.middleware("http")
            async def revalidate_web_ui(request, call_next):  # type: ignore[no-untyped-def]
                # The page and its assets change with each deploy: browsers check the ETag on every load
                # (a cheap 304 when unchanged) instead of guessing a cache lifetime and showing old files.
                response = await call_next(request)
                if request.url.path == "/" or request.url.path.startswith("/assets/"):
                    response.headers["Cache-Control"] = "no-cache"
                return response

            @app.get("/", include_in_schema=False)
            async def serve_index() -> FileResponse:
                return FileResponse(str(frontend_dir / "index.html"))

    return app


app = create_app()
