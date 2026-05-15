"""FastAPI entrypoint for the Anvil PCAM Lab dashboard."""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates

from anvil_pcam.core import PCAMEngine


def create_app() -> FastAPI:
    app = FastAPI(
        title="Anvil PCAM Lab",
        version="1.0.0",
        description="Precision-controlled associative memory retrieval laboratory.",
    )

    template_dir = Path(__file__).parent / "templates"
    app.state.templates = Jinja2Templates(directory=str(template_dir))
    app.state.pcam = PCAMEngine()

    from anvil_pcam.web.routes import router

    app.include_router(router)
    return app
