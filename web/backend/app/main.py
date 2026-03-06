"""FastAPI application for Spectra web UI."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import get_settings
from .routers import auto_detect, compare, datasets, experiments, llm, models, presets, results, share

app = FastAPI(
    title="Spectra API",
    description="Web API for Spectra evaluation engine",
    version="1.0.0",
)

# Dynamic CORS origins from config (includes localhost + any SPECTRA_CORS_ORIGINS)
settings = get_settings()
settings.validate_startup()
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auto_detect.router)
app.include_router(compare.router)
app.include_router(datasets.router)
app.include_router(experiments.router)
app.include_router(models.router)
app.include_router(presets.router)
app.include_router(results.router)
app.include_router(share.router)
app.include_router(llm.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "ok",
        "hosted": settings.is_hosted,
        "auth_enabled": settings.auth_enabled,
    }
