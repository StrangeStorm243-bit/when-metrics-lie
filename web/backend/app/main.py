"""FastAPI application for Spectra web UI."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .routers import experiments, presets, results

app = FastAPI(
    title="Spectra API",
    description="Web API for Spectra evaluation engine",
    version="0.2.0",
)

# Enable CORS for local frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite default
        "http://localhost:3000",  # React default
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(experiments.router)
app.include_router(presets.router)
app.include_router(results.router)


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok"}
