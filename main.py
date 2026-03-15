"""
Application entry point.

This file:
    - Creates the FastAPI app
    - Registers routers
    - Handles startup/shutdown lifecycle
    - Configures logging

Nothing else. Business logic never lives here.
"""

import structlog
from fastapi.responses import FileResponse
import os
import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import webbrowser
from app.db.session import init_db
from app.api.routes import completion, feedback, analytics
from app.config import get_settings

settings = get_settings()

# ─────────────────────────────────────────────────────────
# STRUCTURED LOGGING SETUP
# structlog gives us JSON logs in production —
# every log line is machine-parseable by tools like
# Datadog, Splunk, CloudWatch.
# In development it gives us readable colored output.
# ─────────────────────────────────────────────────────────

structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.dev.ConsoleRenderer(),  # Switch to JSONRenderer in prod
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
)

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────
# LIFESPAN — runs on startup and shutdown
# This replaces the old @app.on_event("startup") pattern
# which is deprecated in modern FastAPI.
# ─────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before yield: runs on startup
    Code after yield:  runs on shutdown

    Interview talking point:
        "We use the lifespan context manager pattern instead of
        deprecated startup/shutdown events. On startup we initialize
        the database schema. In production this would be replaced
        by Alembic migrations running in a pre-deploy step — you
        never auto-migrate in production because it can cause
        downtime on large tables."
    """
    # ── Startup ───────────────────────────────────────────
    logger.info("Starting LLM Router API")
    await init_db()
    logger.info("Database initialized")
    logger.info("LLM Router API ready")
    webbrowser.open("http://127.0.0.1:8000/dashboard")

    yield  # App runs here

    # ── Shutdown ──────────────────────────────────────────
    logger.info("Shutting down LLM Router API")


# ─────────────────────────────────────────────────────────
# APP FACTORY
# ─────────────────────────────────────────────────────────

app = FastAPI(
    title="Smart LLM Model Routing Platform",
    description="""
    Intelligently routes prompts to the optimal LLM based on
    complexity, confidence, and historical feedback.

    Minimizes cost while preserving output quality.
    """,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",  # Swagger UI
    redoc_url="/redoc",  # ReDoc UI (alternative docs)
)

# CORS — allows browser clients to call the API
# In production, restrict origins to your actual domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────
# ROUTE REGISTRATION
# Each router handles a group of related endpoints.
# Prefix groups them in the Swagger UI.
# ─────────────────────────────────────────────────────────

app.include_router(completion.router, prefix="/api/v1", tags=["Completion"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(analytics.router, prefix="/api/v1", tags=["Analytics"])
@app.get("/dashboard", tags=["System"])
async def dashboard():
    path = os.path.join(os.path.dirname(__file__), "cost_control_llm_router_dashboard.html")
    return FileResponse(path, media_type="text/html")

@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "service": "llm-router",
        "version": "1.0.0",
    }

# uvicorn main:app --reload --port 8000