from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# from app.api.auth import router as auth_router
from app.api.documents import router as document_router
from app.api.query import router as query_router
# from app.api.health import router as health_router

# from app.core.config import settings
# from app.core.logging import configure_logging

# from prometheus_fastapi_instrumentator import Instrumentator


# -------------------------
# Lifespan Events
# -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    # configure_logging()
    print("🚀 Application starting up...")
    yield
    # Shutdown
    print("🛑 Application shutting down...")


# -------------------------
# App Initialization
# -------------------------

app = FastAPI(
    title="Agentic Generative AI Platform",
    description="Multi-tenant Agentic AI Platform API",
    version="1.0.0",
    lifespan=lifespan,
)


# -------------------------
# Middleware
# -------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------
# Metrics (Prometheus)
# -------------------------

# Instrumentator().instrument(app).expose(app)


# -------------------------
# Routers
# -------------------------

# app.include_router(auth_router, prefix="/auth", tags=["Auth"])
app.include_router(document_router, prefix="/documents", tags=["Documents"])
app.include_router(query_router, prefix="/query", tags=["Query"])


# -------------------------
# Root Endpoint
# -------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agentic Generative AI Platform"}
