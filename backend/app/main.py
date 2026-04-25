from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

# from app.api.auth import router as auth_router
from app.api.documents import router as document_router
from app.api.query import router as query_router
from app.api.agent import router as agent_router
# from app.api.health import router as health_router
from langgraph.checkpoint.redis import RedisSaver  

from app.core.workflows import create_agent_workflow


# from app.core.config import settings
# from app.core.logging import configure_logging

# from prometheus_fastapi_instrumentator import Instrumentator

import os
from dotenv import load_dotenv

load_dotenv()

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")
DATABASE_URL = os.getenv("DATABASE_URL")



# -------------------------
# Lifespan Events
# -------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    with RedisSaver.from_conn_string(REDIS_URL) as checkpointer:
        checkpointer.setup()
        graph = create_agent_workflow().compile(
            checkpointer=checkpointer
        )

        app.state.graph = graph
        app.state.checkpointer = checkpointer

        print("🚀 Application starting up...")

        yield

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
app.include_router(agent_router, prefix="/agent", tags=["Agent"])

# -------------------------
# Root Endpoint
# -------------------------

@app.get("/")
async def root():
    return {"status": "ok", "service": "Agentic Generative AI Platform"}
