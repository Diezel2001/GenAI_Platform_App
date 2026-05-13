# app/services/vector_store/__init__.py

import os
from typing import Dict, Any
from .vs_base import VectorStore   # ✅ FIXED (relative import)


def get_vector_store(provider: str, **kwargs) -> VectorStore:
    provider = provider.lower()

    if provider == "faiss":
        from .vs_faiss import FaissVectorStore
        return FaissVectorStore(**kwargs)

    elif provider == "pinecone":
        from .vs_pinecone import PineconeVectorStore
        return PineconeVectorStore(**kwargs)

    elif provider == "qdrant":
        from .vs_qdrant import QdrantVectorStore
        return QdrantVectorStore(**kwargs)

    elif provider == "milvus":
        from .vs_milvus import MilvusVectorStore
        return MilvusVectorStore(**kwargs)

    else:
        raise ValueError(f"Unknown provider: {provider}")


class VectorStoreConfig:
    PROVIDER: str = os.environ.get("VECTOR_STORE_PROVIDER", "faiss")

    FAISS_CONFIG: Dict[str, Any] = {
        "dim": 1536,
        "index_path": "faiss.index",
        "meta_path": "faiss_meta.pkl",
    }


def get_configured_vector_store() -> VectorStore:
    config = VectorStoreConfig()

    if config.PROVIDER == "faiss":
        return get_vector_store("faiss", **config.FAISS_CONFIG)

    raise ValueError(f"Unsupported provider: {config.PROVIDER}")