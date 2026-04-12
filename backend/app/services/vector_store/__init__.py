"""
Vector Store Factory

This module provides a unified interface for creating vector stores.
To switch between providers, simply change the provider name in get_vector_store().

Supported providers:
- faiss: Local FAISS vector store (no external dependencies)
- pinecone: Pinecone managed vector store
- qdrant: Qdrant (local or cloud)
- milvus: Milvus (local or Zilliz cloud)

Usage:
    from vector_store import get_vector_store
    
    # Using Pinecone
    vector_store = get_vector_store("pinecone", index_name="my-index", dimension=1536)
    
    # Using Qdrant
    vector_store = get_vector_store("qdrant", location=":memory:", dimension=1536)
    
    # Using Milvus
    vector_store = get_vector_store("milvus", collection_name="documents", dimension=1536)
    
    # Using FAISS (local)
    vector_store = get_vector_store("faiss", dim=1536)
"""

import os
from typing import Optional, Dict, Any, List
from vector_store.vs_base import VectorStore


def get_vector_store(
    provider: str,
    **kwargs
) -> VectorStore:
    """
    Get a vector store instance based on the provider.
    
    Args:
        provider: The vector store provider ("faiss", "pinecone", "qdrant", "milvus")
        **kwargs: Provider-specific configuration options
    
    Returns:
        VectorStore: An instance of the requested vector store
    
    Raises:
        ValueError: If the provider is not supported
    """
    provider = provider.lower()
    
    if provider == "faiss":
        from vector_store.vs_faiss import FaissVectorStore
        return FaissVectorStore(**kwargs)
    
    elif provider == "pinecone":
        from vector_store.vs_pinecone import PineconeVectorStore
        return PineconeVectorStore(**kwargs)
    
    elif provider == "qdrant":
        from vector_store.vs_qdrant import QdrantVectorStore
        return QdrantVectorStore(**kwargs)
    
    elif provider == "milvus":
        from vector_store.vs_milvus import MilvusVectorStore
        return MilvusVectorStore(**kwargs)
    
    else:
        raise ValueError(
            f"Unknown vector store provider: {provider}. "
            f"Supported providers: faiss, pinecone, qdrant, milvus"
        )


# ============================================================
# Configuration-based vector store (recommended for production)
# ============================================================

class VectorStoreConfig:
    """Configuration for vector store."""
    
    # Change this to switch between providers
    # Options: "faiss", "pinecone", "qdrant", "milvus"
    PROVIDER: str = os.environ.get("VECTOR_STORE_PROVIDER", "faiss")
    
    # Default configurations for each provider
    FAISS_CONFIG: Dict[str, Any] = {
        "dim": 1536,  # Embedding dimension
        "index_path": "faiss.index",
        "meta_path": "faiss_meta.pkl",
    }
    
    PINECONE_CONFIG: Dict[str, Any] = {
        "index_name": os.environ.get("PINECONE_INDEX", "documents"),
        "dimension": 1536,
        "metric": "cosine",
        "cloud": "aws",
        "region": "us-west-2",
    }
    
    QDRANT_CONFIG: Dict[str, Any] = {
        "location": os.environ.get("QDRANT_LOCATION", ":memory:"),
        "url": os.environ.get("QDRANT_URL"),  # For cloud
        "api_key": os.environ.get("QDRANT_API_KEY"),
        "collection_name": os.environ.get("QDRANT_COLLECTION", "documents"),
        "dimension": 1536,
        "distance": "Cosine",
    }
    
    MILVUS_CONFIG: Dict[str, Any] = {
        "collection_name": os.environ.get("MILVUS_COLLECTION", "documents"),
        "dimension": 1536,
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "host": os.environ.get("MILVUS_HOST", "localhost"),
        "port": os.environ.get("MILVUS_PORT", "19530"),
    }


def get_configured_vector_store() -> VectorStore:
    """
    Get a vector store instance based on the configuration.
    
    This is the recommended way to get a vector store in production,
    as it allows switching providers via environment variables.
    
    Returns:
        VectorStore: Configured vector store instance
    """
    config = VectorStoreConfig()
    provider = config.PROVIDER.lower()
    
    if provider == "faiss":
        return get_vector_store("faiss", **config.FAISS_CONFIG)
    elif provider == "pinecone":
        return get_vector_store("pinecone", **config.PINECONE_CONFIG)
    elif provider == "qdrant":
        return get_vector_store("qdrant", **config.QDRANT_CONFIG)
    elif provider == "milvus":
        return get_vector_store("milvus", **config.MILVUS_CONFIG)
    else:
        raise ValueError(f"Unknown VECTOR_STORE_PROVIDER: {config.PROVIDER}")


# Export all vector stores for direct import
from vector_store.vs_base import VectorStore
from vector_store.vs_faiss import FaissVectorStore
from vector_store.vs_pinecone import PineconeVectorStore
from vector_store.vs_qdrant import QdrantVectorStore
from vector_store.vs_milvus import MilvusVectorStore

__all__ = [
    "VectorStore",
    "FaissVectorStore",
    "PineconeVectorStore", 
    "QdrantVectorStore",
    "MilvusVectorStore",
    "get_vector_store",
    "get_configured_vector_store",
    "VectorStoreConfig",
]
