"""
RAG Pipeline

This module handles the complete RAG pipeline:
1. Load document (PDF)
2. Preprocess/normalize text
3. Chunk text
4. Embed chunks
5. Store in vector store

To switch between vector stores (Pinecone, Qdrant, Milvus, FAISS),
use the VECTOR_STORE_PROVIDER environment variable or configure manually.
"""

import os
from typing import List, Dict, Any, Optional
from fastapi import UploadFile

from app.services.rag.loaders import PDFLoader
from app.services.rag.chunking import PDFChunker
from app.services.rag import preprocessing
from app.services.rag.rag_schemas import RawDocument, RecordData
from app.services.rag.embedding_models import get_embedding_model, EmbeddingModel


# ============================================================
# Vector Store Configuration
# ============================================================

# DEFAULT_EMBEDDING_DIMENSION = 1536  # For OpenAI text-embedding-3-large
DEFAULT_EMBEDDING_DIMENSION = 1024 # For BGE large
# DEFAULT_EMBEDDING_DIMENSION = 768 # For instructor-large: 768

VECTOR_STORE_PROVIDER = os.environ.get("VECTOR_STORE_PROVIDER", "faiss").lower()


def get_vector_store():
    """Get the configured vector store instance."""
    from app.services.vector_store import get_vector_store as _get_vector_store
    
    # Configuration for each provider
    configs = {
        "faiss": {"dim": DEFAULT_EMBEDDING_DIMENSION},
        "pinecone": {
            "index_name": os.environ.get("PINECONE_INDEX", "documents"),
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
        },
        "qdrant": {
            "location": os.environ.get("QDRANT_LOCATION", ":memory:"),
            "collection_name": os.environ.get("QDRANT_COLLECTION", "documents"),
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
        },
        "milvus": {
            "collection_name": os.environ.get("MILVUS_COLLECTION", "documents"),
            "dimension": DEFAULT_EMBEDDING_DIMENSION,
        },
    }
    
    config = configs.get(VECTOR_STORE_PROVIDER, configs["faiss"])
    return _get_vector_store(VECTOR_STORE_PROVIDER, **config)


def get_embedding_model_instance() -> EmbeddingModel:
    """Get the embedding model instance."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "bge").lower()
    
    if provider == "openai":
        api_key = os.environ.get("OPENAI_API_KEY")
        return get_embedding_model("openai", api_key=api_key)
    elif provider == "bge":
        return get_embedding_model("bge")
    elif provider == "instructor":
        return get_embedding_model("instructor")
    else:
        # Default to BGE
        return get_embedding_model("bge")


# ============================================================
# RAG Store Function
# ============================================================

def rag_store(
    contents: bytes,
    file: UploadFile,
    ext: str,
    vector_store=None,
    embedding_model=None,
) -> Dict[str, Any]:
    """Complete RAG pipeline to store document in vector database."""
    # 1. Load document
    content = PDFLoader().load(contents)
    content_p = preprocessing.normalize_text(content)
    
    # 2. Create raw document
    raw_docu = preprocessing._extract_uploaded_raw_document(file, ext)
    raw_docu.content = content_p
    
    # 3. Chunk document
    chunks = PDFChunker().chunk_by_fixed_size(raw_docu)
    # Alternative: chunks = PDFChunker.chunk_by_page(raw_docu)
    # Alternative: chunks = PDFChunker.chunk_by_semantics(raw_docu)
    
    # 4. Get embedding model (or use provided)
    if embedding_model is None:
        embedding_model = get_embedding_model_instance()
    
    # 5. Embed chunks
    chunk_texts = [chunk.text for chunk in chunks]
    embeddings = embedding_model.embed_documents(chunk_texts)
    
    # 6. Create records with embeddings
    records = []
    for chunk, embedding in zip(chunks, embeddings):
        record = RecordData(
            id=chunk.id,
            text=chunk.text,
            embedding=embedding,
            metadata=chunk.metadata,
        )
        records.append(record)
    
    # 7. Get vector store (or use provided)
    if vector_store is None:
        vector_store = get_vector_store()
    
    # 8. Store in vector database
    vector_store.add_records(records)
    vector_store.persist()
    
    # 9. Return result
    return {
        "status": "success",
        "document_id": raw_docu.doc_id,
        "file_name": raw_docu.metadata.get("file_name"),
        "chunks_count": len(chunks),
        "vector_store": VECTOR_STORE_PROVIDER,
    }


# ============================================================
# Async Version (for FastAPI)
# ============================================================

async def rag_store_async(
    contents: bytes,
    file: UploadFile,
    ext: str,
    vector_store=None,
    embedding_model=None,
) -> Dict[str, Any]:
    """Async version of rag_store for use with FastAPI."""
    # For now, just call the sync version
    # In production, you might want to offload to a background task
    return rag_store(contents, file, ext, vector_store, embedding_model)
