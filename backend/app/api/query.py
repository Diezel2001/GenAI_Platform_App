"""Query API endpoints for semantic document search."""

from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, Field
from langchain_community.llms import Ollama

# from app.services.llm.LocalLlmWrapper import OllamaLLM

from app.services.rag.rag_pipeline import (
    get_vector_store,
    get_embedding_model_instance,
    VECTOR_STORE_PROVIDER,
)
from app.services.rag.rag_schemas import SearchResult


class QueryRequest(BaseModel):
    """Request schema for query endpoint."""
    
    query: str = Field(
        ...,
        description="The query text to search for",
        min_length=1,
        max_length=5000,
        examples=["What is the main topic of the document?"]
    )
    k: int = Field(
        default=5,
        description="Number of results to return",
        ge=1,
        le=100,
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters for the search",
        examples=[{"source": "pdf", "tags": ["important"]}]
    )


class QueryResponse(BaseModel):
    """Response schema for query endpoint."""
    
    query: str = Field(..., description="The original query text")
    results: List[SearchResult] = Field(
        ...,
        description="List of search results sorted by relevance"
    )
    total_results: int = Field(..., description="Total number of results found")
    vector_store: str = Field(..., description="Vector store provider used")

router = APIRouter()

@router.post("/", response_model=QueryResponse)
async def query_documents(request: QueryRequest):
    """Search for documents using semantic similarity."""
    try:
        # Get embedding model and vector store
        embedding_model = get_embedding_model_instance()
        vector_store = get_vector_store()
        
        # Embed the query text
        query_embedding = embedding_model.embed_query(request.query)
        
        # Search the vector store
        results = vector_store.query(
            text=request.query,
            k=request.k,
            filters=request.filters,
            embedding=query_embedding,
        )
        
        # Return response
        return QueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            vector_store=VECTOR_STORE_PROVIDER,
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}"
        )


# ============================================================
# Alternative: Query with LLM Response (RAG)
# ============================================================


class RAGQueryRequest(BaseModel):
    """Request schema for RAG query with LLM-generated response."""
    
    query: str = Field(
        ...,
        description="The query text to search for",
        min_length=1,
        max_length=5000,
    )
    k: int = Field(
        default=5,
        description="Number of context chunks to retrieve",
        ge=1,
        le=20,
    )
    include_llm_response: bool = Field(
        default=False,
        description="Whether to generate an LLM response using retrieved context",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters",
    )


class RAGQueryResponse(BaseModel):
    """Response schema for RAG query."""
    
    query: str
    results: List[SearchResult]
    total_results: int
    vector_store: str
    llm_response: Optional[str] = None
    sources: Optional[List[Dict[str, Any]]] = None


@router.post("/rag", response_model=RAGQueryResponse)
async def rag_query_documents(request: RAGQueryRequest):
    """Search for documents and optionally generate an LLM response."""
    try:
        # Get embedding model and vector store
        embedding_model = get_embedding_model_instance()
        vector_store = get_vector_store()
        
        # Embed the query text
        query_embedding = embedding_model.embed_query(request.query)
        
        # Search the vector store
        results = vector_store.query(
            text=request.query,
            k=request.k,
            filters=request.filters,
            embedding=query_embedding,
        )
        
        # Prepare response
        response = RAGQueryResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            vector_store=VECTOR_STORE_PROVIDER,
        )
        
        # Generate LLM response if requested
        if request.include_llm_response and results:
            # Extract context from results
            context = "\n\n".join([
                f"Source {i+1}:\n{result.text}"
                for i, result in enumerate(results)
            ])
            
            llm = Ollama(model="mistral", url="http://localhost:11434")

            prompt = f"""Based on the following context, answer the user's question. 
            If the context doesn't contain relevant information, say so.

            Context:
            {context}

            Question: {request.query}

            Answer:"""
            llmresponse = llm.invoke(prompt)

            response.llm_response = llmresponse
            response.sources = [
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.text[:200] + "..." if len(result.text) > 200 else result.text,
                }
                for result in results
            ]
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing RAG query: {str(e)}"
        )


# ============================================================
# Health Check for Query Service
# ============================================================


@router.get("/health")
async def query_health():
    """Health check for the query service."""
    try:
        vector_store = get_vector_store()
        
        return {
            "status": "healthy",
            "vector_store": VECTOR_STORE_PROVIDER,
            "embedding_model": "configured",
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
        }
