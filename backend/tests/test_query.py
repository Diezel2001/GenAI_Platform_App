"""Tests for query API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from app.services.rag.rag_schemas import SearchResult


class TestQueryAPI:
    """Test class for query API endpoints."""
    
    @patch("app.api.query.get_embedding_model_instance")
    @patch("app.api.query.get_vector_store")
    def test_query_documents_success(self, mock_get_vector_store, mock_get_embedding, client):
        """Test that a valid query returns results."""
        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 384
        mock_get_embedding.return_value = mock_embedding_model
        
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = [
            SearchResult(
                id="doc-1",
                text="Test document content",
                score=0.95,
                metadata={"source": "test.pdf"}
            )
        ]
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make query request
        response = client.post(
            "/query/",
            json={
                "query": "What is this document about?",
                "k": 5
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["query"] == "What is this document about?"

        print(data["results"])
    
    @patch("app.api.query.get_embedding_model_instance")
    @patch("app.api.query.get_vector_store")
    def test_query_with_filters(self, mock_get_vector_store, mock_get_embedding, client):
        """Test query with metadata filters."""
        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 384
        mock_get_embedding.return_value = mock_embedding_model
        
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = []
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make query request with filters
        response = client.post(
            "/query/",
            json={
                "query": "Search term",
                "k": 3,
                "filters": {"source": "pdf", "tags": ["important"]}
            },
        )
        
        assert response.status_code == 200
        # Verify the vector store was called with filters
        mock_vector_store.query.assert_called_once()
        call_kwargs = mock_vector_store.query.call_args.kwargs
        assert call_kwargs["filters"] == {"source": "pdf", "tags": ["important"]}
    
    @patch("app.api.query.get_embedding_model_instance")
    @patch("app.api.query.get_vector_store")
    def test_query_empty_results(self, mock_get_vector_store, mock_get_embedding, client):
        """Test query that returns no results."""
        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 384
        mock_get_embedding.return_value = mock_embedding_model
        
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = []
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make query request
        response = client.post(
            "/query/",
            json={
                "query": "Nonexistent topic",
                "k": 5
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["results"] == []
        assert data["total_results"] == 0
    
    def test_query_invalid_request_missing_query(self, client):
        """Test that missing query field returns validation error."""
        response = client.post(
            "/query/",
            json={"k": 5},  # Missing "query" field
        )
        
        assert response.status_code == 422
    
    def test_query_invalid_request_empty_query(self, client):
        """Test that empty query string returns validation error."""
        response = client.post(
            "/query/",
            json={"query": "", "k": 5},
        )
        
        assert response.status_code == 422
    
    def test_query_invalid_k_value(self, client):
        """Test that invalid k value returns validation error."""
        response = client.post(
            "/query/",
            json={
                "query": "Test query",
                "k": 0  # Invalid: k must be >= 1
            },
        )
        
        assert response.status_code == 422
    
    @patch("app.api.query.get_embedding_model_instance")
    @patch("app.api.query.get_vector_store")
    def test_query_large_k_value(self, mock_get_vector_store, mock_get_embedding, client):
        """Test query with maximum allowed k value."""
        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 384
        mock_get_embedding.return_value = mock_embedding_model
        
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = []
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make query request with max k
        response = client.post(
            "/query/",
            json={
                "query": "Test query",
                "k": 100  # Maximum allowed
            },
        )
        
        assert response.status_code == 200


class TestQueryRAGEndpoint:
    """Test class for RAG query endpoint with LLM response."""
    
    @patch("app.api.query.get_embedding_model_instance")
    @patch("app.api.query.get_vector_store")
    def test_rag_query_without_llm_response(self, mock_get_vector_store, mock_get_embedding, client):
        """Test RAG query without LLM response generation."""
        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 384
        mock_get_embedding.return_value = mock_embedding_model
        
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = [
            SearchResult(
                id="doc-1",
                text="Document content here",
                score=0.9,
                metadata={"source": "test.pdf"}
            )
        ]
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make RAG query request
        response = client.post(
            "/query/rag",
            json={
                "query": "What is the content?",
                "k": 5,
                "include_llm_response": False
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert data["llm_response"] is None
    
    @patch("app.api.query.get_embedding_model_instance")
    @patch("app.api.query.get_vector_store")
    def test_rag_query_with_llm_response_requested(self, mock_get_vector_store, mock_get_embedding, client):
        """Test RAG query with LLM response requested."""
        # Setup mocks
        mock_embedding_model = MagicMock()
        mock_embedding_model.embed_query.return_value = [0.1] * 384
        mock_get_embedding.return_value = mock_embedding_model
        
        mock_vector_store = MagicMock()
        mock_vector_store.query.return_value = [
            SearchResult(
                id="doc-1",
                text="Sample document text for testing purposes.",
                score=0.95,
                metadata={"source": "test.pdf"}
            )
        ]
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make RAG query request with LLM response enabled
        response = client.post(
            "/query/rag",
            json={
                "query": "Summarize this document",
                "k": 5,
                "include_llm_response": True
            },
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "sources" in data
        assert data["llm_response"] is not None  # Not implemented yet
        print(data["llm_response"])


class TestQueryHealthEndpoint:
    """Test class for query health check endpoint."""
    
    @patch("app.api.query.get_vector_store")
    def test_query_health_healthy(self, mock_get_vector_store, client):
        """Test health check when query service is healthy."""
        # Setup mock
        mock_vector_store = MagicMock()
        mock_get_vector_store.return_value = mock_vector_store
        
        # Make health check request
        response = client.get("/query/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "vector_store" in data
    
    @patch("app.api.query.get_vector_store")
    def test_query_health_unhealthy(self, mock_get_vector_store, client):
        """Test health check when query service is unhealthy."""
        # Setup mock to raise exception
        mock_get_vector_store.side_effect = Exception("Vector store unavailable")
        
        # Make health check request
        response = client.get("/query/health")
        
        # Should still return 200 but with unhealthy status
        assert response.status_code == 200
        data = response.json()
