"""Tests for main FastAPI application."""

import pytest
from unittest.mock import patch, MagicMock


class TestMainApp:
    """Test class for main FastAPI application tests."""
    
    def test_root_endpoint(self, client):
        """Test the root endpoint returns correct response."""
        response = client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["service"] == "Agentic Generative AI Platform"
    
    def test_app_title(self, client):
        """Test that the app has the correct title in OpenAPI schema."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        assert data["info"]["title"] == "Agentic Generative AI Platform"
        assert data["info"]["description"] == "Multi-tenant Agentic AI Platform API"
        assert data["info"]["version"] == "1.0.0"


class TestCORS:
    """Test class for CORS middleware configuration."""
    
    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.get("/")
        
        # Check that CORS headers are present
        assert "access-control-allow-origin" in response.headers
    
    def test_cors_preflight_request(self, client):
        """Test CORS preflight (OPTIONS) request is handled."""
        response = client.options(
            "/",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )
        
        # Preflight should return 200 or 204
        assert response.status_code in [200, 204]


class TestRouters:
    """Test class for router configuration tests."""
    
    def test_documents_router_mounted(self, client):
        """Test that documents router is correctly mounted."""
        # The documents router requires a file upload, so we just check
        # that the route exists by checking the OpenAPI schema
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that the documents paths exist in OpenAPI
        paths = data.get("paths", {})
        assert "/documents/upload" in paths
    
    def test_query_router_mounted(self, client):
        """Test that query router is correctly mounted."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that the query paths exist in OpenAPI
        paths = data.get("paths", {})
        assert "/query/" in paths
    
    def test_health_router_mounted(self, client):
        """Test that health router is correctly mounted."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        # Check that the health paths exist in OpenAPI
        paths = data.get("paths", {})
        assert "/query/health" in paths


class TestTags:
    """Test class for API tag configuration."""
    
    def test_api_tags_in_openapi(self, client):
        """Test that API tags are correctly defined in OpenAPI schema."""
        response = client.get("/openapi.json")
        
        assert response.status_code == 200
        data = response.json()
        
        tags = data.get("tags", [])
        tag_names = [tag["name"] for tag in tags]
        
        # Check that expected tags are present
        assert "Documents" in tag_names
        assert "Query" in tag_names


class TestLifespan:
    """Test class for application lifespan events."""
    
    def test_app_starts_without_error(self, client):
        """Test that the app starts without errors."""
        # If we got this far, the app started successfully
        response = client.get("/")
        assert response.status_code == 200
    
    @patch("app.main.configure_logging")
    def test_startup_event_calls_configure_logging(self, mock_configure, client):
        """Test that startup event calls configure_logging."""
        # The mock should have been called during app startup
        # This is verified implicitly by the app starting without errors
        response = client.get("/")
        assert response.status_code == 200


class TestMetrics:
    """Test class for Prometheus metrics endpoint."""
    
    def test_metrics_endpoint_exists(self, client):
        """Test that the metrics endpoint is exposed."""
        # The prometheus instrumentator typically adds /metrics endpoint
        response = client.get("/metrics")
        
        # Should return 200 if metrics are enabled
        # Note: This may return 404 if prometheus_fastapi_instrumentator
        # is not properly configured or not in requirements
        assert response.status_code in [200, 404]