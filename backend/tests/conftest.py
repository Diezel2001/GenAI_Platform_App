"""Pytest configuration for backend tests."""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO

# Import FastAPI app
from app.main import app

## Fixtures ##

@pytest.fixture
def client():
    """Create a TestClient for the FastAPI application."""
    from fastapi.testclient import TestClient
    return TestClient(app)


@pytest.fixture
def mock_pdf_content():
    """Mock PDF content for testing document upload."""
    pdf_content = b"%PDF-1.4\nMock PDF content for testing"
    return BytesIO(pdf_content)


@pytest.fixture
def mock_upload_file(mock_pdf_content):
    """Create a mock UploadFile."""
    from fastapi import UploadFile
    
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = "test.pdf"
    mock_file.content_type = "application/pdf"
    mock_file.read = lambda: mock_pdf_content.getvalue()
    mock_file.seek = lambda pos: None
    
    return mock_file


@pytest.fixture
def mock_search_result():
    """Mock search result for testing query endpoints."""
    from app.services.rag.rag_schemas import SearchResult
    
    return SearchResult(
        id="test-doc-1",
        text="This is a sample document text for testing.",
        score=0.95,
        metadata={"source": "test.pdf", "page": 1}
    )


@pytest.fixture
def mock_vector_store():
    """Mock vector store for testing query endpoints."""
    mock_store = MagicMock()
    mock_store.query.return_value = []
    return mock_store


@pytest.fixture
def mock_embedding_model():
    """Mock embedding model for testing."""
    mock_model = MagicMock()
    mock_model.embed_query.return_value = [0.1] * 384  # Mock embedding vector
    return mock_model


## Test Helpers ##

def create_mock_file(content: bytes, filename: str, content_type: str):
    """Create a mock UploadFile for testing."""
    from fastapi import UploadFile
    
    mock_file = MagicMock(spec=UploadFile)
    mock_file.filename = filename
    mock_file.content_type = content_type
    mock_file.read = lambda: content
    mock_file.seek = lambda pos: None
    
    return mock_file
