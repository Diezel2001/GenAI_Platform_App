"""Tests for documents API endpoints."""

import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO


class TestDocumentsAPI:
    """Test class for documents API endpoints."""
    
    def test_upload_document_invalid_content_type(self, client):
        """Test that uploading a non-PDF file returns a 400 error."""
        # Create a text file instead of PDF
        file_content = b"This is a text file"
        response = client.post(
            "/documents/upload",
            files={"file": ("test.txt", file_content, "text/plain")},
        )
        
        assert response.status_code == 400
        assert "Invalid file type" in response.json()["detail"]
    
    def test_upload_document_invalid_extension(self, client):
        """Test that uploading a file with invalid extension returns a 400 error."""
        # Create a file with .doc extension
        file_content = b"%PDF-1.4\nMock content"
        response = client.post(
            "/documents/upload",
            files={"file": ("test.doc", file_content, "application/pdf")},
        )
        
        assert response.status_code == 400
        assert "Invalid file extension" in response.json()["detail"]
    
    @patch("app.api.documents.rag_store_async")
    def test_upload_document_success(self, mock_rag_store, client):
        """Test that uploading a valid PDF returns a 200 status code."""
        # Mock the rag_store_async function
        mock_rag_store.return_value = {
            "status": "success",
            "document_id": "test-doc-123",
            "message": "Document processed successfully",
        }
        
        # Create a valid PDF file
        file_content = b"%PDF-1.4\nMock PDF content for testing"
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", file_content, "application/pdf")},
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert "document_id" in data
    
    @patch("app.api.documents.rag_store_async")
    def test_upload_document_processing_error(self, mock_rag_store, client):
        """Test that errors during processing are handled correctly."""
        # Mock the rag_store_async function to raise an exception
        mock_rag_store.side_effect = Exception("Processing failed")
        
        # Create a valid PDF file
        file_content = b"%PDF-1.4\nMock PDF content"
        
        # Use raise_server_exceptions=False to catch the 500 response
        response = client.post(
            "/documents/upload",
            files={"file": ("test.pdf", file_content, "application/pdf")},
        )
        
        # The endpoint returns 500 when an unhandled exception occurs
        assert response.status_code == 500


class TestDocumentsAPIEdgeCases:
    """Test class for edge cases in documents API."""
    
    def test_upload_empty_file(self, client):
        """Test that uploading an empty file raises ValueError."""
        file_content = b""
        response = client.post(
            "/documents/upload",
            files={"file": ("empty.pdf", file_content, "application/pdf")},
        )
        
        # Should return 400 with the specific error message
        assert response.status_code == 400
        assert "Cannot read an empty file" in response.json()["detail"]
    
    def test_upload_no_file(self, client):
        """Test that not providing a file is handled."""
        response = client.post("/documents/upload")
        
        # FastAPI should return a validation error
        assert response.status_code == 422
    
    @patch("app.api.documents.rag_store_async")
    def test_upload_large_filename(self, mock_rag_store, client):
        """Test handling of files with very long filenames."""
        mock_rag_store.return_value = {"status": "success"}
        
        long_filename = "a" * 200 + ".pdf"
        file_content = b"%PDF-1.4\nContent"
        
        response = client.post(
            "/documents/upload",
            files={"file": (long_filename, file_content, "application/pdf")},
        )
        
        # Should still be processed (or return appropriate error)
        assert response.status_code in [200, 400]