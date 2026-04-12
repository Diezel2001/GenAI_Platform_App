"""
Pinecone Vector Store Implementation

To switch to Pinecone:
1. Install: pip install pinecone
2. Set environment variables:
   - PINECONE_API_KEY
   - PINECONE_ENVIRONMENT (e.g., "us-west1-gcp")
3. In your config, set: VECTOR_STORE_PROVIDER = "pinecone"

Usage:
    from vector_store import get_vector_store
    vector_store = get_vector_store("pinecone", index_name="my-index", dimension=1536)
"""

from typing import List, Dict, Any, Optional
from vector_store.vs_base import VectorStore
from app.services.rag.rag_schemas import RecordData, SearchResult


class PineconeVectorStore(VectorStore):
    """Pinecone vector store implementation."""
    
    def __init__(
        self,
        index_name: str,
        dimension: int,
        metric: str = "cosine",
        cloud: str = "aws",
        region: str = "us-west-2",
        api_key: Optional[str] = None,
        environment: Optional[str] = None,
    ):
        """
        Initialize Pinecone vector store.
        
        Args:
            index_name: Name of the Pinecone index
            dimension: Dimension of the embedding vectors
            metric: Distance metric ("cosine", "euclidean", "dotproduct")
            cloud: Cloud provider ("aws", "gcp", "azure")
            region: Cloud region
            api_key: Pinecone API key (optional, reads from PINECONE_API_KEY env)
            environment: Pinecone environment (optional, reads from PINECONE_ENVIRONMENT env)
        """
        import os
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        # Get API key from param or environment
        self.api_key = api_key or os.environ.get("PINECONE_API_KEY")
        self.environment = environment or os.environ.get("PINECONE_ENVIRONMENT")
        
        if not self.api_key:
            raise ValueError("Pinecone API key is required. Set PINECONE_API_KEY env var or pass api_key parameter.")
        
        # Initialize Pinecone client
        from pinecone import Pinecone, ServerlessSpec
        
        self.client = Pinecone(
            api_key=self.api_key,
            environment=self.environment
        )
        
        # Create index if it doesn't exist
        self._ensure_index_exists()
        
        # Connect to index
        self.index = self.client.Index(self.index_name)
        
        # In-memory tracking for upsert operations
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
    
    def _ensure_index_exists(self):
        """Create index if it doesn't exist."""
        from pinecone import ServerlessSpec
        
        # List existing indexes
        existing_indexes = self.client.list_indexes()
        
        # Check if our index exists
        index_exists = any(idx.name == self.index_name for idx in existing_indexes)
        
        if not index_exists:
            spec = ServerlessSpec(
                cloud=self.cloud,
                region=self.region
            )
            self.client.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=spec
            )
    
    def add(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the vector store."""
        # This method is for compatibility; use add_records for RecordData
        if not texts:
            return
        
        from uuid import uuid4
        
        # Generate IDs if not provided
        if ids is None:
            ids = [str(uuid4()) for _ in texts]
        
        # Store texts and metadata for reference
        self.ids.extend(ids)
        self.texts.extend(texts)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            self.metadatas.extend([{} for _ in texts])
    
    def add_records(self, records: List[RecordData]):
        """Add RecordData objects to the vector store."""
        if not records:
            return
        
        # Prepare vectors for upsert
        vectors = []
        for record in records:
            vectors.append({
                "id": record.id,
                "values": record.embedding,
                "metadata": {
                    "text": record.text,
                    **(record.metadata.model_dump() if hasattr(record.metadata, 'model_dump') else record.metadata)
                }
            })
        
        # Upsert to Pinecone
        self.index.upsert(vectors=vectors)
    
    def query(
        self,
        text: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store."""
        # If no embedding provided, raise error (caller should provide)
        if embedding is None:
            raise ValueError("Query embedding is required. Pass embedding parameter or use a wrapper that handles embeddings.")
        
        # Query Pinecone
        query_params = {
            "vector": embedding,
            "top_k": k,
            "include_metadata": True
        }
        
        if filters:
            query_params["filter"] = filters
        
        results = self.index.query(**query_params)
        
        # Convert to our format
        return [
            {
                "id": match.id,
                "score": match.score,
                "text": match.metadata.get("text", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "text"}
            }
            for match in results.matches
        ]
    
    def persist(self):
        """Pinecone is a managed service - no local persistence needed."""
        pass
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        self.index.delete(ids=ids)
    
    def describe_index_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return self.index.describe_index_stats()
