"""
Qdrant Vector Store Implementation

To switch to Qdrant:
1. Install: pip install qdrant
2. Set environment variables:
   - QDRANT_API_KEY (for cloud)
   - Or use local mode with a local Qdrant instance
3. In your config, set: VECTOR_STORE_PROVIDER = "qdrant"

Usage:
    from vector_store import get_vector_store
    vector_store = get_vector_store("qdrant", location=":memory:", dimension=1536)
    # Or for cloud: vector_store = get_vector_store("qdrant", url="https://xxx.cloud.qdrant.io", api_key="...")
"""

from typing import List, Dict, Any, Optional
from vs_base import VectorStore
from app.services.rag.rag_schemas import RecordData, SearchResult


class QdrantVectorStore(VectorStore):
    """Qdrant vector store implementation."""
    
    def __init__(
        self,
        location: Optional[str] = ":memory:",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        collection_name: str = "documents",
        dimension: int = 1536,
        distance: str = "Cosine",
        host: Optional[str] = None,
        port: int = 6333,
        **kwargs
    ):
        """
        Initialize Qdrant vector store.
        
        Args:
            location: Location string (":memory:" for in-memory, or path for local)
            url: URL for Qdrant cloud
            api_key: API key for Qdrant cloud
            collection_name: Name of the collection
            dimension: Dimension of embedding vectors
            distance: Distance metric ("Cosine", "Euclid", "Dot")
            host: Host for local Qdrant server
            port: Port for local Qdrant server
            **kwargs: Additional Qdrant client options
        """
        import os
        
        self.collection_name = collection_name
        self.dimension = dimension
        self.distance = distance
        
        # Map distance names to Qdrant constants
        distance_map = {
            "Cosine": "Cosine",
            "Euclid": "Euclid",
            "Dot": "Dot",
            "cosine": "Cosine",
            "euclidean": "Euclid",
            "dotproduct": "Dot",
        }
        self.qdrant_distance = distance_map.get(distance, "Cosine")
        
        # Get API key from param or environment
        self.api_key = api_key or os.environ.get("QDRANT_API_KEY")
        
        # Initialize Qdrant client
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams
        
        # Determine client initialization parameters
        client_kwargs = {}
        
        if url:
            # Cloud mode
            client_kwargs["url"] = url
            if api_key:
                client_kwargs["api_key"] = api_key
        elif location == ":memory:":
            # In-memory mode
            client_kwargs["location"] = location
        elif host:
            # Local server mode
            client_kwargs["host"] = host
            client_kwargs["port"] = port
        else:
            # Default: try local Qdrant
            client_kwargs["host"] = "localhost"
            client_kwargs["port"] = port
        
        # Add any additional kwargs
        client_kwargs.update(kwargs)
        
        self.client = QdrantClient(**client_kwargs)
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
        # In-memory tracking
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        from qdrant_client.models import Distance, VectorParams
        
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        
        if self.collection_name not in collection_names:
            # Map distance to Qdrant enum
            distance_map = {
                "Cosine": Distance.COSINE,
                "Euclid": Distance.EUCLID,
                "Dot": Distance.DOT,
            }
            
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.dimension,
                    distance=distance_map.get(self.qdrant_distance, Distance.COSINE)
                )
            )
    
    def add(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to the vector store."""
        if not texts:
            return
        
        from uuid import uuid4
        from qdrant_client.models import PointStruct
        
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
        
        # Note: embeddings need to be added separately via add_records
        # This method stores text/metadata for later embedding
    
    def add_records(self, records: List[RecordData]):
        """Add RecordData objects to the vector store."""
        if not records:
            return
        
        from qdrant_client.models import PointStruct
        
        points = []
        for record in records:
            # Convert metadata to dict
            metadata = record.metadata
            if hasattr(metadata, 'model_dump'):
                meta_dict = metadata.model_dump()
            elif hasattr(metadata, 'dict'):
                meta_dict = metadata.dict()
            else:
                meta_dict = dict(metadata)
            
            point = PointStruct(
                id=record.id,
                vector=record.embedding,
                payload={
                    "text": record.text,
                    **meta_dict
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
    
    def query(
        self,
        text: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
    ) -> List[Dict[str, Any]]:
        """Query the vector store."""
        if embedding is None:
            raise ValueError("Query embedding is required. Pass embedding parameter.")
        
        # Convert filters to Qdrant format if provided
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        
        query_filter = None
        if filters:
            conditions = []
            for key, value in filters.items():
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
            query_filter = Filter(must=conditions)
        
        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=embedding,
            limit=k,
            query_filter=query_filter,
            with_payload=True,
            with_vectors=False
        )
        
        return [
            {
                "id": str(result.id),
                "score": result.score,
                "text": result.payload.get("text", ""),
                "metadata": {k: v for k, v in result.payload.items() if k != "text"}
            }
            for result in results
        ]
    
    def persist(self):
        """Qdrant handles persistence automatically (local or cloud)."""
        pass
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        from qdrant_client.models import Filter, FieldCondition, MatchAny
        
        self.client.delete(
            collection_name=self.collection_name,
            points_selector=ids
        )
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get collection information."""
        return self.client.get_collection(self.collection_name)
