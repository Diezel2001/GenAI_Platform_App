"""
Milvus Vector Store Implementation

To switch to Milvus:
1. Install: pip install pymilvus
2. Set environment variables:
   - MILVUS_HOST (default: "localhost")
   - MILVUS_PORT (default: "19530")
   - Or use Milvus Cloud (Zilliz)
3. In your config, set: VECTOR_STORE_PROVIDER = "milvus"

Usage:
    from vector_store import get_vector_store
    vector_store = get_vector_store("milvus", collection_name="documents", dimension=1536)
    # Or for cloud: vector_store = get_vector_store("milvus", host="<cluster-id>.zillizcloud.com", port="19530", user="<username>", password="<password>", secure=True)
"""

from typing import List, Dict, Any, Optional
from vector_store.vs_base import VectorStore
from app.services.rag.rag_schemas import RecordData, SearchResult


class MilvusVectorStore(VectorStore):
    """Milvus vector store implementation."""
    
    def __init__(
        self,
        collection_name: str = "documents",
        dimension: int = 1536,
        metric_type: str = "COSINE",
        index_type: str = "IVF_FLAT",
        host: str = "localhost",
        port: str = "19530",
        user: Optional[str] = None,
        password: Optional[str] = None,
        secure: bool = False,
        **kwargs
    ):
        """
        Initialize Milvus vector store.
        
        Args:
            collection_name: Name of the collection
            dimension: Dimension of embedding vectors
            metric_type: Distance metric ("COSINE", "IP", "L2")
            index_type: Index type for vectors
            host: Milvus host
            port: Milvus port
            user: Username for Milvus Cloud (Zilliz)
            password: Password for Milvus Cloud (Zilliz)
            secure: Use TLS for connection
            **kwargs: Additional pymilvus connections options
        """
        import os
        
        self.collection_name = collection_name
        self.dimension = dimension
        self.metric_type = metric_type
        self.index_type = index_type
        self.host = host
        self.port = port
        self.user = user or os.environ.get("MILVUS_USER")
        self.password = password or os.environ.get("MILVUS_PASSWORD")
        self.secure = secure
        
        # Initialize Milvus client
        from pymilvus import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
        
        # Connect to Milvus
        connection_kwargs = {"host": host, "port": port}
        
        if user and password:
            connection_kwargs["user"] = user
            connection_kwargs["password"] = password
            connection_kwargs["secure"] = secure
        
        connection_kwargs.update(kwargs)
        
        # Use alias "default" for connection
        try:
            connections.connect(**connection_kwargs)
        except Exception as e:
            # If already connected, this might fail - that's ok
            pass
        
        self.connections = connections
        self.Collection = Collection
        self.CollectionSchema = CollectionSchema
        self.FieldSchema = FieldSchema
        self.DataType = DataType
        self.utility = utility
        
        # Create collection if it doesn't exist
        self._ensure_collection_exists()
        
        # In-memory tracking
        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        
        # Load collection for search
        self.collection.load()
    
    def _ensure_collection_exists(self):
        """Create collection if it doesn't exist."""
        
        # Check if collection exists
        if not self.utility.has_collection(self.collection_name):
            # Define schema
            fields = [
                FieldSchema(name="id", dtype=self.DataType.VARCHAR, max_length=65535, is_primary=True),
                FieldSchema(name="text", dtype=self.DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=self.DataType.FLOAT_VECTOR, dim=self.dimension),
            ]
            
            # Add metadata fields dynamically
            # For simplicity, we'll store metadata as JSON in a separate field
            fields.append(FieldSchema(name="metadata", dtype=self.DataType.VARCHAR, max_length=65535))
            
            schema = self.CollectionSchema(fields=fields, description="Document collection")
            collection = self.Collection(name=self.collection_name, schema=schema)
            
            # Create index
            index_params = {
                "metric_type": self.metric_type,
                "index_type": self.index_type,
                "params": {"nlist": 128}
            }
            collection.create_index(field_name="vector", index_params=index_params)
        else:
            # Collection exists, get it
            pass
        
        self.collection = self.Collection(self.collection_name)
    
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
        import json
        
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
    
    def add_records(self, records: List[RecordData]):
        """Add RecordData objects to the vector store."""
        if not records:
            return
        
        import json
        
        # Prepare data for insertion
        ids = []
        texts = []
        vectors = []
        metadatas = []
        
        for record in records:
            ids.append(record.id)
            texts.append(record.text)
            vectors.append(record.embedding)
            
            # Serialize metadata to JSON
            metadata = record.metadata
            if hasattr(metadata, 'model_dump'):
                meta_dict = metadata.model_dump()
            elif hasattr(metadata, 'dict'):
                meta_dict = metadata.dict()
            else:
                meta_dict = dict(metadata)
            
            metadatas.append(json.dumps(meta_dict))
        
        # Insert into Milvus
        data = [ids, texts, vectors, metadatas]
        self.collection.insert(data)
        self.collection.flush()
    
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
        
        import json
        
        # Prepare search parameters
        search_params = {"metric_type": self.metric_type, "params": {"nprobe": 10}}
        
        # Execute search
        results = self.collection.search(
            data=[embedding],
            anns_field="vector",
            param=search_params,
            limit=k,
            output_fields=["id", "text", "metadata"]
        )
        
        # Process results
        output = []
        for hits in results:
            for hit in hits:
                # Parse metadata
                metadata_str = hit.entity.get("metadata", "{}")
                try:
                    metadata = json.loads(metadata_str)
                except:
                    metadata = {}
                
                output.append({
                    "id": hit.entity.get("id"),
                    "score": hit.distance,
                    "text": hit.entity.get("text", ""),
                    "metadata": metadata
                })
        
        return output
    
    def persist(self):
        """Milvus handles persistence automatically."""
        self.collection.flush()
    
    def delete(self, ids: List[str]):
        """Delete vectors by IDs."""
        # Milvus doesn't support direct delete by ID in the same way
        # This is a placeholder - you'd need to use delete expressions
        pass
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return self.collection.num_entities
    
    def close(self):
        """Close the connection."""
        self.connections.disconnect("default")
