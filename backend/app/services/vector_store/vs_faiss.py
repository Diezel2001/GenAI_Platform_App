import faiss
import numpy as np
import pickle
from typing import List, Dict, Any, Optional
from .vs_base import VectorStore
from app.services.rag.rag_schemas import RecordData, SearchResult

class FaissVectorStore(VectorStore):
    def __init__(
        self,
        dim: int,
        index_path: str = "faiss.index",
        meta_path: str = "faiss_meta.pkl"
    ):
        self.dim = dim
        self.index_path = index_path
        self.meta_path = meta_path

        # Cosine similarity via inner product + normalized vectors
        self.index = faiss.IndexFlatIP(dim)

        self.ids: List[str] = []
        self.texts: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []
        self.records: List[RecordData] = []

    def add(self, texts: List[str], metadatas: Optional[List[Dict[str, Any]]] = None, ids: Optional[List[str]] = None):
        """Add documents (text only, without embeddings)."""
        # This method is for compatibility; use add_records for RecordData
        pass
    
    def add_records(self, records: List[RecordData]):
        """Add RecordData objects with embeddings to the vector store."""
        if not records:
            return

        vectors = np.array([record.embedding for record in records], dtype="float32")
        faiss.normalize_L2(vectors)
        self.index.add(vectors)

        for record in records:
            self.ids.append(record.id if record.id is not None else str(len(self.ids)))
            self.texts.append(record.text if record.text is not None else "")
            
            # Convert metadata to dict
            metadata = record.metadata
            if hasattr(metadata, 'model_dump'):
                meta_dict = metadata.model_dump()
            elif hasattr(metadata, 'dict'):
                meta_dict = metadata.dict()
            else:
                meta_dict = dict(metadata)
            
            self.metadatas.append(meta_dict)
            self.records.append(record)

    def query(
        self,
        text: str = "",
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        vector = np.array([embedding], dtype="float32")
        faiss.normalize_L2(vector)

        scores, indices = self.index.search(vector, k)

        results: List[SearchResult] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue

            metadata = self.metadatas[idx]

            results.append(
                SearchResult(
                    id=self.ids[idx],
                    score=float(score),
                    text=self.texts[idx],
                    metadata=metadata
                )
            )

        return results

    def persist(self):
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "wb") as f:
            pickle.dump(
                {
                    "ids": self.ids,
                    "texts": self.texts,
                    "metadatas": self.metadatas
                },
                f
            )

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.meta_path, "rb") as f:
            data = pickle.load(f)
            self.ids = data.get("ids", [])
            self.texts = data.get("texts", [])
            self.metadatas = data.get("metadatas", [])