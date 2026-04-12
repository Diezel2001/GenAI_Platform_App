from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class VectorStore(ABC):

    @abstractmethod
    def add(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ):
        pass

    @abstractmethod
    def query(
        self,
        text: str,
        k: int = 5,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def persist(self):
        pass
