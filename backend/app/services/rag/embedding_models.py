from abc import ABC, abstractmethod
from typing import List, Optional


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        pass


# =====================================================
# OpenAI Embeddings - Needs api keyyy
# =====================================================

class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model: str = "text-embedding-3-large",
        api_key: Optional[str] = None,
    ):
        from langchain_openai import OpenAIEmbeddings

        self._embeddings = OpenAIEmbeddings(
            model=model,
            api_key=api_key,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


# =====================================================
# BGE Embeddings (BAAI)
# =====================================================

class BGEEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "BAAI/bge-large-en-v1.5",
        device: str = "cpu",
        normalize: bool = True,
    ):
        from langchain_community.embeddings import HuggingFaceBgeEmbeddings

        self._embeddings = HuggingFaceBgeEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            encode_kwargs={"normalize_embeddings": normalize},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


# =====================================================
# HuggingFace Instructor Embeddings
# =====================================================

class InstructorEmbeddingModel(EmbeddingModel):
    def __init__(
        self,
        model_name: str = "hkunlp/instructor-large",
        device: str = "cpu",
        document_instruction: str = "Represent the document for retrieval:",
        query_instruction: str = "Represent the query for retrieval:",
    ):
        from langchain_community.embeddings import HuggingFaceInstructEmbeddings

        self._embeddings = HuggingFaceInstructEmbeddings(
            model_name=model_name,
            model_kwargs={"device": device},
            embed_instruction=document_instruction,
            query_instruction=query_instruction,
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._embeddings.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._embeddings.embed_query(text)


# =====================================================
# Factory (recommended)
# =====================================================

def get_embedding_model(
    provider: str,
    **kwargs,
) -> EmbeddingModel:
    provider = provider.lower()

    if provider == "openai":
        return OpenAIEmbeddingModel(**kwargs)
    elif provider == "bge":
        return BGEEmbeddingModel(**kwargs)
    elif provider == "instructor":
        return InstructorEmbeddingModel(**kwargs)
    else:
        raise ValueError(f"Unknown embedding provider: {provider}")
