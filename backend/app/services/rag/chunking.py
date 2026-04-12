from typing import List, Union
from abc import ABC, abstractmethod
from datetime import datetime
from uuid import uuid4

from app.services.rag.rag_schemas import (
    RawDocument,
    PDFChunkMetadata,
    MarkdownChunkMetadata,
    TextChunkMetadata,
    JSONChunkMetadata,
    CSVChunkMetadata,
    HTMLChunkMetadata,
)

# -----------------------------
# Chunk data structure
# -----------------------------

class Chunk:
    """A single retrievable chunk."""
    def __init__(self, id, text: str, metadata: Union[
        PDFChunkMetadata,
        MarkdownChunkMetadata,
        TextChunkMetadata,
        JSONChunkMetadata,
        CSVChunkMetadata,
        HTMLChunkMetadata,
    ]):
        self.id = id
        self.text = text
        self.metadata = metadata


# -----------------------------
# Base chunker
# -----------------------------

class BaseChunker(ABC):
    """Template interface for all chunkers."""

    @abstractmethod
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        pass


# -----------------------------
# PDF Chunker - page-based chunking
# -----------------------------

class PDFChunker():
    def chunk_by_page(self, doc: RawDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        id = str(uuid4())

        # TODO: integrate pdf parser (pypdf, pdfplumber, pymupdf)
        # Placeholder logic
        pages = doc.content.split("\f") if doc.content else []

        for i, page_text in enumerate(pages, start=1):
            metadata = PDFChunkMetadata(
                doc_id=id,
                chunk_index=i-1,
                path_or_url=doc.path_or_url,
                source=doc.source_type,
                created_at=doc.created_at or None,
                pdf_file_type="scanned" if not page_text.strip() else "text",
                page_number=i,
            )
            chunk_id = f"{id}:{i-1}"
            chunks.append(Chunk(chunk_id, page_text, metadata))

        return chunks
    def chunk_by_fixed_size(
        self,
        doc: RawDocument,
        chunk_size: int = 200,
        overlap: int = 50,
        ) -> List[Chunk]:
        chunks: List[Chunk] = []
        id = str(uuid4())

        text = doc.content or ""
        step = chunk_size - overlap
        index = 0
        chunk_index = 0

        while index < len(text):
            chunk_text = text[index : index + chunk_size]

            metadata = PDFChunkMetadata(
                doc_id=id,
                chunk_index=chunk_index,
                path_or_url=doc.path_or_url,
                source=doc.source_type,
                created_at=doc.created_at or None,
                pdf_file_type="text",
                page_number=None,  # spans pages
            )

            chunk_id = f"{id}:{chunk_index}"
            chunks.append(Chunk(chunk_id, chunk_text, metadata))

            index += step
            chunk_index += 1

        return chunks

    def chunk_by_semantics(self, doc: RawDocument) -> List[Chunk]:
        """Basic semantic chunking placeholder splitting by paragraphs."""
        chunks: List[Chunk] = []
        id = str(uuid4())

        text = doc.content or ""
        paragraphs = [p for p in text.split("\n\n") if p.strip()]

        for i, para in enumerate(paragraphs):
            metadata = PDFChunkMetadata(
                doc_id=id,
                chunk_index=i,
                path_or_url=doc.path_or_url,
                source=doc.source_type,
                created_at=doc.created_at or None,
                pdf_file_type="text",
                page_number=None,
            )
            chunk_id = f"{id}:{i}"
            chunks.append(Chunk(chunk_id, para, metadata))

        return chunks


# -----------------------------
# Markdown Chunker - heading based chunking
# -----------------------------

from uuid import uuid4

class MarkdownChunker(BaseChunker):
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        chunks: List[Chunk] = []
        current_heading = None

        # Generate ONE doc_id per document
        doc_id = str(uuid4())
        chunk_index = 0

        lines = doc.content.splitlines() if doc.content else []
        buffer = []
        paragraph_index = 0

        for line in lines:
            if line.startswith("#"):
                if buffer:
                    metadata = MarkdownChunkMetadata(
                        doc_id=doc_id,
                        chunk_index=chunk_index,
                        path_or_url=doc.path_or_url,
                        source=doc.source_type,
                        created_at=doc.created_at or None,
                        heading=current_heading,
                        paragraph_index=paragraph_index,
                    )
                    chunk_id = f"{doc_id}:{chunk_index}"
                    chunks.append(Chunk(chunk_id, "\n".join(buffer), metadata))

                    buffer = []
                    paragraph_index += 1
                    chunk_index += 1

                current_heading = line.lstrip("#").strip()
            else:
                buffer.append(line)

        # Flush remaining buffer
        if buffer:
            metadata = MarkdownChunkMetadata(
                doc_id=doc_id,
                chunk_index=chunk_index,
                path_or_url=doc.path_or_url,
                source=doc.source_type,
                created_at=doc.created_at or None,
                heading=current_heading,
                paragraph_index=paragraph_index,
            )
            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(Chunk(chunk_id, "\n".join(buffer), metadata))

        return chunks



# -----------------------------
# Text Chunker - line based chunking
# -----------------------------

class TextChunker(BaseChunker):
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        chunks: List[Chunk] = []

        doc_id = str(uuid4())
        chunk_index = 0

        lines = doc.content.splitlines() if doc.content else []

        chunk_size = 70
        overlap = 30
        step = chunk_size - overlap  # 40

        start = 0

        while start < len(lines):
            end = min(start + chunk_size, len(lines))
            text = "\n".join(lines[start:end])

            metadata = TextChunkMetadata(
                doc_id=doc_id,
                chunk_index=chunk_index,
                path_or_url=doc.path_or_url,
                source=doc.source_type,
                created_at=doc.created_at or None,
                line_start=start + 1,
                line_end=end,
            )

            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(Chunk(chunk_id, text, metadata))


            chunk_index += 1
            start += step

        return chunks


# -----------------------------
# JSON Chunker
# -----------------------------

class JSONChunker(BaseChunker):
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        chunks: List[Chunk] = []

        doc_id = str(uuid4())
        chunk_index = 0

        metadata = JSONChunkMetadata(
            doc_id=doc_id,
            chunk_index=chunk_index,
            path_or_url=doc.path_or_url,
            source=doc.source_type,
            created_at=doc.created_at or None,
            json_path="$",
            schema_version=doc.metadata.get("schema_version"),
        )

        chunk_id = f"{doc_id}:{chunk_index}"
        chunks.append(Chunk(chunk_id, doc.content or "", metadata))


        return chunks


# -----------------------------
# CSV Chunker
# -----------------------------

class CSVChunker(BaseChunker):
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        chunks: List[Chunk] = []

        doc_id = str(uuid4())
        chunk_index = 0

        rows = doc.content.splitlines() if doc.content else []

        start = 0
        window = 50

        while start < len(rows):
            end = min(start + window, len(rows))
            text = "\n".join(rows[start:end])

            metadata = CSVChunkMetadata(
                doc_id=doc_id,
                chunk_index=chunk_index,
                path_or_url=doc.path_or_url,
                source=doc.source_type,
                created_at=doc.created_at or None,
                row_start=start + 1,
                row_end=end,
            )

            chunk_id = f"{doc_id}:{chunk_index}"
            chunks.append(Chunk(chunk_id, text, metadata))


            chunk_index += 1
            start = end

        return chunks


# -----------------------------
# HTML Chunker
# -----------------------------

class HTMLChunker(BaseChunker):
    def chunk(self, doc: RawDocument) -> List[Chunk]:
        chunks: List[Chunk] = []

        doc_id = str(uuid4())
        chunk_index = 0

        metadata = HTMLChunkMetadata(
            doc_id=doc_id,
            chunk_index=chunk_index,
            path_or_url=doc.path_or_url,
            source=doc.source_type,
            created_at=doc.created_at or None,
            url=doc.path_or_url,
            html_tag="document",
        )

        chunk_id = f"{doc_id}:{chunk_index}"
        chunks.append(Chunk(chunk_id, doc.content or "", metadata))


        return chunks


# -----------------------------
# Factory
# -----------------------------

def get_chunker(doc: RawDocument) -> BaseChunker:
    return {
        "pdf": PDFChunker(),
        "md": MarkdownChunker(),
        "txt": TextChunker(),
        "json": JSONChunker(),
        "csv": CSVChunker(),
        "html": HTMLChunker(),
    }[doc.file_type]


def chunk_document(doc: RawDocument) -> List[Chunk]:
    chunker = get_chunker(doc)
    return chunker.chunk(doc)
