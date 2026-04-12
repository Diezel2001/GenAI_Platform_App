from pydantic import BaseModel, Field, field_validator
from datetime import datetime
from typing import Optional, Dict, Any, Literal, List, Union
from uuid import uuid4



#----- Metadata Models------

class BaseMetadata(BaseModel):
    doc_id: str
    source: str
    file_type: str
    path_or_url: str
    chunk_index: int
    created_at: Optional[datetime] = None
    tags: List[str] = Field(default_factory=list)

class PDFChunkMetadata(BaseMetadata):
    file_type: Literal["pdf"] = "pdf"
    pdf_file_type: str
    page_number: Optional[int]
    author: Optional[str] = None
    title: Optional[str] = None

class MarkdownChunkMetadata(BaseMetadata):
    file_type: Literal["md"] = "md"
    heading: Optional[str] = None
    paragraph_index: Optional[int]
    frontmatter: dict = Field(default_factory=dict)

class TextChunkMetadata(BaseMetadata):
    file_type: Literal["txt"] = "txt"
    line_start: Optional[int] = None
    line_end: Optional[int] = None

class JSONChunkMetadata(BaseMetadata):
    file_type: Literal["json"] = "json"
    json_path: Optional[str] = None
    schema_version: Optional[str] = None

class CSVChunkMetadata(BaseMetadata):
    file_type: Literal["csv"] = "csv"
    row_start: Optional[int] = None
    row_end: Optional[int] = None

class HTMLChunkMetadata(BaseMetadata):
    file_type: Literal["html"] = "html"
    url: Optional[str] = None
    html_tag: Optional[str] = None       # e.g. section, article, div
    css_selector: Optional[str] = None   # e.g. "#main > article"
    heading: Optional[str] = None

Metadata = Union[
    PDFChunkMetadata,
    MarkdownChunkMetadata,
    TextChunkMetadata,
    JSONChunkMetadata,
    CSVChunkMetadata,
    HTMLChunkMetadata
]

#----- Schemas ------

class RawDocument(BaseModel):
    doc_id: str = Field(default_factory=lambda: str(uuid4()))
    source_type: Literal[
        "local",
        "remote"
    ]
    file_type: Literal[
        "pdf", "md", "txt", "html", "csv", "json"
    ]
    path_or_url: str
    content: Optional[str] = None           # raw text (if already loaded)
    binary_content: Optional[bytes] = None  # for PDFs, images, etc.
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: Optional[datetime] = None

class RecordData(BaseModel):
    id: str
    text: str
    embedding: List[float]
    metadata: Metadata

    @field_validator("embedding")
    @classmethod
    def validate_embedding(cls, v):
        if not v:
            raise ValueError("Embedding cannot be empty")
        return v

class SearchResult(BaseModel):
    id: str
    score: float
    text: str = ""
    metadata: Dict[str, Any] = {}

    class Config:
        # Allow dict-style access with dot notation
        arbitrary_types_allowed = True
        orm_mode = True
