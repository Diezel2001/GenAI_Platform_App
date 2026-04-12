import pytest
from datetime import datetime

from injestion.schemas import RawDocument
from injestion.chunking import (
    chunk_document,
    PDFChunker,
    MarkdownChunker,
    TextChunker,
    JSONChunker,
    CSVChunker,
    HTMLChunker,
)


# -----------------------------
# Fixtures
# -----------------------------

@pytest.fixture
def base_time():
    return datetime(2025, 1, 1)


@pytest.fixture
def text_document(base_time):
    content = "\n".join([f"line {i}" for i in range(1, 201)])
    return RawDocument(
        file_type="txt",
        source_type="local",
        path_or_url="test.txt",
        content=content,
        created_at=base_time,
    )


@pytest.fixture
def markdown_document(base_time):
    content = """# Title1
Paragraph one.

## Section A
Paragraph two.

## Section B
Paragraph three.

# Title2
Paragraph one.

## Section A
Paragraph four.

## Section B
Paragraph five.
"""
    return RawDocument(
        file_type="md",
        source_type="local",
        path_or_url="test.md",
        content=content,
        created_at=base_time,
    )


@pytest.fixture
def pdf_document(base_time):
    content = "Page 1 text\fPage 2 text\fPage 3 text"
    return RawDocument(
        file_type="pdf",
        source_type="local",
        path_or_url="test.pdf",
        content=content,
        created_at=base_time,
    )


@pytest.fixture
def json_document(base_time):
    content = '{"user": {"name": "Alice", "age": 30}}'
    return RawDocument(
        file_type="json",
        source_type="local",
        path_or_url="test.json",
        content=content,
        metadata={"schema_version": "1.0"},
        created_at=base_time,
    )


@pytest.fixture
def csv_document(base_time):
    rows = ["col1,col2"] + [f"{i},{i+1}" for i in range(100)]
    return RawDocument(
        file_type="csv",
        source_type="local",
        path_or_url="test.csv",
        content="\n".join(rows),
        created_at=base_time,
    )


@pytest.fixture
def html_document(base_time):
    content = "<html><body><h1>Title</h1><p>Hello</p></body></html>"
    return RawDocument(
        file_type="html",
        source_type="remote",
        path_or_url="https://example.com",
        content=content,
        created_at=base_time,
    )


# -----------------------------
# TextChunker
# -----------------------------

def test_text_chunker_with_overlap(text_document):
    chunks = TextChunker().chunk(text_document)

    assert len(chunks) > 1
    assert chunks[0].metadata.line_start == 1
    assert chunks[0].metadata.line_end == 70
    assert chunks[1].metadata.line_start == 41  # overlap verified


# -----------------------------
# MarkdownChunker
# -----------------------------

def test_markdown_chunker(markdown_document):
    chunks = MarkdownChunker().chunk(markdown_document)

    assert len(chunks) >= 2
    assert chunks[0].metadata.heading == "Title1"
    assert chunks[1].metadata.heading == "Section A"
    assert chunks[2].metadata.heading == "Section B"
    assert chunks[3].metadata.heading == "Title2"
    assert chunks[4].metadata.heading == "Section A"
    assert chunks[5].metadata.heading == "Section B"


# -----------------------------
# PDFChunker
# -----------------------------

def test_pdf_chunker(pdf_document):
    chunker = PDFChunker()

    # --- page-based chunking ---
    page_chunks = chunker.chunk_by_page(pdf_document)

    assert len(page_chunks) == 3
    assert page_chunks[0].metadata.page_number == 1
    assert page_chunks[2].metadata.page_number == 3

    # --- fixed-size chunking ---
    fixed_chunks = chunker.chunk_by_fixed_size(
        pdf_document,
        chunk_size=10,
        overlap=0,
    )

    assert len(fixed_chunks) > 0
    assert fixed_chunks[0].metadata.page_number is None

    # --- semantic chunking ---
    semantic_chunks = chunker.chunk_by_semantics(pdf_document)

    assert len(semantic_chunks) > 0
    assert semantic_chunks[0].metadata.page_number is None



# -----------------------------
# JSONChunker
# -----------------------------

def test_json_chunker(json_document):
    chunks = JSONChunker().chunk(json_document)

    assert len(chunks) == 1
    assert chunks[0].metadata.json_path == "$"
    assert chunks[0].metadata.schema_version == "1.0"


# -----------------------------
# CSVChunker
# -----------------------------

def test_csv_chunker(csv_document):
    chunks = CSVChunker().chunk(csv_document)

    assert len(chunks) > 1
    assert chunks[0].metadata.row_start == 1
    assert chunks[0].metadata.row_end <= 50


# -----------------------------
# HTMLChunker
# -----------------------------

def test_html_chunker(html_document):
    chunks = HTMLChunker().chunk(html_document)

    assert len(chunks) == 1
    assert chunks[0].metadata.url == "https://example.com"
    assert chunks[0].metadata.html_tag == "document"


# -----------------------------
# Factory / Integration
# -----------------------------

def test_chunk_document_factory(text_document):
    chunks = chunk_document(text_document)
    assert chunks
