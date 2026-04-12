import re
import os
import mimetypes
import requests
import datetime
import unicodedata
from urllib.parse import urlparse

import injestion.utilities as utilities
import injestion.schemas as schemas

def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def remove_boilerplate(text: str, patterns) -> str:
    for pattern in patterns:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text

def _extract_local_raw_document(path: str, ftype: str) -> schemas.RawDocument:
    stat = os.stat(path)
    mime_type, _ = mimetypes.guess_type(path)

    return schemas.RawDocument(
        source_type="local",
        file_type=ftype,
        path_or_url=path,
        created_at=datetime.fromtimestamp(stat.st_ctime),
        metadata={
            "file_name": os.path.basename(path),
            "mime_type": mime_type,
            "size_bytes": stat.st_size,
            "modified_at": datetime.fromtimestamp(stat.st_mtime),
        }
    )

def _extract_uploaded_raw_document(upload_file, kl: str) -> schemas.RawDocument:
    """Extract raw document from an uploaded file (e.g., FastAPI UploadFile).
    
    Args:
        upload_file: FastAPI UploadFile object
        ftype: File type (e.g., "pdf", "txt")
    
    Returns:
        RawDocument with binary content populated
    """
    # Read the file content
    contents = upload_file.file.read()
    
    # Reset file pointer if needed for future reads
    upload_file.file.seek(0)
    
    # Get filename from upload_file
    file_name = upload_file.filename
    
    return schemas.RawDocument(
        source_type="local",
        file_type=ftype,
        path_or_url=file_name,
        content=None,  # Text content - for PDFs this would need extraction
        binary_content=contents,  # <-- THIS IS THE ACTUAL FILE BINARY
        created_at=None,
        metadata={
            "file_name": file_name or None,
            "mime_type": upload_file.content_type,
            "size_bytes": len(contents),
        }
    )


def _extract_remote_raw_document(path: str, ftype: str) -> schemas.RawDocument:
    response = requests.head(path, allow_redirects=True, timeout=5)
    response.raise_for_status()

    parsed = urlparse(path)
    file_name = os.path.basename(parsed.path)

    return schemas.RawDocument(
        source_type="remote",
        file_type=ftype,
        path_or_url=path,
        created_at=None,
        metadata={
            "file_name": file_name or None,
            "mime_type": response.headers.get("content-type"),
            "size_bytes": (
                int(response.headers["content-length"])
                if "content-length" in response.headers
                else None
            ),
            "last_modified": response.headers.get("last-modified"),
        }
    )


def extract_raw_document(path: str) -> schemas.RawDocument:
    file_source = utilities.is_accessible_file(path)
    file_type = utilities.infer_file_type(path)

    if file_source is "local":
        raw_document = _extract_local_raw_document(path, file_type)
    elif file_source is "remote":
        raw_document = _extract_remote_raw_document(path, file_type)
    else:
        if file_source is "unreachable_url":
            print("Url for file unreachable")
        else:
            print("Invalid path")
    
    return raw_document
