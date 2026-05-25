from __future__ import annotations

import os
import json
import chardet

from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field


# =========================================================
# SCHEMAS
# =========================================================

class ReadFileSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the file to read.")
    encoding: Optional[str] = Field(None, description="File encoding. Auto-detected if not provided.")
    max_chars: Optional[int] = Field(None, description="Truncate output to this many characters. None = full file.")


class WriteFileSchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to write to. Parent directories are created if missing.")
    content: str = Field(..., description="Text content to write to the file.")
    mode: str = Field("w", description="Write mode: 'w' (overwrite) or 'a' (append).")
    encoding: str = Field("utf-8", description="File encoding to use when writing.")


class ListDirectorySchema(BaseModel):
    path: str = Field(..., description="Absolute or relative path to the directory to list.")
    recursive: bool = Field(False, description="If True, list all files recursively.")
    extension_filter: Optional[str] = Field(None, description="Only return files with this extension (e.g. '.py', '.csv').")


# =========================================================
# IMPLEMENTATIONS
# =========================================================

def read_file(
    path: str,
    encoding: Optional[str] = None,
    max_chars: Optional[int] = None,
) -> str:
    """
    Read a file and return its text content.
    Handles encoding detection automatically.
    Returns an error string on failure so the agent can observe and recover.
    """

    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        parent = resolved.parent
        available = (
            [f.name for f in parent.iterdir()]
            if parent.exists()
            else []
        )
        return (
            f"ERROR: File not found: {resolved}\n"
            f"Available in {parent}: {available}"
        )

    if not resolved.is_file():
        return f"ERROR: Path is not a file: {resolved}"

    # Detect encoding if not supplied
    if encoding is None:
        raw_bytes = resolved.read_bytes()
        detected = chardet.detect(raw_bytes)
        encoding = detected.get("encoding") or "utf-8"

    try:
        content = resolved.read_text(encoding=encoding, errors="replace")
    except Exception as e:
        return f"ERROR reading file: {e}"

    size = len(content)

    if max_chars and size > max_chars:
        content = content[:max_chars]
        return (
            f"[File: {resolved} | {size} chars total | "
            f"showing first {max_chars}]\n\n{content}\n\n"
            f"[TRUNCATED — {size - max_chars} chars not shown]"
        )

    return f"[File: {resolved} | {size} chars]\n\n{content}"


def write_file(
    path: str,
    content: str,
    mode: str = "w",
    encoding: str = "utf-8",
) -> str:
    """
    Write text content to a file.
    Creates parent directories automatically.
    Returns a confirmation or error string.
    """

    if mode not in ("w", "a"):
        return f"ERROR: Invalid mode '{mode}'. Use 'w' or 'a'."

    resolved = Path(path).expanduser().resolve()

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with resolved.open(mode=mode, encoding=encoding) as f:
            f.write(content)
    except Exception as e:
        return f"ERROR writing file: {e}"

    action = "Written" if mode == "w" else "Appended"
    return (
        f"{action}: {resolved}\n"
        f"Size: {len(content.encode(encoding))} bytes\n"
        f"Encoding: {encoding}"
    )


def list_directory(
    path: str,
    recursive: bool = False,
    extension_filter: Optional[str] = None,
) -> str:
    """
    List files and directories at the given path.
    Returns a formatted tree or error string.
    """

    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        return f"ERROR: Directory not found: {resolved}"

    if not resolved.is_dir():
        return f"ERROR: Path is not a directory: {resolved}"

    ext = (
        extension_filter.lower()
        if extension_filter and not extension_filter.startswith(".")
        else extension_filter
    )

    entries = []

    if recursive:
        for item in sorted(resolved.rglob("*")):
            if ext and item.suffix.lower() != ext:
                continue
            rel = item.relative_to(resolved)
            kind = "DIR " if item.is_dir() else "FILE"
            size = (
                f"{item.stat().st_size:>10} bytes"
                if item.is_file()
                else ""
            )
            entries.append(f"  {kind}  {rel}  {size}")
    else:
        for item in sorted(resolved.iterdir()):
            if ext and item.is_file() and item.suffix.lower() != ext:
                continue
            kind = "DIR " if item.is_dir() else "FILE"
            size = (
                f"{item.stat().st_size:>10} bytes"
                if item.is_file()
                else ""
            )
            entries.append(f"  {kind}  {item.name}  {size}")

    if not entries:
        return f"Directory is empty (or no files match filter): {resolved}"

    header = (
        f"Directory: {resolved}\n"
        f"Filter: {extension_filter or 'none'} | "
        f"Recursive: {recursive} | "
        f"Entries: {len(entries)}\n"
        + "-" * 60
    )

    return header + "\n" + "\n".join(entries)


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

FILE_TOOLS = {
    "read_file": {
        "func": read_file,
        "schema": ReadFileSchema,
        "description": (
            "Read a file from disk and return its text content. "
            "Supports encoding auto-detection and optional truncation."
        ),
    },
    "write_file": {
        "func": write_file,
        "schema": WriteFileSchema,
        "description": (
            "Write or append text content to a file. "
            "Creates parent directories automatically."
        ),
    },
    "list_directory": {
        "func": list_directory,
        "schema": ListDirectorySchema,
        "description": (
            "List files and subdirectories at a given path. "
            "Supports recursive listing and extension filtering."
        ),
    },
}
