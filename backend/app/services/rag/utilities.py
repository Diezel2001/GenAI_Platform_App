import requests
from io import BytesIO
from urllib.parse import urlparse
import os

def is_local_file(path: str) -> bool:
    return os.path.isfile(path)

def is_url(path_or_url: str) -> bool:
    parsed = urlparse(path_or_url)
    return parsed.scheme in ("http", "https", "ftp")

def url_points_to_file(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=5)
        if response.status_code != 200:
            return False
        content_type = response.headers.get("content-type", "")
        # crude check: ignore HTML pages
        if "text/html" in content_type.lower():
            return False
        return True
    except requests.RequestException:
        return False
    
def is_accessible_file(path_or_url: str) -> str:
    if is_local_file(path_or_url):
        return "local"
    elif is_url(path_or_url):
        if url_points_to_file(path_or_url):
            return "remote"
        else:
            return "unreachable_url"
    else:
        return "invalid"

def infer_file_type(path_or_url: str) -> str:
    ext = os.path.splitext(urlparse(path_or_url).path)[1].lower()

    mapping = {
        ".pdf": "pdf",
        ".md": "md",
        ".txt": "txt",
        ".html": "html",
        ".htm": "html",
        ".csv": "csv",
        ".json": "json"
    }

    if ext not in mapping:
        raise ValueError(f"Unsupported file type: {ext}")

    return mapping[ext]
