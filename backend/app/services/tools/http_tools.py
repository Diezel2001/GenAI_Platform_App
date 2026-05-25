from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# =========================================================
# SCHEMAS
# =========================================================

class HttpRequestSchema(BaseModel):
    url: str = Field(..., description="Full URL including scheme (https://...).")
    method: str = Field("GET", description="HTTP method: GET, POST, PUT, PATCH, DELETE.")
    headers: Optional[Dict[str, str]] = Field(None, description="Additional HTTP headers as key-value pairs.")
    params: Optional[Dict[str, str]] = Field(None, description="URL query parameters (appended to the URL).")
    body: Optional[Any] = Field(None, description="Request body. Dicts/lists are serialized as JSON automatically.")
    timeout: int = Field(20, ge=1, le=120, description="Request timeout in seconds.")
    max_response_chars: int = Field(8000, description="Truncate response body to this many characters.")
    follow_redirects: bool = Field(True, description="Whether to follow HTTP redirects.")


# =========================================================
# HELPERS
# =========================================================

_SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
_BODY_METHODS = {"POST", "PUT", "PATCH", "DELETE"}

_DEFAULT_HEADERS = {
    "User-Agent": "AgentHTTPClient/1.0",
    "Accept": "application/json, text/plain, */*",
}


def _build_url(url: str, params: Optional[Dict[str, str]]) -> str:
    if not params:
        return url
    encoded = urllib.parse.urlencode(params)
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{encoded}"


def _serialize_body(body: Any) -> tuple[bytes, str]:
    """
    Serialize request body to bytes.
    Returns (encoded_bytes, content_type).
    """
    if isinstance(body, (dict, list)):
        return json.dumps(body).encode("utf-8"), "application/json"
    if isinstance(body, str):
        return body.encode("utf-8"), "text/plain"
    if isinstance(body, bytes):
        return body, "application/octet-stream"
    return str(body).encode("utf-8"), "text/plain"


def _redact_auth(headers: dict) -> dict:
    """Return headers with Authorization values redacted for safe logging."""
    return {
        k: ("***REDACTED***" if k.lower() == "authorization" else v)
        for k, v in headers.items()
    }


# =========================================================
# IMPLEMENTATION
# =========================================================

def http_request(
    url: str,
    method: str = "GET",
    headers: Optional[Dict[str, str]] = None,
    params: Optional[Dict[str, str]] = None,
    body: Optional[Any] = None,
    timeout: int = 20,
    max_response_chars: int = 8000,
    follow_redirects: bool = True,
) -> str:
    """
    Make an HTTP request and return the response as a formatted string.
    Handles JSON and plain-text responses. Redacts Authorization headers
    from logs. Supports GET, POST, PUT, PATCH, DELETE.
    """

    method = method.upper()
    valid_methods = _SAFE_METHODS | _BODY_METHODS
    if method not in valid_methods:
        return f"ERROR: Unsupported HTTP method '{method}'. Use one of: {sorted(valid_methods)}"

    if not url.startswith(("http://", "https://")):
        return f"ERROR: URL must start with http:// or https://. Got: {url}"

    final_url = _build_url(url, params)

    # Build headers
    merged_headers = dict(_DEFAULT_HEADERS)
    if headers:
        merged_headers.update(headers)

    # Serialize body
    encoded_body: Optional[bytes] = None
    if body is not None:
        if method in _SAFE_METHODS:
            return (
                f"ERROR: HTTP {method} does not support a request body. "
                "Use POST, PUT, or PATCH instead."
            )
        encoded_body, inferred_ct = _serialize_body(body)
        if "Content-Type" not in merged_headers:
            merged_headers["Content-Type"] = inferred_ct

    req = urllib.request.Request(
        url=final_url,
        data=encoded_body,
        headers=merged_headers,
        method=method,
    )

    start = time.monotonic()

    try:
        opener = urllib.request.build_opener()
        if not follow_redirects:
            opener = urllib.request.build_opener(
                urllib.request.HTTPRedirectHandler()
            )

        with opener.open(req, timeout=timeout) as resp:
            elapsed_ms = int((time.monotonic() - start) * 1000)
            status = resp.status
            reason = resp.reason
            resp_headers = dict(resp.headers)
            raw = resp.read()
            content_type = resp_headers.get("Content-Type", "")

            # Decode response
            if "json" in content_type:
                try:
                    parsed = json.loads(raw)
                    body_text = json.dumps(parsed, indent=2, ensure_ascii=False)
                except json.JSONDecodeError:
                    body_text = raw.decode("utf-8", errors="replace")
            else:
                body_text = raw.decode("utf-8", errors="replace")

            total_chars = len(body_text)
            truncated = total_chars > max_response_chars
            if truncated:
                body_text = body_text[:max_response_chars]

            safe_headers_sent = _redact_auth(merged_headers)

            result = (
                f"Request:  {method} {final_url}\n"
                f"Status:   {status} {reason}\n"
                f"Time:     {elapsed_ms}ms\n"
                f"Headers sent: {json.dumps(safe_headers_sent, indent=2)}\n"
                f"Response Content-Type: {content_type}\n"
                f"Response size: {total_chars} chars"
                + (" (truncated)" if truncated else "")
                + f"\n\n--- Response Body ---\n{body_text}"
                + ("\n\n[TRUNCATED]" if truncated else "")
            )
            return result

    except urllib.error.HTTPError as e:
        elapsed_ms = int((time.monotonic() - start) * 1000)
        try:
            error_body = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            error_body = "(could not read error body)"

        return (
            f"HTTP ERROR {e.code} {e.reason}\n"
            f"URL: {final_url}\n"
            f"Time: {elapsed_ms}ms\n"
            f"Error body:\n{error_body}"
        )

    except urllib.error.URLError as e:
        return (
            f"URL ERROR: {e.reason}\n"
            f"URL: {final_url}\n"
            "Check that the URL is reachable and the hostname resolves."
        )

    except TimeoutError:
        return (
            f"TIMEOUT: Request to {final_url} exceeded {timeout} seconds."
        )

    except Exception as e:
        return f"UNEXPECTED ERROR: {e}\nURL: {final_url}"


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

HTTP_TOOLS = {
    "http_request": {
        "func": http_request,
        "schema": HttpRequestSchema,
        "description": (
            "Make an HTTP request (GET, POST, PUT, PATCH, DELETE) to any URL. "
            "Automatically serializes dict/list bodies as JSON. "
            "Redacts Authorization headers from output. "
            "Returns status, timing, and response body."
        ),
    },
}
