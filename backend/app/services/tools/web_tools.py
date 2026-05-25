from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
import urllib.error

from typing import Optional, List

from pydantic import BaseModel, Field


# =========================================================
# SCHEMAS
# =========================================================

class WebSearchSchema(BaseModel):
    query: str = Field(..., description="Search query string.")
    num_results: int = Field(5, ge=1, le=20, description="Number of results to return.")
    site_filter: Optional[str] = Field(None, description="Restrict results to a specific domain (e.g. 'docs.python.org').")


class WebFetchSchema(BaseModel):
    url: str = Field(..., description="Full URL to fetch (must include http:// or https://).")
    max_chars: Optional[int] = Field(8000, description="Truncate returned text to this many characters.")
    timeout: int = Field(15, description="Request timeout in seconds.")


# =========================================================
# HELPERS
# =========================================================

_DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; AgentBot/1.0)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json",
    "Accept-Language": "en-US,en;q=0.9",
}


def _strip_html(html: str) -> str:
    """
    Very lightweight HTML → plain text stripper.
    Removes tags, collapses whitespace, decodes common entities.
    Not a full parser — good enough for content extraction.
    """

    # Remove script and style blocks entirely
    html = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        html,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Remove all remaining tags
    html = re.sub(r"<[^>]+>", " ", html)

    # Decode common HTML entities
    entities = {
        "&amp;": "&", "&lt;": "<", "&gt;": ">",
        "&quot;": '"', "&#39;": "'", "&nbsp;": " ",
        "&mdash;": "—", "&ndash;": "–", "&hellip;": "…",
    }
    for entity, char in entities.items():
        html = html.replace(entity, char)

    # Collapse whitespace
    html = re.sub(r"\s+", " ", html).strip()

    return html


def _fetch_raw(url: str, timeout: int = 15) -> tuple[int, str, str]:
    """
    Fetch a URL and return (status_code, content_type, body_text).
    Returns error info on failure.
    """

    req = urllib.request.Request(url, headers=_DEFAULT_HEADERS)

    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read()

            # Detect encoding from content-type or default utf-8
            encoding = "utf-8"
            ct_match = re.search(r"charset=([^\s;]+)", content_type)
            if ct_match:
                encoding = ct_match.group(1).strip()

            body = raw.decode(encoding, errors="replace")
            return status, content_type, body

    except urllib.error.HTTPError as e:
        return e.code, "", f"HTTP_ERROR:{e.code}:{e.reason}"

    except urllib.error.URLError as e:
        return 0, "", f"URL_ERROR:{e.reason}"

    except Exception as e:
        return 0, "", f"FETCH_ERROR:{e}"


# =========================================================
# IMPLEMENTATIONS
# =========================================================

def web_search(
    query: str,
    num_results: int = 5,
    site_filter: Optional[str] = None,
) -> str:
    """
    Search the web using DuckDuckGo's HTML interface (no API key required).
    Returns a formatted list of results with title, URL, and snippet.

    Note: For production use, replace with a proper Search API
    (SerpAPI, Brave Search API, Google Custom Search, etc.)
    """

    q = query
    if site_filter:
        q = f"site:{site_filter} {q}"

    encoded_query = urllib.parse.quote_plus(q)
    url = f"https://html.duckduckgo.com/html/?q={encoded_query}"

    status, content_type, body = _fetch_raw(url, timeout=15)

    if status != 200 or body.startswith(("HTTP_ERROR", "URL_ERROR", "FETCH_ERROR")):
        return f"SEARCH_ERROR: {body}"

    # Parse DuckDuckGo HTML results
    results: List[dict] = []

    # Extract result blocks
    result_blocks = re.findall(
        r'<div class="result[^"]*"[^>]*>(.*?)</div>\s*</div>',
        body,
        re.DOTALL,
    )

    for block in result_blocks[:num_results * 2]:

        # Title + URL
        title_match = re.search(
            r'<a[^>]+class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )

        # Snippet
        snippet_match = re.search(
            r'<a[^>]+class="result__snippet"[^>]*>(.*?)</a>',
            block,
            re.DOTALL,
        )

        if not title_match:
            continue

        href = title_match.group(1)
        title = _strip_html(title_match.group(2))
        snippet = _strip_html(snippet_match.group(1)) if snippet_match else ""

        # DuckDuckGo wraps URLs in a redirect — extract the real URL
        ud_match = re.search(r"uddg=([^&]+)", href)
        real_url = (
            urllib.parse.unquote(ud_match.group(1))
            if ud_match
            else href
        )

        if real_url and title:
            results.append({
                "title": title,
                "url": real_url,
                "snippet": snippet,
            })

        if len(results) >= num_results:
            break

    if not results:
        return (
            f"No results found for query: {query}\n"
            "Tip: Try a shorter or more specific query."
        )

    lines = [
        f"Search results for: {query}\n"
        + "-" * 60
    ]

    for i, r in enumerate(results, 1):
        lines.append(
            f"\n[{i}] {r['title']}\n"
            f"    URL: {r['url']}\n"
            f"    {r['snippet']}"
        )

    return "\n".join(lines)


def web_fetch(
    url: str,
    max_chars: Optional[int] = 8000,
    timeout: int = 15,
) -> str:
    """
    Fetch a URL and return cleaned plain-text content.
    Strips HTML tags, scripts, and styles.
    Truncates to max_chars if set.
    """

    if not url.startswith(("http://", "https://")):
        return f"ERROR: URL must start with http:// or https://. Got: {url}"

    status, content_type, body = _fetch_raw(url, timeout=timeout)

    if body.startswith(("HTTP_ERROR", "URL_ERROR", "FETCH_ERROR")):
        return f"FETCH_ERROR — {body}"

    if status == 403:
        return (
            f"ERROR 403 Forbidden: Access denied for {url}\n"
            "The server blocked this request."
        )

    if status == 404:
        return f"ERROR 404 Not Found: {url}"

    if status >= 400:
        return f"ERROR HTTP {status}: {url}"

    # Strip HTML if content is HTML
    if "html" in content_type.lower() or body.lstrip().startswith("<"):
        text = _strip_html(body)
    else:
        text = body  # JSON, plain text, etc. returned as-is

    total_chars = len(text)

    if max_chars and total_chars > max_chars:
        text = text[:max_chars]
        truncation_note = (
            f"\n\n[TRUNCATED — showing {max_chars} of "
            f"{total_chars} chars]"
        )
    else:
        truncation_note = ""

    return (
        f"[URL: {url}]\n"
        f"[Status: {status} | Content-Type: {content_type}]\n"
        f"[Length: {total_chars} chars]\n\n"
        f"{text}"
        f"{truncation_note}"
    )


# =========================================================
# TOOL REGISTRY ENTRIES
# =========================================================

WEB_TOOLS = {
    "web_search": {
        "func": web_search,
        "schema": WebSearchSchema,
        "description": (
            "Search the web for a query. Returns titles, URLs, and "
            "snippets for the top results. Uses DuckDuckGo by default."
        ),
    },
    "web_fetch": {
        "func": web_fetch,
        "schema": WebFetchSchema,
        "description": (
            "Fetch a URL and return its plain-text content. "
            "Strips HTML tags and scripts. Truncates large pages."
        ),
    },
}
