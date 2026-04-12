"""
web_search.py — Web search tool for Ollama tool-calling.

Backed by DuckDuckGo (no API key required).
Returns structured results that are fed back to the LLM as tool responses.

Install: pip install ddgs
  (The old package name was duckduckgo-search; it has been renamed to ddgs)
"""

from __future__ import annotations

import asyncio
import random
import time
from typing import Any

from loguru import logger

# ---------------------------------------------------------------------------
# Package detection — support both the new `ddgs` and legacy `duckduckgo_search`
# ---------------------------------------------------------------------------

DDG_AVAILABLE = False
_ddg_impl: str = "none"  # "ddgs" | "legacy" | "none"

try:
    from ddgs import DDGS as _DDGS_new   # type: ignore
    DDG_AVAILABLE = True
    _ddg_impl = "ddgs"
    logger.debug("[WebSearch] Using 'ddgs' package for web search")
except ImportError:
    _DDGS_new = None  # type: ignore

if not DDG_AVAILABLE:
    try:
        from duckduckgo_search import DDGS as _DDGS_legacy  # type: ignore
        DDG_AVAILABLE = True
        _ddg_impl = "legacy"
        logger.debug("[WebSearch] Using 'duckduckgo_search' (legacy) package for web search")
    except ImportError:
        _DDGS_legacy = None  # type: ignore

if not DDG_AVAILABLE:
    logger.warning("[WebSearch] Web search not available — install with: pip3 install ddgs")


def _DDGS():  # type: ignore
    """Return an instance of whichever DDGS class is available."""
    if _ddg_impl == "ddgs":
        return _DDGS_new()
    if _ddg_impl == "legacy":
        return _DDGS_legacy()
    raise RuntimeError("No DDGS implementation available")


# ---------------------------------------------------------------------------
# Rate-limit / backoff state
# ---------------------------------------------------------------------------

# Minimum gap (seconds) between consecutive DDG HTTP calls
_DDG_MIN_INTERVAL: float = 4.0
_last_ddg_call: float = 0.0

# If we hit a hard error (proto failure, repeated 403), pause the whole
# module for this many seconds before trying again.
_DDG_COOLDOWN_UNTIL: float = 0.0
_DDG_COOLDOWN_SECS: float = 120.0   # 2 min cool-down after repeated failures


def _ddg_wait() -> None:
    """Enforce minimum spacing between DuckDuckGo requests (called in-thread)."""
    global _last_ddg_call
    now = time.monotonic()
    gap = _DDG_MIN_INTERVAL - (now - _last_ddg_call)
    if gap > 0:
        time.sleep(gap + random.uniform(0.2, 0.8))
    _last_ddg_call = time.monotonic()


def _is_fatal_error(exc: Exception) -> bool:
    """
    Return True for errors that should trigger a longer cool-down rather than
    an immediate retry:
      - HTTP/2 TLS protocol errors ("Unsupported protocol version 0x304")
      - Persistent 403 / RateLimit responses
    """
    s = str(exc).lower()
    return (
        "unsupported protocol version" in s
        or "protocol" in s and "0x30" in s
        or s.count("ratelimit") > 0
        or "429" in s
    )


# ---------------------------------------------------------------------------
# Ollama tool definitions (passed in the `tools` param of ollama.chat)
# ---------------------------------------------------------------------------

WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "Search the internet for the latest news, prices, announcements, or any "
            "real-time information about a stock, company, or market event. "
            "Use this whenever you need up-to-date data that may not be in your training. "
            "Returns the top search results with titles, snippets, and URLs."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "The search query. Be specific — include company name, ticker, "
                        "and what you are looking for. E.g. 'DBS Group D05 SGX latest earnings 2024'"
                    ),
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of results to return (1-10). Default 5.",
                },
            },
            "required": ["query"],
        },
    },
}

NEWS_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "news_search",
        "description": (
            "Search for the latest financial news, press releases, and SGX announcements "
            "for a specific stock or market topic. Focused on news results only."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "News search query, e.g. 'DBS Group SGX dividend announcement 2024'",
                },
                "max_results": {
                    "type": "integer",
                    "description": "Number of news results to return (1-10). Default 5.",
                },
            },
            "required": ["query"],
        },
    },
}

# All tools exposed to the LLM
ALL_TOOLS = [WEB_SEARCH_TOOL, NEWS_SEARCH_TOOL]


# ---------------------------------------------------------------------------
# Tool executor
# ---------------------------------------------------------------------------

async def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """
    Execute a tool call requested by the LLM.

    Args:
        tool_name:  "web_search" or "news_search"
        arguments:  Dict of arguments from the LLM tool call

    Returns:
        String result to feed back to the LLM as a tool response.
    """
    global _DDG_COOLDOWN_UNTIL

    query = arguments.get("query", "")
    max_results = int(arguments.get("max_results", 5))
    max_results = max(1, min(10, max_results))

    if not DDG_AVAILABLE:
        return "Web search unavailable — install with: pip3 install ddgs"

    if not query:
        return "Error: no query provided."

    # Check if we are in a cool-down period
    now = time.monotonic()
    if now < _DDG_COOLDOWN_UNTIL:
        remaining = int(_DDG_COOLDOWN_UNTIL - now)
        logger.info(f"[WebSearch] Skipping search — cooling down ({remaining}s left). Query: {query!r}")
        return (
            "Web search temporarily paused to avoid rate limiting. "
            "Proceeding with data already available."
        )

    logger.info(f"[WebSearch] Tool call: {tool_name}({query!r}, max={max_results})")

    last_err: Exception | None = None
    for attempt in range(1, 4):
        try:
            if tool_name == "news_search":
                results = await _ddg_news(query, max_results)
            else:
                results = await _ddg_web(query, max_results)

            if not results:
                logger.info(f"[WebSearch] No results for: {query!r}")
                return f"No results found for: {query}"

            # Format results as readable text for the LLM
            lines: list[str] = [f"Search results for: {query}\n"]
            for i, r in enumerate(results, 1):
                title = r.get("title", "")
                body  = r.get("body") or r.get("snippet") or r.get("description") or ""
                url   = r.get("url") or r.get("href") or r.get("link") or ""
                date  = r.get("date") or r.get("published") or ""
                lines.append(f"[{i}] {title}")
                if date:
                    lines.append(f"    Date: {date}")
                if body:
                    lines.append(f"    {body[:300]}")
                if url:
                    lines.append(f"    URL: {url}")
                lines.append("")

            result_text = "\n".join(lines)
            logger.info(f"[WebSearch] Returned {len(results)} results ({len(result_text)} chars)")
            return result_text

        except Exception as exc:
            last_err = exc
            err_lower = str(exc).lower()

            if _is_fatal_error(exc):
                _DDG_COOLDOWN_UNTIL = time.monotonic() + _DDG_COOLDOWN_SECS
                logger.warning(
                    f"[WebSearch] Persistent error from DuckDuckGo "
                    f"({exc.__class__.__name__}: {exc}). "
                    f"Pausing web search for {int(_DDG_COOLDOWN_SECS)}s to avoid further blocks."
                )
                return (
                    "Web search temporarily unavailable (DuckDuckGo rate limit / network issue). "
                    "The LLM will proceed using the financial data already fetched from Finnhub and other sources."
                )

            # Transient error — wait and retry
            wait = 6 * attempt + random.uniform(1, 4)
            logger.warning(
                f"[WebSearch] Attempt {attempt}/3 failed: {exc} — retrying in {wait:.1f}s ..."
            )
            await asyncio.sleep(wait)

    logger.warning(f"[WebSearch] All retries exhausted: {last_err}")
    return (
        "Web search temporarily unavailable. "
        "Proceeding with data already available from Finnhub and other sources."
    )


async def _ddg_web(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run DuckDuckGo web search in a thread executor (SDK is synchronous)."""
    loop = asyncio.get_event_loop()

    def _search() -> list[dict[str, Any]]:
        _ddg_wait()
        with _DDGS() as ddgs:
            return list(ddgs.text(query, max_results=max_results))

    return await loop.run_in_executor(None, _search)


async def _ddg_news(query: str, max_results: int) -> list[dict[str, Any]]:
    """Run DuckDuckGo news search in a thread executor."""
    loop = asyncio.get_event_loop()

    def _search() -> list[dict[str, Any]]:
        _ddg_wait()
        with _DDGS() as ddgs:
            return list(ddgs.news(query, max_results=max_results))

    return await loop.run_in_executor(None, _search)
