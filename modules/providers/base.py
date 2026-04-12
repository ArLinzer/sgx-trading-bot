"""
base.py — Abstract base class for all news/data providers.

Every provider must:
  - Declare a unique NAME and CATEGORY
  - Implement fetch(ticker) → list[ProviderArticle]
  - Read its API key and enabled flag from the config dict passed at init

ProviderArticle is the canonical dict shape used throughout the pipeline.
news_fetcher.py normalises all provider output to this schema before dedup.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


# ---------------------------------------------------------------------------
# Canonical article schema (TypedDict-style, kept as plain dict for compat)
# ---------------------------------------------------------------------------
#
# {
#   "source"       : str   — human-readable provider name
#   "headline"     : str
#   "summary"      : str   — plain-text body / excerpt (max 500 chars)
#   "url"          : str
#   "published_at" : str   — ISO-8601 UTC  e.g. "2024-04-12T08:30:00+00:00"
#   "sentiment"    : float | None  — -1.0 (bearish) … +1.0 (bullish), None if unavailable
#   "ticker"       : str   — the ticker this article was fetched for
#   "tags"         : list[str]
# }

ProviderArticle = dict[str, Any]


# ---------------------------------------------------------------------------
# Base provider
# ---------------------------------------------------------------------------

class BaseProvider(ABC):
    """
    Subclass this for every data provider.

    Attributes
    ----------
    NAME     : short identifier used in logs and config  e.g. "marketaux"
    CATEGORY : "news" | "ohlcv" | "sentiment" | "quote"
    """

    NAME: str = "base"
    CATEGORY: str = "news"

    def __init__(self, cfg: dict[str, Any]):
        """
        Args:
            cfg: The provider's own config block from config.yaml, e.g.
                 { "enabled": true, "api_key": "abc123", ... }
        """
        self._cfg = cfg
        self.enabled: bool = bool(cfg.get("enabled", False))
        self.api_key: str = str(cfg.get("api_key") or cfg.get("api_token") or cfg.get("token") or "")

    # ------------------------------------------------------------------
    # Interface every provider must implement
    # ------------------------------------------------------------------

    @abstractmethod
    async def fetch(self, ticker: str) -> list[ProviderArticle]:
        """
        Fetch articles / data for the given SGX ticker.

        Args:
            ticker: SGX stock code, e.g. "D05"

        Returns:
            List of ProviderArticle dicts.  Return [] on error — never raise.
        """

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _article(
        self,
        headline: str,
        url: str = "",
        summary: str = "",
        published_at: str = "",
        sentiment: Optional[float] = None,
        ticker: str = "",
        tags: Optional[list] = None,
    ) -> ProviderArticle:
        """Convenience constructor that fills in the source name automatically."""
        return {
            "source": self.NAME,
            "headline": headline.strip(),
            "summary": summary.strip()[:500],
            "url": url.strip(),
            "published_at": published_at,
            "sentiment": sentiment,
            "ticker": ticker,
            "tags": tags or [],
        }

    def _guard(self) -> bool:
        """Return False (and log nothing) if provider is disabled or has no key."""
        return self.enabled and bool(self.api_key)

    @staticmethod
    def _normalise_iso(value: str) -> str:
        if not value:
            return ""
        try:
            from dateutil import parser as dp
            from datetime import timezone
            parsed = dp.parse(value, fuzzy=True)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc).isoformat()
        except Exception:
            return value

    def __repr__(self) -> str:
        status = "ON" if self.enabled else "OFF"
        return f"<{self.__class__.__name__} name={self.NAME} status={status}>"
