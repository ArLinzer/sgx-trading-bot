"""
eodhd.py — EODHD Financial News API provider.

SGX coverage notes:
  - Per-ticker news (/api/news?s=TICKER.SG): returns 404 — SGX not in EODHD exchange list
  - Per-ticker EOD price (/api/eod/TICKER.SG): returns 404 — same reason
  - Text/tag news search (/api/news?t=KEYWORD): works ✓
  - General news feed (/api/news): works ✓

Strategy: Use EODHD's keyword news search to find articles about the company
by name and ticker. Falls back to general news filtered in-memory.
"""

from __future__ import annotations

from typing import Any

import aiohttp
from loguru import logger

from .base import BaseProvider, ProviderArticle

API_URL   = "https://eodhd.com/api/news"
TIMEOUT   = aiohttp.ClientTimeout(total=15)


class EODHDProvider(BaseProvider):
    NAME     = "EODHD"
    CATEGORY = "news"

    async def fetch(self, ticker: str, name: str = "") -> list[ProviderArticle]:
        if not self._guard():
            return []

        articles: list[ProviderArticle] = []

        # Build search terms: ticker + meaningful words from the company name
        search_terms: list[str] = [ticker]
        if name:
            meaningful = [w for w in name.split() if len(w) > 3]
            search_terms.extend(meaningful[:2])  # take at most 2 name words

        for term in search_terms:
            hits = await self._fetch_by_keyword(term, ticker)
            # Deduplicate by URL
            existing_urls = {a["url"] for a in articles}
            articles.extend(a for a in hits if a["url"] not in existing_urls)

        if articles:
            logger.info(f"[{self.NAME}] {len(articles)} articles for {ticker}")
        else:
            logger.debug(f"[{self.NAME}] No articles found for {ticker}")

        return articles

    async def _fetch_by_keyword(self, keyword: str, ticker: str) -> list[ProviderArticle]:
        """
        Search EODHD news by keyword/tag (the `t` parameter).
        Returns matching articles attributed to `ticker`.
        """
        params = {
            "t":         keyword,
            "limit":     str(self._cfg.get("limit", 20)),
            "api_token": self.api_key,
            "fmt":       "json",
        }

        try:
            async with aiohttp.ClientSession(timeout=TIMEOUT) as session:
                async with session.get(API_URL, params=params) as resp:
                    if resp.status != 200:
                        logger.debug(f"[{self.NAME}] HTTP {resp.status} for keyword '{keyword}'")
                        return []
                    data = await resp.json(content_type=None)
        except Exception as exc:
            logger.debug(f"[{self.NAME}] Request failed for keyword '{keyword}': {exc}")
            return []

        if not isinstance(data, list):
            return []

        return [
            self._article(
                headline=item.get("title", ""),
                url=item.get("link", ""),
                summary=item.get("content", "")[:500],
                published_at=self._normalise_iso(item.get("date", "")),
                sentiment=self._parse_sentiment(item.get("sentiment")),
                ticker=ticker,
                tags=item.get("tags", []),
            )
            for item in data
            if item.get("title")
        ]

    @staticmethod
    def _parse_sentiment(value: Any) -> float | None:
        if isinstance(value, dict):
            raw = value.get("polarity")
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        if isinstance(value, (int, float)):
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
        return None
