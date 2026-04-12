"""
marketaux.py — Marketaux finance news provider.

API docs : https://www.marketaux.com/documentation
Auth     : ?api_token=<key>  (query param)
Endpoint : https://api.marketaux.com/v1/news/all

SGX coverage notes:
  - Per-ticker symbol search (symbols=D05.SI): returns 0 results — SGX not indexed
  - Keyword/text search (search=company_name): works well ✓

Strategy: Search by company name (and ticker as fallback). Returns relevant
financial news with per-entity sentiment scores.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import aiohttp
from loguru import logger

from .base import BaseProvider, ProviderArticle

API_URL = "https://api.marketaux.com/v1/news/all"
TIMEOUT = aiohttp.ClientTimeout(total=15)


class MarketauxProvider(BaseProvider):
    NAME     = "Marketaux"
    CATEGORY = "news"

    async def fetch(self, ticker: str, name: str = "") -> list[ProviderArticle]:
        if not self._guard():
            return []

        published_after = (
            datetime.now(tz=timezone.utc) - timedelta(hours=24)
        ).strftime("%Y-%m-%dT%H:%M")

        # Build search terms: prefer company name, fall back to ticker
        search_terms: list[str] = []
        if name:
            search_terms.append(name)
            # Also try just the first meaningful word (e.g. "DBS" from "DBS Group")
            first_word = name.split()[0]
            if len(first_word) > 3 and first_word != name:
                search_terms.append(first_word)
        search_terms.append(ticker)

        all_articles: list[ProviderArticle] = []
        seen_urls: set[str] = set()

        for term in search_terms:
            hits = await self._fetch_by_keyword(term, ticker, published_after)
            for a in hits:
                if a["url"] not in seen_urls:
                    seen_urls.add(a["url"])
                    all_articles.append(a)
            # Stop early if we have enough
            if len(all_articles) >= int(self._cfg.get("limit", 20)):
                break

        if all_articles:
            logger.info(f"[{self.NAME}] {len(all_articles)} articles for {ticker} ({name or ticker})")
        else:
            logger.debug(f"[{self.NAME}] No articles for {ticker} ({name or ticker})")

        return all_articles

    async def _fetch_by_keyword(
        self, keyword: str, ticker: str, published_after: str
    ) -> list[ProviderArticle]:
        params = {
            "search":         keyword,
            "published_after": published_after,
            "language":       "en",
            "limit":          str(self._cfg.get("limit", 20)),
            "api_token":      self.api_key,
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

        articles: list[ProviderArticle] = []
        for item in data.get("data", []):
            sentiment = self._extract_sentiment(item, ticker)
            articles.append(
                self._article(
                    headline=item.get("title", ""),
                    url=item.get("url", ""),
                    summary=item.get("description") or item.get("snippet") or "",
                    published_at=self._normalise_iso(item.get("published_at", "")),
                    sentiment=sentiment,
                    ticker=ticker,
                    tags=item.get("keywords", []),
                )
            )
        return articles

    @staticmethod
    def _extract_sentiment(item: dict[str, Any], ticker: str) -> float | None:
        """Pull per-entity sentiment score; fall back to article-level score."""
        for entity in item.get("entities", []):
            sym = entity.get("symbol", "").upper()
            if sym in (ticker.upper(), f"{ticker}.SI".upper()):
                raw = entity.get("sentiment_score")
                if raw is not None:
                    try:
                        return float(raw)
                    except (TypeError, ValueError):
                        pass
        # Fall back to first entity's sentiment
        for entity in item.get("entities", []):
            raw = entity.get("sentiment_score")
            if raw is not None:
                try:
                    return float(raw)
                except (TypeError, ValueError):
                    pass
        return None
